import os
import sys
import argparse
import torch
import torch.utils.tensorboard
from data.dataset import *
from utils.misc import *
from data.data import *
from models.vae_flow import *
from models.flow import add_spectral_norm
from ddpm import Diffusion1d, Unet1d

# Arguments
parser = argparse.ArgumentParser()
# Model arguments
parser.add_argument("--model", type=str, default="flow")
parser.add_argument("--latent_dim", type=int, default=256)
parser.add_argument("--num_steps", type=int, default=100)
parser.add_argument("--beta_1", type=float, default=1e-4)
parser.add_argument("--beta_T", type=float, default=0.02)
parser.add_argument("--sched_mode", type=str, default="linear")
parser.add_argument("--flexibility", type=float, default=0.0)
parser.add_argument("--truncate_std", type=float, default=2.0)
parser.add_argument("--latent_flow_depth", type=int, default=14)
parser.add_argument("--latent_flow_hidden_dim", type=int, default=256)
parser.add_argument("--kl_weight", type=float, default=0.0001)
parser.add_argument("--residual", type=eval, default=True, choices=[True, False])
parser.add_argument("--spectral_norm", type=eval, default=False, choices=[True, False])
parser.add_argument("--seed", type=int, default=3047, help="3047")
parser.add_argument("--logging", type=eval, default=True, choices=[True, False])
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--tag", type=str, default=None)
parser.add_argument("--log_root", type=str, default="./log/Joint-generation")
parser.add_argument("--num_samples", type=int, default=32)
parser.add_argument("--sample_num_points", type=int, default=2048)
parser.add_argument(
    "--categories",
    type=str_list,
    # default=["upper-2048"],
    default=["dress-2048"],
    help="[upper-2048, dress-2048]",
)
parser.add_argument(
    "--stage1_human",
    type=str,
    # default="./ckpt/human-upper/ckpt_0.000000_2550000.pt",
    default="./ckpt/human-dress/ckpt_0.000000_3320000.pt",
)
parser.add_argument(
    "--stage1_garment",
    type=str,
    # default="./ckpt/garment-upper/ckpt_0.000000_2550000.pt",
    default="./ckpt/garment-dress/ckpt_0.000000_4300000.pt",
)
parser.add_argument(
    "--stage2_diffusion",
    type=str,
    # default="./ckpt/1d-upper/model_best.pt",
    default="./ckpt/1d-dress/model_best.pt",
)

args = parser.parse_args()
seed_all(args.seed)
# Logging
if args.logging:
    log_dir = get_new_log_dir(
        args.log_root,
        prefix=args.categories[0],
        postfix="_" + args.tag if args.tag is not None else "",
    )
    logger = get_logger("Joint-generation", log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    ckpt_mgr = CheckpointManager(log_dir)
    log_hyperparams(writer, args)
else:
    logger = get_logger("Joint-generation", None)
    writer = BlackHole()
    ckpt_mgr = BlackHole()
logger.info(args)
# Statistic
stats_save_path_human = os.path.join(
    "./data",
    "human-%s-2048_stats" % args.categories[0].split("-")[0],
    "stats_human-%s-2048.pt" % args.categories[0].split("-")[0],
)
stats_save_path_garment = os.path.join(
    "./data",
    "garment-%s-2048_stats" % args.categories[0].split("-")[0],
    "stats_garment-%s-2048.pt" % args.categories[0].split("-")[0],
)
stats_human = torch.load(stats_save_path_human)
mean_human, std_human = stats_human["mean"], stats_human["std"]
mean_human, std_human = mean_human.to(args.device), std_human.to(args.device)
stats_garment = torch.load(stats_save_path_garment)
mean_garment, std_garment = stats_garment["mean"], stats_garment["std"]
mean_garment, std_garment = mean_garment.to(args.device), std_garment.to(args.device)
# Model
logger.info("Building model...")
model_human = FlowVAE(args).to(args.device)
model_garment = FlowVAE(args).to(args.device)
logger.info(repr(model_human))
logger.info(repr(model_garment))
if args.spectral_norm:
    add_spectral_norm(model_human, logger=logger)
    add_spectral_norm(model_garment, logger=logger)
# latent diffusion
latent_model = Unet1d(64)
diffuser = Diffusion1d(
    time_steps=1000,
    sample_steps=1000,
    model=latent_model,
    device=args.device,
    model_path=args.stage2_diffusion,
)
sampler = diffuser.sampling_z
# Resume
logger.info("Resuming model...")
resume_path_human = args.stage1_human
resume_path_garment = args.stage1_garment
resume_dict_human = torch.load(args.stage1_human)
resume_dict_garment = torch.load(args.stage1_garment)
model_human.load_state_dict(resume_dict_human["state_dict"])
model_garment.load_state_dict(resume_dict_garment["state_dict"])


def validate_inspect(z1, z2):
    x1 = model_human.sample_existz(
        z1, args.sample_num_points, flexibility=args.flexibility
    )
    x2 = model_garment.sample_existz(
        z2, args.sample_num_points, flexibility=args.flexibility
    )

    # De-normalized
    x1 = std_human * x1 + mean_human
    x2 = std_garment * x2 + mean_garment

    x3 = torch.cat((x1, x2), dim=1)

    return x1, x2, x3


logger.info("[Inspect] Generating samples...")
save_path = os.path.join(log_dir, "sample")
os.makedirs(save_path, exist_ok=True)
# Sample
z, _ = sampler(torch.Size([args.num_samples, 1, 512]))
z1, z2 = z.split([256, 256], dim=-1)
z1, z2 = z1.squeeze(1), z2.squeeze(1)
x1, x2, x3 = validate_inspect(z1, z2)
for i in range(x1.size(0)):
    np.savetxt(
        os.path.join(save_path, "human_%d.xyz" % i), x1[i].squeeze(0).cpu().numpy()
    )
    np.savetxt(
        os.path.join(save_path, "%s_%d.xyz" % (args.categories[0].split("-")[0], i)),
        x2[i].squeeze(0).cpu().numpy(),
    )
    np.savetxt(
        os.path.join(
            save_path,
            "dressed-avatar_%d.xyz" % i,
        ),
        x3[i].squeeze(0).cpu().numpy(),
    )
