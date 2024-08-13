import os
import math
import argparse
import torch
import torch.utils.tensorboard
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm

from data.dataset import *
from utils.misc import *
from data.data import *
from models.vae_flow import *
from models.flow import add_spectral_norm, spectral_norm_power_iteration


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
parser.add_argument("--sample_num_points", type=int, default=2048)
parser.add_argument("--kl_weight", type=float, default=0.0001)
parser.add_argument("--residual", type=eval, default=True, choices=[True, False])
parser.add_argument("--spectral_norm", type=eval, default=False, choices=[True, False])
parser.add_argument(
    "--resume_path",
    type=str,
    default="/mnt/zhanggy/Hoodie/log/stage1/human-upper-2048_2020_GEN_2024_05_01__15_18_37/ckpt_0.000000_190000.pt",
)
# Datasets and loaders
parser.add_argument("--dataset_path", type=str, default="./data")
parser.add_argument(
    "--categories",
    type=str_list,
    default=["human-upper-2048"],
    help="[human-upper-2048, garment-upper-2048, human-dress-2048, garment-dress-2048]",
)
parser.add_argument(
    "--split",
    type=str,
    default="upper",
    choices=[
        "upper",
        "dress",
    ],
)
parser.add_argument(
    "--scale_mode",
    type=str,
    default="global_unit",
)
parser.add_argument("--batch_size", type=int, default=48)
# Optimizer and scheduler
parser.add_argument("--lr", type=float, default=2e-3)
parser.add_argument("--weight_decay", type=float, default=0)
parser.add_argument("--max_grad_norm", type=float, default=10)
parser.add_argument("--end_lr", type=float, default=1e-4)
parser.add_argument(
    "--sched_start_epoch",
    type=int,
    default=200 * THOUSAND,
    choices=[True, 200 * THOUSAND],
)
parser.add_argument(
    "--sched_end_epoch",
    type=int,
    default=400 * THOUSAND,
    choices=[True, 400 * THOUSAND],
)

# Training
parser.add_argument("--seed", type=int, default=2020)
parser.add_argument("--logging", type=eval, default=True, choices=[True, False])
parser.add_argument("--log_root", type=str, default="./log/latent")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--tag", type=str, default=None)
args = parser.parse_args()
seed_all(args.seed)

# Logging
if args.logging:
    log_dir = get_new_log_dir(
        os.path.join(args.log_root, args.categories[0]), prefix="save_latent"
    )
    logger = get_logger("train", log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    ckpt_mgr = CheckpointManager(log_dir)
    log_hyperparams(writer, args)
else:
    logger = get_logger("train", None)
    writer = BlackHole()
    ckpt_mgr = BlackHole()
logger.info(args)

logger.info("Loading datasets...")
train_dset = Pointdata(
    path=args.dataset_path,
    cates=args.categories,
    split=args.split,
    scale_mode=args.scale_mode,
    shuffle=True,
)
gen_z_iter = DataLoader(train_dset, batch_size=args.batch_size, shuffle=True)

# Model
logger.info("Building model...")
model = FlowVAE(args).to(args.device)
logger.info(repr(model))
if args.spectral_norm:
    add_spectral_norm(model, logger=logger)

# Resume
logger.info("Resuming model...")
resume_path = args.resume_path
ckpt_resume = torch.load(resume_path)
model.load_state_dict(ckpt_resume["state_dict"])

# Optimizer and scheduler
optimizer = torch.optim.Adam(
    model.parameters(), lr=args.lr, weight_decay=args.weight_decay
)
scheduler = get_linear_scheduler(
    optimizer,
    start_epoch=args.sched_start_epoch,
    end_epoch=args.sched_end_epoch,
    start_lr=args.lr,
    end_lr=args.end_lr,
)

path_txt = f"./data/{args.split}.txt"
with open(path_txt, "r") as file:
    lines = [line.strip() for line in file]
logger.info("Start noise generate pointcloud...")

for data in gen_z_iter:
    x = data["pointcloud"].to(args.device)
    id = data["id"].to(args.device)

    z, pcy = model.pc2z(x, args.flexibility)
    pcy = pcy.cpu()
    stats_dir = os.path.join(args.dataset_path, args.categories[0] + "_stats")
    stats_save_path = os.path.join(stats_dir, "stats_" + args.categories[0] + ".pt")

    if os.path.exists(stats_save_path):
        stats = torch.load(stats_save_path)

    shift = stats["mean"].reshape(1, 3)
    scale = stats["std"].reshape(1, 1)
    pcx = pcy * scale + shift

    pc_path = os.path.join(args.log_root, args.categories[0], "pointcloud")
    z_path = os.path.join(args.log_root, args.categories[0], "latentz")
    os.makedirs(pc_path, exist_ok=True)
    os.makedirs(z_path, exist_ok=True)

    for index, order in enumerate(id):
        filename_pc = f"{pc_path}/{lines[order]}"
        z_name = lines[order].replace(".xyz", ".pt")
        filename_z = f"{z_path}/{z_name}"

        np_array_pc = pcx[index].squeeze(0).numpy()
        torch.save(z[index].squeeze(0), filename_z)
        np.savetxt(filename_pc, np_array_pc, fmt="%f", delimiter=" ")
