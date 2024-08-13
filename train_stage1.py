import os
import warnings

warnings.filterwarnings("ignore")
import argparse
import torch
import torch.utils.tensorboard
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

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
parser.add_argument("--num_samples", type=int, default=48)
parser.add_argument(
    "--sample_num_points", type=int, default=2048
)
parser.add_argument("--kl_weight", type=float, default=0.0001)
parser.add_argument("--residual", type=eval, default=True, choices=[True, False])
parser.add_argument("--spectral_norm", type=eval, default=False, choices=[True, False])
parser.add_argument("--resume_path", type=str, default="")

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
parser.add_argument("--train_batch_size", type=int, default=48)

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
parser.add_argument("--log_root", type=str, default="./log/stage1")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument('--max_iters', type=int, default=3000000)
parser.add_argument("--val_freq", type=int, default=10 * THOUSAND)
parser.add_argument("--tag", type=str, default=None)
args = parser.parse_args()
seed_all(args.seed)

# Logging
if args.logging:
    log_dir = get_new_log_dir(
        args.log_root,
        prefix=f"{args.categories[0]}_{args.seed}_GEN_",
        postfix="_" + args.tag if args.tag is not None else "",
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

# Datasets and loaders
logger.info("Loading datasets...")
# shuffle to train all
train_dset = Pointdata(
    path=args.dataset_path,
    cates=args.categories,
    split=args.split,
    scale_mode=args.scale_mode,
    shuffle=True,
)
# not shuffle to generate z in order
gen_z_dset = Pointdata(
    path=args.dataset_path,
    cates=args.categories,
    split=args.split,
    scale_mode=args.scale_mode,
    shuffle=False,
)
train_iter = get_data_iterator(
    DataLoader(
        train_dset,
        batch_size=args.train_batch_size,
        num_workers=0,
    )
)
gen_z_iter = get_data_iterator(
    DataLoader(
        train_dset,
        batch_size=args.train_batch_size,
        num_workers=0,
    )
)
# Model
logger.info("Building model...")
model = FlowVAE(args).to(args.device)
logger.info(repr(model))
if args.spectral_norm:
    add_spectral_norm(model, logger=logger)

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

def train(it):
    batch = next(train_iter)
    x = batch["pointcloud"].to(args.device)
    # Reset grad and model state
    optimizer.zero_grad()
    model.train()
    if args.spectral_norm:
        spectral_norm_power_iteration(model, n_power_iterations=1)

    # Forward
    kl_weight = args.kl_weight
    loss, loss_entropy, loss_prior, loss_recons = model.get_loss(
        x, kl_weight=kl_weight, writer=writer, it=it
    )

    # Backward and optimize
    loss.backward()
    orig_grad_norm = clip_grad_norm_(model.parameters(), args.max_grad_norm)
    optimizer.step()
    scheduler.step()

    logger.info(
        "[Train] Iter %04d | Loss %.6f | Loss_entropy %.6f | Loss_prior %.6f | Loss_recons %.6f | Grad %.4f"
        % (
            it,
            loss.item(),
            loss_entropy.item(),
            loss_prior.item(),
            loss_recons.item(),
            orig_grad_norm,
        )
    )
    writer.add_scalar("train/loss", loss, it)
    writer.add_scalar("train/loss_entropy", loss_entropy, it)
    writer.add_scalar("train/loss_prior", loss_prior, it)
    writer.add_scalar("train/loss_recons", loss_recons, it)
    writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], it)
    writer.add_scalar("train/grad_norm", orig_grad_norm, it)
    writer.flush()


def validate_inspect(it):
    z = torch.randn([args.num_samples, args.latent_dim]).to(args.device)
    x = model.sample(z, args.sample_num_points, flexibility=args.flexibility)

    # De-normalized
    stats_dir = os.path.join(args.dataset_path, args.categories[0] + "_stats")
    stats_save_path = os.path.join(stats_dir, "stats_" + args.categories[0] + ".pt")
    stats = torch.load(stats_save_path)
    mean = stats["mean"].to(args.device)
    std = stats["std"].to(args.device)
    x = std * x + mean

    writer.add_mesh("pointcloud", x, global_step=it)
    writer.flush()

    save_path = os.path.join(log_dir, "val")
    os.makedirs(save_path, exist_ok=True)
    for i in range(args.num_samples):
        np.savetxt(
            os.path.join(save_path, f"{args.categories[0]}_%d.xyz" % i),
            x[i].cpu().numpy(),
        )

    logger.info("[Inspect] Generating samples...")


# Main loop
logger.info("Start training...")
try:
    it = 0
    while it <= args.max_iters:
        train(it)
        if it % args.val_freq == 0 or it == args.max_iters:
            validate_inspect(it)
            opt_states = {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }
            ckpt_mgr.save(model, args, 0, others=opt_states, step=it)
        it += 1

except KeyboardInterrupt:
    logger.info("Terminating...")
