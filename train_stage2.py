import os
from ddpm import Diffusion1d, Unet1d, Trainer1d, get_dataloader
import argparse
import warnings

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    device = "cuda"
    model = Unet1d(64)
    diffuser = Diffusion1d(
        time_steps=1000, sample_steps=1000, model=model, device=device
    )
    sampler = diffuser.sampling_sequence
    parser = argparse.ArgumentParser()
    # dataset
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_epoch", type=int, default=150000)
    parser.add_argument("--cat_path", type=str, default="./log/latent/cat-upper/latentz")
    parser.add_argument("--save_model", type=str, default="./log/stage2/upper-2048")
    
    args = parser.parse_args()
    
    train_dataloader = get_dataloader(args.batch_size, args.cat_path)
    trainer = Trainer1d(
        dataloader=train_dataloader,
        batch_size=args.batch_size,
        epochs=args.max_epoch,
        device=device,
        diffuser=diffuser,
        sampler=sampler,
        save_model=args.save_model
    )
    trainer.train()
