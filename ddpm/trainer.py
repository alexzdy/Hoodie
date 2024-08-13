import os
import cv2
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt
from .dataloader import get_dataloader

class Trainer1d:
    def __init__(
        self,
        dataloader,
        batch_size,
        epochs: int,
        device,
        diffuser,
        sampler,
        save_model,
    ) -> None:

        self.device = device
        # define diffusion model
        self.diffuser = diffuser
        self.T = self.diffuser.time_steps
        self.forward_diffusion_sample = self.diffuser.forward
        self.unet = self.diffuser.model

        self.sampler = sampler
        self.model_save_dir = save_model
        os.makedirs(self.model_save_dir, exist_ok=True)

        self.optimizer = torch.optim.Adam(self.unet.parameters(), lr=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=1125, T_mult=2, eta_min=1e-6
        )
        self.epochs = epochs
        self.dataloader = dataloader
        self.batch_size = batch_size
        self.losses = []
        self.loss_min = float("inf")

    def get_loss(self, x_0, t):
        x_noisy, noise = self.forward_diffusion_sample(x_0, t)
        noise_pred = self.unet(x_noisy, t)
        return F.l1_loss(noise, noise_pred)

    def save_model_weight(self, epoch):
        torch.save(self.unet.state_dict(), f"{self.model_save_dir}/model_{epoch}.pt")

    def save_model_weight_best(self, epoch):
        torch.save(self.unet.state_dict(), f"{self.model_save_dir}/model_best.pt")

    def save_sampled_image(self, epoch, x_shape: torch.Size):
        sampled_img = self.sampler(x_shape, "image")
        cv2.imwrite(f"{self.model_save_dir}/sampled_{epoch}.jpg", sampled_img)

    def save_sampled_sequence(self, epoch, x_shape: torch.Size):
        sampled_seq = self.sampler(x_shape)

        sampled_seq = sampled_seq.squeeze().detach().cpu().numpy()

        plt.plot(sampled_seq)
        plt.savefig(f"{self.model_save_dir}/sampled_seq_{epoch}.jpg")
        plt.close()

    def plot_metric(self, epoch):
        axis = np.arange(epoch + 1).astype(np.uint8)
        fig = plt.figure()
        plt.title("Loss")
        plt.plot(axis, self.losses)
        plt.legend(labels=["Loss/Epochs"])
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.savefig(f"{self.model_save_dir}/Loss.pdf")
        plt.close(fig)

    def train(self):
        for epoch in range(self.epochs):
            loop = tqdm(self.dataloader, desc=f"Epoch {epoch}")
            loss_list = []
            for data, obs in loop:
                self.optimizer.zero_grad()

                # [0, T)
                t = torch.randint(0, self.T, (self.batch_size,)).to(self.device).long()
                data = data.to(self.device)

                loss = self.get_loss(data, t)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                loop.set_postfix(
                    loss=loss.item(), lr=self.scheduler.get_last_lr()[0], t=t[0].item()
                )

                loss_list.append(loss.item())

            loss_epoch = sum(loss_list) / len(self.dataloader)
            self.losses.append(loss_epoch)
            self.plot_metric(epoch)

            if loss_epoch < self.loss_min:
                self.loss_min = loss_epoch
                self.save_model_weight_best(epoch)

            if (epoch + 1) % 5000 == 0:
                self.save_model_weight(epoch)
