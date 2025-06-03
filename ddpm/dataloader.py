import os
import torch
from torch.utils.data import Dataset, DataLoader
import glob
import numpy as np
import matplotlib.pyplot as plt


# dataset
class PairedDataset(Dataset):
    def __init__(self, cat_list):
        self.cat_list = cat_list

    def __len__(self):
        return len(self.cat_list)

    def __getitem__(self, idx):
        cat_z = torch.load(self.cat_list[idx]).unsqueeze(0)
        file_name = self.cat_list[idx].split("/")[-1].split(".")[0]
        return cat_z, file_name


def get_dataloader(batch_size: int, data_path: str):
    cat_list = glob.glob(os.path.join(data_path, "*.pt"))
    train_dataset = PairedDataset(cat_list)
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    return train_dataloader


def get_dataloader_test(batch_size: int, data_path: str):
    cat_list = glob.glob(os.path.join(data_path, "*.pt"))
    train_dataset = PairedDataset(cat_list)
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, drop_last=False
    )
    return train_dataloader


if __name__ == "__main__":
    data_loader = get_dataloader(64, "./datasets")
    for data in data_loader:
        print(data.shape)
        print(data.max(), data.min())
        break
