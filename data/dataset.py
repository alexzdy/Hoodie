import os
import random
from copy import copy
import torch
from torch.utils.data import Dataset

import numpy as np
import h5py

synsetid_to_cate = {
    "1": "human-upper-2048",
    "2": "garment-upper-2048",
    "3": "human-dress-2048",
    "4": "garment-dress-2048",
}
cate_to_synsetid = {v: k for k, v in synsetid_to_cate.items()}


class Pointdata(Dataset):
    GRAVITATIONAL_AXIS = 1

    def __init__(self, path, cates, split, scale_mode, shuffle, transform=None):
        super().__init__()
        assert isinstance(cates, list), "`cates` must be a list of cate names."
        assert scale_mode is None or scale_mode in (
            "global_unit",
            "shape_unit",
            "shape_bbox",
            "shape_half",
            "shape_34",
            "no",
        )
        self.split = split
        self.path = path
        self.cates = cates
        self.cate_synsetids = [cate_to_synsetid[s] for s in cates]
        self.scale_mode = scale_mode
        self.transform = transform
        self.shuffle = shuffle
        self.pointclouds = []

        self.get_statistics()
        self.load()

    def get_statistics(self):
        for cate in self.cates:
            dsetname = cate
            data_dir = os.path.join(self.path, dsetname)
            stats_dir = os.path.join(self.path, dsetname + "_stats")
            os.makedirs(stats_dir, exist_ok=True)
            stats_save_path = os.path.join(stats_dir, "stats_" + dsetname + ".pt")
            if os.path.exists(stats_save_path):
                self.stats = torch.load(stats_save_path)
                continue
            else:
                pointclouds = []
                cate_files = os.listdir(data_dir)
                for file in cate_files:
                    file = os.path.join(data_dir, file)
                    pcd = np.loadtxt(file)
                    pointclouds.append(torch.from_numpy(np.array(pcd)).unsqueeze(0))
                all_points = torch.cat(pointclouds, dim=0)  # (B, N, 3)
                B, N, _ = all_points.size()
                mean = all_points.view(B * N, -1).mean(dim=0)  # (1, 3)
                std = all_points.view(-1).std(dim=0)  # (1, )
                self.stats = {"mean": mean, "std": std}
                torch.save(self.stats, stats_save_path)

    def load(self):
        path_txt = os.path.join(self.path, self.split + ".txt")

        def _enumerate_pointclouds(f=path_txt):
            idx = -1
            for line in open(f, "r"):
                line = line.strip()
                pcd = np.loadtxt(os.path.join(self.path, self.cates[0], line))
                point = torch.from_numpy(np.array(pcd)).to(torch.float32)
                idx += 1
                yield point, idx

        #'global_unit', 'shape_unit', 'shape_half', 'shape_34', 'no'
        for pc, pc_id in _enumerate_pointclouds():
            if self.scale_mode == "global_unit":
                shift = self.stats["mean"].reshape(1, 3)
                scale = self.stats["std"].reshape(1, 1)
            elif self.scale_mode == "shape_unit":
                shift = pc.mean(dim=0).reshape(1, 3)
                scale = pc.flatten().std().reshape(1, 1)
            elif self.scale_mode == "shape_half":
                shift = pc.mean(dim=0).reshape(1, 3)
                scale = pc.flatten().std().reshape(1, 1) / (0.5)
            elif self.scale_mode == "shape_34":
                shift = pc.mean(dim=0).reshape(1, 3)
                scale = pc.flatten().std().reshape(1, 1) / (0.75)
            elif self.scale_mode == "no":
                shift = torch.zeros([1, 3])
                scale = torch.ones([1, 1])

            pc = ((pc - shift) / scale).to(torch.float32)

            self.pointclouds.append(
                {
                    "pointcloud": pc,
                    "id": pc_id,
                    "shift": shift,
                    "scale": scale,
                }
            )

        # Deterministically shuffle the dataset
        self.pointclouds.sort(key=lambda data: data["id"], reverse=False)
        if self.shuffle:
            random.Random(3047).shuffle(self.pointclouds)

    def __len__(self):
        return len(self.pointclouds)

    def __getitem__(self, idx):
        data = {
            k: v.clone() if isinstance(v, torch.Tensor) else copy(v)
            for k, v in self.pointclouds[idx].items()
        }
        if self.transform is not None:
            data = self.transform(data)
        return data
