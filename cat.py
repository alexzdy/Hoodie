import os
import torch

os.makedirs("./log/latent/cat-upper/latentz", exist_ok=True)
for file in os.listdir("./log/latent/human-upper-2048/latenz"):
    human = torch.load(f"./log/latent/human-upper-2048/latenz/{file}")
    upper = torch.load(f"./log/latent/garment-upper-2048/latenz/{file}")
    print(human.shape, upper.shape)
    cat = torch.cat((human, upper), dim=0)
    print(cat.shape)
    torch.save(cat, f"./log/latent/cat-upper/latentz/{file}")