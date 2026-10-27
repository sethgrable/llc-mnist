import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import DigitNet
from llc import estimate_llc


device = "mps" if torch.backends.mps.is_available() else "cpu"
torch.manual_seed(0)

train_data = datasets.MNIST(root="data", train=True, download=True, transform=transforms.ToTensor())
loader = DataLoader(train_data, batch_size=512, shuffle=True)
n = len(train_data)

import json

results = []
for width in [16, 32, 64, 128, 256]:
    model = DigitNet(width).to(device)
    model.load_state_dict(torch.load(f"models/digitnet_w{width}.pth", map_location=device))
    d = sum(p.numel() for p in model.parameters())

    llcs, traces, loss_star = estimate_llc(model, loader, nn.CrossEntropyLoss(), n,
                                           epsilon=1e-5, gamma=100.0,
                                           num_chains=4, num_steps=6000, burn_in=2500,
                                           device=device)
    mean = sum(llcs) / len(llcs)
    std = (sum((v - mean)**2 for v in llcs) / (len(llcs) - 1)) ** 0.5   # spread across chains
    print(f"width {width:4d} | d/2 {d/2:8.0f} | LLC {mean:7.1f} ± {std:5.1f} | chains {[round(v, 1) for v in llcs]}")

    results.append({"width": width, "d": d, "llc_mean": mean, "llc_std": std, # one dict per width
                    "llc_chains": llcs, "loss_star": loss_star})

# json.dump: saves the results to a text file, so plotting later doesn't require rerunning
with open("results.json", "w") as f:
    json.dump(results, f, indent=2)