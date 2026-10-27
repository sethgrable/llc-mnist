import json
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import DigitNet
from llc import estimate_llc
from train import train_model, test_accuracy, device

train_data = datasets.MNIST(root="data", train=True, download=True, transform=transforms.ToTensor())
loader = DataLoader(train_data, batch_size=512, shuffle=True)
n = len(train_data)


def llc_of(model, gamma):
    llcs, traces, loss_star = estimate_llc(model, loader, nn.CrossEntropyLoss(), n,
                                           epsilon=1e-5, gamma=gamma,
                                           num_chains=4, num_steps=6000, burn_in=2500,
                                           device=device)
    return sum(llcs) / len(llcs), llcs


rows = []

# Check A: localization strength. Same models as before (seed 0), gamma 300 instead of 100.
for width in [16, 256]:
    model = DigitNet(width).to(device)
    model.load_state_dict(torch.load(f"models/digitnet_w{width}.pth", map_location=device))
    mean, chains = llc_of(model, gamma=300.0)
    print(f"[gamma 300] width {width:4d} | LLC {mean:6.1f} | chains {[round(v, 1) for v in chains]}")
    rows.append({"check": "gamma", "width": width, "seed": 0, "gamma": 300.0,
                 "llc_mean": mean, "llc_chains": chains})

# Check B: training seed. Retrain with seed 1, gamma back to 100.
for width in [16, 256]:
    torch.manual_seed(1)
    model = train_model(width)
    acc = test_accuracy(model)
    torch.save(model.state_dict(), f"models/digitnet_w{width}_seed1.pth")
    mean, chains = llc_of(model, gamma=100.0)
    print(f"[seed 1]    width {width:4d} | acc {acc:.2%} | LLC {mean:6.1f} | chains {[round(v, 1) for v in chains]}")
    rows.append({"check": "seed", "width": width, "seed": 1, "gamma": 100.0,
                 "test_acc": acc, "llc_mean": mean, "llc_chains": chains})

with open("robustness.json", "w") as f:
    json.dump(rows, f, indent=2)