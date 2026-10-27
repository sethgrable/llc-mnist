import json
import math
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import DigitNet
from llc import estimate_llc

device = "mps" if torch.backends.mps.is_available() else "cpu"

train_data = datasets.MNIST(root="data", train=True, download=True, transform=transforms.ToTensor())
loader = DataLoader(train_data, batch_size=512, shuffle=True)
n = len(train_data)
nbeta = n / math.log(n)
loss_fn = nn.CrossEntropyLoss()


def full_loss(model):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            total += loss_fn(model(x), y).item() * len(x)
    return total / n


def polish(model, epochs=5, lr=1e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


results = []
for width in [16, 32, 64, 128, 256]:
    torch.manual_seed(0)       # same seed for every width, so each gets identical treatment
    model = DigitNet(width).to(device)
    model.load_state_dict(torch.load(f"models/digitnet_w{width}.pth", map_location=device))
    d = sum(p.numel() for p in model.parameters())

    before = full_loss(model)
    polish(model)
    after = full_loss(model)

    llcs, traces, loss_star = estimate_llc(model, loader, loss_fn, n,
                                           epsilon=1e-5, gamma=100.0,
                                           num_chains=4, num_steps=6000, burn_in=2500,
                                           device=device)
    mean = sum(llcs) / len(llcs)
    std = (sum((v - mean)**2 for v in llcs) / (len(llcs) - 1)) ** 0.5
    early = [nbeta * (sum(t[:100]) / 100 - loss_star) for t in traces]

    print(f"width {width:4d} | d/2 {d/2:8.0f} | loss {before:.5f} -> {after:.5f} "
          f"| LLC {mean:6.1f} ± {std:4.1f} | early offset {[round(v, 1) for v in early]}")

    results.append({"width": width, "d": d, "loss_before": before, "loss_after": after,
                    "llc_mean": mean, "llc_std": std, "llc_chains": llcs, "early_offset": early})

    # save after every width, so a crash or interruption doesn't lose finished results
    with open("results_polished.json", "w") as f:
        json.dump(results, f, indent=2)