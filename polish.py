import json
import math
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import DigitNet
from llc import estimate_llc
from train import device

train_data = datasets.MNIST(root="data", train=True, download=True, transform=transforms.ToTensor())
loader = DataLoader(train_data, batch_size=512, shuffle=True)
n = len(train_data)
nbeta = n / math.log(n)
loss_fn = nn.CrossEntropyLoss()


def full_loss(model):
    # average training loss over all n examples (same as loss_star in llc.py)
    model.eval()
    total = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            total += loss_fn(model(x), y).item() * len(x)
    return total / n


def polish(model, epochs=5, lr=1e-4):
    # extra training at a small learning rate, so w* settles into the minimum
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


rows = []
for width in [16, 256]:
    model = DigitNet(width).to(device)
    model.load_state_dict(torch.load(f"models/digitnet_w{width}.pth", map_location=device))

    before = full_loss(model)
    polish(model)
    after = full_loss(model)
    torch.save(model.state_dict(), f"models/digitnet_w{width}_polished.pth")
    print(f"width {width:4d} | train loss {before:.5f} -> {after:.5f}")

    for gamma in [100.0, 300.0]:
        llcs, traces, loss_star = estimate_llc(model, loader, loss_fn, n,
                                               epsilon=1e-5, gamma=gamma,
                                               num_chains=4, num_steps=6000, burn_in=2500,
                                               device=device)
        mean = sum(llcs) / len(llcs)
        # early offset: LLC formula applied to the first 100 steps only.
        # Negative means the walker immediately found loss below w*, i.e. w* wasn't at the minimum.
        early = [nbeta * (sum(t[:100]) / 100 - loss_star) for t in traces]
        print(f"   gamma {gamma:5.0f} | LLC {mean:6.1f} | chains {[round(v, 1) for v in llcs]} "
              f"| early offset {[round(v, 1) for v in early]}")
        rows.append({"width": width, "gamma": gamma, "loss_before": before, "loss_after": after,
                     "llc_mean": mean, "llc_chains": llcs, "early_offset": early})

with open("polished.json", "w") as f:
    json.dump(rows, f, indent=2)