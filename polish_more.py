import json
import math
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import DigitNet
from llc import estimate_llc

device = "mps" if torch.backends.mps.is_available() else "cpu"
torch.manual_seed(2)

train_data = datasets.MNIST(root="data", train=True, download=True, transform=transforms.ToTensor())
loader = DataLoader(train_data, batch_size=512, shuffle=True)
n = len(train_data)
loss_fn = nn.CrossEntropyLoss()


def full_loss(model):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            total += loss_fn(model(x), y).item() * len(x)
    return total / n


def weight_norm(model):
    # length of the full weight vector: sqrt of the sum of every squared parameter
    return math.sqrt(sum((p**2).sum().item() for p in model.parameters()))


def polish(model, epochs, lr=1e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


width = 256
model = DigitNet(width).to(device)
model.load_state_dict(torch.load(f"models/digitnet_w{width}_polished.pth", map_location=device))

rows = []
total_extra = 0
for round_ in range(2):
    polish(model, epochs=5)
    total_extra += 5
    loss = full_loss(model)
    norm = weight_norm(model)
    llcs, traces, loss_star = estimate_llc(model, loader, loss_fn, n,
                                           epsilon=1e-5, gamma=100.0,
                                           num_chains=4, num_steps=6000, burn_in=2500,
                                           device=device)
    mean = sum(llcs) / len(llcs)
    print(f"+{total_extra:2d} epochs | train loss {loss:.5f} | weight norm {norm:6.1f} "
          f"| LLC {mean:6.1f} | chains {[round(v, 1) for v in llcs]}")
    rows.append({"extra_epochs": total_extra, "train_loss": loss, "weight_norm": norm,
                 "llc_mean": mean, "llc_chains": llcs})
    torch.save(model.state_dict(), f"models/digitnet_w{width}_polished_plus{total_extra}.pth")

with open("polish_more.json", "w") as f:
    json.dump(rows, f, indent=2)