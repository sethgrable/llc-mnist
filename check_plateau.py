import math
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
nbeta = n / math.log(n)

width, eps, steps, window = 256, 1e-5, 12000, 500
model = DigitNet(width).to(device)
model.load_state_dict(torch.load(f"models/digitnet_w{width}.pth", map_location=device))

llcs, traces, loss_star = estimate_llc(model, loader, nn.CrossEntropyLoss(), n,
                                       epsilon=eps, gamma=100.0,
                                       num_chains=2, num_steps=steps, burn_in=steps // 4,
                                       device=device)

for i, t in enumerate(traces):
    lam = [nbeta * (sum(t[s - window:s]) / window - loss_star) for s in range(window, steps)]
    plt.plot(range(window, steps), lam, label=f"chain {i}")
plt.axvline(2500, color="gray", linestyle="--", label="burn-in used")
plt.xlabel("SGLD step")
plt.ylabel("windowed LLC estimate")
plt.title(f"LLC vs time, width {width}, eps {eps:.0e}")
plt.legend()
plt.savefig(f"llc_vs_time_w{width}.png")
plt.show()