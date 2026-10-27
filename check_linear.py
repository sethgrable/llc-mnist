import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from llc import estimate_llc

#checking the estimator returns d/2 for non-singular model (single layer network).

torch.manual_seed(0)

# Data: 10 input features, true linear rule plus Gaussian noise

n, d_in = 10000, 10
X = torch.randn(n, d_in)
y = X @ torch.randn(d_in, 1) + 0.5 + torch.randn(n, 1)

# Train to the minimum (full batch: all n examples each step)
model = nn.Linear(d_in, 1)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
for step in range(2000):
    loss = loss_fn(model(X), y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# TensorDataset(X, y): wraps existing tensors as a Dataset, so data[i] = (X[i], y[i])
loader = DataLoader(TensorDataset(X, y), batch_size=512, shuffle=True)

d = sum(p.numel() for p in model.parameters())
llcs, traces = estimate_llc(model, loader, loss_fn, n,
                            epsilon=1e-5, gamma=100.0,
                            num_steps=2000, burn_in=500)

print(f"d = {d}, expected d/2 = {d/2}")
print("per chain:", [round(v, 2) for v in llcs])
print(f"mean LLC = {sum(llcs)/len(llcs):.2f}")