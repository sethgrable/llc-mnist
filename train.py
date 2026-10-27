import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import DigitNet

device = "mps" if torch.backends.mps.is_available() else "cpu"

# torch.manual_seed(n): fixes the random number generator so runs are reproducible
torch.manual_seed(0)

# transforms.ToTensor() picture into a float tensor of shape [1, 28, 28]

train_data = datasets.MNIST(root="data", train=True, download=True, transform=transforms.ToTensor())
test_data  = datasets.MNIST(root="data", train=False, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(train_data, batch_size=64, shuffle=True) # sequence of tuples (64 images, 64 labels)
test_loader  = DataLoader(test_data, batch_size=1000, shuffle=False)

# Model returns [1000,10] tensor. argmax(dim=1) looks at the largest of the 10 outputs.
def test_accuracy(model):
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            correct += (model(images).argmax(dim=1) == labels).sum().item()
    return correct / len(test_data)


def train_model(width, epochs=10):
    model = DigitNet(width).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#builds a DigitNet of the given width, then runs the five-step loop (predict → loss → zero → backward → step). 60,000 training images gives 938 batches of 64 images per epoch. runs for 10 epochs, and returns the trained model.

    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            loss = loss_fn(model(images), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


# checking the name of train.py. If train is imported into another file its name is no longer the defult __main__. os.makedirs("models", exist_ok=True) creates a folder in your project file.
if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)

# we are not going to check the accuracy of the five widths unless we run the train.py from main

    for width in [16, 32, 64, 128, 256]:
        model = train_model(width)
        d = sum(p.numel() for p in model.parameters())
        acc = test_accuracy(model)
        print(f"width {width:4d} | params {d:7d} | d/2 {d/2:9.1f} | test acc {acc:.2%}")
        torch.save(model.state_dict(), f"models/digitnet_w{width}.pth")

# Saving a dictionary of all the model’s weights and biases, so later scripts can load the trained models.
