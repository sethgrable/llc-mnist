import torch.nn as nn

# Creating a 2 layer nn with relu activation. Width is variable. Flatten takes [64,1,28,28] images to [64,784] where 64 is the batch size, 1 represent one color channel and 28 are pixels.

class DigitNet(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layer1 = nn.Linear(784, width)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(width, 10)


 # Forward is passing x as the input, which will be the images. it starts with flattening then runs through layer1 then relu then layer2. Shapes: [64, 1, 28, 28] → flatten → [64, 784] → layer1 → [64, width] → relu → [64, width] → layer2 → [64, 10]

    def forward(self, x):
        x = self.flatten(x)
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

   