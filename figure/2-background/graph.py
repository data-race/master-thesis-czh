import torch.nn as nn
import torch

class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,6,3)
        self.conv2 = nn.Conv2d(3,6,3)
        self.conv3 = nn.Conv2d(3,6,3)
        self.relu1 = nn.ReLU()
        self.linear = nn.Linear(384,20)
    
    def forward(self, input):
        v1 = self.conv1(input)
        v2 = self.conv2(input)
        v3 = self.conv3(input)
        v4 = self.relu1((v1-v2))
        v5 = torch.flatten(v4, start_dim=1)
        v6 = torch.flatten(v3, start_dim=1)
        return self.linear(v5+v6)