import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F

class Cnn(nn):
    def __init__(self,n_classes, out1 = 10, linear_size = 100):
        self.n_classes = n_classes
        self.linear_size = linear_size
        self.out1 = out1
        self.conv1 = nn.Conv2d(1,10,out1)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(out1,2*out1)
        self.conv3 = nn.Conv2d(2*out1,4*out1)
        self.linear1 = nn.Linear(linear_size,2)

    
    def forward(self,x):
        out = F.relu(self.conv1(x))
        out = self.pool(out)
        out = F.relu(self.conv2)
        out = self.pool(out)
        out = F.relu(self.conv3(out))
        out = F.relu(self.linear1(out))
        return out


