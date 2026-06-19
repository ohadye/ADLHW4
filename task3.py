import torch
import torchvision
from torchvision.transforms import v2
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

seed = 1

torch.manual_seed(seed)
transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                         download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

from Conv import Net

net = Net()

# Pick one random index from each dataset using torch's RNG
train_idx = torch.randint(0, len(trainset), (1,)).item()
test_idx = torch.randint(0, len(testset), (1,)).item()

train_img, train_label = trainset[train_idx]
test_img, test_label = testset[test_idx]

conv_latent1, conv_latent2 = net.task3(train_img.unsqueeze(0))

print(f'{conv_latent1}, {conv_latent2}')
