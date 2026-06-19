import torch
import torchvision
from torchvision.transforms import v2
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import math

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

net.eval()

def unnormalize(img_tensor):
    img = img_tensor.clone().detach().cpu()
    img = img * 0.5 + 0.5
    return img.clamp(0, 1)

def show_task3_results(img, label, title_prefix, ncols=3):
    img_batch = img.unsqueeze(0)
    with torch.no_grad():
        reconstructed1, reconstructed2 = net.task3(img_batch)

    recon_items = [(rec[0], f"Block1_Channel {c}") for c, rec in enumerate(reconstructed1)]
    recon_items += [(rec[0], f"Block2_Channel {c}") for c, rec in enumerate(reconstructed2)]

    nrows = 1 + math.ceil(len(recon_items) / ncols)  # +1 row for original
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2))

    # Top row: original image, centered, other cells in that row hidden
    for c in range(ncols):
        axes[0, c].axis("off")
    center_col = ncols // 2
    axes[0, center_col].imshow(unnormalize(img).permute(1, 2, 0))
    axes[0, center_col].set_title(f"orig ({label})", fontsize=9)

    # Remaining rows: reconstructions, flattened into the grid below
    recon_axes = axes[1:].flatten()
    for i, (tensor, title) in enumerate(recon_items):
        recon_axes[i].imshow(unnormalize(tensor).permute(1, 2, 0))
        recon_axes[i].set_title(title, fontsize=8)
        recon_axes[i].axis("off")
    for i in range(len(recon_items), len(recon_axes)):
        recon_axes[i].axis("off")

    fig.suptitle(title_prefix)
    plt.tight_layout()
    plt.show()

show_task3_results(train_img, train_label, "Train image")
show_task3_results(test_img, test_label, "Test image")


