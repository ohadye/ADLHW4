import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)         #(28x28x6)
        self.pool = nn.MaxPool2d(2, 2, return_indices=True)          #(14x14x6)
        self.conv2 = nn.Conv2d(6, 16, 5)        #(10x10x16)
                                                #(5X5X16)

        self.deconv1 = nn.ConvTranspose2d(6, 3, 5)
        self.deconv2 = nn.ConvTranspose2d(16, 6, 5)
        self.unpool =  nn.MaxUnpool2d(2, 2)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)   
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Encoder forward pass, keeping sizes + indices for unpooling
        x1 = F.relu(self.conv1(x))
        size1 = x1.size()
        x1_pooled, idx1 = self.pool(x1)

        x2 = F.relu(self.conv2(x1_pooled))
        size2 = x2.size()
        x2_pooled, idx2 = self.pool(x2)

        # --- Classifier branch ---
        flat = torch.flatten(x2_pooled, 1)
        c = F.relu(self.fc1(flat))
        c = F.relu(self.fc2(c))
        class_logits = self.fc3(c)

        # --- Decoder branch ---
        d = self.unpool(x2_pooled, idx2, output_size=size2)
        d = self.deconv2(F.relu(d))
        d = self.unpool(d, idx1, output_size=size1)
        reconstructed = self.deconv1(F.relu(d))  # squashes to [0,1] for image output

        return class_logits, reconstructed

    def task3(self, x):
        x1 = F.relu(self.conv1(x))
        size1 = x1.size()
        x1_pooled, idx1 = self.pool(x1)

        reconstructed1 = []
        for c in range(6):
            channel_isolated = torch.zeros_like(x1_pooled)
            channel_isolated[:, c, :, :] = x1_pooled[:, c, :, :]
            d = self.unpool(channel_isolated, idx1, output_size=size1)
            reconstructed1.append(self.deconv1(F.relu(d)))

        x2 = F.relu(self.conv2(x1_pooled))
        size2 = x2.size()
        x2_pooled, idx2 = self.pool(x2)

        reconstructed2 = []
        for c in range(3):  # only first 3 channels of conv2's 16
            channel_isolated = torch.zeros_like(x2_pooled)
            channel_isolated[:, c, :, :] = x2_pooled[:, c, :, :]
            d = self.unpool(channel_isolated, idx2, output_size=size2)
            d = self.deconv2(F.relu(d))
            d = self.unpool(d, idx1, output_size=size1)
            reconstructed2.append(self.deconv1(F.relu(d)))

        return reconstructed1, reconstructed2

class NetBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
