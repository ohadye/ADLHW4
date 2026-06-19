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
num_epochs = 5
lambda_recon = 1.0
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
criterion_cls = nn.CrossEntropyLoss()
criterion_recon = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


train_accs, test_accs = [], []

for epoch in range(num_epochs):
    # --- Training phase ---
    net.train()
    correct, total = 0, 0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        class_logits, reconstructed = net(inputs)
        loss_cls = criterion_cls(class_logits, labels)
        loss_recon = criterion_recon(reconstructed, inputs)
        loss = loss_cls + lambda_recon * loss_recon
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(class_logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if i % 2000 == 1999:
            print(f'CELoss: {loss_cls.item():.3f}, MSELoss: {loss_recon.item():.3f}')

    epoch_train_acc = 100 * correct / total
    train_accs.append(epoch_train_acc)

    # --- Evaluation phase (test set) ---
    net.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            class_logits, reconstructed = net(images)
            _, predicted = torch.max(class_logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_test_acc = 100 * correct / total
    test_accs.append(epoch_test_acc)

    print(f'[Epoch {epoch + 1}] train_acc: {epoch_train_acc:.2f}% | test_acc: {epoch_test_acc:.2f}%')

print('Finished Training')

# --- Plot accuracy curves ---
epochs_range = range(1, num_epochs + 1)
plt.figure(figsize=(6, 4))
plt.plot(epochs_range, train_accs, label='Train')
plt.plot(epochs_range, test_accs, label='Test')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy per epoch')
plt.legend()
plt.tight_layout()
plt.show()

# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
reconstructed_images = []
with torch.no_grad():
    for data in testloader:
        images, labels = data
        class_logits, reconstructed = net(images)
        reconstructed_images.append(reconstructed)
        _, predictions = torch.max(class_logits, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1


# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

# Function to unnormalize and show an image
def imshow(img, ax):
    img = img / 2 + 0.5  # unnormalize (reverses the 0.5/0.5 normalization)
    npimg = img.cpu().numpy()
    ax.imshow(np.transpose(npimg, (1, 2, 0)))
    ax.axis('off')

batch_size = images.shape[0]  # actual number of images in this batch

fig, axes = plt.subplots(2, batch_size, figsize=(3 * batch_size, 6))

for idx in range(batch_size):
    imshow(images[idx], axes[0, idx])
    true_label = classes[labels[idx]]
    pred_label = classes[predicted[idx]]
    color = 'green' if true_label == pred_label else 'red'
    axes[0, idx].set_title(f"True: {true_label}\nPred: {pred_label}", color=color)

    imshow(reconstructed[idx].detach(), axes[1, idx])
    axes[1, idx].set_title("Reconstructed")

plt.tight_layout()
plt.show()