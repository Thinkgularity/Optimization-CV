import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
from model import build_model, classes

batch_size = 4
model_path = "models/cifar_model.pth"


def show_images(trainloader):
    def imshow(img):
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    # get some random training images
    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print(" ".join(f"{classes[labels[j]]:5s}" for j in range()))


def train(optimizer_name="SGD"):
    print("Load data")
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    net = build_model()

    criterion = nn.CrossEntropyLoss()

    # Initialize optimizer based on the argument
    if optimizer_name == "SGD":
        optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    elif optimizer_name == "Adam":
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    # Create a SummaryWriter for TensorBoard
    writer = SummaryWriter()

    for epoch in range(1):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            # Log loss to TensorBoard
            writer.add_scalar(
                "training loss", loss.item(), epoch * len(trainloader) + i
            )

            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}")
                running_loss = 0.0

    print("Finished Training")
    os.makedirs("models", exist_ok=True)

    torch.save(net.state_dict(), model_path)

    writer.close()


if __name__ == "__main__":
    train(optimizer_name="SGD")
