# Model class must be defined somewhere
import time

import torch
import torchvision
import torchvision.transforms as transforms

from model import build_model
from model import classes

model_path = 'models/cifar_weights.pth'

batch_size = 4

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def test():
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=1)


    nnet = build_model()
    nnet.load_state_dict(
        torch.load(model_path))

    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    time_pred = []

    # again no gradients needed
    with torch.no_grad():
        for data in testloader:
            images, labels = data

            t0 = time.time()
            outputs = nnet(images)
            t1 = time.time() - t0
            time_pred.append(t1 * 1000)

            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

    print('Time, ms: ', time_pred)


if __name__ == '__main__':
    test()
