"""
https://pytorch.org/tutorials/beginner/onnx/export_simple_model_to_onnx_tutorial.html
"""

import time

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

from model import build_model
from model import classes

print(torch.__version__)
import onnx
import onnxscript
print(onnxscript.__version__)

import onnxruntime
print(onnxruntime.__version__)

onnx_model_path = 'models/cifar_model.onnx'
torch_model_path = 'models/cifar_weights.pth'

batch_size = 1
torch_input = torch.randn(batch_size, 3, 32, 32)


def convert_to_onnx():
    torch_model = build_model()
    torch_model.load_state_dict(torch.load(torch_model_path))

    onnx_program = torch.onnx.dynamo_export(torch_model, torch_input)
    onnx_program.save(onnx_model_path)


def test_onnx_load():
    onnx_model = onnx.load(onnx_model_path)
    onnx.checker.check_model(onnx_model)


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def test_onnx_runtime():
    torch_model = build_model()
    onnx_program = torch.onnx.dynamo_export(torch_model, torch_input)
    onnx_input = onnx_program.adapt_torch_inputs_to_onnx(torch_input)

    print(f"Input length: {len(onnx_input)}")
    print(f"Sample input: {onnx_input}")

    ort_session = onnxruntime.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])

    onnxruntime_input = {k.name: to_numpy(v) for k, v in zip(ort_session.get_inputs(), onnx_input)}
    t0 = time.time()
    onnxruntime_outputs = ort_session.run(None, onnxruntime_input)
    t1 = time.time() - t0

    print('Onnx output: ', onnxruntime_outputs)
    print('Onnx time, ms: ', t1 * 1000)


def benchmark_onnx():
    torch_model = build_model()

    ort_session = onnxruntime.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=1)

    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    time_pred = []

    for data in testloader:
        images, labels = data
        ort_input_name = ort_session.get_inputs()[0].name
        ort_input = {ort_input_name: to_numpy(images)}
        t0 = time.time()
        onnxruntime_outputs = ort_session.run(None, ort_input)
        t1 = time.time() - t0
        time_pred.append(t1 * 1000_000)

        prediction = np.argmax(onnxruntime_outputs[0], axis=1)
        # collect the correct predictions for each class
        label = labels[0]
        if to_numpy(label) == prediction:
            correct_pred[classes[label]] += 1
        total_pred[classes[label]] += 1

    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

    print('mean time, mcs: ', np.mean(time_pred))
    print('length: ', len(time_pred))


if __name__ == '__main__':
    convert_to_onnx()
    # test_onnx_load()
    # test_onnx_runtime()
    benchmark_onnx()
