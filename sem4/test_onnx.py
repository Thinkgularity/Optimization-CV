import time
import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from model import build_model  # Убедитесь, что эта функция определена
from model import classes      # Убедитесь, что классы определены

# Пути к моделям
onnx_model_path = 'models/cifar_model.onnx'
torch_model_path = "models/cifar_model.pth"

batch_size = 1
torch_input = torch.randn(batch_size, 3, 32, 32)

# Создаем объект SummaryWriter для записи в TensorBoard
writer = SummaryWriter('runs/benchmarking')

def convert_to_onnx():
    # Проверка существования файла весов модели
    if not os.path.exists(torch_model_path):
        print(f"Ошибка: файл весов модели не найден по адресу: {torch_model_path}")
        return

    # Загружаем модель и её веса
    torch_model = build_model()
    torch_model.load_state_dict(torch.load(torch_model_path))
    torch_model.eval()  # Убедитесь, что модель в режиме оценки

    # Экспорт модели в формат ONNX
    torch.onnx.export(torch_model, torch_input, onnx_model_path)

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def benchmark_model(model_type='torch'):
    if model_type == 'torch':
        # Загружаем модель PyTorch
        model = build_model()
        model.load_state_dict(torch.load(torch_model_path))
        model.eval()

        # Подготовка тестового набора
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

        correct = 0
        total = 0
        time_taken = []

        with torch.no_grad():
            for images, labels in testloader:
                start_time = time.time()
                outputs = model(images)
                elapsed_time = (time.time() - start_time) * 1000  # время в миллисекундах
                time_taken.append(elapsed_time)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        mean_time = np.mean(time_taken)
        print(f'Torch Model Accuracy: {accuracy:.2f}%, Mean Inference Time: {mean_time:.2f} ms')

        # Запись метрик в TensorBoard
        writer.add_scalar('Torch Model Accuracy', accuracy)
        writer.add_scalar('Torch Model Mean Inference Time (ms)', mean_time)

    elif model_type == 'onnx':
        import onnx
        import onnxruntime

        ort_session = onnxruntime.InferenceSession(onnx_model_path)

        # Подготовка тестового набора
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

        correct = 0
        total = 0
        time_taken = []

        for images, labels in testloader:
            ort_input_name = ort_session.get_inputs()[0].name
            ort_input = {ort_input_name: to_numpy(images)}

            start_time = time.time()
            ort_outs = ort_session.run(None, ort_input)
            elapsed_time = (time.time() - start_time) * 1000  # время в миллисекундах
            time_taken.append(elapsed_time)

            prediction = np.argmax(ort_outs[0], axis=1)
            total += labels.size(0)
            correct += (prediction == labels.numpy()).sum()

        accuracy = 100 * correct / total
        mean_time = np.mean(time_taken)
        print(f'ONNX Model Accuracy: {accuracy:.2f}%, Mean Inference Time: {mean_time:.2f} ms')

        # Запись метрик в TensorBoard
        writer.add_scalar('ONNX Model Accuracy', accuracy)
        writer.add_scalar('ONNX Model Mean Inference Time (ms)', mean_time)

if __name__ == '__main__':
    convert_to_onnx()  # Конвертация модели в ONNX
    benchmark_model('torch')  # Бенчмаркинг модели PyTorch
    benchmark_model('onnx')  # Бенчмаркинг модели ONNX

    # Закрытие writer после завершения записи
    writer.close()
