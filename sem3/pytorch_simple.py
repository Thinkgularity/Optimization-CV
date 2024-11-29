import os
import optuna
from optuna.trial import TrialState
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCHSIZE = 128
CLASSES = 10
DIR = os.getcwd()
EPOCHS = 10
N_TRAIN_EXAMPLES = BATCHSIZE * 30
N_VALID_EXAMPLES = BATCHSIZE * 10

# Настройка TensorBoard
writer = SummaryWriter(log_dir=os.path.join(DIR, "logs"))

# Определение модели
class ConvNet(nn.Module):
    def __init__(self, trial):
        super(ConvNet, self).__init__()
        n_layers = trial.suggest_int("n_layers", 1, 5)
        kernel_size = trial.suggest_int("kernel_size", 3, 7)

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=kernel_size, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=kernel_size, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        # Прямое вычисление размера выходного тензора после свертки
        self.flatten = nn.Flatten()
        self.flat_input_size = self.get_flattened_size()

        self.fc_layers = nn.ModuleList()
        for i in range(n_layers):
            out_features = trial.suggest_int("n_units_l{}".format(i), 64, 256)
            self.fc_layers.append(nn.Linear(self.flat_input_size, out_features))
            self.fc_layers.append(nn.ReLU())
            p = trial.suggest_float("dropout_l{}".format(i), 0.2, 0.5)
            self.fc_layers.append(nn.Dropout(p))
            self.flat_input_size = out_features

        self.fc_layers.append(nn.Linear(self.flat_input_size, CLASSES))
        self.fc_layers.append(nn.LogSoftmax(dim=1))

    def get_flattened_size(self):
        # Проверка размерности с произвольным входным тензором
        with torch.no_grad():
            x = torch.zeros(1, 3, 32, 32)  # Размер входного изображения CIFAR-10
            x = self.conv_layers(x)
            return x.numel()  # Общее количество признаков после свертки

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        for layer in self.fc_layers:
            x = layer(x)
        return x

def get_cifar10():
    # Загрузка набора данных CIFAR-10.
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(DIR, train=True, download=True, transform=transforms.ToTensor()),
        batch_size=BATCHSIZE,
        shuffle=True,
    )
    valid_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(DIR, train=False, transform=transforms.ToTensor()),
        batch_size=BATCHSIZE,
        shuffle=True,
    )

    return train_loader, valid_loader

def objective(trial):
    # Генерация модели.
    model = ConvNet(trial).to(DEVICE)

    # Генерация оптимизаторов.
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    # Получение набора данных CIFAR-10.
    train_loader, valid_loader = get_cifar10()

    # Обучение модели.
    for epoch in range(EPOCHS):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            if batch_idx * BATCHSIZE >= N_TRAIN_EXAMPLES:
                break

            data, target = data.to(DEVICE), target.to(DEVICE)

            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

        # Валидация модели.
        model.eval()
        correct = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(valid_loader):
                if batch_idx * BATCHSIZE >= N_VALID_EXAMPLES:
                    break
                data, target = data.to(DEVICE), target.to(DEVICE)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        accuracy = correct / min(len(valid_loader.dataset), N_VALID_EXAMPLES)

        # Запись в TensorBoard
        writer.add_scalar("Accuracy", accuracy, epoch)
        writer.add_scalar("Loss", loss.item(), epoch)

        trial.report(accuracy, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return accuracy

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100, timeout=600)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    writer.close()  # Закрытие TensorBoard писателя