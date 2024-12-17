Sem №1

В данном проекте реализована нейронная сеть для классификации изображений, в частности, для датасета CIFAR-10.

Структура проекта:

__pycache__: Папка, в которой Python хранит скомпилированные байт-коды и оптимизированные версии файлов. Эта папка создается автоматически при запуске модулей.

cifar_model.pth: Файл содержит веса предобученной модели, сохраненные в формате PyTorch. Этот файл используется для загрузки и инференса модели.

model.py: Этот файл содержит определение архитектуры нейронной сети, которую можно обучить для выполнения задач классификации изображений.

test.py:  Здесь мы тестируем модель для оценки ее точности на валидационном или тестовом наборе данных.

test_onnx: Этот файл используется для тестирования модели, экспортированной в формат ONNX (Open Neural Network Exchange), что позволяет использовать модель в различных фреймворках.

train.py: Этот файл предназначен для обучения нейронной сети, включая этапы загрузки данных, оптимизации и сохранения модели.


Sem № 2:

В данном проекте проводилось сравнение скорости сходимости модели при использовании различных алгоритмов оптимизации: **SGD (Stochastic Gradient Descent)** и **Adam**. 

## Результаты

На графике представлено изменение значения потерь (loss) по эпохам для обоих алгоритмов. 

![image](https://github.com/user-attachments/assets/5aca40f4-9ad0-4cef-9faa-470f949a93b5)


### Выводы

1. **SGD**: Сходимость модели была более волнообразной, однако в целом значение потерь снижалось.
   
2. **Adam**: Этот алгоритм демонстрировал более стремительное и стабильное снижение потерь, что говорит о более эффективной сходимости.

Выводя итог, можно сказать, что алгоритм **Adam** в большинстве случаев является предпочтительным выбором для обучения моделей, так как он обеспечивает более быструю сходимость по сравнению с **SGD**. Важно учесть, что выбор оптимизатора также может зависеть от конкретной задачи и свойств данных.


Sem № 3

# Оптимизация гиперпараметров с помощью Optuna

В данной работе проводилась настройка гиперпараметров сверточной нейронной сети (CNN), используя Optuna для оптимизации на основе датасета CIFAR-10.

## Описание эксперимента

### Модель
Вместо многослойного перцептрона, была построена сверточная нейронная сеть (CNN) с использованием слоев `Conv2D`, `ReLU`, `MaxPool2D`, `Flatten`, а также полносвязных слоев. При построении модели оптимизировались два гиперпараметра:

- **`n_layers`**: Число сверточных слоев (от 1 до 5).
- **`kernel_size`**: Размер ядра свертки (от 3 до 7).

### Алгоритм оптимизации
Для оптимизации использовались три различных алгоритма: Adam, RMSprop и SGD. Кроме того, был настроен переменный коэффициент обучения (`lr`), который также варьировался в диапазоне от 1e-5 до 1e-1.

### Результаты

После завершения 100 проб, наилучший набор гиперпараметров был определен, отражая лучшие значения точности.

![result](https://github.com/user-attachments/assets/c0b8c629-5ed9-4f3d-92d8-d58b47a5e345)

```
[I 2024-12-17 20:09:01,522] A new study created in memory with name: no-name-239d3f72-c058-42e9-a2a0-46d9c23c460a
Files already downloaded and verified
[I 2024-12-17 20:09:41,447] Trial 0 finished with value: 0.18828125 and parameters: {'n_layers': 1, 'kernel_size': 6, 'n_units_l0': 201, 'dropout_l0': 0.30911100982464534, 'optimizer': 'RMSprop', 'lr': 0.021098068185632465}. Best is trial 0 with value: 0.18828125.
Files already downloaded and verified
[I 2024-12-17 20:10:21,566] Trial 1 finished with value: 0.115625 and parameters: {'n_layers': 3, 'kernel_size': 6, 'n_units_l0': 80, 'dropout_l0': 0.21308743421210632, 'n_units_l1': 88, 'dropout_l1': 0.2777448913174193, 'n_units_l2': 209, 'dropout_l2': 0.24132856034700917, 'optimizer': 'SGD', 'lr': 0.029288957471676615}. Best is trial 0 with value: 0.18828125.
Files already downloaded and verified
[I 2024-12-17 20:11:00,211] Trial 2 finished with value: 0.09296875 and parameters: {'n_layers': 2, 'kernel_size': 3, 'n_units_l0': 76, 'dropout_l0': 0.46533596629939716, 'n_units_l1': 162, 'dropout_l1': 0.43071094844000735, 'optimizer': 'Adam', 'lr': 0.03750920320560918}. Best is trial 0 with value: 0.18828125.
Files already downloaded and verified
[I 2024-12-17 20:11:43,122] Trial 3 finished with value: 0.17734375 and parameters: {'n_layers': 4, 'kernel_size': 5, 'n_units_l0': 76, 'dropout_l0': 0.4021459624497033, 'n_units_l1': 178, 'dropout_l1': 0.29636024054177357, 'n_units_l2': 213, 'dropout_l2': 0.26950154546831206, 'n_units_l3': 71, 'dropout_l3': 0.3208043781408744, 'optimizer': 'Adam', 'lr': 0.015345242791238604}. Best is trial 0 with value: 0.18828125.
Files already downloaded and verified
[I 2024-12-17 20:12:23,189] Trial 4 finished with value: 0.10546875 and parameters: {'n_layers': 3, 'kernel_size': 3, 'n_units_l0': 90, 'dropout_l0': 0.31718349573769583, 'n_units_l1': 133, 'dropout_l1': 0.364362686837544, 'n_units_l2': 165, 'dropout_l2': 0.2990451392240695, 'optimizer': 'SGD', 'lr': 0.01675342428219103}. Best is trial 0 with value: 0.18828125.
Files already downloaded and verified
[I 2024-12-17 20:12:33,404] Trial 5 pruned. 
Files already downloaded and verified
[I 2024-12-17 20:13:10,965] Trial 6 pruned. 
Files already downloaded and verified
[I 2024-12-17 20:13:16,965] Trial 7 pruned. 
Files already downloaded and verified
[I 2024-12-17 20:13:56,065] Trial 8 finished with value: 0.43046875 and parameters: {'n_layers': 1, 'kernel_size': 3, 'n_units_l0': 80, 'dropout_l0': 0.41909576859709596, 'optimizer': 'RMSprop', 'lr': 0.00043420817842303974}. Best is trial 8 with value: 0.43046875.
Files already downloaded and verified
[I 2024-12-17 20:14:38,676] Trial 9 finished with value: 0.20703125 and parameters: {'n_layers': 5, 'kernel_size': 7, 'n_units_l0': 156, 'dropout_l0': 0.4302523622829424, 'n_units_l1': 100, 'dropout_l1': 0.4387924526618807, 'n_units_l2': 139, 'dropout_l2': 0.2713871906083642, 'n_units_l3': 139, 'dropout_l3': 0.40105966711856184, 'n_units_l4': 107, 'dropout_l4': 0.2653372416520615, 'optimizer': 'RMSprop', 'lr': 0.000489496066700236}. Best is trial 8 with value: 0.43046875.
Files already downloaded and verified
[I 2024-12-17 20:15:18,475] Trial 10 finished with value: 0.26953125 and parameters: {'n_layers': 1, 'kernel_size': 4, 'n_units_l0': 131, 'dropout_l0': 0.4973956210774587, 'optimizer': 'RMSprop', 'lr': 2.6563053307277374e-05}. Best is trial 8 with value: 0.43046875.        
Files already downloaded and verified
[I 2024-12-17 20:15:57,992] Trial 11 finished with value: 0.29375 and parameters: {'n_layers': 1, 'kernel_size': 4, 'n_units_l0': 129, 'dropout_l0': 0.4839096961400572, 'optimizer': 'RMSprop', 'lr': 2.3563617163816872e-05}. Best is trial 8 with value: 0.43046875.
Files already downloaded and verified
[I 2024-12-17 20:16:37,417] Trial 12 finished with value: 0.28515625 and parameters: {'n_layers': 1, 'kernel_size': 4, 'n_units_l0': 122, 'dropout_l0': 0.39144550682967383, 'optimizer': 'RMSprop', 'lr': 1.4065533867092136e-05}. Best is trial 8 with value: 0.43046875.       
Files already downloaded and verified
[I 2024-12-17 20:17:18,508] Trial 13 finished with value: 0.38203125 and parameters: {'n_layers': 2, 'kernel_size': 4, 'n_units_l0': 239, 'dropout_l0': 0.4581559310968374, 'n_units_l1': 254, 'dropout_l1': 0.20387068264265365, 'optimizer': 'RMSprop', 'lr': 0.0001229028288014588}. Best is trial 8 with value: 0.43046875.
Files already downloaded and verified
[I 2024-12-17 20:18:00,069] Trial 14 finished with value: 0.42265625 and parameters: {'n_layers': 2, 'kernel_size': 3, 'n_units_l0': 245, 'dropout_l0': 0.372653934010525, 'n_units_l1': 253, 'dropout_l1': 0.20408630231345984, 'optimizer': 'RMSprop', 'lr': 0.00019003276908458172}. Best is trial 8 with value: 0.43046875.
Files already downloaded and verified
[I 2024-12-17 20:18:41,146] Trial 15 finished with value: 0.46875 and parameters: {'n_layers': 2, 'kernel_size': 3, 'n_units_l0': 241, 'dropout_l0': 0.355047876846278, 'n_units_l1': 248, 'dropout_l1': 0.21310712644808932, 'optimizer': 'RMSprop', 'lr': 0.0009095300422742675}. Best is trial 15 with value: 0.46875.
Files already downloaded and verifie

```


Sem № 4 
```
console output example:

Torch Model Accuracy: 45.79%, Mean Inference Time: 0.84 ms
ONNX Model Accuracy: 45.79%, Mean Inference Time: 0.12 ms
```

ONNX использует onnxruntime, который оптимизирует граф вычислений и использует более эффективные механизмы выполнения. Он может автоматически применять различные оптимизации, такие как фуззинг операций и упрощение вычислений. ONNX может более эффективно управлять вычислениями, минимизируя накладные расходы, связанные с планированием и инициализацией в режиме выполнения.

В PyTorch некоторые функции, как например автоматическое дифференцирование и динамическое построение вычислительных графов, могут добавлять накладные расходы на этапе выполнения. ONNX, будучи статической моделью, не требует этих дополнительных накладных расходов.
