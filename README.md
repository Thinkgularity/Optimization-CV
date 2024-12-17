![image](https://github.com/user-attachments/assets/bb34d2bd-f14e-4ef4-9f0d-fdf906c383af)Sem №1

В данном проекте реализована нейронная сеть для классификации изображений, в частности, для датасета CIFAR-10.

Структура проекта:

__pycache__: Папка, в которой Python хранит скомпилированные байт-коды и оптимизированные версии файлов. Эта папка создается автоматически при запуске модулей.

cifar_model.pth: Файл содержит веса предобученной модели, сохраненные в формате PyTorch. Этот файл используется для загрузки и инференса модели.

model.py: Этот файл содержит определение архитектуры нейронной сети, которую можно обучить для выполнения задач классификации изображений.

test.py:  Здесь мы тестируем модель для оценки ее точности на валидационном или тестовом наборе данных.

test_onnx: Этот файл используется для тестирования модели, экспортированной в формат ONNX (Open Neural Network Exchange), что позволяет использовать модель в различных фреймворках.

train.py: Этот файл предназначен для обучения нейронной сети, включая этапы загрузки данных, оптимизации и сохранения модели.


Sem № 2:



![image](https://github.com/user-attachments/assets/5aca40f4-9ad0-4cef-9faa-470f949a93b5)





Sem № 4 
```
console output example:

Torch Model Accuracy: 45.79%, Mean Inference Time: 0.84 ms
ONNX Model Accuracy: 45.79%, Mean Inference Time: 0.12 ms
```

ONNX использует onnxruntime, который оптимизирует граф вычислений и использует более эффективные механизмы выполнения. Он может автоматически применять различные оптимизации, такие как фуззинг операций и упрощение вычислений. ONNX может более эффективно управлять вычислениями, минимизируя накладные расходы, связанные с планированием и инициализацией в режиме выполнения.

В PyTorch некоторые функции, как например автоматическое дифференцирование и динамическое построение вычислительных графов, могут добавлять накладные расходы на этапе выполнения. ONNX, будучи статической моделью, не требует этих дополнительных накладных расходов.
