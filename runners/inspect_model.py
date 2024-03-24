# inspect_model.py
import torch
import matplotlib.pyplot as plt
from entities.simple_cnn import SimpleCNN  # Импорт определения модели из другого файла

# Функция для визуализации весов сверточного слоя
def visualize_weights(model_path, layer_name):
    # Загрузка модели
    model = SimpleCNN()
    model.load_state_dict(torch.load(model_path))

    # Получение весов из указанного слоя
    weights = getattr(model, layer_name).weight.data

    # Визуализация весов
    for i, filter in enumerate(weights):
        plt.subplot(4, 4, i+1)  # Расположение графиков в сетке 4x4
        plt.imshow(filter[0, :, :].detach().numpy(), cmap='gray')  # Визуализация фильтра
        plt.axis('off')
    plt.show()

# Функция для вывода статистики весов
def print_weight_stats(model_path):
    model = SimpleCNN()
    model.load_state_dict(torch.load(model_path))

    # Вывод статистики весов
    for name, param in model.named_parameters():
        print(f'{name}: mean = {param.data.mean()}, std = {param.data.std()}')

# Путь к сохраненной модели
model_path = 'target/saved_models/model.pth'

# Вызов функций для визуализации и статистики
visualize_weights(model_path, 'conv1')
print_weight_stats(model_path)
