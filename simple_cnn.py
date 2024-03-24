import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Пример сверточного слоя
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        # Пример полносвязного слоя
        self.fc1 = nn.Linear(20*12*12, 10)

    def forward(self, x):
        # Пример использования сверточного слоя и функции активации
        x = F.relu(self.conv1(x))
        # Изменение формы тензора
        x = x.view(-1, 20*12*12)
        # Пример использования полносвязного слоя
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)

# Инициализация модели
model = SimpleCNN()
print(model)
