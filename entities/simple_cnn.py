# simple_cnn.py
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        # После определения слоев убедитесь, что размерность подходит под вашу модель
        self.fc1 = nn.Linear(16 * 32 * 32, 3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 16 * 32 * 32)  # Вытягиваем в один вектор
        x = self.fc1(x)
        return x
