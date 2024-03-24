import torch.nn as nn
import torch.nn.functional as F

from constants.constants import BLOOD_TYPES


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(12, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, len(BLOOD_TYPES))  # Предполагая, что len(BLOOD_TYPES) определен где-то в вашем коде

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 16 * 32 * 32)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
