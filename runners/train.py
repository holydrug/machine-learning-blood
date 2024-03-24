import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import cv2
from constants.constants import (
    IMAGE_HEIGHT, IMAGE_WIDTH, MODEL_PATH, TRAIN_IMAGE_PATH, BLOOD_TYPES
)
from entities.simple_cnn import SimpleCNN  # Убедитесь, что SimpleCNN определен в simple_cnn.py

label_to_index = {label: i for i, label in enumerate(BLOOD_TYPES)}

# Определение функции для обнаружения кругов на изображении
def detect_circles(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(
        gray_blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=20,
        param1=50,
        param2=30,
        minRadius=20,
        maxRadius=52
    )
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        # Сортируем круги по горизонтальной координате x (слева направо)
        circles = sorted(circles, key=lambda x: x[0])
        # Проверка, что найдено ровно 4 круга (опционально)
        if len(circles) != 4:
            raise ValueError(f"Expected 4 circles, but found {len(circles)}.")
        return circles
    else:
        raise ValueError("No circles detected.")


# Определение функции для преобразования изображения из формата OpenCV в формат PIL
def cv2_to_pil(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image)
    return pil_image

def one_hot_encode(labels, num_classes):
    encoded_labels = torch.zeros(len(labels), num_classes)
    for i, label in enumerate(labels):
        encoded_labels[i][label] = 1
    return encoded_labels

# Функция для создания списка путей к изображениям и их меток
def create_dataset(image_dir):
    image_paths = []
    labels = []
    for image_name in os.listdir(image_dir):
        if image_name.endswith('.png'):
            image_path = os.path.join(image_dir, image_name)
            image = cv2.imread(image_path)
            circles = detect_circles(image)
            if circles is not None and len(circles) == 4:  # Убедимся, что обнаружено ровно 4 круга
                for circle_label in BLOOD_TYPES:
                    image_paths.append(image_path)
                    labels.append(circle_label)
            else:
                print(f"Warning: Expected 4 circles, but found {len(circles)}. Ignoring this image.")
    print(f"Found {len(image_paths) // 4} complete images in the dataset.")
    return image_paths, labels

# Обновленная функция для создания датасета
class BloodCellDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform  # Эта трансформация теперь будет применяться к каждому кругу

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index // 4]
        image = cv2.imread(image_path)
        circles = detect_circles(image)

        circle_index = index % 4
        x, y, r = circles[circle_index]
        crop = image[y-r:y+r, x-r:x+r]

        if self.transform:
            crop = self.transform(crop)

        label = self.labels[index]
        label_index = torch.tensor(label_to_index[label], dtype=torch.long)

        return crop, label_index

# Трансформации изображения
transform = transforms.Compose([
    transforms.Lambda(lambda x: cv2.resize(x, (IMAGE_HEIGHT, IMAGE_WIDTH))),  # Масштабирование изображения
    transforms.Lambda(lambda x: Image.fromarray(x)),  # Преобразование в PIL для дальнейших трансформаций
    transforms.ToTensor(),  # Преобразование в тензор PyTorch
])

# Обновленная функция train_model
def train_model(model, dataloader, criterion, optimizer, num_epochs=25):
    for epoch in range(num_epochs):
        for inputs, labels in dataloader:
            optimizer.zero_grad()

            # Убедитесь, что inputs имеют размерность [batch_size * 4, C, H, W]
            outputs = model(inputs.float())
            # labels будут иметь размерность [batch_size * 4]
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')



# Создание модели
model = SimpleCNN()

# Функция потерь и оптимизатор
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Запуск обучения
if __name__ == "__main__":
    # Создание экземпляра датасета
    image_paths, labels = create_dataset(os.path.abspath(TRAIN_IMAGE_PATH))
    if not image_paths:
        raise Exception(f"No images found in the directory {TRAIN_IMAGE_PATH}. Check the path and file extensions.")
    dataset = BloodCellDataset(image_paths=image_paths, labels=labels, transform=transform)
    # Убедитесь, что dataset содержит элементы
    if len(dataset) == 0:
        raise RuntimeError('Dataset is empty. Check the directory and the label extraction logic.')

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Запуск обучения
    train_model(model, dataloader, criterion, optimizer)

    # Проверка существования директории для сохранения модели и ее сохранение
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f'Model saved to {MODEL_PATH}')
