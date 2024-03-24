import os
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
        circles = circles[0]  # Извлечение списка кругов из возвращаемого значения
        # Преобразование координат центра круга и его радиуса к целым числам
        circles = [[int(coord) for coord in circle] for circle in circles]
        # Проверка каждого круга на целочисленные координаты и положительный радиус
        for circle in circles:
            if not all(isinstance(coord, int) for coord in circle[:2]) or not isinstance(circle[2], int) or circle[2] <= 0:
                print("Invalid circle parameters detected:", circle)
                raise ValueError("Invalid circle parameters detected.")
    return circles if circles is not None else []  # Вернуть пустой список, если круги не обнаружены

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
            if circles is not None and len(circles) > 0:
                if len(circles) == len(BLOOD_TYPES):
                    for i, (x, y, r) in enumerate(circles):
                        circle_label = BLOOD_TYPES[i]
                        image_paths.append(image_path)
                        labels.append(circle_label)
                else:
                    print(f"Warning: Detected {len(circles)} circles, which does not match the number of blood types. Ignoring this image.")
    print(f"Found {len(image_paths)} images in the dataset.")
    return image_paths, labels

# Обновленная функция для создания датасета
class BloodCellDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = cv2.imread(self.image_paths[index])
        circles = detect_circles(image)
        crops = []
        labels = []
        for circle in circles:
            x, y, r = circle
            crop = image[y-r:y+r, x-r:x+r]
            if self.transform:
                crop = self.transform(crop)
            crops.append(crop)
            label = label_to_index.get(self.labels[index], -1)
            if label == -1:
                print(f"Warning: Unknown label '{self.labels[index]}' encountered.")
            labels.append(label)
        return torch.cat(crops), torch.tensor(labels).long()  # Преобразование меток в тип torch.LongTensor

# Обновленная функция train_model
def train_model(model, dataloader, criterion, optimizer, num_epochs=25):
    for epoch in range(num_epochs):
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs.float())  # Преобразование входных данных модели в тип torch.FloatTensor
            num_classes = len(BLOOD_TYPES)
            labels_one_hot = one_hot_encode(labels.squeeze().long(), num_classes)
            loss = criterion(outputs, labels_one_hot.float())
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

# Трансформации изображения
transform = transforms.Compose([
    transforms.Lambda(cv2_to_pil),  # Преобразование изображения в формат PIL
    transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
    transforms.ToTensor(),
])

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
