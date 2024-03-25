from collections import Counter

import torch
import cv2
from torchvision import transforms
from PIL import Image
import numpy as np

from constants.constants import MODEL_PATH, BLOOD_TYPES, PREDICT_IMAGE_PATH
from entities.simple_cnn import SimpleCNN


# Загрузка модели
model = SimpleCNN()  # Убедитесь, что SimpleCNN определен и импортирован
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()  # Переключение модели в режим предсказания

# Функция для загрузки и преобразования изображения
def transform_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # Используйте реальные размеры, на которых обучалась ваша модель
        transforms.ToTensor(),
    ])

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image = transform(image)
    image = image.unsqueeze(0)  # Добавляем батч размерность
    return image

# Функция для предсказания группы крови
def predict_blood_type(image_path):
    image = transform_image(image_path)
    with torch.no_grad():  # Отключение подсчёта градиентов
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        predicted_blood_type = BLOOD_TYPES[predicted.item()]
    return predicted_blood_type

# Пример использования
if __name__ == "__main__":
    predictions = []
    for _ in range(500):
        predicted_blood_type = predict_blood_type(PREDICT_IMAGE_PATH)
        predictions.append(predicted_blood_type)

    # Подсчет количества для каждой предсказанной группы крови
    predictions_count = Counter(predictions)
    print(f"Predictions Count: {predictions_count}")