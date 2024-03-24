import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from entities.simple_cnn import SimpleCNN

# Ваша обученная модель
model = SimpleCNN()
model.load_state_dict(torch.load('path_to_saved_model.pth'))
model.eval()

# Подготовка изображения и преобразование его в тензор
def prepare_image(image):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    image = transform(image).unsqueeze(0)  # Добавление размерности batch
    return image

# Функция для предсказания класса крови изображения
def predict(image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs.data, 1)
        return predicted.item()

# Функция для выделения кругов с помощью OpenCV
def extract_circles(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                               param1=50, param2=30, minRadius=0, maxRadius=0)
    circle_images = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            radius = i[2]
            circle_img = gray[center[1]-radius:center[1]+radius, center[0]-radius:center[0]+radius]
            circle_images.append(circle_img)
    return circle_images

# Загрузите и обработайте новое изображение
image_path = '/mnt/data/image.png'
circle_images = extract_circles(image_path)

# Предсказания для каждого круга
predictions = []
for circle_img in circle_images:
    pil_img = Image.fromarray(circle_img)
    image_tensor = prepare_image(pil_img)
    prediction = predict(image_tensor)
    predictions.append(prediction)

# Распечатать предсказания
predictions = ['A' if p == 0 else 'B' if p == 1 else 'D' for p in predictions]
print('Predicted blood groups:', predictions)
