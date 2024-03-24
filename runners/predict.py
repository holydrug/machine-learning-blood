from PIL import Image
from torchvision import transforms
import torch
from entities.simple_cnn import SimpleCNN  # Или откуда вы импортируете вашу модель

# Предполагаем, что классы соответствуют следующему порядку
classes = ['A', 'B', 'D']
photo_to_predict_path = '../target/photos_to_predict/1.png'

# Загрузка и предобработка изображения
def prepare_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # Измените размер согласно вашей модели
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Добавляем размерность batch
    return image

# Загрузка модели и весов
model = SimpleCNN()
model.load_state_dict(torch.load('target/saved_models/model.pth'))
model.eval()  # Переключение модели в режим оценки

# Функция для предсказания
def predict(image_path):
    image = prepare_image(image_path)
    with torch.no_grad():  # Отключение расчета градиентов
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        return classes[predicted.item()]

# Пример использования
image_path = photo_to_predict_path  # Укажите путь к вашему изображению
prediction = predict(image_path)
print(f'Predicted blood group: {prediction}')
