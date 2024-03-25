# constants.py
# Размеры для изменения размера изображения
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64

# Параметры для выделения кругов
HOUGH_GRADIENT = 1
MIN_DIST = 20
PARAM1 = 50
PARAM2 = 30
MIN_RADIUS = 0
MAX_RADIUS = 0

# Путь к сохраненной модели
MODEL_PATH = '../target/saved_models/model.pth'
TRAIN_IMAGE_PATH = '../target/photos'
PREDICT_IMAGE_PATH = '../target/photos_to_predict/2.png'

# Классы группы крови
BLOOD_TYPES = ['A', 'B', 'D', 'Control']
