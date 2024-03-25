# Определение функции для обнаружения кругов на изображении
import cv2
from PIL import Image


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
        minRadius=1,
        maxRadius=70
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