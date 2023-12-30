import cv2
import pytesseract
import numpy as np
import os
import fitz  # PyMuPDF
from sklearn.cluster import KMeans

class PDFImageExtractor:
    def __init__(self, pdf_path, images_folder='extracted_images'):
        self.pdf_path = pdf_path        
        self.images_folder = images_folder        
        os.makedirs(images_folder, exist_ok=True)

    def isolate_dark_shades(self, image_path, dark_shades_count=2):
        """
        Изолирует самые темные оттенки на изображении, заменяя остальные на белый цвет.

        :param image_path: Путь к исходному изображению.
        :param dark_shades_count: Количество темных оттенков для сохранения.
        :return: Изображение только с темными оттенками и белым фоном.
        """
        # Загрузка и подготовка изображения
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_flat = img.reshape((-1, 3))

        # Применение k-средних для нахождения основных цветов
        kmeans = KMeans(n_clusters=5, n_init=10)
        kmeans.fit(img_flat)
        dominant_colors = kmeans.cluster_centers_

        # Сортировка цветов по их яркости
        colors_brightness = np.sum(dominant_colors, axis=1)
        sorted_indices = np.argsort(colors_brightness)

        # Выбор самых темных оттенков
        dark_colors = sorted_indices[:dark_shades_count]

        # Замена пикселей, не относящихся к темным оттенкам, на белый
        for i in range(len(img_flat)):
            if kmeans.labels_[i] not in dark_colors:
                img_flat[i] = [255, 255, 255]

        return img_flat.reshape(img.shape)

    def remove_spots(self, image, kernel_size=(3, 3), area_threshold=100):
        """
        Удаляет мелкие изолированные элементы (кляксы) из изображения.

        :param image: Входное изображение для обработки.
        :param kernel_size: Размер ядра для морфологических операций.
        :param area_threshold: Пороговое значение площади для удаления мелких контуров.
        :return: Обработанное изображение.
        """
        # Морфологическая эрозия и дилатация
        kernel = np.ones(kernel_size, np.uint8)
        erosion = cv2.erode(image, kernel, iterations=1)
        dilation = cv2.dilate(erosion, kernel, iterations=1)

        # Найти контуры на изображении
        contours, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Создать маску для удаления мелких контуров
        mask = np.ones_like(dilation) * 255
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < area_threshold:
                cv2.drawContours(mask, [contour], -1, 0, -1)

        # Применить маску к изображению
        return cv2.bitwise_and(dilation, dilation, mask=mask)
    