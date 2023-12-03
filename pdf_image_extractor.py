import cv2
import pytesseract
import numpy as np
import os
import fitz  # PyMuPDF
from sklearn.cluster import KMeans

class PDFImageExtractor:
    def __init__(self, pdf_path, tesseract_path, images_folder='extracted_images'):
        self.pdf_path = pdf_path
        self.tesseract_path = tesseract_path
        self.images_folder = images_folder
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
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

    def extract_images_from_pdf(self, start_page=0, end_page=None, scaling_factor=5):
        # Открытие PDF файла
        doc = fitz.open(self.pdf_path)
        end_page = end_page or doc.page_count

        for page_num in range(start_page, end_page):
            print(f"Обработка страницы {page_num + 1}...")
            page = doc.load_page(page_num)
            pix = page.get_pixmap(matrix=fitz.Matrix(scaling_factor, scaling_factor))
            img_data = pix.tobytes("png")
            img = cv2.imdecode(np.frombuffer(img_data, np.uint8), 1)

            # Сохранение оригинального изображения
            original_img_path = os.path.join(self.images_folder, f'page_{page_num + 1}_processed.png')
            cv2.imwrite(original_img_path, img)

            # Вызов метода обработки изображения
            img = self.isolate_dark_shades(original_img_path, dark_shades_count=2)
            if img is not None:
                cv2.imwrite(original_img_path, img)

        # Закрыть PDF файл
        doc.close()

    def extract_text_from_images(self, custom_config=r'--oem 3 --psm 6', start_page=0, end_page=None):
        # Открытие файла для записи извлеченного текста
        extracted_text_path = os.path.join(self.images_folder, 'extracted_text.txt')
        with open(extracted_text_path, 'w', encoding='utf-8') as text_file:
            end_page = end_page or len(os.listdir(self.images_folder))

            for page_num in range(start_page, end_page):
                img_path = os.path.join(self.images_folder, f'page_{page_num + 1}_processed.png')
                if os.path.exists(img_path):
                    # Адаптивная пороговая обработка и удаление "клякс"
                    img = cv2.imread(img_path)
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
                    processed_img = self.remove_spots(thresh, kernel_size=(5, 5), area_threshold=100)

                    # OCR и сохранение текста
                    text = pytesseract.image_to_string(processed_img, config=custom_config, lang='rus+eng')
                    text_file.write(f'--- Страница {page_num + 1} ---\n{text}\n')
                    self.text = pytesseract.image_to_data(processed_img, output_type=pytesseract.Output.DICT, config='--oem 3 --psm 6', lang='rus+eng')
