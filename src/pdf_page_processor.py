import cv2
import pytesseract
import numpy as np
import os
import fitz  # PyMuPDF
from sklearn.cluster import KMeans
from .pdf_image_extractor import PDFImageExtractor

class PageImage:
    def __init__(self, page_num: int, extractor: PDFImageExtractor):
        self.page_num = page_num
        self.extractor = extractor        
        self.max_height = None        
        self.max_width = None
        self.original_img_path = None
        self.extract_images_from_pdf()        
        #self.extract_text_from_images()
        
    
    def extract_images_from_pdf(self, scaling_factor=5):
        # Открытие PDF файла
        doc = fitz.open(self.extractor.pdf_path)        
        
        print(f"Обработка страницы {self.page_num}")
        page = doc.load_page(self.page_num)
        pix = page.get_pixmap(matrix=fitz.Matrix(scaling_factor, scaling_factor))
        img_data = pix.tobytes("png")
        img = cv2.imdecode(np.frombuffer(img_data, np.uint8), 1)
        
        # Обновление минимума и максимума для высоты и ширины
        h, w, _ = img.shape        
        self.max_height = max(self.max_height, h) if self.max_height is not None else h        
        self.max_width = max(self.max_width, w) if self.max_width is not None else w

        # Сохранение оригинального изображения
        self.original_img_path = os.path.join(self.extractor.images_folder, f'page_{self.page_num}_processed.png')
        cv2.imwrite(self.original_img_path, img)

        # Вызов метода обработки изображения
        img = self.extractor.isolate_dark_shades(self.original_img_path, dark_shades_count=2)
        if img is not None:
            print('Метод isolate_dark_shades выполнен')
            cv2.imwrite(self.original_img_path, img)

        # Закрыть PDF файл        
        doc.close()
    
    def extract_text_from_images(self, custom_config=r'--oem 3 --psm 6 -c preserve_interword_spaces=1'):
        extracted_text_path = os.path.join(self.extractor.images_folder, f'{self.page_num}_text.txt')
        with open(extracted_text_path, 'w', encoding='utf-8') as text_file:                     
            img_path = os.path.join(self.extractor.images_folder, f'page_{self.page_num}_processed.png')
            if os.path.exists(img_path):
                # Адаптивная пороговая обработка и удаление "клякс"
                img = cv2.imread(img_path)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
                processed_img = self.extractor.remove_spots(thresh, kernel_size=(5, 5), area_threshold=100)

                # OCR и сохранение текста
                text = pytesseract.image_to_string(processed_img, config=custom_config, lang='rus+eng')
                text_file.write(f'--- Страница {self.page_num} ---\n{text}\n')
                self.text = pytesseract.image_to_data(processed_img, output_type=pytesseract.Output.DICT, config=custom_config, lang='rus+eng')

        