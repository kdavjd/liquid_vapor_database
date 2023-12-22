import sys
sys.path.append('C:/IDE/repository/liquid_vapor_database')
from src.pdf_image_extractor import PDFImageExtractor
from src.pdf_page_processor import Page


if __name__ == '__main__':
    # Путь к PDF файлу и путь к Tesseract
    NUM_PAGE = 134
    pdf_path = 'C:/IDE/repository/liquid_vapor_database/data/Kogan_1.pdf'
    tesseract_path = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
    image = PDFImageExtractor(pdf_path, tesseract_path, images_folder=f'pdf_data/page_{NUM_PAGE}')        
    page = Page(NUM_PAGE, image)