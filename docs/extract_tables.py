import pandas as pd
import easyocr 
import sys
sys.path.append('C:/IDE/repository/liquid_vapor_database')
from src.column_names_detector import ColumnLineDetector

if __name__ == '__main__':
    PAGE_NUM = 134
    img_path = f'C:/IDE/repository/liquid_vapor_database/pdf_data/page_{PAGE_NUM}/page_{PAGE_NUM}_processed.png'

    reader = easyocr.Reader(['ru', 'en'])
    result = reader.readtext(img_path)
    easyocr_df = pd.DataFrame(result, columns=['bbox','text','conf'])
    detector = ColumnLineDetector(img_path, easyocr_df, debug=True)
    detector.detect_column_names_area()

