import logging
import sys
import unittest
import pandas as pd
import os

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)


class TestRegionSequence(unittest.TestCase):
    def setUp(self):
        # Загрузка всех csv файлов из папки 'assets' в список DataFrame
        assets_path = 'C:/IDE/repository/liquid_vapor_database/tests/assets'  
        self.data = [pd.read_csv(os.path.join(assets_path, file)) 
                     for file in os.listdir(assets_path) if file.endswith('.csv')]

    def test_metadata_starts_first(self):        
        for df in self.data:
            # Проверка, что первое значение в столбце 'region' равно 'metadata'
            self.assertEqual(
                df['region'].iloc[0], 'metadata', "Столбец 'region' должен начинаться с 'metadata'")

    def test_no_single_values_in_region(self):        
        for df in self.data:
            regions = df['region'].tolist()            
            for i in range(1, len(regions) - 1):
                # Проверка, что текущее значение равно предыдущему или следующему
                if regions[i] != regions[i - 1] and regions[i] != regions[i + 1]:
                    self.fail(f"Single value '{regions[i]}' found at position {i} in 'region' column")

if __name__ == '__main__':    
    unittest.main(argv=[''], exit=False)
