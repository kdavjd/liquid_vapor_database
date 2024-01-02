import logging
import sys
import unittest


logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)


class TestRegionSequence(unittest.TestCase):
    def setUp(self):
        # Предполагается, что df уже определен и содержит столбец 'region'
        self.data = '' # pd.read_csv()

    def test_metadata_starts_first(self):
        # Поиск первого вхождения 'value' в столбце 'region'
        first_value_index = self.data['region'].tolist().index('value')

        # Проверка, что все элементы до первого 'value' являются 'metadata'
        for region in self.data['region'][:first_value_index]:
            self.assertEqual(region, 'metadata', "Metadata should start first")



if __name__ == '__main__':    
    unittest.main(argv=[''], exit=False)
