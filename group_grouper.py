import numpy as np
import pandas as pd
import re

class GaussianCurveGrouper:
    def __init__(self, hystogram_data, ocr_text_path, regex_pattern):
        self.hystogram_data = hystogram_data
        self.ocr_text_path = ocr_text_path
        self.regex_pattern = regex_pattern
        self.n_curves = self.count_matching_lines(ocr_text_path, regex_pattern)

    @staticmethod
    def count_matching_lines(hystogram_data, pattern):
        count = 0
        with open(hystogram_data, 'r', encoding='utf-8') as file:
            for line in file:
                if re.search(pattern, line):
                    count += 1
        return count

    @staticmethod
    def gaussian(x, mu, sigma, height):
        return height * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sigma, 2.)))

    def group_data(self):
        MIN_DISTANCE = 500
        AGGREGATION_STEP = 50

        df = pd.read_csv(self.hystogram_data)
        df['Center_Y_Aggregated'] = (df['Center_Y'] // AGGREGATION_STEP) * AGGREGATION_STEP
        aggregated_counts = df['Center_Y_Aggregated'].value_counts().sort_index()    

        point = aggregated_counts.sort_values(ascending=False).index[0]
        other_points = [value for value in aggregated_counts.sort_values(ascending=False).index if abs(value - point) > MIN_DISTANCE]
        points = [point, *other_points]
        valid_points = points[:self.n_curves]

        if len(valid_points) < self.n_curves:
            raise ValueError("Не удалось найти достаточное количество точек, удовлетворяющих условиям.")

        groups = [[] for _ in range(self.n_curves)]
        for idx in aggregated_counts.index:
            gauss_values = [self.gaussian(idx, point, 500, aggregated_counts[point]) for point in valid_points]
            assigned_group = np.argmax(gauss_values)
            groups[assigned_group].append(idx)

        return groups



# # Пример использования
# grouper = GaussianCurveGrouper('histogram_data.csv', "extracted_images/extracted_text.txt", r"([NnеНЕмт№]\s?\d*\.?\d*\s?.*?\[.*?\])")
# grouped_data = grouper.group_data()
