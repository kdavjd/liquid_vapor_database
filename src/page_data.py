import pandas as pd
import numpy as np
from scipy.optimize import differential_evolution


class PageData():
    def __init__(self, data: pd.DataFrame):        
        self.data = data
        self.min_x, self.min_y = float('inf'), float('inf')
        self.max_x, self.max_y = float('-inf'), float('-inf')
        self.update_min_max()
        self.add_bbox_dimensions()
        self.df = self.process_df()
    
    def update_min_max(self):
        for bbox in self.data['bbox']:
            for x, y in bbox:
                self.min_x = min(self.min_x, x)
                self.max_x = max(self.max_x, x)
                self.min_y = min(self.min_y, y)
                self.max_y = max(self.max_y, y)
                
    def add_bbox_dimensions(self):
        self.data['bbox_width'] = self.data['bbox'].apply(lambda bbox: abs(bbox[1][0] - bbox[0][0]))
        self.data['bbox_height'] = self.data['bbox'].apply(lambda bbox: abs(bbox[2][1] - bbox[0][1]))
        
    def scale_position(self, bbox, scale_x, scale_y):
        center_x = sum([point[0] for point in bbox]) / 4
        center_y = sum([point[1] for point in bbox]) / 4
        scaled_x = scale_x * (center_x - self.min_x) / (self.max_x - self.min_x)
        scaled_y = scale_y * (center_y - self.min_y) / (self.max_y - self.min_y)
        return scaled_x, scaled_y
    
    def unscale_width_position(self, start_scaled_x, end_scaled_x):
        pixel_per_unit_x = (self.max_x - self.min_x) / (self.best_scale_x)
        start_pixel_x = self.min_x + start_scaled_x * pixel_per_unit_x
        end_pixel_x = self.min_x + end_scaled_x * pixel_per_unit_x
        return int(start_pixel_x), int(end_pixel_x)

    def unscale_height_position(self, start_scaled_y, end_scaled_y):
        pixel_per_unit_y = (self.max_y - self.min_y) / (self.best_scale_y)
        start_pixel_y = self.min_y + start_scaled_y * pixel_per_unit_y
        end_pixel_y = self.min_y + end_scaled_y * pixel_per_unit_y
        return int(start_pixel_y), int(end_pixel_y)
    
    def optimize_scale_factors(self):
        def objective(scale_factors):
            scale_x, scale_y = map(int, scale_factors)
            minimize_scale = scale_x + scale_y
            heatmap_data = np.zeros((scale_y + 1, scale_x + 1))
            for bbox in self.data['bbox']:
                x, y = self.scale_position(bbox, scale_x, scale_y)
                heatmap_data[int(y), int(x)] += 1
            if np.all(heatmap_data <= 1):
                return minimize_scale
            else:
                return minimize_scale + np.sum(heatmap_data)
        
        bounds = [(5, 25), (25, 75)]
        result = differential_evolution(objective, bounds)
        return map(int, result.x)
    
    def apply_scale_and_create_heatmap(self, scale_x, scale_y):
        self.data[['bbox_scale_position_x', 'bbox_scale_position_y']] = self.data['bbox'].apply(
            lambda bbox: self.scale_position(bbox, scale_x, scale_y)).tolist()
        
        self.data['bbox_scale_position_x'] = self.data['bbox_scale_position_x'].apply(
            lambda x: min(max(int(x), 0), scale_x - 1))
        self.data['bbox_scale_position_y'] = self.data['bbox_scale_position_y'].apply(
            lambda y: min(max(int(y), 0), scale_y - 1))
        
        heatmap_data = np.empty((scale_y + 1, scale_x + 1), dtype=object)
        for y in range(scale_y + 1):
            for x in range(scale_x + 1):
                heatmap_data[y, x] = []
        
        for index, row in self.data.iterrows():
            x, y, text = row['bbox_scale_position_x'], row['bbox_scale_position_y'], row['text']
            heatmap_data[y, x].append(text)
        
        return pd.DataFrame(heatmap_data)
    
    def reassign_regions(self, heatmap_df):
        empty_indices = []  # Индексы всех пустых рядов
        transition_indices = []  # Индексы переходов от 'value' к 'metadata'
        current_empty_series = []  # Текущая серия пустых 'metadata' рядов
        series_to_reassign = []  # Список серий для переназначения

        for idx, row in heatmap_df.iterrows():
            # Определение пустой строки
            row_values = [' '.join(map(str, cell)) if isinstance(cell, list) else str(cell) for cell in row[:-1]]
            is_empty = all(value == "" for value in row_values)

            if is_empty:
                empty_indices.append(idx)

            if row['region'] == 'metadata':
                if is_empty:
                    # Если текущая строка- пустая 'metadata', добавляем в текущую серию
                    current_empty_series.append(idx)
                else:
                    # Если серия пустых 'metadata' закончилась, сохраняем ее и сбрасываем
                    if current_empty_series:
                        series_to_reassign.append(current_empty_series)
                        current_empty_series = []
            elif row['region'] == 'value' and current_empty_series:
                # Если мы нашли 'value' после пустых 'metadata', значит это переход
                transition_indices.append(current_empty_series[0] - 1)  # Предполагаем, что строка перед серией - 'value'
                current_empty_series = []

        # Обработка последней серии в df
        if current_empty_series:
            series_to_reassign.append(current_empty_series)

        # Переназначение половины индексов в каждой серии
        for series in series_to_reassign:
            mid_point = len(series) // 2
            for idx in series[:mid_point]:
                heatmap_df.at[idx, 'region'] = 'value'

        return heatmap_df
    
    def process_df(self):        
        self.best_scale_x, self.best_scale_y = self.optimize_scale_factors()
        heatmap_df = self.apply_scale_and_create_heatmap(self.best_scale_x, self.best_scale_y)        
        
        heatmap_df['region'] = heatmap_df.apply(self.identify_table_region, axis=1)
        heatmap_df = self.reassign_regions(heatmap_df)
        heatmap_df['prev_region'] = heatmap_df['region'].shift(1, fill_value='metadata')
        heatmap_df['next_region'] = heatmap_df['region'].shift(-1, fill_value='metadata')        
        
        def update_region(row):
            if row['region'] == 'value' and row['prev_region'] != 'value' and row['next_region'] != 'value':
                return 'metadata'
            elif row['region'] == 'metadata' and row['prev_region'] != 'metadata' and row['next_region'] != 'metadata':
                return 'value'
            return row['region']
        
        heatmap_df['region'] = heatmap_df.apply(update_region, axis=1)
        
        # Назначение номеров таблиц
        heatmap_df['table_number'] = self.assign_table_numbers(heatmap_df['region'])
        
        return heatmap_df

    @staticmethod
    def identify_table_region(row):
        string_row = [' '.join(map(str, cell)) if isinstance(cell, list) else str(cell) for cell in row]
        float_count = sum([1 for cell in string_row if PageData.is_convertible_to_float(cell)])
        return 'value' if float_count >= 2 else 'metadata'

    @staticmethod
    def is_convertible_to_float(value):
        try:
            float(value)
            return True
        except ValueError:
            return False
    
    @staticmethod
    def assign_table_numbers(region_series):
        table_number = 0
        table_numbers = []
        last_region = None
        for region in region_series:
            if last_region == 'value' and region == 'metadata':
                table_number += 1
            table_numbers.append(table_number)
            last_region = region
        return table_numbers

