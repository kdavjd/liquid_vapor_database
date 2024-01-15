import os
import cv2
import matplotlib.pyplot as plt


class CroppedTables:
    def __init__(self, page, img_path, work_folder_path):
        self.page = page
        self.img_path = img_path
        self.work_folder_path = work_folder_path
        self.metadata = []
        self.value = []

        # Создаем папку для сохранения, если она не существует
        if not os.path.exists(self.work_folder_path):
            os.makedirs(self.work_folder_path)

        self.process_tables()

    def adjust_min_max(self, _min, _max, bbox_mean):
        if (_min_adjusted := _min - bbox_mean) < 0:
            _min_adjusted = 0
        _max_adjusted = _max + bbox_mean
        return _min_adjusted, _max_adjusted

    def crop_image(self, adjusted_min, adjusted_max, region, table_num):
        img = cv2.imread(self.img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Переключение цветовых каналов из BGR в RGB.
        cropped_img = img[adjusted_min:adjusted_max, :, :]
        image_path = os.path.join(self.work_folder_path, f'{region}_{table_num}.png')
        plt.imsave(image_path, cropped_img)
        return image_path

    def process_tables(self):
        for table_num in self.page.df['table_number'].unique():
            for region in ['metadata', 'value']:                
                temp_df = self.page.df.query(f"region == '{region}' and table_number == {table_num}")
                                
                if temp_df.empty:
                    continue

                _min, _max = self.page.unscale_height_position(temp_df.index.min(), temp_df.index.max())
                adjusted_min, adjusted_max = self.adjust_min_max(
                    _min, _max, self.page.data['bbox_height'].mean().astype('int'))

                image_path = self.crop_image(adjusted_min, adjusted_max, region, table_num)

                # Добавление пути к списку соответствующего атрибута
                if region == 'metadata':
                    self.metadata.append(image_path)
                else:
                    self.value.append(image_path)
