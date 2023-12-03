import cv2
import numpy as np
import pytesseract
import matplotlib.pyplot as plt
import csv

class ColumnValuesDetector:
    def __init__(self, img_path, ocr_data, debug=False):
        self.img_path = img_path
        self.debug = debug
        self.ocr_data = ocr_data
        self.img = cv2.imread(img_path)
        self.img_height, self.img_width = self.img.shape[:2]

    def line_intersects_rect(self, line, rect):        
        rx, ry, rw, rh = rect
        # Конечные точки прямоугольника
        left_line = (rx, ry, rx, ry + rh)
        right_line = (rx + rw, ry, rx + rw, ry + rh)
        top_line = (rx, ry, rx + rw, ry)
        bottom_line = (rx, ry + rh, rx + rw, ry + rh)

        # Проверка пересечения с каждой стороной прямоугольника
        return self.line_intersects_line(line, left_line) or \
               self.line_intersects_line(line, right_line) or \
               self.line_intersects_line(line, top_line) or \
               self.line_intersects_line(line, bottom_line)

    def line_intersects_line(self, line1, line2): 
        def to_homog(point):
            return np.array([point[0], point[1], 1])

        def line_from_points(p1, p2):
            return np.cross(to_homog(p1), to_homog(p2))

        p1, p2 = (line1[0], line1[1]), (line1[2], line1[3])
        p3, p4 = (line2[0], line2[1]), (line2[2], line2[3])

        l1 = line_from_points(p1, p2)
        l2 = line_from_points(p3, p4)

        intersection = np.cross(l1, l2)
        if intersection[2] == 0:  # параллельные линии
            return False

        # преобразование обратно в неоднородные координаты
        intersection = (intersection[0] / intersection[2], intersection[1] / intersection[2])

        # Проверка нахождения точки пересечения внутри отрезков
        in_segment1 = min(p1[0], p2[0]) <= intersection[0] <= max(p1[0], p2[0]) and \
                    min(p1[1], p2[1]) <= intersection[1] <= max(p1[1], p2[1])
        in_segment2 = min(p3[0], p4[0]) <= intersection[0] <= max(p3[0], p4[0]) and \
                    min(p3[1], p4[1]) <= intersection[1] <= max(p3[1], p4[1])

        return in_segment1 and in_segment2

    def filter_lines_crossing_text(self, lines, ocr_data, threshold=1):
        filtered_lines = []
        for line in lines:
            intersect_count = 0
            line_coords = (line[0], line[1], line[2], line[3])

            for i in range(len(ocr_data['text'])):
                if int(ocr_data['conf'][i]) > 60:
                    rect_coords = (ocr_data['left'][i], ocr_data['top'][i], ocr_data['width'][i], ocr_data['height'][i])
                    if self.line_intersects_rect(line_coords, rect_coords):
                        intersect_count += 1

            if intersect_count <= threshold:
                filtered_lines.append(line)

        return filtered_lines

    def find_vertical_lines(self, min_height):
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=min_height, maxLineGap=20)
        vertical_lines = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(x2 - x1) < 10:  # Угол почти вертикальный
                vertical_lines.append((x1, y1, x2, y2))

        return vertical_lines

    def combine_close_vertical_lines(self, lines, width_threshold):
        def can_combine(line1, line2):
            _, y1_1, _, y2_1 = line1
            _, y1_2, _, y2_2 = line2
            return not (y2_1 < y1_2 or y2_2 < y1_1)

        while True:
            combined_lines = []
            lines = sorted(lines, key=lambda x: x[0])  # Сортируем линии по X координате
            used_lines = set()

            for i, current_line in enumerate(lines):
                if i in used_lines:
                    continue

                x1, y1, x2, y2 = current_line
                to_combine = [current_line]
                used_lines.add(i)

                for j, other_line in enumerate(lines):
                    if j in used_lines:
                        continue

                    ox1, oy1, ox2, oy2 = other_line
                    if abs(ox1 - x1) < width_threshold and can_combine(current_line, other_line):
                        to_combine.append(other_line)
                        used_lines.add(j)

                # Вычисляем среднюю X координату
                avg_x = int(sum([l[0] for l in to_combine]) / len(to_combine))
                min_y = min([l[1] for l in to_combine])
                max_y = max([l[3] for l in to_combine])
                combined_lines.append((avg_x, min_y, avg_x, max_y))

            if len(combined_lines) == len(lines):
                break  # Прекращаем, если не произошло изменений
            lines = combined_lines

        return combined_lines

    def detect_column_values_area(self):
        print("Начало функции detect_column_values_area")

        vertical_lines = self.find_vertical_lines(self.img_height / 10)
        print(f"Найдено вертикальных линий: {len(vertical_lines)}")

        combined_lines = self.combine_close_vertical_lines(vertical_lines, self.img_width / 10)
        print(f"Линии после объединения: {len(combined_lines)}")

        filtered_lines = self.filter_lines_crossing_text(combined_lines, self.ocr_data)
        print(f"Линии после фильтрации: {len(filtered_lines)}")

        if self.debug:
            print("Отладка включена, начинаем отрисовку линий и сохранение гистограмм")

            # Отрисовка линий на изображении
            for line in filtered_lines:
                x1, y1, x2, y2 = line
                cv2.line(self.img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.imwrite('debug_column_values_area.png', self.img)
            print("Изображение с линиями сохранено: debug_column_values_area.png")

            # Подготовка данных для гистограммы
            centers_x = [(line[0] + line[2]) / 2 for line in filtered_lines]
            centers_y = [(line[1] + line[3]) / 2 for line in filtered_lines]

            if not centers_x or not centers_y:
                print("Предупреждение: центры линий отсутствуют, гистограммы не будут созданы")
            else:
                # Создание и сохранение гистограмм
                fig, axs = plt.subplots(2, 1, figsize=(5, 8))
                axs[0].hist(centers_x, bins=20, color='blue')
                axs[0].set_title('Гистограмма распределения по ширине')
                axs[1].hist(centers_y, bins=20, color='red')
                axs[1].set_title('Гистограмма распределения по высоте')
                plt.tight_layout()
                plt.savefig('debug_histograms.png')
                print("Гистограммы сохранены: debug_histograms.png")

                # Сохранение данных гистограммы в CSV файл
                with open('histogram_data.csv', 'w', newline='') as csvfile:
                    csvwriter = csv.writer(csvfile)
                    csvwriter.writerow(['Center_X', 'Center_Y'])
                    csvwriter.writerows(zip(centers_x, centers_y))
                print("Данные гистограммы сохранены в CSV: histogram_data.csv")

        print("Завершение функции detect_column_values_area")
        return filtered_lines


