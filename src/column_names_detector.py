import cv2
import numpy as np
import pandas as pd

class ColumnLineDetector:
    def __init__(self, img_path, ocr_df, debug=False):
        self.img_path = img_path
        self.debug = debug
        self.img = cv2.imread(img_path)
        self.img_height, self.img_width = self.img.shape[:2]
        self.ocr_df = ocr_df

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

    def filter_lines_crossing_text(self, lines):
        filtered_lines = []
        for line in lines:
            intersect_count = 0
            line_coords = (line[0], line[1], line[2], line[3])

            for index, row in self.ocr_df.iterrows():
                if row['conf'] > 0:
                    rect_coords = self.get_rect_from_bbox(row['bbox'])
                    if self.line_intersects_rect(line_coords, rect_coords):
                        intersect_count += 1

            if intersect_count <= 1:  # Вы можете изменить порог в зависимости от потребностей
                filtered_lines.append(line)

        return filtered_lines

    def get_rect_from_bbox(self, bbox):
        # Преобразует bbox из формата [[x1, y1], [x2, y2], ...] в (x, y, width, height)
        x_coordinates, y_coordinates = zip(*bbox)
        x1, y1 = min(x_coordinates), min(y_coordinates)
        width, height = max(x_coordinates) - x1, max(y_coordinates) - y1
        return (x1, y1, width, height)
    
    def find_horizontal_lines(self, min_width):
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=min_width, maxLineGap=20)
        horizontal_lines = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(y2 - y1) < 10:  # Угол почти горизонтальный
                horizontal_lines.append((x1, y1, x2, y2))

        return horizontal_lines

    def combine_lines(self, lines, height_tolerance):
        combined_lines = []

        for line in lines:
            x1, y1, x2, y2 = line
            combined = False

            for combined_line in combined_lines:
                cx1, cy1, cx2, cy2 = combined_line
                if abs(y1 - cy1) <= height_tolerance:
                    new_x1 = min(x1, cx1)
                    new_x2 = max(x2, cx2)
                    combined_lines.remove(combined_line)
                    combined_lines.append((new_x1, cy1, new_x2, cy2))
                    combined = True
                    break
            
            if not combined:
                combined_lines.append(line)

        return combined_lines

    def filter_lines_by_width(self, lines, min_width):
        return [line for line in lines if line[2] - line[0] >= min_width]

    def detect_column_names_area(self):
        horizontal_lines = self.find_horizontal_lines(self.img_width / 20)
        combined_lines = self.combine_lines(horizontal_lines, self.img_height / 100)
        filtered_lines = self.filter_lines_by_width(combined_lines, self.img_width / 5)        
        text_filtered_lines = self.filter_lines_crossing_text(filtered_lines)

        if self.debug:
            for line in text_filtered_lines:
                x1, y1, x2, y2 = line
                cv2.line(self.img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.imwrite('debug_column_names_area.png', self.img)

        return text_filtered_lines


