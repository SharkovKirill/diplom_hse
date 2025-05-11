import cv2
import math
from ultralytics import YOLO
import cv2
import numpy as np


class PhotoQualityAnalyzer:
    def __init__(self, pil_image):
        self.image_rgb = np.array(pil_image)
        self.image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        self.h, self.w = self.gray.shape

    def check_blur(self, threshold=100):
        """Проверка на размытость (используем вариацию Лапласа)"""
        fm = cv2.Laplacian(self.gray, cv2.CV_64F, ksize=1).var()
        return fm < threshold, fm

    def check_exposure_on_gray(self, dark_thresh=50, bright_thresh=220):
        """Проверка на слишком темное или пересвеченное изображение в градациях серого"""
        hist = cv2.calcHist([self.gray], [0], None, [256], [0, 256])
        dark_pixels = np.sum(hist[:dark_thresh])
        bright_pixels = np.sum(hist[bright_thresh:])
        total_pixels = self.h * self.w

        is_dark = (dark_pixels / total_pixels) > 0.3  # >30% темных пикселей
        is_bright = (bright_pixels / total_pixels) > 0.3  # >30% светлых пикселей

        return (
            is_dark,
            is_bright,
            dark_pixels / total_pixels,
            bright_pixels / total_pixels,
        )

    def check_exposure_on_hsv(self, dark_thresh=50, bright_thresh=220):
        """Проверка на слишком темное или пересвеченное изображение в hsv"""
        hist = self.hsv[:, :, 2].ravel()
        dark_pixels = np.sum(hist <= dark_thresh)
        bright_pixels = np.sum(hist >= bright_thresh)
        total_pixels = self.h * self.w

        is_dark = (dark_pixels / total_pixels) > 0.3  # >30% темных пикселей
        is_bright = (bright_pixels / total_pixels) > 0.3  # >30% светлых пикселей

        return (
            is_dark,
            is_bright,
            dark_pixels / total_pixels,
            bright_pixels / total_pixels,
        )

    def check_noise(self, threshold=10):
        """Проверка на шумы (используем медианный фильтр для сравнения)"""
        denoised = cv2.medianBlur(self.gray, 3)
        mse = np.mean((self.gray - denoised) ** 2)
        return mse > threshold, mse

    def check_horizon(self, angle_threshold=3):
        """Проверка на заваленный горизонт"""
        temp_img = self.image_rgb.copy()
        blurred = cv2.GaussianBlur(self.gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 150, 250, apertureSize=3)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10
        )

        if lines is None:
            return False, 0

        total_lines, total_horiizontal_lines, good_horizontal_lines = 0, 0, 0
        angles = []
        for line in lines:
            total_lines += 1
            x1, y1, x2, y2 = line[0]
            angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
            if angle <= 30:  # учитываем только горизонтальные линии
                cv2.line(temp_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                total_horiizontal_lines += 1
                angles.append(angle)
                if angle <= angle_threshold:
                    good_horizontal_lines += 1
        avg_angle = np.mean(angles)
        if total_horiizontal_lines == 0:

            return (False, 0, temp_img)
        else:
            return (
                (good_horizontal_lines / total_horiizontal_lines) >= 0.5,
                avg_angle,
                temp_img,
            )

    def person_detection(self):
        model = YOLO("yolo11x.pt")
        model.to("cpu")
        output_image = self.image.copy()
        results = model.predict(source=self.image, save=False)

        person_count = 0

        for result in results:
            for box, cls, conf in zip(
                result.boxes.xyxy, result.boxes.cls, result.boxes.conf
            ):
                if cls == 0:
                    person_count += 1
                    x1, y1, x2, y2 = map(int, box[:4])

                    cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

                    label = f"Person {conf:.2f}"
                    cv2.putText(
                        output_image,
                        label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )

        return person_count, cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
