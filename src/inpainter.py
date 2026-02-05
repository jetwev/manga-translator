import logging
import textwrap
from typing import Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


logger = logging.getLogger(__name__)


class TextInpainter:
    def __init__(self, selected_font: str, font_dir: str):
        self.fonts = {}
        self.selected_font = selected_font
        self._load_fonts(font_dir)

    def _load_fonts(self, font_dir: str):
        import os
        for fname in os.listdir(font_dir):
            if fname.endswith(".ttf") or fname.endswith(".otf"):
                font_path = os.path.join(font_dir, fname)
                try:
                    _ = ImageFont.truetype(font_path, 20)
                    font_name = os.path.splitext(fname)[0]
                    self.fonts[font_name] = font_path
                except Exception as e:
                    logger.warning(f"Не удалось загрузить шрифт {fname}: {e}")

    def remove_text(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [largest_contour], -1, 255, cv2.FILLED)
        image[mask == 255] = (255, 255, 255)
        # Использование KMeans для заполнения фона
        # from sklearn.cluster import KMeans
        # kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        # kmeans.fit(pixels)
        # labels, counts = np.unique(kmeans.labels_, return_counts=True)
        # color = kmeans.cluster_centers_[labels[np.argmax(counts)]].astype(int)
        return image, largest_contour

    def calculate_font_size(self, text: str, bbox_size: Tuple[int, int],
                           max_font_size: int = 36, min_font_size: int = 7) -> Tuple[ImageFont.FreeTypeFont, list]:
        width, height = bbox_size

        for font_size in range(max_font_size, min_font_size, -1):
            try:
                font = ImageFont.truetype(self.fonts[self.selected_font], font_size)

                avg_char_width = font_size * 0.9
                max_chars_per_line = int(width / avg_char_width)
                if max_chars_per_line <= 0:
                    continue

                lines = textwrap.wrap(text, width=max_chars_per_line, break_long_words=True)
                line_height = font_size * 1.2
                total_height = len(lines) * line_height

                if total_height < height * 0.9:
                    return font, lines
            except Exception:
                continue

        font = ImageFont.truetype(self.fonts[self.selected_font], min_font_size)
        lines = textwrap.wrap(text, width=20)
        return font, lines

    def draw_text(self, image: np.ndarray, largest_contour: np.ndarray, text: str) -> np.ndarray:
        x, y, w, h = cv2.boundingRect(largest_contour)

        pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        font, lines = self.calculate_font_size(text, (w, h))

        line_height = font.size * 1.2
        total_text_height = len(lines) * line_height

        y_offset = y + (h - total_text_height) // 2

        for _, line in enumerate(lines):
            line_width = draw.textlength(line, font=font)

            x_offset = x + (w - line_width) // 2

            for dx, dy in [(-1,-1), (-1,1), (1,-1), (1,1), (0,-1), (0,1), (-1,0), (1,0)]:
                draw.text((x_offset+dx, y_offset+dy), line, font=font, fill=(255, 255, 255))
            draw.text((x_offset, y_offset), line, font=font, fill=(0, 0, 0))

            y_offset += line_height
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
