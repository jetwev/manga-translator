from typing import List, Tuple

import cv2
import numpy as np
from loguru import logger
from ultralytics import YOLO


class BubbleDetector:
    def __init__(self, model_path: str, conf_threshold: float = 0.25):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        logger.info(f"Детектор инициализирован с моделью {model_path}")

    def detect(self, image_path: str) -> Tuple[np.ndarray, List[Tuple[int, int, int, int]]]:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Не удалось загрузить изображение: {image_path}")

        results = self.model(img, conf=self.conf_threshold)
        bboxes = []

        for r in results:
            for box in r.boxes:
                if box.conf.item() < self.conf_threshold:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                padding = 5
                h, w = img.shape[:2]
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(w, x2 + padding)
                y2 = min(h, y2 + padding)

                bboxes.append((x1, y1, x2, y2))

        logger.info(f"Обнаружено {len(bboxes)} пузырей")
        return img, bboxes

    def detect_batch(self, image_paths: List[str]):
        results = []
        for path in image_paths:
            try:
                img, bboxes = self.detect(path)
                results.append((path, img, bboxes))
            except Exception as e:
                logger.error(f"Ошибка при обработке {path}: {e}")
                continue
        return results
