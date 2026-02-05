from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from loguru import logger
from tqdm import tqdm

from config.settings import settings

from .detector import BubbleDetector
from .inpainter import TextInpainter
from .ocr import TextRecognizer
from .translator import MultiLanguageTranslator


class MangaTranslatorPipeline:
    def __init__(self, yolo_model_path: str = None,
                 source_lang: str = None,
                 selected_font: str = None,
                 ocr_type: Literal["manga", "doctr", "easy", "paddle"] = None,
                 translator_type: Literal["google", "transformers"] = None):
        model_path = yolo_model_path or settings.YOLO_MODEL_PATH
        self.ocr_type = ocr_type
        self.detector = BubbleDetector(str(model_path))
        self.ocr = TextRecognizer([source_lang] if source_lang else settings.SUPPORTED_LANGUAGES, settings.OCR_GPU, ocr_type)
        self.translator = MultiLanguageTranslator(translator_type)
        self.inpainter = TextInpainter(selected_font, str(settings.FONT_DIR))

        self.source_lang = source_lang
        logger.info("Пайплайн инициализирован")

    def process_single_bubble(self, image: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        crop = image[y1:y2, x1:x2].copy()
        if self.ocr_type == "manga":
            text = self.ocr.recognize_mangaocr(crop)
        elif self.ocr_type == "doctr":
            text = self.ocr.recognize_doctr(crop)
        elif self.ocr_type == "easy":
            text = self.ocr.recognize_easyocr(crop)
        elif self.ocr_type == "paddle":
            text = self.ocr.recognize_paddle(crop)
        else:
            logger.warning("Этот тип OCR не поддерживается...")
            text = None

        if text:
            translated = self.translator.translate(text, self.source_lang)
            cleaned, largest_contour = self.inpainter.remove_text(crop)
            final_crop = self.inpainter.draw_text(cleaned, largest_contour, translated)
            image[y1:y2, x1:x2] = final_crop
        return image

    def process_image(self, input_path: str, output_path: str,
                     show_progress: bool = True) -> bool:
        try:
            img, bboxes = self.detector.detect(input_path)
            if not bboxes:
                logger.warning(f"На изображении {input_path} не найдено пузырей")
                cv2.imwrite(output_path, img)
                return True

            if show_progress:
                bboxes_iter = tqdm(bboxes, desc="Обработка пузырей")
            else:
                bboxes_iter = bboxes

            for bbox in bboxes_iter:
                img = self.process_single_bubble(img, bbox)
            cv2.imwrite(output_path, img)
            logger.info(f"Изображение сохранено: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Ошибка при обработке {input_path}: {e}")
            return False

    def process_batch(self, input_dir: str, output_dir: str) -> dict:
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        image_paths = []
        for ext in settings.SUPPORTED_EXTENSIONS:
            image_paths.extend(input_dir.glob(f"*{ext}"))

        results = {
            "total": len(image_paths),
            "success": 0,
            "failed": 0,
            "failed_files": []
        }

        def _process_file(img_path):
            output_path = output_dir / img_path.name
            success = self.process_image(str(img_path), str(output_path), show_progress=False)
            return img_path.name, success

        for img_path in tqdm(image_paths, desc="Пакетная обработка"):
            fname, success = _process_file(img_path)
            if success:
                results["success"] += 1
            else:
                results["failed"] += 1
                results["failed_files"].append(fname)
        logger.info(f"Пакетная обработка завершена: {results['success']}/{results['total']} успешно")
        return results
