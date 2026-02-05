from typing import Literal

import cv2
import numpy as np
from PIL import Image
from loguru import logger


class TextRecognizer:
    def __init__(self, languages: list[str], gpu: bool = False, ocr_type: Literal["manga", "doctr", "easy", "paddle"] = None):
        if ocr_type == "manga":
            from manga_ocr import MangaOcr
            self.model = MangaOcr()
        elif ocr_type == "doctr":
            from doctr.models import ocr_predictor
            self.model = ocr_predictor(pretrained=True)
        elif ocr_type == "easy":
            import easyocr
            self.model = easyocr.Reader(languages, gpu=gpu)
        elif ocr_type == "paddle":
            from paddleocr import PaddleOCR
            self.model = PaddleOCR(
                lang=languages[0],
                text_recognition_model_name="PP-OCRv5_mobile_rec",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
        else:
            logger.error("OCR не инициализирован...")
            raise ValueError("OCR не инициализирован")

    def recognize_paddle(self, image: np.ndarray) -> str:
        result = self.model.predict(image)
        if isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict) and "rec_texts" in result[0]:
            result = result[0]

        texts = []
        if isinstance(result, dict) and "rec_texts" in result:
            rec_texts = result["rec_texts"]
            for text in rec_texts:
                texts.append(text)
        return " ".join(texts).lower()

    def recognize_doctr(self, image: np.ndarray) -> str:
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        result = self.model([image])
        texts = []
        for page in result.pages:
            for block in page.blocks:
                for line in block.lines:
                    for word in line.words:
                        if hasattr(word, "value"):
                            texts.append(word.value)

        text = " ".join(texts).strip().lower()
        logger.debug(f"Doctr распознал: {text}...")
        return text

    def recognize_mangaocr(self, image: np.ndarray) -> str:
        if isinstance(image, np.ndarray):
            if len(image.shape) == 3:
                if image.shape[2] == 3:
                    pil_image = Image.fromarray(image)
                else:
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(rgb_image)
            else:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                pil_image = Image.fromarray(rgb_image)
        else:
            pil_image = image
        text = self.model(pil_image)
        logger.debug(f"Manga OCR распознал: {text}...")
        return text

    def recognize_easyocr(self, image: np.ndarray) -> str:
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        try:
            results = self.reader.readtext(image, detail=0, paragraph=True)
            if results:
                text = " ".join(results).strip()
                logger.debug(f"Easy OCR распознал: {text}...")
                return text
            return ""
        except Exception as e:
            logger.error(f"Ошибка OCR: {e}")
            return ""

    def recognize_with_confidence(self, image: np.ndarray, confidence_threshold: float = 0.5):
        try:
            results = self.reader.readtext(image)
            texts = []
            for (bbox, text, prob) in results:
                if prob >= confidence_threshold:
                    texts.append(text)
            return " ".join(texts).strip()
        except Exception as e:
            logger.error(f"Ошибка OCR с confidence: {e}")
            return ""
