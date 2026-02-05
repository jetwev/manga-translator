import asyncio
from typing import Literal

from langdetect import LangDetectException, detect
from loguru import logger
from transformers import pipeline

from config.settings import settings


class MultiLanguageTranslator:
    def __init__(self, translator_type: Literal["google", "transformers"] = None):
        self.translator = None
        if translator_type == "google":
            from googletrans import Translator
            self.translator = Translator()
            logger.info("Переводчик инициализирован")
        elif translator_type == "transformers":
            self.translator = None
            self.models = {
                "ja": pipeline("translation", model="Helsinki-NLP/opus-mt-ja-ru"),
                "en": pipeline("translation", model="Helsinki-NLP/opus-mt-en-ru")
            }
            logger.info("Переводчик инициализирован")
        else:
            logger.error("Неизвестный тип переводчика")

    def detect_language(self, text: str) -> str | None:
        try:
            lang = detect(text)
            return lang if lang in settings.SUPPORTED_LANGUAGES else None
        except LangDetectException:
            return None

    def translate(self, text: str, source_lang: str = None, target_lang: str = "ru") -> str:
        if not text.strip():
            return ""

        if source_lang is None:
            source_lang = self.detect_language(text)
            if source_lang is None:
                logger.warning(f"Не удалось определить язык текста: {text}...")
                return text

        logger.info(f"Перевод с {source_lang} на {target_lang}: {text}...")

        if source_lang == target_lang:
            return text

        if self.translator:
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            ru_text = loop.run_until_complete(self.translator.translate(text, src=source_lang, dest="ru"))
            return ru_text.text
        else:
            try:
                ru_text = self.models[source_lang](text)[0]["translation_text"]
                return ru_text
            except Exception as e:
                logger.error(f"Ошибка перевода: {e}")
                return text
