from pathlib import Path

from loguru import logger
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    BASE_DIR: Path = Path(__file__).parent.parent
    MODEL_DIR: Path = BASE_DIR / "models"
    DATA_DIR: Path = BASE_DIR / "data"
    FONT_DIR: Path = BASE_DIR / "data" / "fonts"
    INPUT_DIR: Path = BASE_DIR / "data" / "in"
    OUTPUT_DIR: Path = BASE_DIR / "data" / "out"

    YOLO_MODEL_PATH: Path = MODEL_DIR / "yolo_best.pt"
    OCR_GPU: bool = False

    MAX_FILE_SIZE_MB: int = 5
    SUPPORTED_EXTENSIONS: list[str] = [".jpg"]
    SUPPORTED_LANGUAGES: list[str] = ["ja", "en"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._validate_directories()

    def _validate_directories(self):
        dirs_to_check = [
            # tuple(dir, is_empty)
            (self.MODEL_DIR, False),
            (self.DATA_DIR, False),
            (self.FONT_DIR, False),
            (self.INPUT_DIR, True),
            (self.OUTPUT_DIR, True)
        ]

        for directory, is_empty in dirs_to_check:
            if not directory.exists():
                logger.info(f"Создаем директорию: {directory}")
                directory.mkdir(parents=True, exist_ok=True)

            if not directory.is_dir():
                raise ValueError(f"{directory} существует, но это не директория")

            if not is_empty:
                bowels = list(directory.glob("*"))

                if not bowels:
                    raise ValueError(f"{directory} должна быть не пустой")


settings = Settings()
