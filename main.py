import base64
import io
import os
import tempfile
import zipfile
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw, ImageFont

from config.settings import settings
from src import MangaTranslatorPipeline


if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "results" not in st.session_state:
    st.session_state.results = None


def _get_available_fonts() -> list:
    font_files = []
    for ext in [".ttf", ".otf"]:
        font_files.extend(list(settings.FONT_DIR.glob(f"*{ext}")))
    return sorted([f.stem for f in font_files if f.is_file()])


def _create_font_preview(font_name: str, font_size: int = 24) -> str:
    try:
        font_files = list(settings.FONT_DIR.glob(f"{font_name}.*"))
        if not font_files:
            return ""

        font_path = str(font_files[0])
        img = Image.new("RGB", (300, 60), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype(font_path, font_size)

        text = "Привет"
        draw.text((10, 10), text, font=font, fill=(0, 0, 0))
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"
    except Exception:
        return ""


with st.sidebar:
    st.title("⚙️ Настройки")
    source_lang = st.selectbox(
        "Исходный язык комикса",
        options=["ja", "en", "auto"],
        help="Выберите язык оригинала или 'auto' для автоопределения"
    )

    ocr_type = st.selectbox(
        "Тип модели OCR (перевод)",
        options=[ "manga", "paddle", "doctr", "easy"],
        help="Выберите тип OCR."
    )

    available_fonts = _get_available_fonts()

    if not available_fonts:
        st.sidebar.error("В папке шрифтов нет шрифтов")
        st.sidebar.info(f"Добавьте .ttf/.otf файлы в: {settings.FONT_DIR}")

    selected_font = st.sidebar.selectbox(
        "Выберите шрифт для текста",
        options=available_fonts,
        index=0,
        help="Шрифт будет использоваться для вставки переведенного текста"
    )

    if st.sidebar.checkbox("Показать предпросмотр шрифта", value=False):
        preview_size = st.sidebar.slider("Размер предпросмотра", 16, 36, 24)
        preview_image = _create_font_preview(selected_font, preview_size)
        if preview_image:
            st.sidebar.markdown(
                f'<img src="{preview_image}" width="100%">',
                unsafe_allow_html=True
            )
            st.sidebar.caption(f"Шрифт: {selected_font} ({preview_size}px)")
        else:
            st.sidebar.warning("Не удалось создать предпросмотр")

    translator_type = st.sidebar.selectbox(
        "Выберите переводчика",
        options=["google", "transformers"],
        index=0,
        help= ("В зависимости от типа переводчика будут использованы модели "
               "transformers (Helsinki-NLP/opus-mt) или стороннее API (Google).")
    )

    if st.button("🔄 Инициализировать пайплайн", type="primary"):
        try:
            st.session_state.pipeline = MangaTranslatorPipeline(
                source_lang=None if source_lang == "auto" else source_lang,
                selected_font=selected_font,
                ocr_type=ocr_type,
                translator_type=translator_type
            )
            st.success("Пайплайн готов к работе!")
        except Exception as e:
            st.error(f"Ошибка инициализации: {e}")


st.title("📚 Переводчик манги")
st.markdown("---")

tab1, tab2= st.tabs(["📤 Одно изображение", "📁 Пакетная обработка"])


with tab1:
    st.header("Перевод одного изображения")

    delete_checkbox = st.checkbox(
        "Удалить изображения",
        value=False,
        help="Если отмечено, изображения будут удалены после обработки"
    )

    uploaded_file = st.file_uploader(
        f"Выберите изображение (ограничение {settings.MAX_FILE_SIZE_MB}MB)",
        type=settings.SUPPORTED_EXTENSIONS,
        max_upload_size=settings.MAX_FILE_SIZE_MB,
        key="single_upload",
    )
    if uploaded_file and st.session_state.pipeline:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Оригинал")
            image = Image.open(uploaded_file)
            st.image(image, width="content")

        if st.button("🚀 Начать перевод", type="primary"):
            with st.spinner("Обработка..."):
                with tempfile.NamedTemporaryFile(dir=settings.INPUT_DIR, delete=False, suffix=".jpg") as tmp_file:
                    image.save(tmp_file.name)
                    input_path = tmp_file.name

                with tempfile.NamedTemporaryFile(dir=settings.OUTPUT_DIR, delete=False, suffix=".jpg") as tmp_output:
                    output_path = tmp_output.name

                success = st.session_state.pipeline.process_image(input_path, output_path)
                if success:
                    with col2:
                        st.subheader("Результат перевода")
                        result_image = Image.open(output_path)
                        st.image(result_image, width="content")

                    with open(output_path, "rb") as f:
                        img_bytes = f.read()

                    st.download_button(
                        label="💾 Скачать результат",
                        data=img_bytes,
                        file_name=f"translated_{uploaded_file.name}",
                        mime="image/jpg"
                    )

                if delete_checkbox:
                    os.unlink(input_path)
                    os.unlink(output_path)


with tab2:
    st.header("Пакетная обработка")

    uploaded_files = st.file_uploader(
        f"Выберите несколько изображений (ограничение {settings.MAX_FILE_SIZE_MB*10}MB)",
        type=settings.SUPPORTED_EXTENSIONS,
        accept_multiple_files=True,
        max_upload_size=settings.MAX_FILE_SIZE_MB*10,
        key="batch_upload"
    )

    if uploaded_files and st.session_state.pipeline:
        if st.button("🚀 Перевести все", type="primary"):
            with st.spinner(f"Перевод {len(uploaded_files)} изображений..."):
                with tempfile.TemporaryDirectory(dir=settings.DATA_DIR) as tmp_input:
                    with tempfile.TemporaryDirectory(dir=settings.DATA_DIR) as tmp_output:
                        for i, uploaded_file in enumerate(uploaded_files):
                            file_path = Path(tmp_input) / uploaded_file.name
                            with open(file_path, "wb") as f:
                                f.write(uploaded_file.getvalue())

                        results = st.session_state.pipeline.process_batch(tmp_input, tmp_output)

                        zip_buffer = io.BytesIO()
                        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                            for file in Path(tmp_output).iterdir():
                                zip_file.write(file, file.name)

                        zip_buffer.seek(0)

                        st.success(f"Обработано: {results['success']}/{results['total']}")

                        if results["failed"] > 0:
                            st.warning(f"Не удалось обработать: {results['failed']}")
                            for fname in results["failed_files"]:
                                st.write(f"- {fname}")

                        st.download_button(
                            label=f"📦 Скачать все результаты ({results['success']} файлов)",
                            data=zip_buffer,
                            file_name="translated.zip",
                            mime="application/zip"
                        )


st.markdown("""
<style>
.stButton > button {
    width: 100%;
}
.stDownloadButton > button {
    width: 100%;
}
</style>
""", unsafe_allow_html=True)
