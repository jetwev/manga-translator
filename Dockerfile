FROM python:3.11

ENV LANG=en_US.utf-8
ENV LC_ALL=en_US.utf-8
ENV UV_SYSTEM_PYTHON=1

WORKDIR /app

RUN apt-get update && \
    apt-get install --no-install-recommends -y \
    wget \
    locales \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.9.7 /uv /uvx /bin/

COPY pyproject.toml uv.lock ./

RUN uv sync --locked

COPY . .

VOLUME [ "/app/data"]

ENTRYPOINT ["uv", "run", "streamlit", "run", "main.py"]
