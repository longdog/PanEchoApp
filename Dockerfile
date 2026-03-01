FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY app/requirements.txt /app/app/requirements.txt

RUN pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision \
    && pip install -r /app/app/requirements.txt

COPY . /app

ENV TORCH_HOME=/models/torch_home \
    PYTHONPATH=/app

VOLUME ["/models", "/output"]

ENTRYPOINT ["python", "/app/app/main.py"]
