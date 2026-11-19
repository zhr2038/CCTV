FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/tmp \
    XDG_CACHE_HOME=/tmp \
    HF_HUB_OFFLINE=1 \
    OMP_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        intel-media-va-driver \
        libglib2.0-0 \
        fonts-noto-cjk \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
        "numpy==1.26.4" \
        "tflite-runtime==2.14.0" \
        "onnxruntime==1.19.2" \
        "faster-whisper==1.1.1" \
        "opencv-python-headless==4.10.0.84" \
        "Pillow>=10.4,<12" \
        "requests>=2.32,<3"

RUN pip install --no-cache-dir "cryptography>=42,<47"

WORKDIR /app
COPY models/ /app/models/
COPY config.toml /app/config.toml
COPY nas_growth/ /app/nas_growth/

RUN mkdir -p /app/conf && chmod -R a+rX /app

USER 1000:10
EXPOSE 18795

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:18795/health.json', timeout=3).read()"

ENTRYPOINT ["python", "-m", "nas_growth.app", "--config", "/app/config.toml"]
CMD ["serve", "--host", "0.0.0.0", "--port", "18795"]
