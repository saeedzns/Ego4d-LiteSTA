FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    YOLO_CONFIG_DIR=/tmp/ultralytics

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends -y libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY deploy/requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.9.0 torchvision==0.24.0 \
    && python -m pip install --no-cache-dir -r /tmp/requirements.txt

COPY local_extraction /app/local_extraction

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p \
        /artifacts \
        /data \
        /tmp/ultralytics \
        /app/local_extraction/toolkit_yolo/runs/sta_yolov8s_singlecls_20251113_002330/weights \
        /app/local_extraction/runs/Track_B/checkpoints \
        /app/local_extraction/v2/org_annotations \
    && chown -R appuser:appuser /artifacts /data /tmp/ultralytics

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).read()"]

CMD ["python", "-m", "local_extraction.inference.api", "--host", "0.0.0.0", "--port", "8000", "--tokenizer-weights", "/artifacts/resnet18-f37072fd.pth"]
