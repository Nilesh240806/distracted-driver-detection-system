FROM python:3.11-slim

# Install system dependencies including OpenGL and GLib for OpenCV/MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip uninstall -y opencv-contrib-python opencv-python || true && \
    pip install --no-cache-dir opencv-python-headless opencv-contrib-python-headless

COPY . .

EXPOSE 10000

ENV PORT=10000
ENV HOST=0.0.0.0

CMD ["python", "app.py"]
