# Stage 1: Build wheels
FROM --platform=linux/amd64 python:3.11-slim AS builder

WORKDIR /build

# Install only essential build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# For x86 architecture, we can optimize with CPU-only PyTorch
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

# Stage 2: Download and cache YOLOv8 model - NOW WITH OPENCV DEPENDENCIES
FROM --platform=linux/amd64 python:3.11-slim AS model-downloader

WORKDIR /models

# Install necessary dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install only ultralytics to download the model
RUN pip install --no-cache-dir ultralytics

# Download the YOLOv8 model - this layer will be cached if the command doesn't change
RUN python -c "from ultralytics import YOLO; model = YOLO('yolov8n.pt', task='detect')" && \
    mkdir -p /models/yolo && \
    mv yolov8n.pt /models/yolo/

# Stage 3: Runtime application
FROM --platform=linux/amd64 python:3.11-slim

WORKDIR /app

# Install only runtime dependencies (minimal set)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    libgomp1 \
    curl \
    gnupg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Properly install Microsoft ODBC driver by removing conflicting packages first
RUN curl -sSL https://packages.microsoft.com/keys/microsoft.asc | apt-key add - \
    && curl -sSL https://packages.microsoft.com/config/debian/11/prod.list > /etc/apt/sources.list.d/mssql-release.list \
    && apt-get update \
    # First remove any conflicting packages to avoid dependency conflicts
    && apt-get remove -y unixodbc-dev unixodbc odbcinst odbcinst1debian2 libodbc1 \
    # Force install with --no-install-recommends to minimize dependencies
    && ACCEPT_EULA=Y apt-get install -y --no-install-recommends msodbcsql18 \
    # Now install the development package needed for pyodbc
    && apt-get install -y --no-install-recommends unixodbc-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set ODBC environment variables
ENV ODBCINI=/etc/odbc.ini \
    ODBCSYSINI=/etc \
    LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu \
    PYTHONWARNINGS="ignore::DeprecationWarning:OpenSSL,ignore:Could not initialize NNPACK" \
    OPENSSL_CONF=/dev/null

# Create model directory structure
RUN mkdir -p /app/models/yolo /app/models/insightface_data /app/models/eye_state_model

# Copy wheels from builder stage and install
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir --no-index --find-links=/wheels /wheels/* && \
    rm -rf /wheels

# Copy pre-downloaded YOLOv8 model
COPY --from=model-downloader /models/yolo/yolov8n.pt /app/models/yolo/yolov8n.pt

# Copy application files - do this last to leverage caching
COPY . .

# Run the FastAPI application with uvicorn server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]