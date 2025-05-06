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

# Runtime stage
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

# Copy wheels from builder stage and install
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir --no-index --find-links=/wheels /wheels/* && \
    rm -rf /wheels

# Copy application files
COPY . .

# Use smaller YOLOv8 model and pre-download
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt', task='detect')" && \
    mkdir -p /app/models/yolo && \
    mv yolov8n.pt /app/models/yolo/

# Run the FastAPI application with uvicorn server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]