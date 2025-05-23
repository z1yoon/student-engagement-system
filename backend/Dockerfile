FROM --platform=linux/amd64 python:3.11-slim

WORKDIR /app

# Install system dependencies for ARM64/Apple Silicon
RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    cmake \
    libopenblas-dev \
    uuid-dev \
    libssl-dev \
    curl \
    gnupg \
    unixodbc-dev \
    # MediaPipe and OpenCV dependencies \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    # InsightFace specific dependencies \
    libgfortran5 \
    libatlas-base-dev \
    libhdf5-dev \
    libprotobuf-dev \
    protobuf-compiler \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Microsoft ODBC Driver - with fix for package conflicts
RUN curl -sSL https://packages.microsoft.com/keys/microsoft.asc | apt-key add - \
    && curl -sSL https://packages.microsoft.com/config/debian/11/prod.list > /etc/apt/sources.list.d/mssql-release.list \
    && apt-get update \
    # Remove any conflicting packages
    && apt-get remove -y unixodbc-dev unixodbc-common libodbc2 libodbccr2 libodbcinst2 \
    # Install Microsoft ODBC
    && ACCEPT_EULA=Y apt-get install -y msodbcsql18 \
    && apt-get install -y -f \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set ODBC environment variables
ENV ODBCINI=/etc/odbc.ini
ENV ODBCSYSINI=/etc
ENV LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu

# Copy requirements.txt
COPY requirements.txt .

# Set environment variables to address OpenSSL deprecation warnings
ENV PYTHONWARNINGS="ignore::DeprecationWarning:OpenSSL"
ENV OPENSSL_CONF=/dev/null

# Install Python dependencies using requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Run the FastAPI application with uvicorn server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]