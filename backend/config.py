import os
import logging

# Database configuration
DB_CONNECTION_STRING = os.getenv("DB_CONNECTION_STRING", "")

# Azure Blob Storage configuration
BLOB_CONNECTION_STRING = os.getenv("BLOB_CONNECTION_STRING", "")
BLOB_CONTAINER_NAME = os.getenv("BLOB_CONTAINER_NAME", "students")

# Azure Vision API configuration
VISION_API_ENDPOINT = os.getenv("VISION_API_ENDPOINT", "")
VISION_API_KEY = os.getenv("VISION_API_KEY", "")

# IoT Hub configuration
IOT_HUB_CONNECTION_STRING = os.getenv("IOT_HUB_CONNECTION_STRING", "")
DEVICE_ID = os.getenv("DEVICE_ID", "Mac01")

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Validate required configuration
if not DB_CONNECTION_STRING:
    logging.warning("⚠️ DB_CONNECTION_STRING environment variable is not set!")

if not BLOB_CONNECTION_STRING:
    logging.warning("⚠️ BLOB_CONNECTION_STRING environment variable is not set!")

if not IOT_HUB_CONNECTION_STRING:
    logging.warning("⚠️ IOT_HUB_CONNECTION_STRING environment variable is not set!")
    
if not VISION_API_ENDPOINT or not VISION_API_KEY:
    logging.warning("⚠️ Azure Vision API credentials (VISION_API_ENDPOINT, VISION_API_KEY) are not set! Phone detection will be disabled.")