import os
import logging
from azure.storage.blob import BlobServiceClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BLOB_CONNECTION_STRING = os.getenv("BLOB_CONNECTION_STRING", "")
CONTAINER_NAME = os.getenv("BLOB_CONTAINER_NAME", "student-images")

def upload_to_blob(image_bytes: bytes, file_name: str) -> str:
    try:
        blob_service_client = BlobServiceClient.from_connection_string(BLOB_CONNECTION_STRING)
        container_client = blob_service_client.get_container_client(CONTAINER_NAME)
        # Create container if not exists
        container_client.create_container()
        blob_client = container_client.get_blob_client(file_name)

        blob_client.upload_blob(image_bytes, overwrite=True)
        blob_url = blob_client.url
        logger.info(f"✅ Uploaded {file_name} to blob storage.")
        return blob_url
    except Exception as e:
        logger.error(f"❌ Error uploading to blob: {e}")
        return ""
