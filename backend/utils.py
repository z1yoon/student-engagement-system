import os
import logging
from azure.storage.blob import BlobServiceClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

AZURE_BLOB_CONNECTION_STRING = os.getenv("AZURE_BLOB_CONNECTION_STRING", "")
AZURE_BLOB_CONTAINER_NAME = os.getenv("AZURE_BLOB_CONTAINER_NAME", "students")


def upload_to_blob(image_bytes: bytes, file_name: str) -> str:
    """
    Uploads the image bytes to Azure Blob Storage and returns the blob URL.
    """
    try:
        blob_service_client = BlobServiceClient.from_connection_string(AZURE_BLOB_CONNECTION_STRING)
        container_client = blob_service_client.get_container_client(AZURE_BLOB_CONTAINER_NAME)

        # Create the container if it does not exist
        if not container_client.exists():
            container_client.create_container()

        blob_client = container_client.get_blob_client(file_name)
        blob_client.upload_blob(image_bytes, overwrite=True)

        blob_url = blob_client.url
        logger.info(f"✅ Uploaded image to blob storage: {blob_url}")
        return blob_url
    except Exception as e:
        logger.error(f"❌ Error uploading to blob: {e}")
        raise e
