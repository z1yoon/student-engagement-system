import os
import logging
from azure.storage.blob import BlobServiceClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BLOB_CONNECTION_STRING = os.getenv("BLOB_CONNECTION_STRING", "")
BLOB_CONTAINER_NAME = os.getenv("BLOB_CONTAINER_NAME", "students")


def upload_to_blob(image_bytes: bytes, file_name: str) -> str:
    """
    Uploads the image bytes to Azure Blob Storage and returns the blob URL.
    """
    try:
        if not BLOB_CONNECTION_STRING:
            logger.warning("⚠️ BLOB_CONNECTION_STRING not set, saving to local file instead")
            # Save to local file as fallback
            os.makedirs("static/images", exist_ok=True)
            file_path = f"static/images/{file_name}"
            with open(file_path, "wb") as f:
                f.write(image_bytes)
            return f"/static/images/{file_name}"
            
        blob_service_client = BlobServiceClient.from_connection_string(BLOB_CONNECTION_STRING)
        container_client = blob_service_client.get_container_client(BLOB_CONTAINER_NAME)

        # Try to create the container (this will catch permissions issues early)
        try:
            container_client.create_container(exist_ok=True)
            logger.info(f"✅ Container '{BLOB_CONTAINER_NAME}' exists or was created successfully")
        except Exception as container_error:
            logger.error(f"❌ Error with blob container: {container_error}")
            # Save to local file as fallback
            os.makedirs("static/images", exist_ok=True)
            file_path = f"static/images/{file_name}"
            with open(file_path, "wb") as f:
                f.write(image_bytes)
            return f"/static/images/{file_name}"

        blob_client = container_client.get_blob_client(file_name)
        blob_client.upload_blob(image_bytes, overwrite=True)

        blob_url = blob_client.url
        logger.info(f"✅ Uploaded image to blob storage: {blob_url}")
        return blob_url
    except Exception as e:
        logger.error(f"❌ Error uploading to blob: {e}")
        # Save to local file as fallback
        try:
            os.makedirs("static/images", exist_ok=True)
            file_path = f"static/images/{file_name}"
            with open(file_path, "wb") as f:
                f.write(image_bytes)
            logger.info(f"✅ Saved image to local file as fallback: {file_path}")
            return f"/static/images/{file_name}"
        except Exception as fallback_error:
            logger.error(f"❌ Fallback to local file also failed: {fallback_error}")
            raise e
