import os
from azure.storage.blob import BlobServiceClient

AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
BLOB_CONTAINER_NAME = os.getenv("BLOB_CONTAINER_NAME", "")
blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)

def upload_to_blob(image_bytes):
    container_client = blob_service_client.get_container_client(BLOB_CONTAINER_NAME)
    try:
        container_client.create_container()
    except:
        pass

    blob_name = f"captures/{os.urandom(8).hex()}.jpg"
    blob_client = container_client.get_blob_client(blob=blob_name)
    blob_client.upload_blob(image_bytes, overwrite=True)

    return blob_client.url
