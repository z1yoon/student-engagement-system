import os
import json
import requests
import logging
from azure.iot.device import IoTHubModuleClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

IOT_HUB_CONNECTION_STRING = os.getenv("IOT_HUB_CONNECTION_STRING", "YOUR_DEVICE_CONN_STRING")
BACKEND_URL = "http://localhost:8000/api/analyze-image"

def message_handler(message):
    """Handles messages received from IoT Hub"""
    payload = json.loads(message.data)
    image_data = payload.get("image_data")

    if image_data:
        logger.info("Received image from IoT Hub, forwarding to FastAPI backend.")
        response = requests.post(BACKEND_URL, json={"image": f"data:image/jpeg;base64,{image_data}"})
        logger.info(f"Backend Response: {response.json()}")

def start_consumer():
    """Starts IoT Hub message listener"""
    client = IoTHubModuleClient.create_from_connection_string(IOT_HUB_CONNECTION_STRING)
    client.on_message_received = message_handler
    client.connect()
    logger.info("IoT Hub Consumer started...")
