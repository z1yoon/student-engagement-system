import os
import time
import cv2
import base64
import json
import logging
import requests
from azure.iot.device import IoTHubDeviceClient, Message

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

IOT_HUB_CONNECTION_STRING = os.getenv("IOT_HUB_CONNECTION_STRING", "")
FASTAPI_BACKEND_URL = "http://localhost:8000/api/capture-status"  # Change if deployed

CAPTURE_INTERVAL = 60  # Capture every 60 seconds

def get_capture_status():
    """Check the backend to see if capture is enabled."""
    try:
        response = requests.get(FASTAPI_BACKEND_URL)
        if response.status_code == 200:
            return response.json().get("capture_active", False)
        return False
    except requests.RequestException as e:
        logger.error(f"Error checking capture status: {e}")
        return False

def start_capture_loop():
    """Continuously captures and sends images to IoT Hub when enabled"""
    client = IoTHubDeviceClient.create_from_connection_string(IOT_HUB_CONNECTION_STRING)
    client.connect()
    logger.info("Connected to IoT Hub.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Failed to open camera.")
        return

    try:
        while True:
            if get_capture_status():  # Capture only if status is enabled
                ret, frame = cap.read()
                if ret:
                    success, encoded_jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if success:
                        b64_image = base64.b64encode(encoded_jpg).decode("utf-8")
                        json_payload = json.dumps({"timestamp": time.time(), "image_data": b64_image})
                        message = Message(json_payload)
                        client.send_message(message)
                        logger.info("Sent frame to IoT Hub.")
                else:
                    logger.warning("Failed to capture frame from camera.")

            time.sleep(CAPTURE_INTERVAL)

    except KeyboardInterrupt:
        logger.info("Stopping capture script.")
    finally:
        cap.release()
        client.disconnect()
        logger.info("Disconnected from IoT Hub.")
