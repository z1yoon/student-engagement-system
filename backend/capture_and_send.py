import os
import time
import cv2
import base64
import json
import logging
import threading
from azure.iot.device import IoTHubDeviceClient, Message
from analyze import analyze_image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# IoT Hub Configuration
IOT_HUB_CONNECTION_STRING = os.getenv("IOT_HUB_CONNECTION_STRING", "")
CAPTURE_INTERVAL = 60  # Capture every 60 seconds

stop_capture_event = threading.Event()

def start_capture_loop():
    """Captures images from webcam and sends them to IoT Hub."""
    client = IoTHubDeviceClient.create_from_connection_string(IOT_HUB_CONNECTION_STRING)
    client.connect()
    logger.info("Connected to IoT Hub.")

    cap = cv2.VideoCapture(0,cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        logger.error("Failed to open camera.")
        return

    try:
        while not stop_capture_event.is_set():
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
                    logger.error("Failed to encode image.")
            else:
                logger.warning("Camera capture failed.")
            time.sleep(CAPTURE_INTERVAL)
    except KeyboardInterrupt:
        logger.info("Capture loop stopped.")
    finally:
        cap.release()
        client.disconnect()
        logger.info("Disconnected from IoT Hub.")

def stop_capture_loop():
    """Stops the capture loop."""
    stop_capture_event.set()

def message_handler(message):
    """
    Processes images received from IoT Hub by analyzing them and updating engagement records.
    """
    payload = json.loads(message.data)
    image_data = payload.get("image_data")
    if image_data:
        result = analyze_image(image_data)
        logger.info(f"Analysis result: {result}")

def stop_consumer():
    """Stops IoT Hub Consumer."""
    logger.info("IoT Hub Consumer stopped.")
