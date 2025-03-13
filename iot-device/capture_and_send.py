import os
import time
import cv2
import base64
import json
import logging
import threading
from azure.iot.device import IoTHubDeviceClient, Message, MethodResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEVICE_CONNECTION_STRING = os.getenv("DEVICE_CONNECTION_STRING", "")
CAPTURE_INTERVAL = 60  # seconds between captures

stop_capture_event = threading.Event()
capture_thread = None

# Create a global IoTHubDeviceClient instance
iot_client = IoTHubDeviceClient.create_from_connection_string(DEVICE_CONNECTION_STRING)

def capture_loop():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("❌ Failed to open camera.")
        return

    last_sent_time = 0  # ✅ Track last message time

    try:
        while not stop_capture_event.is_set():
            current_time = time.time()
            if current_time - last_sent_time >= CAPTURE_INTERVAL:
                ret, frame = cap.read()
                if ret and iot_client.connected:  # ✅ Ensure IoT connection is active
                    success, encoded_jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if success:
                        b64_image = base64.b64encode(encoded_jpg).decode("utf-8")
                        payload = json.dumps({"timestamp": current_time, "image_data": b64_image})
                        message = Message(payload)
                        iot_client.send_message(message)
                        logger.info("📷 ✅ Sent frame to IoT Hub.")
                        last_sent_time = current_time
                    else:
                        logger.error("❌ Failed to encode image.")
                elif not ret:
                    logger.warning("⚠️ Camera capture failed.")
                elif not iot_client.connected:
                    logger.warning("⚠️ IoT Client disconnected. Skipping message send.")
            time.sleep(1)  # ✅ Prevent CPU overuse
    except Exception as e:
        logger.error(f"❌ Error in capture loop: {e}")
    finally:
        cap.release()
        logger.info("🔌 Camera released.")



def start_capture():
    """
    Starts the capture loop in a background thread using the global client.
    Ensures only one capture thread runs at a time.
    """
    global capture_thread, stop_capture_event
    if capture_thread and capture_thread.is_alive():
        logger.info("Capture thread already running.")
        return {"status": "capture already running"}

    stop_capture_event.clear()
    capture_thread = threading.Thread(target=capture_loop, daemon=True)
    capture_thread.start()
    logger.info("✅ Capture thread started.")
    return {"status": "capture started"}

def stop_capture():
    """
    Stops the capture loop and ensures thread cleanup.
    """
    global capture_thread
    stop_capture_event.set()
    if capture_thread:
        capture_thread.join()  # ✅ Ensure the thread stops properly
        capture_thread = None  # ✅ Reset the thread reference
    logger.info("🛑 Capture stopped.")
    return {"status": "capture stopped"}


def direct_method_handler(method_request):
    try:
        logger.info(f"Received direct method: {method_request.name}")

        if method_request.name == "startCapture":
            logger.info("Starting capture thread...")
            threading.Thread(target=start_capture, daemon=True).start()
            response_payload = {"status": "capture started"}
        elif method_request.name == "stopCapture":
            logger.info("Stopping capture thread...")
            threading.Thread(target=stop_capture, daemon=True).start()
            response_payload = {"status": "capture stopped"}
        else:
            response_payload = {"status": "unknown method"}

        logger.info(f"Sending response: {response_payload}")

        method_response = MethodResponse.create_from_method_request(method_request, 200, response_payload)
        iot_client.send_method_response(method_response)  # ✅ Explicitly send response
        logger.info("✅ Successfully sent method response to IoT Hub.")

    except Exception as e:
        logger.error(f"❌ Error handling direct method: {e}")
        error_response = MethodResponse.create_from_method_request(method_request, 500,
                                                                   {"status": "error", "message": str(e)})
        iot_client.send_method_response(error_response)  # ✅ Send error response if failed


def main():
    """
    Main function that connects to IoT Hub and listens for direct method commands using the shared client.
    """
    try:
        iot_client.connect()
        logger.info("Device connected to IoT Hub, waiting for direct method calls...")
        iot_client.on_method_request_received = direct_method_handler

        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        logger.info("Shutting down device.")
    finally:
        iot_client.disconnect()
        logger.info("Disconnected from IoT Hub.")

if __name__ == "__main__":
    main()
