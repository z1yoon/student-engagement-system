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
CAPTURE_INTERVAL = int(os.getenv("CAPTURE_INTERVAL", "30"))  # Default: 300s (5 minutes)
JPEG_QUALITY = 50  # Lower means smaller file size
FRAME_WIDTH = 320
FRAME_HEIGHT = 240

stop_capture_event = threading.Event()
capture_thread = None

# Create a global IoTHubDeviceClient instance
iot_client = IoTHubDeviceClient.create_from_connection_string(DEVICE_CONNECTION_STRING)

# Track how many messages we’ve sent, just for local logging
message_count = 0

def capture_loop():
    global message_count

    cap = cv2.VideoCapture(0)
    # Optionally set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        logger.error("❌ Failed to open camera.")
        return

    # Warm up the camera to avoid capturing a black frame
    logger.info("Warming up camera for 2 seconds to avoid black frames...")
    time.sleep(2)

    # Optionally discard a few initial frames
    for i in range(5):
        ret, frame = cap.read()
        if not ret:
            logger.warning("⚠️ Warm-up frame not ready, waiting a bit longer.")
        time.sleep(0.1)

    last_sent_time = 0

    try:
        while not stop_capture_event.is_set():
            current_time = time.time()
            # Send a message only if the interval has passed
            if current_time - last_sent_time >= CAPTURE_INTERVAL:
                ret, frame = cap.read()
                if ret and iot_client.connected:
                    # Encode frame to JPEG with specified quality
                    success, encoded_jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                    if success:
                        b64_image = base64.b64encode(encoded_jpg).decode("utf-8")

                        # Build JSON payload
                        payload_dict = {
                            "timestamp": current_time,
                            "image_data": b64_image
                        }
                        payload = json.dumps(payload_dict)

                        # Send message to IoT Hub
                        message = Message(payload)
                        iot_client.send_message(message)
                        message_count += 1
                        logger.info(f"📷 ✅ Sent frame #{message_count} to IoT Hub at {time.ctime(current_time)}")
                        last_sent_time = current_time
                    else:
                        logger.error("❌ Failed to encode image.")
                elif not ret:
                    logger.warning("⚠️ Camera capture failed.")
                elif not iot_client.connected:
                    logger.warning("⚠️ IoT Client disconnected. Skipping message send.")

            # Sleep a bit to avoid CPU spinning
            time.sleep(1)
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
        capture_thread.join()  # Wait for thread to exit
        capture_thread = None
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
        iot_client.send_method_response(method_response)
        logger.info("✅ Successfully sent method response to IoT Hub.")

    except Exception as e:
        logger.error(f"❌ Error handling direct method: {e}")
        error_response = MethodResponse.create_from_method_request(
            method_request, 500, {"status": "error", "message": str(e)}
        )
        iot_client.send_method_response(error_response)

def main():
    """
    Main function that connects to IoT Hub and listens for direct method calls using the shared client.
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
