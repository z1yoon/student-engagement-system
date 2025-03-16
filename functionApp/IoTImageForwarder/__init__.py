import logging
import json
import os
import azure.functions as func
import requests  # Ensure requests is in your requirements.txt

# Replace with your publicly accessible backend API URL (e.g. via ngrok)
BACKEND_API_URL = os.getenv("BACKEND_API_URL", "")

def main(event: func.EventHubEvent):
    """
    This Azure Function is triggered by IoT Hub messages (via Event Hub trigger).
    It expects a JSON payload with an "image_data" field containing a Base64‑encoded image.
    The function then forwards the image data to the backend API for analysis.
    """
    logging.info("Azure Function triggered for image analysis forwarding.")
    try:
        # Decode event message from bytes to string and parse JSON
        message_body = event.get_body().decode("utf-8")
        payload = json.loads(message_body)
    except Exception as e:
        logging.error("❌ Failed to decode/parse message: %s", e)
        return

    image_data = payload.get("image_data")
    if not image_data:
        logging.error("❌ 'image_data' field missing in payload: %s", payload)
        return

    try:
        # Forward the image data to the backend API (no additional decoding is needed here)
        response = requests.post(BACKEND_API_URL, json={"image_data": image_data})
        if response.status_code == 200:
            logging.info("✅ Successfully forwarded image to backend. Response: %s", response.text)
        else:
            logging.error("❌ Backend API error: %s", response.text)
    except Exception as e:
        logging.error("❌ Error calling backend API: %s", e)
