import logging
import json
import azure.functions as func

# Import the analysis module (analyze.py)
from . import analyze

def main(event: func.EventHubEvent):
    """
    This function is triggered by IoT Hub messages.
    It expects a JSON payload that contains "image_data" as a Base64‑encoded image.
    """
    # Decode the event message (bytes to string)
    try:
        message_body = event.get_body().decode('utf-8')
        payload = json.loads(message_body)
    except Exception as e:
        logging.error(f"❌ Failed to decode and parse IoT message: {e}")
        return

    # Get the image_data field
    image_data = payload.get("image_data")
    if not image_data:
        logging.error(f"❌ 'image_data' missing in payload: {payload}")
        return

    # Analyze the image using our analysis logic
    result = analyze.analyze_image(image_data)
    logging.info(f"✅ Analysis result: {json.dumps(result)}")
