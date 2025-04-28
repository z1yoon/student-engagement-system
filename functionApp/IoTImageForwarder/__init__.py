import logging
import json
import os
import azure.functions as func
import aiohttp  # Replace requests with aiohttp for async support
import asyncio

# Replace with your publicly accessible backend API URL (e.g. via ngrok)
BACKEND_API_URL = os.getenv("BACKEND_API_URL", "")

async def main(event: func.EventHubEvent):
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
        # Forward the image data to the backend API using async pattern
        async with aiohttp.ClientSession() as session:
            async with session.post(BACKEND_API_URL, json={"image_data": image_data}) as response:
                if response.status == 200:
                    result = await response.text()
                    logging.info("✅ Successfully forwarded image to backend. Response: %s", result)
                else:
                    error_text = await response.text()
                    logging.error("❌ Backend API error: %s", error_text)
    except Exception as e:
        logging.error("❌ Error calling backend API: %s", e)
    
    # Make sure all telemetry is flushed before function completes
    await asyncio.sleep(0.1)  # Small delay to ensure logging completes
