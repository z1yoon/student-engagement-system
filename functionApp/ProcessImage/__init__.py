import logging
import json
import azure.functions as func

# Import the analyze logic and database code from the same folder
from . import analyze

def main(event: func.EventGridEvent):
    logging.info("🔔 Event Grid trigger received an event.")

    # Extract the event data as a Python dict
    event_data = event.get_json()  # The JSON payload from IoT Hub

    # You must ensure your IoT device message includes "image_data" in the payload
    image_data = event_data.get("image_data")
    if not image_data:
        logging.error("No 'image_data' found in event payload.")
        return func.HttpResponse(
            "No 'image_data' found in event payload.",
            status_code=400
        )

    # Analyze the image
    result = analyze.analyze_image(image_data)

    # Return the analysis result
    return func.HttpResponse(
        json.dumps(result),
        status_code=200,
        mimetype="application/json"
    )
