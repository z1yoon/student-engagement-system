import os
import base64
import io
import logging
from datetime import datetime
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from azure.iot.hub import IoTHubRegistryManager
from azure.iot.hub.models import CloudToDeviceMethod
from PIL import Image
import numpy as np
import uvicorn

# Import your helper modules
from db import (
    create_tables,
    add_student,
    update_capture_status,
    get_analyze_results,
    get_capture_status
)
from utils import upload_to_blob
from analyze import analyze_image, get_face_app, prepare_image_for_face_detection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize DB tables if they don't exist
create_tables()

IOT_HUB_CONNECTION_STRING = os.getenv("IOT_HUB_CONNECTION_STRING", "")
DEVICE_ID = os.getenv("DEVICE_ID", "Mac01")

def invoke_direct_method(method_name, payload=None):
    registry_manager = IoTHubRegistryManager(IOT_HUB_CONNECTION_STRING)
    method_request = CloudToDeviceMethod(method_name=method_name, payload=payload or {})
    try:
        response = registry_manager.invoke_device_method(DEVICE_ID, method_request)
        logger.info(f"✅ Invoked {method_name}. Response: {response.as_dict()}")
        return response
    except Exception as e:
        logger.error(f"❌ Failed to invoke {method_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Global variable to store the latest annotated image (updated by analyze_image)
latest_annotated_image = None

@app.post("/api/enroll-student/")
def enroll_student(payload: dict = Body(...)):
    """
    Enrolls a new student by storing their name and uploading their image.
    Computes a face embedding for later recognition using InsightFace.
    """
    student_name = payload.get("student_name")
    image_data = payload.get("image_data")
    if not student_name or not image_data:
        raise HTTPException(status_code=400, detail="Missing student name or image data.")
    try:
        image_bytes = base64.b64decode(
            image_data.split(",")[1] if "," in image_data else image_data
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 data: {e}")
    try:
        # Use the unified image processing function for enrollment
        processed_image, image_np, _, _ = prepare_image_for_face_detection(image_bytes, 320)
        faces = get_face_app().get(image_np)
        if faces:
            embedding = getattr(faces[0], "embedding", None)
            if embedding is not None:
                try:
                    embedding = np.array(embedding)
                    face_embedding = embedding.tolist()
                except Exception as ee:
                    logger.error(f"Error converting embedding to numpy array: {ee}")
                    face_embedding = None
            else:
                face_embedding = None
            if face_embedding is not None:
                logger.info("✅ Face embedding computed for enrollment.")
            else:
                logger.warning("⚠️ No valid face embedding computed during enrollment.")
        else:
            face_embedding = None
            logger.warning("⚠️ No face detected during enrollment.")
    except Exception as e:
        logger.error(f"Error computing face embedding: {e}")
        face_embedding = None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"{student_name}_{timestamp}.jpg"
    blob_url = upload_to_blob(image_bytes, file_name)
    add_student(student_name, None, face_embedding, blob_url)
    return {"message": "Student enrolled successfully!", "blob_url": blob_url}

@app.post("/api/start-capture")
def start_capture():
    """
    Tells the device to start capturing images.
    """
    update_capture_status(True)
    response = invoke_direct_method("startCapture")
    return {"message": "Capture command sent.", "response": response.as_dict()}

@app.post("/api/stop-capture")
def stop_capture():
    """
    Tells the device to stop capturing images.
    """
    update_capture_status(False)
    response = invoke_direct_method("stopCapture")
    return {"message": "Stop capture command sent.", "response": response.as_dict()}

@app.get("/api/analyze_results")
def get_analyze(start_date: str = None, end_date: str = None):
    """
    Returns a summary of engagement and attendance data.
    Also includes the latest annotated image from the analysis.
    """
    summary = get_analyze_results(start_date, end_date)
    if latest_annotated_image:
        summary["latest_annotated_image"] = latest_annotated_image
    return summary

@app.post("/api/analyze_image")
def analyze_endpoint(payload: dict = Body(...)):
    """
    Endpoint to analyze an image (provided as a Base64 string).
    Delegates the analysis to the analyze module.
    This endpoint is triggered by your Azure Function every minute.
    """
    global latest_annotated_image
    image_data = payload.get("image_data")
    if not image_data:
        raise HTTPException(status_code=400, detail="Missing image data.")
    try:
        result = analyze_image(image_data)
        if result.get("annotated_image"):
            latest_annotated_image = result["annotated_image"]
        return result
    except Exception as e:
        logger.error(f"Error during image analysis: {e}")
        return {"error": "Failed to analyze image", "face_detected": False}

@app.get("/api/capture-status")
def capture_status():
    """
    Returns the current capture status from the SystemSettings table.
    """
    status = get_capture_status()
    return status

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
