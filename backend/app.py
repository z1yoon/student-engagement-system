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
import cv2  # OpenCV for face detection and recognition

# Import your helper modules
from db import (
    create_tables,
    add_student,
    update_capture_status,
    get_analyze_results,
    get_capture_status,
    student_exists
)
from utils import upload_to_blob
from analyze import (
    analyze_image, 
    prepare_image_for_processing, 
    detect_faces_opencv_dnn,  # Using DNN detector instead of Haar Cascade
    extract_face_features
)

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
    Uses OpenCV DNN for accurate face detection and LBPH for feature extraction.
    """
    student_name = payload.get("student_name")
    image_data = payload.get("image_data")
    if not student_name or not image_data:
        raise HTTPException(status_code=400, detail="Missing student name or image data.")
    
    # Check if student already exists
    if student_exists(student_name):
        raise HTTPException(status_code=409, detail=f"Student '{student_name}' is already registered.")
        
    try:
        image_bytes = base64.b64decode(
            image_data.split(",")[1] if "," in image_data else image_data
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 data: {e}")
        
    face_embedding = None
    try:
        # Process image using CPU-efficient OpenCV
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_np = np.array(img)
        
        # Detect faces using OpenCV DNN (more accurate than Haar Cascade)
        faces = detect_faces_opencv_dnn(img_np)
        
        if len(faces) > 0:
            # Use the first face detected (assumed to be the student)
            face_rect = faces[0]  # (x, y, w, h)
            
            # Extract face features for identification
            face_embedding = extract_face_features(img_np, face_rect)
            
            if face_embedding is not None:
                # Convert embedding to list for storage
                face_embedding = face_embedding.tolist()
                logger.info("✅ Face embedding computed using LBPH after DNN detection")
            else:
                logger.warning("⚠️ No face embedding could be generated")
        else:
            logger.warning("⚠️ No faces detected during enrollment")
    except Exception as e:
        logger.error(f"Error computing face embedding: {e}")
        face_embedding = None
        
    if not face_embedding:
        logger.warning("⚠️ Unable to compute face embedding. Student may not be recognized later.")
        
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
    Uses OpenCV for face detection and recognition, and MediaPipe for gaze tracking and engagement analysis.
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
