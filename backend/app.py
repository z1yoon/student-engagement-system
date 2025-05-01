import os
import base64
import io
import logging
from datetime import datetime
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from azure.iot.hub import IoTHubRegistryManager
from azure.iot.hub.models import CloudToDeviceMethod
import numpy as np
import json
from db import (
    create_tables,
    add_student,
    update_capture_status,
    get_analyze_results,
    get_capture_status,
    student_exists,
    DB_CONNECTION_STRING
)
from utils import upload_to_blob
from analyze import analyze_image, prepare_image_for_processing

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
IOT_HUB_CONNECTION_STRING = os.getenv("IOT_HUB_CONNECTION_STRING", "")
DEVICE_ID = os.getenv("DEVICE_ID", "Mac01")

class NumpyJSONResponse(JSONResponse):
    """Custom JSON response class that handles NumPy data types"""
    def render(self, content) -> bytes:
        return json.dumps(
            content,
            default=self._serialize_numpy,
            ensure_ascii=False,
            allow_nan=True,
            indent=None,
            separators=(",", ":")
        ).encode("utf-8")

    @staticmethod
    def _serialize_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

def convert_numpy_types(obj):
    """Recursively convert NumPy types to standard Python types"""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj

def invoke_direct_method(method_name, payload=None):
    """Invoke a direct method on the IoT device"""
    registry_manager = IoTHubRegistryManager(IOT_HUB_CONNECTION_STRING)
    method_request = CloudToDeviceMethod(method_name=method_name, payload=payload or {})
    try:
        response = registry_manager.invoke_device_method(DEVICE_ID, method_request)
        logger.info(f"✅ Invoked {method_name}. Response: {response.as_dict()}")
        return response
    except Exception as e:
        logger.error(f"❌ Failed to invoke {method_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Initialize FastAPI app
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return NumpyJSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    return NumpyJSONResponse(
        status_code=500,
        content={"detail": str(exc)},
    )

# Create database tables at startup
@app.on_event("startup")
def startup_db_client():
    logger.info("🚀 Starting up application...")
    try:
        # Do not log database connection string for security
        logger.info("📝 Attempting database connection...")
        
        # Explicitly create tables
        if not create_tables():
            logger.error("❌ Failed to create necessary database tables. Application may not function correctly.")
        else:
            logger.info("✅ Database tables ready.")
    except Exception as e:
        logger.error(f"❌ Error during startup: {e}")

@app.post("/api/enroll-student/")
def enroll_student(payload: dict = Body(...)):
    """API endpoint for enrolling a new student with face recognition"""
    student_name = payload.get("student_name")
    image_data = payload.get("image_data")
    
    # Input validation
    if not student_name or not image_data:
        raise HTTPException(status_code=400, detail="Missing student name or image data.")
    if student_exists(student_name):
        raise HTTPException(status_code=409, detail=f"Student '{student_name}' is already registered.")
    
    try:
        # Decode base64 image
        image_bytes = base64.b64decode(
            image_data.split(",")[1] if "," in image_data else image_data
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 data: {e}")
    
    try:
        # Process image and extract face embedding
        processed_image, image_np, image_np_bgr = prepare_image_for_processing(image_bytes)
        face_embedding = None
        
        from analyze import get_face_app
        face_app = get_face_app()
        faces = face_app.get(image_np_bgr)
        
        if faces and len(faces) > 0:
            embedding = getattr(faces[0], 'embedding', None)
            if embedding is not None:
                face_embedding = embedding.tolist()
                logger.info(f"✅ Face embedding computed for {student_name}")
                # Print the face embedding (first 10 values for brevity)
                embedding_sample = face_embedding[:10]
                logger.info(f"Face embedding sample (first 10 values): {embedding_sample}")
            else:
                logger.warning("⚠️ Face detected but no embedding computed")
        else:
            logger.warning("⚠️ No face detected during enrollment")
    except Exception as e:
        logger.error(f"Error during image processing: {e}")
        face_embedding = None
    
    # Save student image and add to database
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"{student_name}_{timestamp}.jpg"
    blob_url = upload_to_blob(image_bytes, file_name)
    add_student(student_name, None, face_embedding, blob_url)
    
    return {
        "message": "Student enrolled successfully!",
        "blob_url": blob_url,
        "face_embedding_sample": face_embedding[:10] if face_embedding else None
    }

@app.post("/api/start-capture")
def start_capture():
    """API endpoint for starting the image capture process"""
    update_capture_status(True)
    response = invoke_direct_method("startCapture")
    return {"message": "Capture command sent.", "response": response.as_dict()}

@app.post("/api/stop-capture")
def stop_capture():
    """API endpoint for stopping the image capture process"""
    update_capture_status(False)
    response = invoke_direct_method("stopCapture")
    return {"message": "Stop capture command sent.", "response": response.as_dict()}

@app.get("/api/analyze_results")
def get_analyze(start_date: str = None, end_date: str = None):
    """API endpoint for getting student engagement analysis results"""
    summary = get_analyze_results(start_date, end_date)
    # Remove latest_annotated_image from the response as requested
    if "latest_annotated_image" in summary:
        del summary["latest_annotated_image"]
    return convert_numpy_types(summary)

@app.post("/api/analyze_image")
def analyze_endpoint(payload: dict = Body(...)):
    """API endpoint for analyzing a single image"""
    image_data = payload.get("image_data")
    if not image_data:
        raise HTTPException(status_code=400, detail="Missing image data.")
    
    try:
        result = analyze_image(image_data)
        # Store the latest analysis result for the /api/latest_analysis endpoint
        app.latest_analysis_result = result
        return NumpyJSONResponse(content=convert_numpy_types(result))
    except Exception as e:
        logger.error(f"Error during image analysis: {e}")
        return NumpyJSONResponse(
            content={"error": "Failed to analyze image", "face_detected": False}
        )

@app.get("/api/capture-status")
def capture_status():
    """API endpoint for checking the current capture status"""
    status = get_capture_status()
    return convert_numpy_types(status)

@app.get("/api/latest_analysis")
def get_latest_analysis():
    """API endpoint for getting the latest analyzed image with student status information"""
    # Store the latest analysis result in memory
    # This will be updated by the analyze_image function
    global latest_analysis_result
    
    if not hasattr(app, 'latest_analysis_result') or app.latest_analysis_result is None:
        return {"message": "No analysis available yet"}
        
    return convert_numpy_types(app.latest_analysis_result)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
