import base64
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
    check_similar_face_exists
)
from utils import upload_to_blob
from analyze import analyze_image, prepare_image_for_processing
from config import IOT_HUB_CONNECTION_STRING, DEVICE_ID

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    if not IOT_HUB_CONNECTION_STRING:
        logger.error("❌ IOT_HUB_CONNECTION_STRING not configured")
        raise HTTPException(status_code=500, detail="IoT Hub not configured")
        
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
    image_center = payload.get("image_center")
    image_left = payload.get("image_left")
    image_right = payload.get("image_right")
    
    # Input validation
    if not student_name:
        raise HTTPException(status_code=400, detail="Missing student name.")
    if not image_center:
        raise HTTPException(status_code=400, detail="Missing center image.")
    if not image_left:
        raise HTTPException(status_code=400, detail="Missing left image.")
    if not image_right:
        raise HTTPException(status_code=400, detail="Missing right image.")
    
    if student_exists(student_name):
        raise HTTPException(status_code=409, detail=f"Student '{student_name}' is already registered.")
    
    # Process all images to compute face embeddings
    try:
        # Decode base64 images
        image_bytes_center = base64.b64decode(
            image_center.split(",")[1] if "," in image_center else image_center
        )
        image_bytes_left = base64.b64decode(
            image_left.split(",")[1] if "," in image_left else image_left
        )
        image_bytes_right = base64.b64decode(
            image_right.split(",")[1] if "," in image_right else image_right
        )
        
        # Process images
        from analyze import prepare_image_for_processing, get_face_app
        
        # Process center image
        processed_center, image_np_center, image_np_bgr_center = prepare_image_for_processing(image_bytes_center)
        
        # Process left image
        processed_left, image_np_left, image_np_bgr_left = prepare_image_for_processing(image_bytes_left)
        
        # Process right image
        processed_right, image_np_right, image_np_bgr_right = prepare_image_for_processing(image_bytes_right)
        
        # Initialize face app once
        face_app = get_face_app()
        
        # Extract embeddings from each position
        face_embedding_center = None
        face_embedding_left = None
        face_embedding_right = None
        
        # Get center face embedding
        faces_center = face_app.get(image_np_bgr_center)
        if faces_center and len(faces_center) > 0:
            embedding = getattr(faces_center[0], 'embedding', None)
            if embedding is not None:
                face_embedding_center = embedding.tolist()
                logger.info(f"✅ Center face embedding computed for {student_name}")
        else:
            logger.warning(f"⚠️ No face detected in center image for {student_name}")
            raise HTTPException(status_code=400, detail="No face detected in center image. Please ensure face is clearly visible.")
        
        # Get left face embedding
        faces_left = face_app.get(image_np_bgr_left)
        if faces_left and len(faces_left) > 0:
            embedding = getattr(faces_left[0], 'embedding', None)
            if embedding is not None:
                face_embedding_left = embedding.tolist()
                logger.info(f"✅ Left face embedding computed for {student_name}")
        else:
            logger.warning(f"⚠️ No face detected in left image for {student_name}")
            # Not raising exception for left image, it's optional
        
        # Get right face embedding
        faces_right = face_app.get(image_np_bgr_right)
        if faces_right and len(faces_right) > 0:
            embedding = getattr(faces_right[0], 'embedding', None)
            if embedding is not None:
                face_embedding_right = embedding.tolist()
                logger.info(f"✅ Right face embedding computed for {student_name}")
        else:
            logger.warning(f"⚠️ No face detected in right image for {student_name}")
            # Not raising exception for right image, it's optional
        
        # Check if we have at least one valid embedding
        if not face_embedding_center and not face_embedding_left and not face_embedding_right:
            raise HTTPException(status_code=400, detail="Could not detect face in any of the provided images.")
        
        # We'll use the center embedding for duplicate checking since it's the most reliable
        if face_embedding_center:
            # Check for similar existing faces (de-duplication)
            has_similar_face, similar_students = check_similar_face_exists(face_embedding_center, similarity_threshold=0.80)
            
            if has_similar_face:
                # Format the similar students for the error message
                similar_faces_info = []
                for name, similarity in similar_students:
                    similarity_percent = round(similarity * 100, 2)
                    similar_faces_info.append(f"{name} ({similarity_percent}% similar)")
                
                similar_faces_str = ", ".join(similar_faces_info)
                raise HTTPException(
                    status_code=409, 
                    detail=f"This face appears to be already enrolled as: {similar_faces_str}. Cannot enroll the same person with different names."
                )
        
    except HTTPException:
        # Re-raise HTTPExceptions
        raise
    except Exception as e:
        logger.error(f"Error during image processing: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing images: {str(e)}")
    
    # Save student images and add to database
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Upload images to blob storage
    center_file_name = f"{student_name}_center_{timestamp}.jpg"
    left_file_name = f"{student_name}_left_{timestamp}.jpg"
    right_file_name = f"{student_name}_right_{timestamp}.jpg"
    
    blob_url_center = upload_to_blob(image_bytes_center, center_file_name)
    blob_url_left = upload_to_blob(image_bytes_left, left_file_name)
    blob_url_right = upload_to_blob(image_bytes_right, right_file_name)
    
    # Add student with all embeddings and image URLs
    add_student(
        student_name, 
        None, 
        face_embedding_center, 
        face_embedding_left, 
        face_embedding_right,
        blob_url_center,
        blob_url_left,
        blob_url_right
    )
    
    return {
        "message": "Student enrolled successfully with all face positions!",
        "blob_urls": {
            "center": blob_url_center,
            "left": blob_url_left,
            "right": blob_url_right
        },
        "embeddings_computed": {
            "center": face_embedding_center is not None,
            "left": face_embedding_left is not None,
            "right": face_embedding_right is not None
        }
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
    # No need to convert simple Python types
    return status

@app.get("/api/latest_analysis")
def get_latest_analysis():
    """API endpoint for getting the latest analyzed image with student status information"""
    if not hasattr(app, 'latest_analysis_result') or app.latest_analysis_result is None:
        return {"message": "No analysis available yet"}
        
    return convert_numpy_types(app.latest_analysis_result)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)