import os
import base64
import logging
from datetime import datetime
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from azure.iot.hub import IoTHubRegistryManager
from azure.iot.hub.models import CloudToDeviceMethod
from db import create_tables, update_capture_status, add_student
from utils import upload_to_blob
from analyze_result import get_analyze_results

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Enable CORS for demonstration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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

@app.post("/api/enroll-student/")
def enroll_student(payload: dict = Body(...)):
    """
    Enrolls a new student by storing their name and uploading their image to blob storage.
    """
    student_name = payload.get("student_name")
    image_data = payload.get("image_data")

    if not student_name or not image_data:
        raise HTTPException(status_code=400, detail="Missing student name or image data.")

    # Decode base64
    try:
        image_bytes = base64.b64decode(image_data.split(",")[1])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 data: {e}")

    # Upload to blob
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"{student_name}_{timestamp}.jpg"
    blob_url = upload_to_blob(image_bytes, file_name)

    # Add to DB
    add_student(student_name, None, None, blob_url)
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
    Returns the summary of engagement & attendance.
    """
    return get_analyze_results(start_date, end_date)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
