import os
import base64
import logging
from datetime import datetime
from fastapi import FastAPI, HTTPException, Body, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from db import create_tables, update_capture_status, add_student
from utils import upload_to_blob
from capture_and_send import start_capture_loop, stop_capture_loop, stop_consumer
from analyze_result import get_analyze_results

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Database tables
create_tables()

# Global task flag for image capture
capture_task = None

@app.post("/api/enroll-student/")
def enroll_student(payload: dict = Body(...)):
    """
    Enrolls a new student by storing their name and image.
    The image is uploaded to Azure Blob Storage.
    """
    student_name = payload.get("student_name")
    image_data = payload.get("image_data")

    if not student_name or not image_data:
        raise HTTPException(status_code=400, detail="Missing student name or image data.")

    try:
        # Decode image from data URI format
        image_bytes = base64.b64decode(image_data.split(",")[1])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image data. Error: {e}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"{student_name}_{timestamp}.jpg"
    blob_url = upload_to_blob(image_bytes, file_name)

    add_student(student_name, None, blob_url)

    return {"message": "Student enrolled successfully!", "blob_url": blob_url}

@app.post("/api/start-capture")
async def start_capture(background_tasks: BackgroundTasks):
    """
    Starts the real-time image capture loop.
    """
    global capture_task
    update_capture_status(True)
    if not capture_task:
        capture_task = background_tasks.add_task(start_capture_loop)
    return {"message": "Capture started. Images are being sent to IoT Hub."}

@app.post("/api/stop-capture")
def stop_capture():
    """
    Stops image capture and IoT consumer processing.
    """
    update_capture_status(False)
    stop_capture_loop()
    stop_consumer()
    return {"message": "Capture and engagement analysis have stopped."}

@app.get("/api/analyze_results")
def get_analyze(start_date: str = None, end_date: str = None):
    """
    Returns the latest summary of engagement and attendance results.
    Supports optional date filtering with start_date and end_date (YYYY-MM-DD).
    """
    return get_analyze_results(start_date, end_date)
