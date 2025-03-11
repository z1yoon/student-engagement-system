import os
import base64
from datetime import datetime
from fastapi import FastAPI, HTTPException, Body, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from faces.analyze import analyze_image
from db import (
    create_tables, update_capture_status, get_capture_status,
    add_attendance_record, get_attendance_records,
    add_student, get_all_students
)
from utils import upload_to_blob
from faces.engagement import summarize_engagement

# Import IoT and Capture Modules
from consumer import start_consumer  # IoT Hub Consumer
from iot_device.capture_and_send import start_capture_loop  # Image Capture

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Database
create_tables()

# Global Task Flags
consumer_task = None
capture_task = None


@app.get("/")
def root():
    return {"message": "Backend is running."}


@app.post("/api/start-capture")
async def start_capture(background_tasks: BackgroundTasks):
    """Starts Image Capture & IoT Consumer as background tasks"""
    global capture_task, consumer_task

    update_capture_status(True)

    # Start Image Capture and IoT Consumer in the background
    if not capture_task:
        capture_task = background_tasks.add_task(start_capture_loop)
    if not consumer_task:
        consumer_task = background_tasks.add_task(start_consumer)

    return {"message": "Capture and IoT Consumer started"}


@app.post("/api/stop-capture")
def stop_capture():
    """Stops Image Capture by setting DB flag"""
    update_capture_status(False)
    return {"message": "Capture stopped"}


@app.get("/api/capture-status")
def capture_status():
    return get_capture_status()


@app.post("/api/analyze-image")
def analyze_image_endpoint(payload: dict = Body(...)):
    """Receives image as base64 and analyzes faces."""
    b64_str = payload.get("image")
    if not b64_str:
        raise HTTPException(status_code=400, detail="No image in payload")

    try:
        image_bytes = base64.b64decode(b64_str.split(",")[1])
    except:
        raise HTTPException(status_code=400, detail="Invalid base64 data")

    # Upload to Azure Blob Storage
    blob_url = upload_to_blob(image_bytes)

    # Analyze Image (Only Face Detection)
    result = analyze_image(image_bytes)

    return {"message": "Frame analyzed", "faces": result["faces"], "blob_url": blob_url}


from datetime import datetime

@app.post("/api/enroll-student/")
def enroll_student(payload: dict = Body(...)):
    """Enrolls a new student by storing their image encoding and uploading the image with a formatted filename."""
    student_name = payload.get("student_name")
    image_data = payload.get("image_data")

    if not student_name or not image_data:
        raise HTTPException(status_code=400, detail="Missing student name or image data.")

    try:
        image_bytes = base64.b64decode(image_data.split(",")[1])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image data. Error: {e}")

    # Analyze the image for face detection
    result = analyze_image(image_bytes)
    faces = result.get("faces", [])

    print(f"Detected Faces: {faces}")  # Debugging

    if not faces:
        raise HTTPException(status_code=400, detail="No face detected. Try again.")

    # Extract face encoding
    face_encoding = faces[0].get("face_encoding") if faces else None

    # Ensure the student is saved before uploading the image
    add_student(student_name, face_encoding)

    # Format filename as `student_name_YYYYMMDD_HHMMSS.jpg`
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"{student_name}_{timestamp}.jpg"

    # Upload file to Azure Blob Storage with formatted filename
    blob_url = upload_to_blob(image_bytes, file_name)  # ✅ Pass file_name to upload_to_blob()

    return {
        "message": "Student enrolled successfully!",
        "blob_url": blob_url
    }




@app.get("/api/attendance")
def get_attendance():
    return get_attendance_records()


@app.get("/api/summarize_engagement")
def get_summary():
    return summarize_engagement()
