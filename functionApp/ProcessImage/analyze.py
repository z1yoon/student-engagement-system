import os
import io
import json
import base64
import logging

from azure.core.credentials import AzureKeyCredential
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.face import FaceClient
from azure.ai.vision.face.models import (
    FaceDetectionModel,
    FaceRecognitionModel,
    FaceAttributeType
)

# Import DB methods from local db.py
from .db import add_engagement_record, mark_attendance, get_enrolled_students

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables (set these in Azure Function App Settings!)
FACE_API_ENDPOINT = os.getenv("FACE_API_ENDPOINT", "")
FACE_API_KEY = os.getenv("FACE_API_KEY", "")
VISION_API_ENDPOINT = os.getenv("VISION_API_ENDPOINT", "")
VISION_API_KEY = os.getenv("VISION_API_KEY", "")

if not FACE_API_ENDPOINT or not FACE_API_KEY:
    raise ValueError("Azure Face API credentials are missing!")
if not VISION_API_ENDPOINT or not VISION_API_KEY:
    raise ValueError("Azure Vision API credentials are missing!")

# Initialize Azure Clients
face_client = FaceClient(FACE_API_ENDPOINT, AzureKeyCredential(FACE_API_KEY))
vision_client = ImageAnalysisClient(VISION_API_ENDPOINT, AzureKeyCredential(VISION_API_KEY))

def analyze_image(image_data: str):
    """
    Analyzes the image for engagement and phone usage, and marks attendance.
    image_data: base64-encoded image string (with or without data URI prefix)
    """
    try:
        logger.info("🔍 Analyzing image for engagement & attendance...")

        # Decode Base64
        try:
            if "," in image_data:
                image_bytes = base64.b64decode(image_data.split(",")[1])
            else:
                image_bytes = base64.b64decode(image_data)
        except Exception as e:
            logger.error(f"❌ Error decoding Base64 image: {e}")
            return {"error": "Invalid base64 image format"}

        # Face detection & recognition
        faces_info, recognized_students = detect_faces_with_recognition(image_bytes)

        # Object detection (check for phone usage)
        phone_detected = detect_objects(image_bytes)

        if not faces_info:
            logger.info("⚠️ No faces detected. Skipping analysis.")
            return {"message": "No faces detected."}

        # Process each face
        for face in faces_info:
            student_name = recognized_students.get(face["face_id"], "Unknown")
            gaze = face.get("gaze", "unknown")
            sleeping = face.get("sleeping", False)

            # Mark attendance if recognized
            if student_name != "Unknown":
                mark_attendance(student_name, True, image_data)

            # Record engagement
            add_engagement_record(student_name, phone_detected, gaze, sleeping)

        logger.info(f"✅ Image analysis complete. Phone Detected: {phone_detected}")
        return {
            "message": "Analysis complete",
            "faces": faces_info,
            "recognized_students": recognized_students,
            "phone_detected": phone_detected
        }
    except Exception as e:
        logger.error(f"❌ Error analyzing image: {e}")
        return {"error": "Failed to analyze image"}

def detect_faces_with_recognition(image_bytes: bytes):
    """
    Detect faces & attributes, match them against enrolled students.
    """
    try:
        face_attrs = [FaceAttributeType.HEAD_POSE, FaceAttributeType.OCCLUSION]
        enrolled_students = get_enrolled_students()

        with io.BytesIO(image_bytes) as stream:
            faces = face_client.detect(
                image_content=stream.read(),
                detection_model=FaceDetectionModel.DETECTION03,
                recognition_model=FaceRecognitionModel.RECOGNITION04,
                return_face_id=True,
                return_face_attributes=face_attrs
            )

        results = []
        recognized_students = {}

        for face in faces:
            face_id = face.face_id
            rect = face.face_rectangle
            attributes = face.face_attributes

            if not attributes:
                logger.warning("⚠️ No face attributes found.")
                continue

            head_pose = attributes.head_pose
            eye_occluded = attributes.occlusion.eye_occluded

            # Basic engagement check
            gaze = "focused" if abs(head_pose.yaw) <= 30 and abs(head_pose.pitch) <= 20 else "distracted"
            sleeping = eye_occluded  # Simplistic assumption

            # Attempt to match with enrolled students
            match_name = match_face_with_students(face_id, enrolled_students)
            if match_name:
                recognized_students[face_id] = match_name

            results.append({
                "face_id": face_id,
                "face_rectangle": {
                    "left": rect.left,
                    "top": rect.top,
                    "width": rect.width,
                    "height": rect.height
                },
                "gaze": gaze,
                "eye_occluded": eye_occluded,
                "sleeping": sleeping
            })

        logger.info(f"✅ Face detection complete. Detected {len(results)} faces.")
        return results, recognized_students
    except Exception as e:
        logger.error(f"❌ Error detecting faces: {e}")
        return [], {}

def match_face_with_students(face_id, enrolled_students):
    """
    Compare detected face_id with stored face_ids for each enrolled student.
    """
    try:
        # Collect all known face_ids from DB
        enrolled_face_ids = [s["face_id"] for s in enrolled_students if s["face_id"]]

        # If no enrolled face IDs, skip
        if not enrolled_face_ids:
            return None

        # Verify face to each known face_id
        matches = face_client.verify_face_to_face(face_id, enrolled_face_ids)

        # Return the best match above a confidence threshold
        for match in matches:
            if match.confidence >= 0.8:
                # Find which student has match.face_id
                for s in enrolled_students:
                    if s["face_id"] == match.face_id:
                        return s["name"]
    except Exception as e:
        logger.error(f"❌ Error matching faces: {e}")
    return None

def detect_objects(image_bytes: bytes):
    """
    Detect objects in the image to check for phone usage.
    """
    try:
        with io.BytesIO(image_bytes) as stream:
            analysis_result = vision_client.analyze(
                image_data=stream.read(),
                visual_features=["Objects"]
            )
        # If "phone" is in the detected objects, we consider phone usage
        detected_objects = [obj.name.lower() for obj in analysis_result.objects if hasattr(obj, "name")]
        phone_detected = "phone" in detected_objects
        logger.info(f"📱 Phone detection: {phone_detected}")
        return phone_detected
    except Exception as e:
        logger.error(f"❌ Error detecting objects: {e}")
        return False

