import os
import io
import logging
import base64
from azure.core.credentials import AzureKeyCredential
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.ai.vision.face import FaceClient
from azure.ai.vision.face.models import (
    FaceDetectionModel,
    FaceRecognitionModel,
    FaceAttributeType,
)
from db import add_engagement_record

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load Azure API Credentials from environment variables
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
    Analyzes the image for engagement and phone usage.
    image_data: base64 encoded image string (with or without data URI scheme)
    """
    try:
        logger.info("🔍 Analyzing image for engagement...")

        # Decode Base64 Image (remove data URI scheme if present)
        try:
            if "," in image_data:
                image_bytes = base64.b64decode(image_data.split(",")[1])
            else:
                image_bytes = base64.b64decode(image_data)
        except Exception as e:
            logger.error(f"❌ Error decoding Base64 image: {e}")
            return {"error": "Invalid base64 image format"}

        # Face & Engagement Detection
        faces_info = detect_faces_with_attributes(image_bytes)

        # Phone Usage Detection
        phone_detected = detect_objects(image_bytes)

        if not faces_info:
            logger.info("⚠️ No faces detected. Skipping analysis.")
            return {"message": "No faces detected."}

        # Process Engagement Data and store each record in the DB
        for face in faces_info:
            # In real use, you might perform face recognition to assign a student name.
            student_name = "Unknown"
            gaze = face.get("gaze", "unknown")
            sleeping = face.get("sleeping", False)
            add_engagement_record(student_name, phone_detected, gaze, sleeping)

        logger.info(f"✅ Image analysis complete. Phone Detected: {phone_detected}")
        return {
            "message": "Analysis complete",
            "faces": faces_info,
            "phone_detected": phone_detected
        }
    except Exception as e:
        logger.error(f"❌ Error analyzing image: {e}")
        return {"error": "Failed to analyze image"}

def detect_faces_with_attributes(image_bytes: bytes):
    """
    Detects faces and extracts attributes to determine focus and sleepiness.
    """
    try:
        face_attrs = [FaceAttributeType.HEAD_POSE, FaceAttributeType.OCCLUSION]

        with io.BytesIO(image_bytes) as image_stream:
            faces = face_client.detect(
                image_content=image_stream.read(),
                detection_model=FaceDetectionModel.DETECTION03,
                recognition_model=FaceRecognitionModel.RECOGNITION04,
                return_face_id=False,
                return_face_attributes=face_attrs
            )

        results = []
        for face in faces:
            rect = face.face_rectangle
            attributes = face.face_attributes

            if not attributes:
                logger.warning("⚠️ No face attributes found. Skipping this face.")
                continue

            head_pose = attributes.head_pose
            eye_occluded = attributes.occlusion.eye_occluded

            # Determine focus level based on head pose
            if abs(head_pose.yaw) > 30 or abs(head_pose.pitch) > 20:
                gaze = "distracted"
            else:
                gaze = "focused"

            sleeping = eye_occluded

            results.append({
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
        return results
    except Exception as e:
        logger.error(f"❌ Error detecting faces: {e}")
        return []

def detect_objects(image_bytes: bytes):
    """
    Detects objects in the image to identify items like phones.
    """
    try:
        with io.BytesIO(image_bytes) as image_stream:
            analysis_result = vision_client.analyze(
                image_data=image_stream.read(),
                visual_features=[VisualFeatures.OBJECTS]
            )
        detected_objects = [obj.name.lower() for obj in analysis_result.objects if hasattr(obj, "name")]
        phone_detected = "phone" in detected_objects
        logger.info(f"📱 Phone detection result: {phone_detected}")
        return phone_detected
    except Exception as e:
        logger.error(f"❌ Error detecting objects: {e}")
        return False
