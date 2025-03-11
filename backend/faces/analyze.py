import os
import io
import logging
from azure.core.credentials import AzureKeyCredential
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.ai.vision.face import FaceClient
from azure.ai.vision.face.models import (
    FaceDetectionModel,
    FaceRecognitionModel,
    FaceAttributeType,
)

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load Azure API Credentials from Environment Variables
FACE_API_ENDPOINT = os.getenv("FACE_API_ENDPOINT", "")
FACE_API_KEY = os.getenv("FACE_API_KEY", "")
VISION_API_ENDPOINT = os.getenv("VISION_API_ENDPOINT", "")
VISION_API_KEY = os.getenv("VISION_API_KEY", "")

# Ensure credentials are set
if not FACE_API_ENDPOINT or not FACE_API_KEY:
    raise ValueError("Azure Face API credentials are missing!")
if not VISION_API_ENDPOINT or not VISION_API_KEY:
    raise ValueError("Azure Vision API credentials are missing!")

# Initialize Clients
face_client = FaceClient(FACE_API_ENDPOINT, AzureKeyCredential(FACE_API_KEY))
vision_client = ImageAnalysisClient(VISION_API_ENDPOINT, AzureKeyCredential(VISION_API_KEY))


def analyze_image(image_bytes: bytes):
    """
    Analyze an image for engagement:
    - Detects faces and attributes (head pose, eye occlusion)
    - Detects objects (e.g., phones)
    """
    try:
        # Detect Faces
        faces_info = detect_faces_with_attributes(image_bytes)

        # Detect Objects (Phone Detection)
        phone_detected = detect_objects(image_bytes)

        return {
            "face_count": len(faces_info),
            "faces": faces_info,
            "phone_detected": phone_detected
        }
    except Exception as e:
        logger.error(f"Error analyzing image: {e}")
        return {"error": "Failed to analyze image"}


def detect_faces_with_attributes(image_bytes: bytes):
    """
    Detects faces in an image and extracts attributes:
    - Head pose (yaw, pitch)
    - Eye occlusion
    """
    try:
        face_attrs = [
            FaceAttributeType.HEAD_POSE,
            FaceAttributeType.OCCLUSION
        ]

        with io.BytesIO(image_bytes) as image_stream:
            faces = face_client.detect(
                image_content=image_stream.read(),
                detection_model=FaceDetectionModel.DETECTION03,
                recognition_model=FaceRecognitionModel.RECOGNITION04,
                return_face_id=False,  # No face recognition
                return_face_attributes=face_attrs
            )

        results = []
        for face in faces:
            rect = face.face_rectangle
            attributes = face.face_attributes
            head_pose = attributes.head_pose
            eye_occluded = attributes.occlusion.eye_occluded

            # Determine focus based on head pose
            if abs(head_pose.yaw) > 30 or abs(head_pose.pitch) > 20:
                gaze = "distracted"
            else:
                gaze = "focused"

            # Determine if the student is sleeping
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

        return results
    except Exception as e:
        logger.error(f"Error detecting faces: {e}")
        return []


def detect_objects(image_bytes: bytes):
    """Detect objects in the image to identify items like phones."""
    try:
        with io.BytesIO(image_bytes) as image_stream:
            analysis_result = vision_client.analyze(
                image_data=image_stream.read(),
                visual_features=[VisualFeatures.OBJECTS]
            )

        # Extract object names
        detected_objects = [
            obj.name.lower() for obj in analysis_result.objects if hasattr(obj, "name")
        ]

        return "phone" in detected_objects

    except Exception as e:
        logger.error(f"Error detecting objects: {e}")
        return False
