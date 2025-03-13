import os
import io
import json
import base64
import logging
import azure.functions as func

from azure.core.credentials import AzureKeyCredential
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.face import FaceClient
from azure.ai.vision.face.models import (
    FaceDetectionModel,
    FaceRecognitionModel,
    FaceAttributeType,
)

from db import (
    add_attendance_record,
    add_engagement_record,
    mark_attendance,
    get_enrolled_students
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load credentials and configuration from environment variables
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
    """
    try:
        logger.info("🔍 Analyzing image for engagement & attendance...")

        # Decode Base64 image (remove data URI scheme if present)
        try:
            if "," in image_data:
                image_bytes = base64.b64decode(image_data.split(",")[1])
            else:
                image_bytes = base64.b64decode(image_data)
        except Exception as e:
            logger.error("❌ Error decoding Base64 image: %s", e)
            return {"error": "Invalid base64 image format"}

        # Face detection and engagement analysis
        faces_info, recognized_students = detect_faces_with_recognition(image_bytes)
        # Object detection (e.g., phone usage)
        phone_detected = detect_objects(image_bytes)

        if not faces_info:
            logger.info("⚠️ No faces detected. Skipping analysis.")
            return {"message": "No faces detected."}

        # Process engagement data and record it in the DB
        for face in faces_info:
            student_name = recognized_students.get(face["face_id"], "Unknown")
            gaze = face.get("gaze", "unknown")
            sleeping = face.get("sleeping", False)

            if student_name != "Unknown":
                mark_attendance(student_name, True, image_data)
            add_engagement_record(student_name, phone_detected, gaze, sleeping)

        logger.info("✅ Image analysis complete. Phone detected: %s", phone_detected)
        return {
            "message": "Analysis complete",
            "faces": faces_info,
            "recognized_students": recognized_students,
            "phone_detected": phone_detected
        }
    except Exception as e:
        logger.error("❌ Error analyzing image: %s", e)
        return {"error": "Failed to analyze image"}


def detect_faces_with_recognition(image_bytes: bytes):
    """
    Detects faces, extracts engagement attributes, and matches detected faces against enrolled students.
    """
    try:
        face_attrs = [FaceAttributeType.HEAD_POSE, FaceAttributeType.OCCLUSION]
        enrolled_students = get_enrolled_students()

        with io.BytesIO(image_bytes) as image_stream:
            faces = face_client.detect(
                image_content=image_stream.read(),
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
                logger.warning("⚠️ No face attributes found. Skipping this face.")
                continue

            head_pose = attributes.head_pose
            eye_occluded = attributes.occlusion.eye_occluded

            gaze = "focused" if abs(head_pose.yaw) <= 30 and abs(head_pose.pitch) <= 20 else "distracted"
            sleeping = eye_occluded

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

        logger.info("✅ Face detection complete. Detected %d faces.", len(results))
        return results, recognized_students
    except Exception as e:
        logger.error("❌ Error detecting faces: %s", e)
        return [], {}


def match_face_with_students(face_id, enrolled_students):
    """
    Matches the detected face with enrolled student images.
    """
    try:
        enrolled_face_ids = [student["face_id"] for student in enrolled_students]
        matches = face_client.verify_face_to_face(face_id, enrolled_face_ids)

        for match in matches:
            if match.confidence >= 0.8:
                return next(
                    student["name"]
                    for student in enrolled_students
                    if student["face_id"] == match.face_id
                )
    except Exception as e:
        logger.error("❌ Error matching faces: %s", e)
    return None


def detect_objects(image_bytes: bytes):
    """
    Detects objects (like phones) in the image.
    """
    try:
        with io.BytesIO(image_bytes) as image_stream:
            analysis_result = vision_client.analyze(
                image_data=image_stream.read(),
                visual_features=["Objects"]
            )
        detected_objects = [obj.name.lower() for obj in analysis_result.objects if hasattr(obj, "name")]
        phone_detected = "phone" in detected_objects
        logger.info("📱 Phone detection result: %s", phone_detected)
        return phone_detected
    except Exception as e:
        logger.error("❌ Error detecting objects: %s", e)
        return False


# Define an HTTP-triggered function. This endpoint can be subscribed to via Event Grid.
app = func.FunctionApp()


@app.function_name(name="ProcessImageHTTP")
@app.route(route="process-image", methods=["POST"])
def process_image(req: func.HttpRequest) -> func.HttpResponse:
    logger.info("📩 Received HTTP request for image processing")
    try:
        payload = req.get_json()
        logger.info(f"Payload: {payload}")

        image_data = payload.get("image_data")
        if image_data:
            logger.info("🔍 Calling analyze_image()...")
            analysis_result = analyze_image(image_data)
            logger.info(f"✅ Analysis result: {analysis_result}")

            # Record results in the database
            if "recognized_students" in analysis_result:
                for face_id, recognized_name in analysis_result["recognized_students"].items():
                    add_attendance_record(
                        student_id=None,
                        recognized_name=recognized_name,
                        image_url=payload.get("image_url", ""),
                        is_attended=True
                    )
            if analysis_result.get("message") == "Analysis complete":
                if "faces" in analysis_result:
                    for face in analysis_result["faces"]:
                        recognized_name = analysis_result["recognized_students"].get(face["face_id"], "Unknown")
                        add_engagement_record(
                            student_name=recognized_name,
                            phone_detected=analysis_result.get("phone_detected", False),
                            gaze=face.get("gaze", "unknown"),
                            sleeping=face.get("sleeping", False)
                        )
            return func.HttpResponse(json.dumps(analysis_result), mimetype="application/json", status_code=200)
        else:
            logger.warning("⚠️ Received request without image data.")
            return func.HttpResponse("No image data found in request.", status_code=400)
    except Exception as e:
        logger.error(f"❌ Error processing image: {e}", exc_info=True)
        return func.HttpResponse("Error processing image.", status_code=500)

