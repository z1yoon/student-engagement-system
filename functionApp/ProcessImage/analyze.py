import os
import io
import base64
import json
import logging
import numpy as np
from PIL import Image
from azure.core.credentials import AzureKeyCredential
from azure.ai.vision.imageanalysis import ImageAnalysisClient
import insightface
from functools import lru_cache

# Import DB methods (adjust relative import based on your folder structure)
from .db import add_engagement_record, mark_attendance, get_enrolled_students

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
VISION_API_ENDPOINT = os.getenv("VISION_API_ENDPOINT", "")
VISION_API_KEY = os.getenv("VISION_API_KEY", "")

if not VISION_API_ENDPOINT or not VISION_API_KEY:
    raise ValueError("Azure Vision API credentials are missing!")


# Initialize Azure Vision client
vision_client = ImageAnalysisClient(VISION_API_ENDPOINT, AzureKeyCredential(VISION_API_KEY))

# Use a writable directory instead of `/root/`
MODEL_DIR = "/tmp/insightface/models/buffalo_l"

def download_models():
    os.makedirs(MODEL_DIR, exist_ok=True)
    logger.info("📥 Checking and downloading InsightFace models...")

    try:
        face_app = insightface.app.FaceAnalysis(name="buffalo_l", root=MODEL_DIR, download=True)
        face_app.prepare(ctx_id=-1, det_size=(640, 640))
        logger.info("✅ Models downloaded and ready.")

        # Log the contents of the model directory
        model_files = os.listdir(MODEL_DIR)
        logger.info(f"Model directory contents: {model_files}")
    except Exception as e:
        logger.error(f"❌ Error downloading models: {e}")
        raise

@lru_cache(maxsize=1)
def get_face_app():
    """
    Ensures models exist before initializing InsightFace.
    """
    if not os.path.exists(MODEL_DIR) or not os.listdir(MODEL_DIR):
        logger.warning("⚠️ Models missing, triggering download...")
        download_models()

    try:
        face_app = insightface.app.FaceAnalysis(name="buffalo_l", root=MODEL_DIR, download=False)
        face_app.prepare(ctx_id=-1, det_size=(640, 640))
        logger.info("✅ FaceAnalysis initialized successfully.")
        return face_app
    except Exception as e:
        logger.error(f"❌ Error initializing FaceAnalysis: {e}")
        raise

# 🚀 Trigger model download after Azure Function starts
download_models()

# Instantiate face_app after models are ensured
face_app = get_face_app()


def compute_head_pose(landmarks):
    try:
        left_eye = np.array(landmarks[0])
        right_eye = np.array(landmarks[1])
        nose = np.array(landmarks[2])
        eye_center = (left_eye + right_eye) / 2.0
        eye_distance = np.linalg.norm(left_eye - right_eye)
        horizontal_disp = abs(nose[0] - eye_center[0])
        norm_disp = horizontal_disp / eye_distance if eye_distance != 0 else 0
        THRESHOLD = 0.15
        return "distracted (talking)" if norm_disp > THRESHOLD else "focused"
    except Exception as e:
        logger.error(f"Error computing head pose: {e}")
        return "unknown"


def analyze_image(image_data: str):
    try:
        logger.info("Starting image analysis...")
        try:
            image_bytes = base64.b64decode(image_data.split(",")[1]) if "," in image_data else base64.b64decode(
                image_data)
        except Exception as e:
            logger.error(f"Error decoding Base64 image: {e}")
            return {"error": "Invalid base64 image format"}

        faces_info, recognized_students = detect_faces_with_recognition(image_bytes)
        phone_detected = detect_objects(image_bytes)

        if not faces_info:
            logger.info("No faces detected in the image.")
            return {"message": "No faces detected."}

        for face in faces_info:
            face_id = face["face_id"]
            student_name = recognized_students.get(face_id, "Unknown")
            gaze = face.get("gaze", "unknown")
            sleeping = face.get("sleeping", False)
            if student_name != "Unknown":
                mark_attendance(student_name, True, image_data)
            add_engagement_record(student_name, phone_detected, gaze, sleeping)

        logger.info(f"Image analysis complete. Phone detected: {phone_detected}")
        return {
            "message": "Analysis complete",
            "faces": faces_info,
            "recognized_students": recognized_students,
            "phone_detected": phone_detected
        }
    except Exception as e:
        logger.error(f"Error during image analysis: {e}")
        return {"error": "Failed to analyze image"}


def detect_faces_with_recognition(image_bytes: bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_np = np.array(image)
        faces = face_app.get(image_np)
        results = []
        recognized_students = {}
        enrolled_students = get_enrolled_students()
        enrolled_embeddings = []
        enrolled_names = []
        for student in enrolled_students:
            if "face_embedding" in student and student["face_embedding"]:
                enrolled_embeddings.append(np.array(student["face_embedding"]))
                enrolled_names.append(student["name"])
        for idx, face in enumerate(faces):
            face_id = f"face_{idx}"
            bbox = face.bbox
            landmarks = face.landmark.tolist()
            head_pose = compute_head_pose(landmarks)
            face_info = {
                "face_id": face_id,
                "face_rectangle": {
                    "left": int(bbox[0]),
                    "top": int(bbox[1]),
                    "width": int(bbox[2] - bbox[0]),
                    "height": int(bbox[3] - bbox[1])
                },
                "gaze": head_pose,
                "sleeping": False
            }
            results.append(face_info)
            recognized = "Unknown"
            if enrolled_embeddings:
                distances = [np.linalg.norm(face.embedding - emb) for emb in enrolled_embeddings]
                if distances:
                    min_idx = int(np.argmin(distances))
                    if distances[min_idx] < 0.6:
                        recognized = enrolled_names[min_idx]
            recognized_students[face_id] = recognized
        logger.info(f"Detected {len(results)} faces.")
        return results, recognized_students
    except Exception as e:
        logger.error(f"Error detecting faces: {e}")
        return [], {}


def detect_objects(image_bytes: bytes):
    try:
        with io.BytesIO(image_bytes) as stream:
            analysis_result = vision_client.analyze(
                image_data=stream.read(),
                visual_features=["Objects"]
            )
        detected_objects = [obj.name.lower() for obj in analysis_result.objects if hasattr(obj, "name")]
        phone_detected = "phone" in detected_objects
        logger.info(f"Phone detection result: {phone_detected}")
        return phone_detected
    except Exception as e:
        logger.error(f"Error detecting objects: {e}")
        return False
