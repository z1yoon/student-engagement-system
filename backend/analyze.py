import os
import io
import base64
import logging
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from azure.core.credentials import AzureKeyCredential
from azure.ai.vision.imageanalysis import ImageAnalysisClient
import insightface
from functools import lru_cache
import threading

from db import add_engagement_record, mark_attendance, get_enrolled_students

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Azure Vision API credentials
VISION_API_ENDPOINT = os.getenv("VISION_API_ENDPOINT", "")
VISION_API_KEY = os.getenv("VISION_API_KEY", "")
if not VISION_API_ENDPOINT or not VISION_API_KEY:
    raise ValueError("Azure Vision API credentials are missing!")

vision_client = ImageAnalysisClient(VISION_API_ENDPOINT, AzureKeyCredential(VISION_API_KEY))

# Model directory for InsightFace
MODEL_DIR = "/models"
download_lock = threading.Lock()

def ensure_models_exist():
    """
    Checks if the InsightFace models exist and downloads them if necessary.
    """
    expected_folder = os.path.join(MODEL_DIR, "models")
    os.makedirs(expected_folder, exist_ok=True)
    exists = os.path.exists(expected_folder) and len(os.listdir(expected_folder)) > 0
    logger.info("models_exist check: expected_folder=%s exists=%s", expected_folder, exists)
    if exists:
        logger.info("✅ Models already exist in %s, skipping download.", MODEL_DIR)
        return
    with download_lock:
        if os.path.exists(expected_folder) and len(os.listdir(expected_folder)) > 0:
            logger.info("✅ Models already exist after acquiring lock, skipping download.")
            return
        os.makedirs(MODEL_DIR, exist_ok=True)
        logger.info("📥 Downloading InsightFace models into %s...", MODEL_DIR)
        try:
            temp_face_app = insightface.app.FaceAnalysis(name="buffalo_l", root=MODEL_DIR, download=True)
            temp_face_app.prepare(ctx_id=-1, det_size=(640, 640))
            logger.info("✅ Models downloaded and ready.")
        except Exception as e:
            logger.error(f"❌ Error downloading models: {e}")
            raise

@lru_cache(maxsize=1)
def get_face_app():
    """
    Initializes and returns the face analysis app instance.
    """
    ensure_models_exist()
    try:
        face_app_instance = insightface.app.FaceAnalysis(name="buffalo_l", root=MODEL_DIR, download=False)
        face_app_instance.prepare(ctx_id=-1, det_size=(640, 640))
        logger.info("✅ FaceAnalysis initialized successfully.")
        return face_app_instance
    except Exception as e:
        logger.error(f"❌ Error initializing FaceAnalysis: {e}")
        raise

def prepare_image_for_face_detection(image_bytes: bytes, max_size: int = 320):
    """
    Decodes image bytes, converts to RGB, and resizes the image if needed.
    Returns the processed PIL image, its numpy array, and (orig_size, new_size).
    """
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        original_size = image.size  # (width, height) before resize
        if max(image.size) > max_size:
            image.thumbnail((max_size, max_size))
            logger.info("Resized image from %s to %s", original_size, image.size)
        resized_size = image.size
        image_np = np.array(image)
        return image, image_np, original_size, resized_size
    except Exception as e:
        logger.error("Error processing image: %s", e)
        raise

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

def scale_bounding_boxes(faces_info, orig_size, resized_size):
    """
    Scales bounding box coordinates from resized image back to the original image size.
    faces_info is a list of dicts with 'face_rectangle' keys.
    """
    (orig_w, orig_h) = orig_size
    (res_w, res_h) = resized_size

    if (orig_w, orig_h) == (res_w, res_h):
        return faces_info

    scale_x = orig_w / float(res_w)
    scale_y = orig_h / float(res_h)

    for face in faces_info:
        rect = face["face_rectangle"]
        rect["left"] = int(rect["left"] * scale_x)
        rect["top"] = int(rect["top"] * scale_y)
        rect["width"] = int(rect["width"] * scale_x)
        rect["height"] = int(rect["height"] * scale_y)

    return faces_info

def draw_faces_on_image(image_bytes, faces_info, recognized_students):
    """
    Draws rectangles and labels on the original image bytes.
    """
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        for face in faces_info:
            rect = face["face_rectangle"]
            label = recognized_students.get(face["face_id"], "Unknown")
            draw.rectangle(
                [(rect["left"], rect["top"]),
                 (rect["left"] + rect["width"], rect["top"] + rect["height"])],
                outline="red",
                width=2
            )
            draw.text((rect["left"], rect["top"] - 10), label, fill="red", font=font)
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    except Exception as e:
        logger.error(f"Error drawing faces on image: {e}")
        return None

def normalize_embedding(embedding):
    """
    Normalizes an embedding to unit length.
    """
    emb = np.array(embedding)
    norm = np.linalg.norm(emb)
    if norm == 0:
        return emb
    return emb / norm

def average_embeddings(embeddings):
    """
    Averages a list of normalized embeddings.
    """
    if not embeddings:
        return None
    avg_emb = np.mean(np.stack(embeddings), axis=0)
    return normalize_embedding(avg_emb)

def detect_faces_with_recognition_from_np(image_np):
    """
    Accepts a numpy array (from the processed image) and performs face detection.
    Returns face info, recognized students, and a boolean indicating if any face was detected.
    """
    try:
        # Wrap face detection in a try/except in case InsightFace raises an error.
        try:
            faces = get_face_app().get(image_np)
        except Exception as e:
            logger.error("Error during face detection: %s", e)
            return [], {}, False

        results = []
        recognized_students = {}
        enrolled_students = get_enrolled_students()

        # Process enrolled embeddings: assume each student has an averaged embedding.
        enrolled_embeddings = [
            normalize_embedding(student["face_embedding"])
            for student in enrolled_students if student.get("face_embedding")
        ]
        enrolled_names = [
            student["name"]
            for student in enrolled_students if student.get("face_embedding")
        ]

        if not faces:
            logger.warning("⚠️ No faces detected in the image (resized).")
            return [], {}, False

        for idx, face in enumerate(faces):
            logger.info(f"Face {idx}: bbox={face.bbox}, landmark={face.landmark}")
            face_id = f"face_{idx}"
            bbox = face.bbox
            head_pose = "unknown"
            if face.landmark is not None:
                head_pose = compute_head_pose(face.landmark.tolist())

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
            embedding = getattr(face, "embedding", None)
            if embedding is not None:
                try:
                    embedding = np.array(embedding)
                    embedding = normalize_embedding(embedding)
                except Exception as e:
                    logger.error(f"Error converting embedding to numpy array: {e}")
                    embedding = None
                if embedding is not None and len(enrolled_embeddings) > 0:
                    distances = [np.linalg.norm(embedding - emb) for emb in enrolled_embeddings]
                    min_distance = min(distances) if distances else 9999
                    logger.info(f"Face {idx} min_distance: {min_distance}")
                    # For normalized embeddings, a typical threshold might be around 1.2 (adjust as necessary)
                    if min_distance < 1.2:
                        recognized = enrolled_names[np.argmin(distances)]
            recognized_students[face_id] = recognized

        return results, recognized_students, True
    except Exception as e:
        logger.error(f"Error detecting faces: {e}")
        return [], {}, False

def detect_objects(image_bytes: bytes):
    """
    Optional object detection with Azure Vision.
    """
    try:
        with io.BytesIO(image_bytes) as stream:
            analysis_result = vision_client.analyze(
                image_data=stream.read(),
                visual_features=["Objects"]
            )
        if not hasattr(analysis_result, "objects") or not analysis_result.objects:
            logger.warning("⚠️ No objects detected or unexpected response format.")
            return False
        detected_objects = []
        for obj in analysis_result.objects:
            if isinstance(obj, str):
                detected_objects.append(obj.lower())
            elif hasattr(obj, "name"):
                detected_objects.append(obj.name.lower())
        phone_detected = "phone" in detected_objects
        logger.info(f"✅ Phone detection result: {phone_detected}")
        return phone_detected
    except Exception as e:
        logger.error(f"❌ Error detecting objects: {e}")
        return False

def analyze_image(image_data: str):
    """
    Decodes a Base64 image, processes it for face detection, performs face and object detection,
    updates attendance and engagement records, and returns a summary.
    If no face is detected, returns the processed image for inspection.
    """
    try:
        logger.info("Starting image analysis...")
        raw_data = image_data.split(",")[1] if "," in image_data else image_data
        image_bytes = base64.b64decode(raw_data)

        processed_image, image_np, orig_size, resized_size = prepare_image_for_face_detection(image_bytes, 320)

        faces_info, recognized_students, face_detected = detect_faces_with_recognition_from_np(image_np)

        faces_info = scale_bounding_boxes(faces_info, orig_size, resized_size)

        phone_detected = detect_objects(image_bytes)

        if not face_detected:
            logger.info("No faces detected in the image.")
            buffered = io.BytesIO()
            processed_image.save(buffered, format="JPEG")
            resized_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            return {
                "message": "No faces detected.",
                "face_detected": False,
                "annotated_image": resized_b64
            }

        for face in faces_info:
            student_name = recognized_students.get(face["face_id"], "Unknown")
            if student_name != "Unknown":
                mark_attendance(student_name, True, image_data)
                logger.info(f"✅ Marked attendance for {student_name}")
            add_engagement_record(student_name, phone_detected, face["gaze"], face["sleeping"])
            logger.info(f"✅ Added engagement record for {student_name}")

        annotated_image = draw_faces_on_image(image_bytes, faces_info, recognized_students)
        return {
            "message": "Analysis complete",
            "faces": faces_info,
            "recognized_students": recognized_students,
            "phone_detected": phone_detected,
            "annotated_image": annotated_image,
            "face_detected": True
        }
    except Exception as e:
        logger.error(f"Error during image analysis: {e}")
        return {"error": "Failed to analyze image", "face_detected": False}
