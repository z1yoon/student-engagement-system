import os
import io
import base64
import logging
import numpy as np
import cv2
import mediapipe as mp
from PIL import Image, ImageDraw, ImageFont
from functools import lru_cache
import threading
import time
import platform
from sklearn.metrics.pairwise import cosine_similarity
import urllib.request

from db import add_engagement_record, mark_attendance, get_enrolled_students

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Threshold for eye aspect ratio to determine if eyes are closed
EAR_THRESHOLD = 0.2

# Initialize MediaPipe solutions
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Load MediaPipe models with global variables for reuse
face_detection_mp = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=10,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
pose_detector = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# DNN model paths
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_FILE = os.path.join(MODEL_DIR, "opencv_face_detector_uint8.pb")
CONFIG_FILE = os.path.join(MODEL_DIR, "opencv_face_detector.pbtxt")

# Load OpenCV face recognition model
@lru_cache(maxsize=1)
def get_face_recognizer():
    """
    Initialize and return OpenCV's LBPH face recognizer.
    """
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    logger.info("✅ OpenCV LBPH Face Recognizer initialized")
    return recognizer

# Load OpenCV Haar Cascade face detector (as fallback)
@lru_cache(maxsize=1)
def get_face_detector():
    """
    Load OpenCV's Haar Cascade face detector.
    """
    # Path to the Haar cascade XML file for face detection
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_detector = cv2.CascadeClassifier(cascade_path)
    logger.info("✅ OpenCV Haar Cascade face detector loaded")
    return face_detector

# Load OpenCV DNN face detector
@lru_cache(maxsize=1)
def get_face_detector_dnn():
    """
    Load OpenCV's DNN-based face detector.
    Uses a pre-trained Single Shot Multibox Detector (SSD) with ResNet base network.
    """
    try:
        # Path to local model files (already downloaded)
        model_file = MODEL_FILE
        config_file = CONFIG_FILE
        
        # Check if files exist and are not empty
        if not os.path.exists(model_file) or os.path.getsize(model_file) == 0:
            logger.error(f"DNN model file missing or empty: {model_file}")
            return None
            
        if not os.path.exists(config_file) or os.path.getsize(config_file) == 0:
            logger.error(f"DNN config file missing or empty: {config_file}")
            return None
        
        # Make sure directory exists
        os.makedirs(os.path.dirname(model_file), exist_ok=True)
        
        # Load the model (protobuf + text config)
        logger.info(f"Loading DNN model from {model_file} and {config_file}")
        try:
            # Use readNet instead of readNetFromTensorflow for better compatibility
            net = cv2.dnn.readNet(model_file, config_file)
            logger.info("✅ OpenCV DNN face detector loaded successfully")
            return net
        except cv2.error as e:
            logger.error(f"OpenCV error loading model: {e}")
            # Try alternative loading method
            try:
                logger.info("Trying alternative loading method...")
                net = cv2.dnn.readNetFromTensorflow(model_file, config_file)
                logger.info("✅ OpenCV DNN face detector loaded successfully with alternative method")
                return net
            except Exception as e2:
                logger.error(f"Alternative loading method failed: {e2}")
                return None
    except Exception as e:
        logger.error(f"❌ Error loading DNN face detector: {e}")
        return None

def detect_faces_opencv(image_np):
    """
    Detect faces using OpenCV's Haar cascade classifier.
    Returns a list of face rectangles (x, y, w, h).
    """
    try:
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        
        # Get face detector
        face_detector = get_face_detector()
        
        # Detect faces
        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        logger.info(f"✅ Detected {len(faces)} faces using OpenCV Haar Cascade")
        return faces
    except Exception as e:
        logger.error(f"❌ Error in OpenCV Haar Cascade face detection: {e}")
        return []

def detect_faces_opencv_dnn(image_np):
    """
    Detect faces using OpenCV's DNN-based face detector.
    Returns a list of face rectangles (x, y, w, h).
    """
    try:
        # Get the DNN face detector
        net = get_face_detector_dnn()
        if net is None:
            # Don't fall back to Haar cascade, just return empty result
            logger.error("❌ DNN face detector not available and fallback disabled")
            return []
        
        # Get image dimensions
        h, w = image_np.shape[:2]
        
        # Create a blob from the image
        blob = cv2.dnn.blobFromImage(
            image_np, 
            1.0,  # scale factor
            (300, 300),  # spatial size required by the model
            [104, 117, 123],  # mean subtraction
            False,  # no crop
            False  # BGR to RGB conversion not needed
        )
        
        # Set the input to the network
        net.setInput(blob)
        
        # Run a forward pass to get face detections
        detections = net.forward()
        
        faces = []
        confidence_threshold = 0.7
        
        # Loop over all detections
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            # If we're confident enough it's a face
            if confidence > confidence_threshold:
                # Get the coordinates of the face rectangle
                x1 = int(detections[0, 0, i, 3] * w)
                y1 = int(detections[0, 0, i, 4] * h)
                x2 = int(detections[0, 0, i, 5] * w)
                y2 = int(detections[0, 0, i, 6] * h)
                
                # Ensure coordinates are within image bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                # Convert to (x, y, w, h) format
                if x2 > x1 and y2 > y1:  # Ensure valid rectangle
                    faces.append((x1, y1, x2 - x1, y2 - y1))
        
        logger.info(f"✅ Detected {len(faces)} faces using OpenCV DNN")
        return faces
    except Exception as e:
        logger.error(f"❌ Error in OpenCV DNN face detection: {e}")
        # Don't fall back to Haar cascade if DNN fails
        return []

def extract_face_features(image_np, face_rect):
    """
    Extract face features using Local Binary Patterns Histograms (LBPH)
    Returns a feature vector that can be used for face recognition.
    """
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        
        # Extract the face region
        x, y, w, h = face_rect
        roi_gray = gray[y:y+h, x:x+w]
        
        # Resize to a standard size for consistent feature extraction
        roi_gray = cv2.resize(roi_gray, (100, 100))
        
        # Calculate LBP features - this will serve as our face embedding
        # We flatten the histogram to create a feature vector
        lbp = cv2.face.LBPHFaceRecognizer_create()
        
        # We need to train on a single image to get the histogram
        # This is a workaround as we can't directly extract features
        lbp.train([roi_gray], np.array([0]))
        
        # Get the histogram from the model
        hist = lbp.getHistograms()[0]
        feature_vector = hist.flatten()
        
        return feature_vector
    except Exception as e:
        logger.error(f"❌ Error extracting face features: {e}")
        return None

def compare_face_with_students_opencv(face_features, enrolled_students, threshold=0.6):
    """
    Compare extracted face features with enrolled student embeddings.
    Returns the best match name and confidence score.
    """
    best_match = "Unknown"
    best_score = 0
    
    if face_features is None:
        return best_match, best_score
    
    for student in enrolled_students:
        if not student.get("face_embedding"):
            continue
            
        # Convert student face embedding to proper format
        student_embedding = np.array(student["face_embedding"])
        
        # Check if embeddings are compatible (same length)
        if len(face_features) != len(student_embedding):
            continue
            
        # Calculate cosine similarity between feature vectors
        similarity = cosine_similarity([face_features], [student_embedding])[0][0]
        
        if similarity > best_score and similarity > threshold:
            best_score = similarity
            best_match = student["name"]
    
    return best_match, best_score

def prepare_image_for_processing(image_bytes: bytes, max_size: int = 320):
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
        # Convert PIL Image to numpy array (RGB)
        image_np = np.array(image)
        # Convert RGB to BGR for OpenCV processing
        image_np_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        return image, image_np, image_np_bgr, original_size, resized_size
    except Exception as e:
        logger.error("Error processing image: %s", e)
        raise

def calculate_eye_aspect_ratio(eye_landmarks, face_landmarks):
    """
    Calculate the Eye Aspect Ratio (EAR) to detect drowsiness
    """
    # Extract eye landmarks
    # MediaPipe indices for the eye landmarks:
    # Left eye: 362, 385, 387, 263, 373, 380
    # Right eye: 33, 160, 158, 133, 153, 144
    
    # Get the 3D coordinates of the eye landmarks
    if not face_landmarks:
        return 1.0  # Default to open eyes if no landmarks
    
    left_eye_pts = [face_landmarks.landmark[idx] for idx in [362, 385, 387, 263, 373, 380]]
    right_eye_pts = [face_landmarks.landmark[idx] for idx in [33, 160, 158, 133, 153, 144]]
    
    # Calculate the eye aspect ratio for both eyes
    def distance(p1, p2):
        return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
    
    # Vertical eye distances
    left_eye_v1 = distance(left_eye_pts[1], left_eye_pts[5])
    left_eye_v2 = distance(left_eye_pts[2], left_eye_pts[4])
    
    right_eye_v1 = distance(right_eye_pts[1], right_eye_pts[5])
    right_eye_v2 = distance(right_eye_pts[2], right_eye_pts[4])
    
    # Horizontal eye distances
    left_eye_h = distance(left_eye_pts[0], left_eye_pts[3])
    right_eye_h = distance(right_eye_pts[0], right_eye_pts[3])
    
    # Calculate EAR for each eye
    left_ear = (left_eye_v1 + left_eye_v2) / (2.0 * left_eye_h)
    right_ear = (right_eye_v1 + right_eye_v2) / (2.0 * right_eye_h)
    
    # Average both eyes
    ear = (left_ear + right_ear) / 2.0
    return ear

def detect_gaze_direction(face_landmarks, image_shape):
    """
    Detect gaze direction using MediaPipe face landmarks
    Returns "focused" or "distracted"
    """
    if not face_landmarks:
        return "unknown"
    
    # Iris landmarks (left eye: 468, right eye: 473)
    left_iris = np.array([face_landmarks.landmark[468].x, face_landmarks.landmark[468].y])
    right_iris = np.array([face_landmarks.landmark[473].x, face_landmarks.landmark[473].y])
    
    # Eye corner landmarks
    left_eye_left_corner = np.array([face_landmarks.landmark[362].x, face_landmarks.landmark[362].y])
    left_eye_right_corner = np.array([face_landmarks.landmark[263].x, face_landmarks.landmark[263].y])
    
    right_eye_left_corner = np.array([face_landmarks.landmark[133].x, face_landmarks.landmark[133].y])
    right_eye_right_corner = np.array([face_landmarks.landmark[33].x, face_landmarks.landmark[33].y])
    
    # Calculate horizontal ratio for each eye
    def calculate_ratio(iris, left_corner, right_corner):
        eye_width = np.linalg.norm(right_corner - left_corner)
        if eye_width == 0:
            return 0.5  # Default to center
        iris_to_left = np.linalg.norm(iris - left_corner)
        return iris_to_left / eye_width
    
    left_ratio = calculate_ratio(left_iris, left_eye_left_corner, left_eye_right_corner)
    right_ratio = calculate_ratio(right_iris, right_eye_left_corner, right_eye_right_corner)
    
    # Average ratio (0.5 means looking straight)
    avg_ratio = (left_ratio + right_ratio) / 2.0
    
    # Determine if looking left, center, or right
    looking_direction = "center"
    if avg_ratio < 0.45:
        looking_direction = "left"
    elif avg_ratio > 0.55:
        looking_direction = "right"
    
    # Consider "center" as focused, others as potentially distracted
    gaze_status = "focused" if looking_direction == "center" else "distracted"
    
    return gaze_status

def detect_phone(pose_landmarks, image_shape):
    """
    Detect phone usage based on hand position relative to face
    """
    if not pose_landmarks:
        return False
    
    # Get wrist, shoulder and ear positions
    landmarks = pose_landmarks.landmark
    
    # Check if landmarks exist
    try:
        left_wrist = np.array([landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, 
                              landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y])
        right_wrist = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y])
        
        left_ear = np.array([landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y])
        right_ear = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].x,
                             landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y])
        
        # Calculate distances between wrists and ears
        left_wrist_to_ear_dist = np.linalg.norm(left_wrist - left_ear)
        right_wrist_to_ear_dist = np.linalg.norm(right_wrist - right_ear)
        
        # Threshold for phone detection (wrist close to ear)
        threshold = 0.2  # Adjust based on testing
        
        phone_detected = (left_wrist_to_ear_dist < threshold or right_wrist_to_ear_dist < threshold)
        return phone_detected
    except:
        return False

def detect_sleeping(ear_value):
    """
    Detect if a person is sleeping based on Eye Aspect Ratio
    """
    return ear_value < EAR_THRESHOLD

def analyze_face_with_mediapipe(image_np, image_np_bgr, image_shape):
    """
    Analyze face using MediaPipe for engagement metrics
    """
    # Process with MediaPipe Face Detection
    face_detection_results = face_detection_mp.process(image_np)
    
    # Process with MediaPipe Face Mesh
    face_mesh_results = face_mesh.process(image_np)
    
    # Process with MediaPipe Pose
    pose_results = pose_detector.process(image_np)
    
    face_info = []
    
    # If face detection results are available
    if face_detection_results.detections:
        for idx, detection in enumerate(face_detection_results.detections):
            # Get bounding box
            bbox = detection.location_data.relative_bounding_box
            h, w, _ = image_shape
            
            # Convert relative coordinates to absolute
            x_min = int(bbox.xmin * w)
            y_min = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)
            
            face_id = f"face_{idx}"
            
            # Default values
            gaze = "unknown"
            sleeping = False
            
            # Get face mesh landmarks if available for this face
            face_landmarks = None
            if face_mesh_results.multi_face_landmarks:
                # Try to match face detection with face mesh
                # Simple approach: use the first face mesh for the first face detection
                if idx < len(face_mesh_results.multi_face_landmarks):
                    face_landmarks = face_mesh_results.multi_face_landmarks[idx]
            
            # Calculate engagement metrics if face landmarks are available
            if face_landmarks:
                # 1. Gaze detection
                gaze = detect_gaze_direction(face_landmarks, image_shape)
                
                # 2. Drowsiness/sleeping detection using Eye Aspect Ratio (EAR)
                ear_value = calculate_eye_aspect_ratio(None, face_landmarks)
                sleeping = detect_sleeping(ear_value)
            
            # Add face info
            face_info.append({
                "face_id": face_id,
                "face_rectangle": {
                    "left": x_min,
                    "top": y_min,
                    "width": width,
                    "height": height
                },
                "gaze": gaze,
                "sleeping": sleeping
            })
    
    # Detect phone usage using pose landmarks
    phone_detected = False
    if pose_results.pose_landmarks:
        phone_detected = detect_phone(pose_results.pose_landmarks, image_shape)
    
    return face_info, phone_detected

def normalize_embedding(embedding):
    """
    Normalizes an embedding to unit length.
    """
    emb = np.array(embedding)
    norm = np.linalg.norm(emb)
    if norm == 0:
        return emb
    return emb / norm

# Main face detection and recognition function using OpenCV DNN + LBPH
def detect_faces_with_recognition_from_np(image_np):
    """
    Accepts a numpy array and performs face detection using DNN detector
    and recognition using LBPH. Returns face info, recognized students, 
    and a boolean indicating if any face was detected.
    """
    try:
        # Use OpenCV DNN for face detection (no fallback to Haar Cascade)
        faces = detect_faces_opencv_dnn(image_np)
        
        if len(faces) == 0:
            logger.warning("⚠️ No faces detected in the image using DNN detector.")
            return [], {}, False
            
        results = []
        recognized_students = {}
        enrolled_students = get_enrolled_students()
        
        # Process each detected face
        for idx, (x, y, w, h) in enumerate(faces):
            face_id = f"face_{idx}"
            
            # Create face info for the UI
            face_info = {
                "face_id": face_id,
                "face_rectangle": {
                    "left": int(x),
                    "top": int(y),
                    "width": int(w),
                    "height": int(h)
                },
                # These will be filled in later with MediaPipe analysis
                "gaze": "unknown",
                "sleeping": False
            }
            results.append(face_info)
            
            # Extract features for recognition using LBPH
            face_features = extract_face_features(image_np, (x, y, w, h))
            
            # Match with enrolled students
            recognized_name, score = compare_face_with_students_opencv(face_features, enrolled_students)
            recognized_students[face_id] = recognized_name
            logger.info(f"Face {idx} recognized as {recognized_name} with score {score:.2f}")
        
        return results, recognized_students, True
    except Exception as e:
        logger.error(f"Error detecting faces with DNN: {e}")
        return [], {}, False

def draw_faces_on_image(image_bytes, faces_info, recognized_students, phone_detected=False):
    """
    Draws rectangles and labels on the original image bytes.
    """
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        
        # Draw each face with its information
        for face in faces_info:
            rect = face["face_rectangle"]
            student_name = recognized_students.get(face["face_id"], "Unknown")
            gaze_status = face["gaze"]
            sleeping_status = "Sleeping" if face["sleeping"] else "Awake"
            
            # Set color based on engagement (red for disengaged, green for engaged)
            if gaze_status == "focused" and not face["sleeping"]:
                outline_color = "green"
                text_color = "green"
            else:
                outline_color = "red"
                text_color = "red"
            
            # Draw face rectangle
            draw.rectangle(
                [(rect["left"], rect["top"]),
                 (rect["left"] + rect["width"], rect["top"] + rect["height"])],
                outline=outline_color,
                width=2
            )
            
            # Draw name label
            draw.text((rect["left"], rect["top"] - 30), student_name, fill=text_color, font=font)
            
            # Draw engagement status
            status_text = f"Gaze: {gaze_status}, {sleeping_status}"
            draw.text((rect["left"], rect["top"] - 15), status_text, fill=text_color, font=font)
        
        # Draw phone usage indicator if detected
        if phone_detected:
            draw.text((10, 10), "Phone Detected!", fill="red", font=font)
        
        # Convert the annotated image back to base64
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    except Exception as e:
        logger.error(f"Error drawing faces on image: {e}")
        return None

def analyze_image(image_data: str):
    """
    Decodes a Base64 image, processes it for face detection, performs face and engagement detection,
    updates attendance and engagement records, and returns a summary.
    If no face is detected, returns the processed image for inspection.
    """
    try:
        logger.info("Starting image analysis...")
        raw_data = image_data.split(",")[1] if "," in image_data else image_data
        image_bytes = base64.b64decode(raw_data)

        # Prepare the image for processing
        processed_image, image_np, image_np_bgr, orig_size, resized_size = prepare_image_for_processing(image_bytes, 320)
        
        # Analyze faces using OpenCV for recognition
        faces_info, recognized_students, face_detected = detect_faces_with_recognition_from_np(image_np)
        
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

        # Use MediaPipe for engagement analysis
        mediapipe_faces, phone_detected = analyze_face_with_mediapipe(image_np, image_np_bgr, image_np.shape)
        
        # Merge MediaPipe analysis results with OpenCV recognition results
        if len(faces_info) > 0 and len(mediapipe_faces) > 0:
            # Match faces based on the overlap of bounding boxes
            for opencv_face in faces_info:
                best_match = None
                max_overlap = 0
                
                # Get the OpenCV face rectangle
                rect1 = opencv_face["face_rectangle"]
                box1 = (rect1["left"], rect1["top"], rect1["left"] + rect1["width"], rect1["top"] + rect1["height"])
                
                # Find the best matching MediaPipe face
                for mediapipe_face in mediapipe_faces:
                    rect2 = mediapipe_face["face_rectangle"]
                    box2 = (rect2["left"], rect2["top"], rect2["left"] + rect2["width"], rect2["top"] + rect2["height"])
                    
                    # Calculate overlap
                    x_overlap = max(0, min(box1[2], box2[2]) - max(box1[0], box2[0]))
                    y_overlap = max(0, min(box1[3], box2[3]) - max(box1[1], box2[1]))
                    overlap_area = x_overlap * y_overlap
                    
                    if overlap_area > max_overlap:
                        max_overlap = overlap_area
                        best_match = mediapipe_face
                
                # Update the OpenCV face info with MediaPipe engagement analysis
                if best_match:
                    opencv_face["gaze"] = best_match["gaze"]
                    opencv_face["sleeping"] = best_match["sleeping"]
        
        # Process attendance and engagement for each recognized face
        for face in faces_info:
            student_name = recognized_students.get(face["face_id"], "Unknown")
            if student_name != "Unknown":
                mark_attendance(student_name, True, image_data)
                logger.info(f"✅ Marked attendance for {student_name}")
            
            # Add engagement record
            add_engagement_record(student_name, phone_detected, face["gaze"], face["sleeping"])
            logger.info(f"✅ Added engagement record for {student_name}")

        # Draw annotations on the image
        annotated_image = draw_faces_on_image(image_bytes, faces_info, recognized_students, phone_detected)

        return {
            "message": "Image analyzed successfully.",
            "face_detected": True,
            "recognized_students": list(set(recognized_students.values())),
            "phone_detected": phone_detected,
            "annotated_image": annotated_image
        }
    except Exception as e:
        logger.error(f"Error during image analysis: {e}")
        return {
            "message": f"Error during analysis: {str(e)}",
            "face_detected": False
        }
