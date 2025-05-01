import os
import io
import base64
import logging
import numpy as np
import cv2
from PIL import Image
from functools import lru_cache
import insightface
from insightface.app import FaceAnalysis
from numpy.linalg import norm
from db import add_engagement_record, mark_attendance, get_enrolled_students
import requests
from collections import defaultdict

# Student tracking dictionaries to maintain state across frames
student_eye_closed_count = defaultdict(int)  # Tracks consecutive frames with closed eyes
student_head_down_count = defaultdict(int)   # Tracks consecutive frames with head down
student_head_turned_count = defaultdict(int)  # Tracks consecutive frames with head turned

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_DIR = "/models"

def prepare_image_for_processing(image_bytes: bytes):
    """Convert image bytes to formats needed for processing"""
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_np = np.array(image)
        image_np_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        return image, image_np, image_np_bgr
    except Exception as e:
        logger.error("Error processing image: %s", e)
        raise

def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors"""
    return np.dot(a, b) / (norm(a) * norm(b))

@lru_cache(maxsize=1)
def get_face_app():
    """Initialize and return the face analysis app instance"""
    try:
        providers = ['CPUExecutionProvider']
        app = FaceAnalysis(
            name="buffalo_s",
            root=MODEL_DIR,
            providers=providers,
            download=True,
            allowed_modules=['detection', 'recognition']
        )
        app.prepare(ctx_id=-1, det_size=(640, 640))
        return app
    except Exception as e:
        logger.error(f"Error initializing FaceAnalysis: {e}")
        raise

def compare_embeddings(embedding, enrolled_students, threshold=0.5):
    """Compare face embedding with enrolled students using cosine similarity"""
    best_name = "Unknown"
    best_similarity = -1
    
    for student in enrolled_students:
        student_name = student.get("name", "Unknown")
        emb = student.get("face_embedding")
        if emb is not None:
            similarity = cosine_similarity(embedding, np.array(emb))
            logger.debug(f"Similarity to {student_name}: {similarity:.4f}")
            if similarity > best_similarity and similarity > threshold:
                best_similarity = similarity
                best_name = student["name"]
    
    if best_name != "Unknown":
        logger.info(f"Recognized as {best_name} with similarity {best_similarity:.4f}")
    
    return best_name

def detect_phone_with_azure_vision(image_bytes):
    """Detect phones in an image using Azure Computer Vision API"""
    endpoint = os.getenv("VISION_API_ENDPOINT")
    key = os.getenv("VISION_API_KEY")
    
    if not endpoint or not key:
        logger.error("Azure Vision API credentials not found")
        return False, {}
    
    analyze_url = f"{endpoint}/vision/v3.2/analyze"
    headers = {
        'Ocp-Apim-Subscription-Key': key,
        'Content-Type': 'application/octet-stream'
    }
    params = {
        'visualFeatures': 'Objects',
        'language': 'en'
    }
    
    try:
        response = requests.post(
            analyze_url,
            headers=headers,
            params=params,
            data=image_bytes
        )
        response.raise_for_status()
        
        result = response.json()
        objects = result.get('objects', [])
        
        phone_related_terms = ['phone', 'cellphone', 'mobile', 'smartphone']
        phone_detections = {}
        
        for obj in objects:
            object_name = obj.get('object', '').lower()
            if any(term in object_name for term in phone_related_terms):
                confidence = obj.get('confidence', 0)
                rectangle = obj.get('rectangle', {})
                logger.info(f"Phone detected with confidence {confidence}")
                
                phone_detections[object_name] = {
                    'confidence': confidence,
                    'position': rectangle
                }
                
        return len(phone_detections) > 0, phone_detections
    
    except Exception as e:
        logger.error(f"Error calling Azure Vision API: {str(e)}")
        return False, {}

def detect_eye_closure(face):
    """Detect if a person's eyes are closed based on face landmarks"""
    landmarks = getattr(face, 'landmark_2d_106', None)
    if landmarks is None:
        return False
    
    try:
        left_eye_height = np.linalg.norm(landmarks[37] - landmarks[41])
        left_eye_width = np.linalg.norm(landmarks[36] - landmarks[39])
        
        right_eye_height = np.linalg.norm(landmarks[43] - landmarks[47])
        right_eye_width = np.linalg.norm(landmarks[42] - landmarks[45])
        
        left_ear = left_eye_height / max(left_eye_width, 1e-5)
        right_ear = right_eye_height / max(right_eye_width, 1e-5)
        
        ear = (left_ear + right_ear) / 2
        ear_threshold = 0.2
        
        return ear < ear_threshold
    except (IndexError, AttributeError):
        logger.warning("Could not analyze eye landmarks")
        return False

def detect_head_position(face):
    """Detect head position (down, left, right) based on face pose"""
    pose = getattr(face, 'pose', None)
    if pose is None:
        return 'center'
    
    try:
        yaw = pose[0]  # left/right
        pitch = pose[1]  # up/down
        
        yaw_threshold = 25  # degrees
        pitch_threshold = 20  # degrees
        
        if pitch > pitch_threshold:
            return 'down'
        elif yaw < -yaw_threshold:
            return 'left'
        elif yaw > yaw_threshold:
            return 'right'
        else:
            return 'center'
    except (IndexError, AttributeError):
        logger.warning("Could not analyze head pose")
        return 'center'

def draw_faces_with_status(image_bytes, faces, recognized_students, student_status):
    """Draw faces with detailed status information on the image"""
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        for idx, face in enumerate(faces):
            bbox = face.bbox.astype(np.int32)
            student_name = recognized_students.get(idx, "Unknown")
            
            # Handle unknown faces
            if student_name == "Unknown":
                cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
                cv2.putText(img, "Unknown", (bbox[0], max(0, bbox[1] - 10)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                continue
            
            status = student_status.get(student_name, {})
            
            # Determine box color and status text based on student's status
            if status.get('is_sleeping', False):
                color = (0, 0, 128)  # Dark red for sleeping
                status_text = "SLEEPING"
            elif status.get('is_distracted', False):
                color = (0, 165, 255)  # Orange for distracted
                status_text = "DISTRACTED"
            elif status.get('using_phone', False):
                color = (255, 0, 0)  # Blue for phone usage
                status_text = "PHONE"
            else:
                color = (0, 255, 0)  # Green for focused
                status_text = "FOCUSED"
            
            # Draw rectangle around face
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            # Draw student name and status
            y_pos = max(0, bbox[1] - 35)
            cv2.putText(img, student_name, (bbox[0], y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(img, status_text, (bbox[0], y_pos + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Add counter indicators when applicable
            if student_eye_closed_count[student_name] > 0:
                cv2.putText(img, f"Eyes closed: {student_eye_closed_count[student_name]}/3", 
                          (bbox[0], bbox[3] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            if student_head_down_count[student_name] > 0:
                cv2.putText(img, f"Head down: {student_head_down_count[student_name]}/3", 
                          (bbox[0], bbox[3] + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
            if student_head_turned_count[student_name] > 0:
                cv2.putText(img, f"Head turned: {student_head_turned_count[student_name]}/3", 
                          (bbox[0], bbox[3] + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Draw summary at the top of the image
        recognized_count = len([n for n in recognized_students.values() if n != "Unknown"])
        cv2.putText(img, f"Students detected: {recognized_count}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw phone detection summary if applicable
        phone_using_students = [name for name, status in student_status.items() 
                              if status.get('using_phone', False)]
        if phone_using_students:
            cv2.putText(img, f"Phone detected: {', '.join(phone_using_students)}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Convert the image back to bytes for web display
        is_success, buffer = cv2.imencode(".jpg", img)
        if is_success:
            return base64.b64encode(buffer).decode('utf-8')
        else:
            logger.error("Failed to encode annotated image")
            return None
    except Exception as e:
        logger.error(f"Error drawing faces with status: {e}")
        return None

def analyze_image(image_data: str):
    """Main function to analyze an image for student engagement"""
    try:
        logger.info("Starting image analysis...")
        raw_data = image_data.split(",")[1] if "," in image_data else image_data
        image_bytes = base64.b64decode(raw_data)
        pil_image, image_np, image_np_bgr = prepare_image_for_processing(image_bytes)
        
        # Get face app and detect faces
        face_app = get_face_app()
        faces = face_app.get(image_np_bgr)
        
        # Handle case where no faces detected
        if not faces:
            logger.info("No faces detected in the image.")
            buffered = io.BytesIO()
            pil_image.save(buffered, format="JPEG")
            img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            return {
                "message": "No faces detected.",
                "face_detected": False,
                "annotated_image": img_b64
            }
        
        # Get enrolled students from database
        enrolled_students = get_enrolled_students()
        recognized_students = {}
        student_status = {}
        
        # Detect phone usage using Azure Vision API
        phone_detected, phone_detections = detect_phone_with_azure_vision(image_bytes)
        
        # Process each detected face
        for idx, face in enumerate(faces):
            embedding = face.embedding
            name = compare_embeddings(embedding, enrolled_students)
            recognized_students[idx] = name
            
            # Skip unknown faces for engagement tracking
            if name == "Unknown":
                continue
                
            # Mark attendance for recognized students
            mark_attendance(name, True, image_data)
            
            # Get face bounding box for phone detection association
            bbox = face.bbox.astype(np.int32)
            face_center_x = (bbox[0] + bbox[2]) / 2
            face_center_y = (bbox[1] + bbox[3]) / 2
            
            # Initialize status for this student
            student_status[name] = {
                'is_sleeping': False,
                'is_distracted': False,
                'using_phone': False,
                'gaze': 'focused'
            }
            
            # Detect eye closure and head position
            eyes_closed = detect_eye_closure(face)
            head_position = detect_head_position(face)
            
            # Update consecutive frame counters for this student
            if eyes_closed:
                student_eye_closed_count[name] += 1
            else:
                student_eye_closed_count[name] = 0
                
            if head_position == 'down':
                student_head_down_count[name] += 1
            else:
                student_head_down_count[name] = 0
                
            if head_position in ['left', 'right']:
                student_head_turned_count[name] += 1
            else:
                student_head_turned_count[name] = 0
            
            # Determine if student is sleeping (3 consecutive frames with eyes closed or head down)
            is_sleeping = (student_eye_closed_count[name] >= 3 or 
                          student_head_down_count[name] >= 3)
            
            # Determine if student is distracted (3 consecutive frames with head turned)
            is_distracted = student_head_turned_count[name] >= 3
            
            # Determine if student is using a phone by checking proximity to detected phones
            student_using_phone = False
            if phone_detected:
                for phone_info in phone_detections.values():
                    phone_rect = phone_info['position']
                    phone_center_x = phone_rect['x'] + (phone_rect['w'] / 2)
                    phone_center_y = phone_rect['y'] + (phone_rect['h'] / 2)
                    
                    # Calculate distance between face and phone
                    distance = np.sqrt((face_center_x - phone_center_x)**2 + 
                                      (face_center_y - phone_center_y)**2)
                    
                    # If phone is close to this face, associate it with this student
                    if distance < 300:  # Threshold for phone proximity
                        student_using_phone = True
                        break
            
            # Update student status
            student_status[name]['is_sleeping'] = is_sleeping
            student_status[name]['is_distracted'] = is_distracted
            student_status[name]['using_phone'] = student_using_phone
            
            # Determine gaze status
            if is_sleeping:
                gaze_status = "sleeping"
            elif is_distracted:
                gaze_status = "distracted" 
            elif head_position == 'down':
                gaze_status = "looking down"
            elif head_position in ['left', 'right']:
                gaze_status = "looking away"
            else:
                gaze_status = "focused"
                
            student_status[name]['gaze'] = gaze_status
            
            # Record engagement with improved criteria
            add_engagement_record(name, student_using_phone, gaze_status, is_sleeping)
        
        # Create annotated image with status information
        annotated_image = draw_faces_with_status(image_bytes, faces, recognized_students, student_status)
        
        return {
            "message": "Image analyzed successfully.",
            "face_detected": True,
            "recognized_students": list(set(recognized_students.values())),
            "student_status": student_status,
            "annotated_image": annotated_image
        }
    except Exception as e:
        logger.error(f"Error during image analysis: {e}")
        return {
            "message": f"Error during analysis: {str(e)}",
            "face_detected": False
        }
