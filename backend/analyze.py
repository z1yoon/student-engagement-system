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
from db import mark_attendance, get_enrolled_students, record_student_engagement
import requests
from collections import defaultdict
from config import VISION_API_ENDPOINT, VISION_API_KEY
import math
import traceback
import mediapipe as mp

# Student tracking dictionaries to maintain state across frames
student_eye_closed_count = defaultdict(int)  # Tracks consecutive frames with closed eyes
student_head_turned_count = defaultdict(int)  # Tracks consecutive frames with head turned

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_DIR = "/models"

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=10,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

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
    """
    Detect phones in an image using Azure Computer Vision API
    Only detects actual phones with exact matches
    """
    endpoint = VISION_API_ENDPOINT
    key = VISION_API_KEY
    
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
        # Call Azure Vision API
        response = requests.post(
            analyze_url,
            headers=headers,
            params=params,
            data=image_bytes
        )
        response.raise_for_status()
        
        result = response.json()
        objects = result.get('objects', [])
        
        # Log all detected objects for debugging
        logger.info(f"Azure Vision detected {len(objects)} objects")
        for obj in objects:
            logger.info(f"Detected object: {obj.get('object', '').lower()} with confidence {obj.get('confidence', 0)}")
        
        # ONLY detect exact phone objects
        phone_terms = ['phone', 'cellphone', 'mobile', 'smartphone', 'iphone', 'android']
        phone_detections = {}
        
        for obj in objects:
            object_name = obj.get('object', '').lower()
            if object_name in phone_terms:
                confidence = obj.get('confidence', 0)
                rectangle = obj.get('rectangle', {})
                logger.info(f"Phone detected with confidence {confidence}")
                
                phone_detections[object_name] = {
                    'confidence': confidence,
                    'position': rectangle
                }
        
        # Return True only if we found explicit phone objects
        has_phones = len(phone_detections) > 0
        return has_phones, phone_detections
    
    except Exception as e:
        logger.error(f"Error calling Azure Vision API: {str(e)}")
        return False, {}

def detect_eye_closure(face_landmarks, width, height):
    """
    Detect if eyes are closed for a single face using individual landmark coordinates
    Avoids any list operations that might cause errors
    
    Returns: True if eyes are closed, False otherwise
    """
    try:
        # Define landmark indices for eyes
        # Right eye - upper and lower points
        right_eye_upper = 159  # Upper eyelid
        right_eye_lower = 145  # Lower eyelid
        
        # Left eye - upper and lower points
        left_eye_upper = 386  # Upper eyelid
        left_eye_lower = 374  # Lower eyelid
        
        # Get y coordinates for these landmarks
        right_upper_y = face_landmarks.landmark[right_eye_upper].y * height
        right_lower_y = face_landmarks.landmark[right_eye_lower].y * height
        
        left_upper_y = face_landmarks.landmark[left_eye_upper].y * height
        left_lower_y = face_landmarks.landmark[left_eye_lower].y * height
        
        # Calculate vertical distances
        right_eye_height = abs(right_upper_y - right_lower_y)
        left_eye_height = abs(left_upper_y - left_lower_y)
        
        # Average eye height
        avg_eye_height = (right_eye_height + left_eye_height) / 2
        
        # More sensitive threshold for detecting closed eyes (especially for Asian eyes)
        # Lower values mean eyes more likely to be considered closed
        CLOSURE_THRESHOLD = 2.2
        
        # Log actual values for debugging
        logger.info(f"Eye height values - Right: {right_eye_height:.2f}, Left: {left_eye_height:.2f}, Avg: {avg_eye_height:.2f}")
        
        # Check both eyes individually and also the average
        is_closed = avg_eye_height < CLOSURE_THRESHOLD
        
        if is_closed:
            logger.info(f"Eyes detected as CLOSED - height: {avg_eye_height:.2f}")
        else:
            logger.info(f"Eyes detected as OPEN - height: {avg_eye_height:.2f}")
            
        # Return True if eyes are closed
        return is_closed
        
    except Exception as e:
        logger.error(f"Error in eye closure detection: {e}")
        # Default to open if there's an error
        return False

def detect_eye_closure_with_mediapipe(image_np):
    """
    Detect eye state (open or closed) using MediaPipe face mesh.
    Simple, reliable implementation that avoids complex calculations.
    Returns a dictionary mapping face index to eye state ('open', 'closed')
    """
    results = {}
    
    try:
        # Convert to RGB format for MediaPipe
        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        # Get dimensions
        height, width = image_np.shape[:2]
        
        # Process the image with MediaPipe
        mp_results = face_mesh.process(image_rgb)
        
        if not mp_results.multi_face_landmarks:
            logger.debug("No facial landmarks found by MediaPipe for eye state")
            return results
            
        # For each detected face
        for face_idx, face_landmarks in enumerate(mp_results.multi_face_landmarks):
            # Detect if eyes are closed
            eyes_closed = detect_eye_closure(face_landmarks, width, height)
            
            # Store result for this face
            results[face_idx] = "closed" if eyes_closed else "open"
            
    except Exception as e:
        logger.error(f"Error in eye detection: {e}")
        logger.error(traceback.format_exc())
        
    return results

def detect_head_position_with_mediapipe(image_np):
    """
    Simple and reliable head position detection
    Returns a dictionary mapping face index to head position.
    """
    results = {}
    try:
        # Convert to RGB format for MediaPipe
        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        # Process the image with MediaPipe
        mp_results = face_mesh.process(image_rgb)
        
        if not mp_results.multi_face_landmarks:
            logger.debug("No facial landmarks found by MediaPipe for head position")
            return results
            
        height, width = image_np.shape[:2]
        
        for face_idx, face_landmarks in enumerate(mp_results.multi_face_landmarks):
            try:
                # Default to "center"
                position = "center"
                
                # Use nose tip (landmark 1) and face center to determine head position
                nose_tip_x = face_landmarks.landmark[1].x * width
                
                # Find left and right bounds of face
                min_x = width
                max_x = 0
                for landmark in face_landmarks.landmark:
                    x = landmark.x * width
                    if x < min_x:
                        min_x = x
                    if x > max_x:
                        max_x = x
                
                # Calculate face center
                face_center_x = (min_x + max_x) / 2
                
                # Determine head position based on nose tip relative to face center
                offset = (nose_tip_x - face_center_x) / (max_x - min_x)  # Normalized offset
                
                # Use thresholds to determine position
                if offset < -0.05:
                    position = "left"
                elif offset > 0.05:
                    position = "right"
                
                logger.info(f"Head position detected: {position} (offset: {offset:.4f})")
                results[face_idx] = position
                
            except Exception as e:
                logger.warning(f"Error analyzing head position: {str(e)}")
                # Default to center if there's an error
                results[face_idx] = "center"
                
    except Exception as e:
        logger.error(f"Error in head position detection: {str(e)}")
        logger.error(traceback.format_exc())
            
    return results

def draw_faces_with_status(image_bytes, faces, recognized_students, student_status, phone_detections=None):
    """Draw faces with detailed status information on the image"""
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Draw phone bounding boxes if detected
        if phone_detections:
            for object_name, phone_info in phone_detections.items():
                rect = phone_info['position']
                confidence = phone_info['confidence']
                
                # Extract coordinates and dimensions of phone
                x = int(rect.get('x', 0))
                y = int(rect.get('y', 0))
                w = int(rect.get('w', 0))
                h = int(rect.get('h', 0))
                
                # Draw red rectangle around detected phone
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                
                # Add label with confidence score
                label = f"{object_name}: {confidence:.2f}"
                cv2.putText(img, label, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                logger.info(f"Drawing bounding box for {object_name} at position x={x}, y={y}, w={w}, h={h}")
        
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
                
            if student_head_turned_count[student_name] > 0:
                cv2.putText(img, f"Head turned: {student_head_turned_count[student_name]}/3", 
                          (bbox[0], bbox[3] + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
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
        
        # Get face app and detect faces - USING INSIGHTFACE FOR FACE DETECTION
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
        
        # Detect phone usage using Azure Vision API - but only trust explicit phone detections
        logger.info("Detecting phones in the image...")
        phone_detected, phone_detections = detect_phone_with_azure_vision(image_bytes)
        logger.info(f"Phone detection result: {phone_detected}")
        if phone_detected:
            logger.info(f"Phones detected: {list(phone_detections.keys())}")
        
        # Process MediaPipe detections for eye closure and head position
        logger.info("Detecting eye closure with MediaPipe...")
        eye_states = detect_eye_closure_with_mediapipe(image_np)
        logger.info("Detecting head position with MediaPipe...")
        head_positions = detect_head_position_with_mediapipe(image_np)
        
        # Process each detected face
        for idx, face in enumerate(faces):
            # USING INSIGHTFACE FOR RECOGNITION
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
            
            # Get MediaPipe detection results for this face
            eyes_closed = (eye_states.get(idx, "open") == "closed")
            head_position = head_positions.get(idx, "center")
            
            # Log current state before updating counters
            logger.info(f"Student {name} - Current state - Eyes closed: {eyes_closed}, Head position: {head_position}")
            logger.info(f"Student {name} - Current counters - Eyes closed: {student_eye_closed_count[name]}/3, Head turned: {student_head_turned_count[name]}/3")
            
            # Update consecutive frame counters for this student
            if eyes_closed:
                student_eye_closed_count[name] += 1
                logger.info(f"Student {name} has closed eyes: frame count now {student_eye_closed_count[name]}/3")
            else:
                student_eye_closed_count[name] = 0
                logger.info(f"Student {name} has open eyes: frame count reset to 0")
                
            # Only track left/right head positions for distraction detection
            if head_position in ['left', 'right']:
                student_head_turned_count[name] += 1
                logger.info(f"Student {name} head turned {head_position}: frame count now {student_head_turned_count[name]}/3")
            else:
                student_head_turned_count[name] = 0
                logger.info(f"Student {name} head position centered: head turned count reset to 0")
            
            # SLEEPING DETECTION
            # When eyes are closed in the current frame or for 3 consecutive frames
            is_sleeping = eyes_closed or (student_eye_closed_count[name] >= 3)
            
            # DISTRACTION DETECTION
            is_distracted = (student_head_turned_count[name] >= 3)
            
            # Associate phone detection with this student if applicable
            is_using_phone = False
            if phone_detected and phone_detections:
                for object_name, phone_info in phone_detections.items():
                    rect = phone_info['position']
                    phone_center_x = (rect['x'] + rect['w']/2)
                    phone_center_y = (rect['y'] + rect['h']/2)
                    
                    # Strict criteria for phone-face association
                    distance = math.sqrt((face_center_x - phone_center_x)**2 + 
                                         (face_center_y - phone_center_y)**2)
                    
                    if distance < 200:
                        is_using_phone = True
                        logger.info(f"Student {name} detected using phone - distance: {distance:.2f}px")
                        break
            
            # Update the student status
            student_status[name]['is_sleeping'] = is_sleeping
            student_status[name]['is_distracted'] = is_distracted
            student_status[name]['using_phone'] = is_using_phone
            student_status[name]['gaze'] = 'focused' if not is_distracted else 'distracted'
                
            # Determine the event type for database recording
            if is_sleeping:
                event_type = "sleeping"
            elif is_using_phone:
                event_type = "phone_usage"
            elif is_distracted:
                event_type = "distracted" 
            else:
                event_type = "focused"
            
            # Log detection results
            if is_sleeping:
                logger.info(f"Student {name} classified as SLEEPING")
            if is_distracted:
                logger.info(f"Student {name} classified as DISTRACTED")
                
            # Record engagement in the consolidated database table
            record_student_engagement(
                student_name=name,
                event_type=event_type,
                confidence=0.85,
                frame_data=image_data
            )
            
            # Log the final engagement classification
            logger.info(f"Final engagement status for {name}: " + 
                      f"sleeping={is_sleeping}, " + 
                      f"distracted={is_distracted}, " + 
                      f"using_phone={is_using_phone}")
        
        # Create annotated image with detection results
        annotated_image = draw_faces_with_status(
            image_bytes, 
            faces, 
            recognized_students, 
            student_status,
            phone_detections if phone_detected else None
        )
        
        # Use the returned base64 string directly
        img_b64 = annotated_image
        
        logger.info("Image analysis completed successfully.")
        return {
            "message": "Analysis complete",
            "face_detected": True,
            "recognized_students": [name for name in recognized_students.values() if name != "Unknown"],
            "student_status": student_status,
            "annotated_image": img_b64
        }
        
    except Exception as e:
        logger.error(f"Error in image analysis: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "error": str(e),
            "face_detected": False
        }