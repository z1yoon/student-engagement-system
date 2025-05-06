import json
import logging
import pyodbc
import numpy as np
import time
from contextlib import contextmanager
from config import DB_CONNECTION_STRING

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@contextmanager
def db_connection():
    """Context manager for database connections to ensure proper closing"""
    conn = None
    try:
        conn = pyodbc.connect(DB_CONNECTION_STRING)
        yield conn
    except pyodbc.Error as e:
        logger.error(f"❌ Database connection error: {e}")
        raise
    finally:
        if conn:
            conn.close()

def query_db(sql, params=None, fetchall=True):
    """Generic function to execute database queries"""
    with db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(sql, params or ())
        
        if fetchall:
            return cursor.fetchall()
        result = cursor.fetchone()
        return result

def create_tables():
    """Create required database tables if they don't exist"""
    logger.info("🔍 Attempting to create database tables...")
    
    # Check DB connection string
    if not DB_CONNECTION_STRING:
        logger.error("❌ Database connection string is empty! Check your environment variables.")
        return False
    
    try:
        with db_connection() as conn:
            cursor = conn.cursor()
            
            # Log database info
            try:
                cursor.execute("SELECT @@version")
                version = cursor.fetchone()[0]
                logger.info(f"✅ Connected to database: {version}")
            except Exception as e:
                logger.error(f"❌ Error getting database version: {e}")
                
            # Define tables to create
            tables = {
                "Students": """
                    IF OBJECT_ID('dbo.Students', 'U') IS NULL BEGIN 
                        CREATE TABLE Students (
                            id INT IDENTITY(1,1) PRIMARY KEY, 
                            student_name VARCHAR(100) NOT NULL UNIQUE, 
                            face_id VARCHAR(255) NULL, 
                            face_embedding TEXT NULL, 
                            image_url VARCHAR(255) NULL
                        ) 
                    END
                """,
                "AttendanceRecords": """
                    IF OBJECT_ID('dbo.AttendanceRecords', 'U') IS NULL BEGIN 
                        CREATE TABLE AttendanceRecords (
                            id INT IDENTITY(1,1) PRIMARY KEY, 
                            timestamp DATETIME DEFAULT GETUTCDATE(), 
                            student_id INT NULL, 
                            recognized_name VARCHAR(100) NOT NULL, 
                            image_url TEXT NULL, 
                            is_attended BIT NOT NULL
                        ) 
                    END
                """,
                "StudentEngagement": """
                    IF OBJECT_ID('dbo.StudentEngagement', 'U') IS NULL BEGIN 
                        CREATE TABLE StudentEngagement (
                            id INT IDENTITY(1,1) PRIMARY KEY,
                            timestamp DATETIME DEFAULT GETDATE(),
                            student_name VARCHAR(100) NOT NULL,
                            event_type VARCHAR(50) NOT NULL,
                            phone_detected BIT NOT NULL,
                            head_position VARCHAR(50) NOT NULL,
                            sleeping BIT NOT NULL,
                            confidence FLOAT NOT NULL,
                            frame_reference VARCHAR(255) NULL
                        )
                    END
                """,
                "SystemSettings": """
                    IF OBJECT_ID('dbo.SystemSettings', 'U') IS NULL BEGIN 
                        CREATE TABLE SystemSettings (
                            id INT PRIMARY KEY, 
                            capture_active BIT NOT NULL DEFAULT 0
                        ); 
                        INSERT INTO SystemSettings (id, capture_active) VALUES (1, 0) 
                    END
                """
            }
            
            # Create each table
            success = True
            for table_name, create_sql in tables.items():
                try:
                    cursor.execute(create_sql)
                    logger.info(f"✅ {table_name} table checked/created")
                except pyodbc.Error as e:
                    logger.error(f"❌ Error creating {table_name} table: {e}")
                    success = False
            
            # Verify tables were created
            try:
                cursor.execute("SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = 'BASE TABLE'")
                created_tables = [row[0] for row in cursor.fetchall()]
                logger.info(f"📊 Available tables: {created_tables}")
                
                for table in tables.keys():
                    if table not in created_tables:
                        logger.error(f"❌ Table {table} was not created successfully!")
                        success = False
            except pyodbc.Error as e:
                logger.error(f"❌ Error verifying tables: {e}")
                success = False

            # Migrate data from old tables if they exist
            try:
                # Check if old tables exist and the new one is empty
                cursor.execute("SELECT COUNT(*) FROM StudentEngagement")
                student_engagement_count = cursor.fetchone()[0]
                
                if student_engagement_count == 0:
                    # Try to migrate data from EngagementRecords if it exists
                    if 'EngagementRecords' in created_tables:
                        cursor.execute("""
                            INSERT INTO StudentEngagement (
                                timestamp, student_name, event_type, 
                                phone_detected, head_position, sleeping, confidence
                            )
                            SELECT 
                                timestamp, student_name, 
                                CASE 
                                    WHEN sleeping = 1 THEN 'sleeping'
                                    WHEN phone_detected = 1 THEN 'phone_usage'
                                    WHEN gaze <> 'focused' THEN 'distracted'
                                    ELSE 'focused'
                                END as event_type,
                                phone_detected, gaze, sleeping, 0.85
                            FROM EngagementRecords
                        """)
                        
                        # Get count of migrated records
                        cursor.execute("SELECT @@ROWCOUNT")
                        migrated_records = cursor.fetchone()[0]
                        logger.info(f"✅ Migrated {migrated_records} records from EngagementRecords")
                    
                    # Try to migrate data from EngagementEvents if it exists
                    if 'EngagementEvents' in created_tables:
                        cursor.execute("""
                            INSERT INTO StudentEngagement (
                                timestamp, student_name, event_type, 
                                phone_detected, head_position, sleeping, confidence, frame_reference
                            )
                            SELECT 
                                timestamp, student_name, event_type,
                                CASE WHEN event_type = 'phone_usage' THEN 1 ELSE 0 END,
                                CASE WHEN event_type = 'focused' THEN 'focused' ELSE 'distracted' END,
                                CASE WHEN event_type = 'sleeping' THEN 1 ELSE 0 END,
                                confidence, frame_reference
                            FROM EngagementEvents
                            WHERE NOT EXISTS (
                                SELECT 1 FROM StudentEngagement 
                                WHERE StudentEngagement.timestamp = EngagementEvents.timestamp
                                AND StudentEngagement.student_name = EngagementEvents.student_name
                            )
                        """)
                        
                        # Get count of migrated records
                        cursor.execute("SELECT @@ROWCOUNT")
                        migrated_records = cursor.fetchone()[0]
                        logger.info(f"✅ Migrated {migrated_records} unique records from EngagementEvents")
                
            except Exception as e:
                logger.warning(f"⚠️ Data migration note: {e}")
                # Continue even if migration fails - this is not critical
                pass

            conn.commit()
            if success:
                logger.info("✅ All tables created or already exist.")
            return success
    except Exception as e:
        logger.error(f"❌ Unexpected error in create_tables: {e}")
        return False

def get_enrolled_students():
    """Get all students with face embeddings for recognition"""
    try:
        with db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT student_name, face_id, face_embedding, image_url FROM Students WHERE face_embedding IS NOT NULL")
            rows = cursor.fetchall()
            result = []
            for row in rows:
                name = row[0]
                face_id = row[1]
                face_embedding_json = row[2]
                image_url = row[3]
                face_embedding = json.loads(face_embedding_json) if face_embedding_json else None
                result.append({
                    "name": name,
                    "face_id": face_id,
                    "face_embedding": face_embedding,
                    "image_url": image_url
                })
            return result
    except pyodbc.Error as e:
        logger.error(f"❌ Error fetching enrolled students: {e}")
        return []

def student_exists(name):
    """Check if a student with the given name already exists"""
    try:
        with db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM Students WHERE student_name = ?", (name,))
            count = cursor.fetchone()[0]
            return count > 0
    except pyodbc.Error as e:
        logger.error(f"❌ Error checking if student exists: {e}")
        return False

def add_student(name, face_id=None, face_embedding=None, image_url=None):
    """Add a new student to the database"""
    try:
        with db_connection() as conn:
            cursor = conn.cursor()
            face_emb_json = json.dumps(face_embedding) if face_embedding else None
            cursor.execute(
                "INSERT INTO Students (student_name, face_id, face_embedding, image_url) VALUES (?, ?, ?, ?)",
                (name, face_id, face_emb_json, image_url)
            )
            conn.commit()
            logger.info(f"✅ Student {name} added.")
    except pyodbc.Error as e:
        logger.error(f"❌ Error adding student {name}: {e}")

def mark_attendance(student_name, is_attended, image_data):
    """Record student attendance with image data"""
    try:
        with db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM Students WHERE student_name = ?", (student_name,))
            row = cursor.fetchone()
            student_id = row[0] if row else None

            cursor.execute("""
                INSERT INTO AttendanceRecords (student_id, recognized_name, image_url, is_attended)
                VALUES (?, ?, ?, ?)
            """, (student_id, student_name, image_data, 1 if is_attended else 0))
            conn.commit()
            logger.info(f"✅ Attendance recorded for {student_name}.")
    except pyodbc.Error as e:
        logger.error(f"❌ Error inserting attendance record: {e}")

def record_student_engagement(student_name, event_type, confidence=0.85, frame_data=None):
    """
    Record student engagement in the consolidated StudentEngagement table
    
    Parameters:
    - student_name: Name of the student
    - event_type: Type of engagement event (sleeping, distracted, phone_usage, focused)
    - confidence: Detection confidence score
    - frame_data: Optional image frame data
    """
    try:
        with db_connection() as conn:
            cursor = conn.cursor()
            
            # Create frame reference if needed
            frame_reference = None
            if frame_data:
                frame_reference = f"frame_{student_name}_{event_type}_{int(time.time())}"
            
            # Determine values for each field based on event type
            phone_detected = (event_type == "phone_usage")
            sleeping = (event_type == "sleeping")
            head_position = "focused" if event_type == "focused" else "distracted"
            
            # Insert into consolidated table
            query = """
            INSERT INTO StudentEngagement (
                student_name, event_type, phone_detected, 
                head_position, sleeping, confidence, frame_reference
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """
            
            cursor.execute(query, (
                student_name, 
                event_type, 
                1 if phone_detected else 0,
                head_position,
                1 if sleeping else 0,
                confidence,
                frame_reference
            ))
            
            conn.commit()
            logger.info(f"✅ Student engagement recorded: {student_name} - {event_type}")
            
    except Exception as e:
        logger.error(f"❌ Error recording student engagement: {str(e)}")
        # Continue execution even if recording fails
        pass

# Legacy functions that now use the consolidated table
def add_engagement_record(student_name, phone_detected, gaze, sleeping):
    """Legacy function - Add an engagement record for a student"""
    try:
        # Determine event type based on old parameters
        if sleeping:
            event_type = "sleeping"
        elif phone_detected:
            event_type = "phone_usage"
        elif gaze != "focused":
            event_type = "distracted"
        else:
            event_type = "focused"
        
        # Use new consolidated function
        record_student_engagement(
            student_name=student_name,
            event_type=event_type,
            confidence=0.85,
            frame_data=None
        )
        
    except Exception as e:
        logger.error(f"❌ Error in legacy add_engagement_record: {e}")

def record_engagement_event(student_name, event_type, confidence=0.75, frame_data=None):
    """Legacy function - Record a student engagement event"""
    # Simply call the new consolidated function
    record_student_engagement(student_name, event_type, confidence, frame_data)

def get_engagement_records(start_date=None, end_date=None):
    """Get engagement records with optional date filtering"""
    try:
        with db_connection() as conn:
            cursor = conn.cursor()
            query = """
                SELECT timestamp, student_name, phone_detected, 
                       head_position, sleeping
                FROM StudentEngagement
            """
            conditions = []
            params = []
            if start_date:
                conditions.append("timestamp >= ?")
                params.append(start_date)
            if end_date:
                conditions.append("timestamp <= ?")
                params.append(end_date)
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            query += " ORDER BY timestamp DESC"
            cursor.execute(query, tuple(params))
            rows = cursor.fetchall()
            return [
                {
                    "timestamp": str(r[0]),
                    "student_name": r[1],
                    "phone_detected": bool(r[2]),
                    "head_position": r[3],
                    "sleeping": bool(r[4])
                } for r in rows
            ]
    except pyodbc.Error as e:
        logger.error(f"❌ Error fetching engagement records: {e}")
        return []

def get_attendance_records(start_date=None, end_date=None):
    """Get attendance records with optional date filtering"""
    try:
        with db_connection() as conn:
            cursor = conn.cursor()
            query = "SELECT timestamp, recognized_name, image_url, is_attended FROM AttendanceRecords"
            conditions = []
            params = []
            if start_date:
                conditions.append("timestamp >= ?")
                params.append(start_date)
            if end_date:
                conditions.append("timestamp <= ?")
                params.append(end_date)
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            query += " ORDER BY timestamp DESC"
            cursor.execute(query, tuple(params))
            rows = cursor.fetchall()
            return [
                {
                    "timestamp": str(r[0]),
                    "recognized_name": r[1],
                    "image_url": r[2],
                    "is_attended": bool(r[3])
                } for r in rows
            ]
    except pyodbc.Error as e:
        logger.error(f"❌ Error fetching attendance records: {e}")
        return []

def update_capture_status(active: bool):
    """Update system capture status"""
    try:
        with db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE SystemSettings SET capture_active=? WHERE id=1", (1 if active else 0,))
            conn.commit()
    except pyodbc.Error as e:
        logger.error(f"❌ Error updating capture status: {e}")

def get_enrolled_student_details():
    """Get basic details about enrolled students without images"""
    try:
        with db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT student_name FROM Students")
            rows = cursor.fetchall()
            result = {}
            for row in rows:
                name = row[0]
                result[name] = {
                    "focused": 0,
                    "distracted": 0, 
                    "phone_usage": 0,
                    "sleeping": 0,
                    "attended": False
                }
            return result
    except pyodbc.Error as e:
        logger.error(f"❌ Error fetching enrolled student details: {e}")
        return {}

def get_capture_status():
    """Get the current capture active status"""
    try:
        with db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT capture_active FROM SystemSettings WHERE id = 1")
            row = cursor.fetchone()
            if row:
                return {"active": bool(row[0])}
            else:
                return {"active": False}
    except pyodbc.Error as e:
        logger.error(f"❌ Error fetching capture status: {e}")
        return {"active": False}

def get_analyze_results(start_date=None, end_date=None):
    """Aggregate engagement and attendance data for all students"""
    # Get raw records
    engagement_records = get_engagement_records(start_date, end_date)
    attendance_records = get_attendance_records(start_date, end_date)

    if not engagement_records:
        logger.warning("⚠️ No engagement records found!")
    if not attendance_records:
        logger.warning("⚠️ No attendance records found!")

    # Start with base student data
    summary = get_enrolled_student_details()
    
    # Process engagement records
    for record in engagement_records:
        name = record["student_name"]
        if name in summary:
            # Prioritize phone usage detection
            if record["phone_detected"]:
                summary[name]["phone_usage"] += 1
            # If no phone detected, then check for sleeping
            elif record["sleeping"]:
                summary[name]["sleeping"] += 1
            # If neither phone nor sleeping, then check head position for distracted/focused
            elif record["head_position"].lower() == "focused":
                summary[name]["focused"] += 1
            else:
                summary[name]["distracted"] += 1
    
    # Process attendance records
    for record in attendance_records:
        name = record["recognized_name"]
        if name in summary and record["is_attended"]:
            summary[name]["attended"] = True
    
    # Calculate percentages
    for name, stats in summary.items():
        total_records = stats["focused"] + stats["distracted"]
        
        if total_records > 0:
            stats["focused_percentage"] = round((stats["focused"] / total_records) * 100, 1)
            stats["distracted_percentage"] = round((stats["distracted"] / total_records) * 100, 1)
            stats["phone_usage_percentage"] = round((stats["phone_usage"] / total_records) * 100, 1) 
            stats["sleeping_percentage"] = round((stats["sleeping"] / total_records) * 100, 1)
        else:
            stats["focused_percentage"] = 0
            stats["distracted_percentage"] = 0
            stats["phone_usage_percentage"] = 0
            stats["sleeping_percentage"] = 0
    
    return summary

def check_similar_face_exists(face_embedding, similarity_threshold=0.80):
    """
    Check if a similar face already exists in the database
    Returns (exists, similar_students) tuple where:
    - exists: Boolean indicating if a face exists with similarity above threshold
    - similar_students: List of (name, similarity) tuples for similar faces
    """
    if not face_embedding:
        return False, []
    
    try:
        # Convert the new embedding to numpy array for calculations
        new_embedding = np.array(face_embedding)
        
        # Get all enrolled students with embeddings
        enrolled_students = get_enrolled_students()
        similar_students = []
        
        # Check similarity against each enrolled student
        for student in enrolled_students:
            if student["face_embedding"]:
                existing_embedding = np.array(student["face_embedding"])
                
                # Calculate cosine similarity
                dot_product = np.dot(new_embedding, existing_embedding)
                norm_new = np.linalg.norm(new_embedding)
                norm_existing = np.linalg.norm(existing_embedding)
                similarity = dot_product / (norm_new * norm_existing)
                
                # If similarity is above threshold, consider it a match
                if similarity >= similarity_threshold:
                    similar_students.append((student["name"], float(similarity)))
        
        # Return whether any face is similar enough to be considered a duplicate
        return len(similar_students) > 0, similar_students
    except Exception as e:
        logger.error(f"❌ Error checking for similar faces: {e}")
        return False, []