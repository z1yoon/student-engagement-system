import os
import json
import logging
import pyodbc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_CONNECTION_STRING = os.getenv("DB_CONNECTION_STRING", "")

def get_db_connection():
    try:
        conn = pyodbc.connect(DB_CONNECTION_STRING)
        return conn
    except pyodbc.Error as e:
        logger.error(f"❌ Database connection error: {e}")
        raise

def create_tables():
    conn = get_db_connection()
    if not conn:
        return
    try:
        cursor = conn.cursor()

        cursor.execute(
            "IF OBJECT_ID('Students', 'U') IS NULL BEGIN "
            "CREATE TABLE Students ("
            "id INT IDENTITY(1,1) PRIMARY KEY, "
            "student_name VARCHAR(100) NOT NULL UNIQUE, "
            "face_id VARCHAR(255) NULL, "
            "face_embedding TEXT NULL, "
            "image_url VARCHAR(255) NULL) "
            "END"
        )

        cursor.execute(
            "IF OBJECT_ID('AttendanceRecords', 'U') IS NULL BEGIN "
            "CREATE TABLE AttendanceRecords ("
            "id INT IDENTITY(1,1) PRIMARY KEY, "
            "timestamp DATETIME DEFAULT GETUTCDATE(), "
            "student_id INT NULL, "
            "recognized_name VARCHAR(100) NOT NULL, "
            "image_url TEXT NULL, "
            "is_attended BIT NOT NULL) "
            "END"
        )

        cursor.execute(
            "IF OBJECT_ID('EngagementRecords', 'U') IS NULL BEGIN "
            "CREATE TABLE EngagementRecords ("
            "id INT IDENTITY(1,1) PRIMARY KEY, "
            "timestamp DATETIME DEFAULT GETUTCDATE(), "
            "student_name VARCHAR(100) NOT NULL, "
            "phone_detected BIT NOT NULL, "
            "gaze VARCHAR(50) NOT NULL, "
            "sleeping BIT NOT NULL) "
            "END"
        )

        cursor.execute(
            "IF OBJECT_ID('SystemSettings', 'U') IS NULL BEGIN "
            "CREATE TABLE SystemSettings ("
            "id INT PRIMARY KEY, "
            "capture_active BIT NOT NULL DEFAULT 0) "
            "INSERT INTO SystemSettings (id, capture_active) VALUES (1, 0) "
            "END"
        )

        conn.commit()
        logger.info("✅ Tables created or already exist.")
    except pyodbc.Error as e:
        logger.error(f"❌ Error creating/updating tables: {e}")
    finally:
        conn.close()

def get_enrolled_students():
    conn = get_db_connection()
    try:
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
    finally:
        conn.close()

def student_exists(name):
    """
    Check if a student with the given name already exists in the database.
    Returns True if student exists, False otherwise.
    """
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM Students WHERE student_name = ?", (name,))
        count = cursor.fetchone()[0]
        return count > 0
    except pyodbc.Error as e:
        logger.error(f"❌ Error checking if student exists: {e}")
        return False
    finally:
        conn.close()

def add_student(name, face_id=None, face_embedding=None, image_url=None):
    conn = get_db_connection()
    try:
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
    finally:
        conn.close()

def mark_attendance(student_name, is_attended, image_data):
    conn = get_db_connection()
    try:
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
    finally:
        conn.close()

def add_engagement_record(student_name, phone_detected, gaze, sleeping):
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO EngagementRecords (student_name, phone_detected, gaze, sleeping)
            VALUES (?, ?, ?, ?)
        """, (student_name, 1 if phone_detected else 0, gaze, 1 if sleeping else 0))
        conn.commit()
        logger.info(f"✅ Engagement record added for {student_name}.")
    except pyodbc.Error as e:
        logger.error(f"❌ Error adding engagement record: {e}")
    finally:
        conn.close()

def get_engagement_records(start_date=None, end_date=None):
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        query = "SELECT timestamp, student_name, phone_detected, gaze, sleeping FROM EngagementRecords"
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
                "gaze": r[3],
                "sleeping": bool(r[4])
            } for r in rows
        ]
    except pyodbc.Error as e:
        logger.error(f"❌ Error fetching engagement records: {e}")
        return []
    finally:
        conn.close()

def get_attendance_records(start_date=None, end_date=None):
    conn = get_db_connection()
    try:
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
    finally:
        conn.close()

def update_capture_status(active: bool):
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("UPDATE SystemSettings SET capture_active=? WHERE id=1", (1 if active else 0,))
        conn.commit()
    except pyodbc.Error as e:
        logger.error(f"❌ Error updating capture status: {e}")
    finally:
        conn.close()

def get_enrolled_student_details():
    """
    Get basic details about all enrolled students regardless of engagement or attendance.
    """
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT student_name, image_url FROM Students")
        rows = cursor.fetchall()
        result = {}
        for row in rows:
            name = row[0]
            image_url = row[1]
            result[name] = {
                "focused": 0,
                "distracted": 0, 
                "phone_usage": 0,
                "sleeping": 0,
                "attended": False,
                "image_url": image_url
            }
        return result
    except pyodbc.Error as e:
        logger.error(f"❌ Error fetching enrolled student details: {e}")
        return {}
    finally:
        conn.close()

def get_analyze_results(start_date=None, end_date=None):
    engagement_records = get_engagement_records(start_date, end_date)
    attendance_records = get_attendance_records(start_date, end_date)

    if not engagement_records:
        logger.warning("⚠️ No engagement records found!")
    if not attendance_records:
        logger.warning("⚠️ No attendance records found!")

    # Get all enrolled students to use as base data
    summary = get_enrolled_student_details()

    for record in engagement_records:
        name = record["student_name"]
        if name not in summary:
            summary[name] = {
                "focused": 0,
                "distracted": 0,
                "phone_usage": 0,
                "sleeping": 0,
                "attended": False,
                "image_url": None
            }
        if record["gaze"] == "focused":
            summary[name]["focused"] += 1
        else:
            summary[name]["distracted"] += 1
        if record["phone_detected"]:
            summary[name]["phone_usage"] += 1
        if record["sleeping"]:
            summary[name]["sleeping"] += 1

    for record in attendance_records:
        name = record["recognized_name"]
        if name not in summary:
            summary[name] = {
                "focused": 0,
                "distracted": 0,
                "phone_usage": 0,
                "sleeping": 0,
                "attended": False,
                "image_url": None
            }
        summary[name]["attended"] = record["is_attended"]
        # Only update image_url if it's not set yet or the current one is None
        if not summary[name]["image_url"] and record["image_url"]:
            summary[name]["image_url"] = record["image_url"]

    if not summary:
        return {"message": "No students enrolled or engagement data found."}

    return summary

def get_capture_status():
    conn = get_db_connection()
    try:
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
    finally:
        conn.close()
