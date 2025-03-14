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
    """
    Creates or alters tables: Students, AttendanceRecords, EngagementRecords, SystemSettings.
    The Students table now stores face_embedding as JSON.
    """
    conn = get_db_connection()
    if not conn:
        return
    try:
        cursor = conn.cursor()

        # Students table (stores face_embedding as JSON)
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

        # AttendanceRecords table
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

        # EngagementRecords table
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

        # SystemSettings table
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

def add_student(name, face_id=None, face_embedding=None, image_url=None):
    """
    Inserts a new student record into the Students table.
    The face_embedding is stored as JSON (if available).
    """
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

def update_capture_status(active: bool):
    """
    Updates the SystemSettings table to indicate whether image capture is active.
    """
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("UPDATE SystemSettings SET capture_active=? WHERE id=1", (1 if active else 0,))
        conn.commit()
    except pyodbc.Error as e:
        logger.error(f"❌ Error updating capture status: {e}")
    finally:
        conn.close()

def get_engagement_records(start_date=None, end_date=None):
    """
    Retrieves engagement records (phone usage, gaze, and sleeping).
    """
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
    """
    Retrieves attendance records.
    """
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
