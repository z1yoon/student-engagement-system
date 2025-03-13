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

def get_enrolled_students():
    """
    Returns a list of { "name": student_name, "face_id": face_id, "image_url": ... }
    for all students who have a face_id stored.
    """
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT student_name, face_id, image_url FROM Students WHERE face_id IS NOT NULL")
        rows = cursor.fetchall()
        return [
            {"name": row[0], "face_id": row[1], "image_url": row[2]}
            for row in rows
        ]
    except pyodbc.Error as e:
        logger.error(f"❌ Error fetching enrolled students: {e}")
        return []
    finally:
        conn.close()

def mark_attendance(student_name, is_attended, image_data):
    """
    Inserts an attendance record for recognized student.
    """
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
    """
    Inserts an engagement record into DB.
    """
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

