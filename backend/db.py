import os
import pyodbc
import json
import numpy as np

# Get DB connection string from environment variables
DB_CONNECTION_STRING = os.getenv("DB_CONNECTION_STRING", "")

def get_db_connection():
    """Establish a connection to the database."""
    try:
        conn = pyodbc.connect(DB_CONNECTION_STRING)
        return conn
    except pyodbc.Error as e:
        print(f"Database connection error: {e}")
        return None


def create_tables():
    """Creates necessary tables if they do not exist."""
    conn = get_db_connection()
    if not conn:
        return

    try:
        cursor = conn.cursor()

        # Students Table
        cursor.execute("""
        IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'Students')
        CREATE TABLE Students (
            id INT IDENTITY(1,1) PRIMARY KEY,
            student_name VARCHAR(100) NOT NULL,
            face_encoding TEXT NULL
        )
        """)

        # Attendance Table
        cursor.execute("""
        IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'AttendanceRecords')
        CREATE TABLE AttendanceRecords (
            id INT IDENTITY(1,1) PRIMARY KEY,
            timestamp DATETIME DEFAULT GETUTCDATE(),
            student_id INT NULL,
            recognized_name VARCHAR(100) NOT NULL,
            is_attended BIT NOT NULL
        )
        """)

        # Engagement Table
        cursor.execute("""
        IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'EngagementRecords')
        CREATE TABLE EngagementRecords (
            id INT IDENTITY(1,1) PRIMARY KEY,
            timestamp DATETIME DEFAULT GETUTCDATE(),
            phone_detected BIT NOT NULL,
            gaze VARCHAR(50) NOT NULL,
            sleeping BIT NOT NULL
        )
        """)

        # System Settings Table
        cursor.execute("""
        IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'SystemSettings')
        CREATE TABLE SystemSettings (
            id INT PRIMARY KEY CHECK (id = 1),
            capture_active BIT NOT NULL DEFAULT 0
        )
        """)

        # Ensure system settings exist
        cursor.execute("""
        IF NOT EXISTS (SELECT * FROM SystemSettings WHERE id=1)
        INSERT INTO SystemSettings (id, capture_active) VALUES (1, 0)
        """)

        conn.commit()
    except pyodbc.Error as e:
        print(f"Error creating tables: {e}")
    finally:
        conn.close()


def update_capture_status(active: bool):
    """Updates system capture status (1 for active, 0 for inactive)."""
    conn = get_db_connection()
    if not conn:
        return

    try:
        cursor = conn.cursor()
        cursor.execute("UPDATE SystemSettings SET capture_active=? WHERE id=1", (1 if active else 0,))
        conn.commit()
    except pyodbc.Error as e:
        print(f"Error updating capture status: {e}")
    finally:
        conn.close()


def get_capture_status():
    """Retrieves the current capture status (active or not)."""
    conn = get_db_connection()
    if not conn:
        return {"capture_active": False}

    try:
        cursor = conn.cursor()
        cursor.execute("SELECT capture_active FROM SystemSettings WHERE id=1")
        row = cursor.fetchone()
        return {"capture_active": bool(row[0])} if row else {"capture_active": False}
    except pyodbc.Error as e:
        print(f"Error fetching capture status: {e}")
        return {"capture_active": False}
    finally:
        conn.close()


def add_attendance_record(student_id, recognized_name, is_attended):
    """Records attendance with student ID and name."""
    conn = get_db_connection()
    if not conn:
        return

    try:
        cursor = conn.cursor()
        cursor.execute("""
        INSERT INTO AttendanceRecords (student_id, recognized_name, is_attended)
        VALUES (?, ?, ?)
        """, (student_id, recognized_name, 1 if is_attended else 0))
        conn.commit()
    except pyodbc.Error as e:
        print(f"Error inserting attendance record: {e}")
    finally:
        conn.close()


def get_attendance_records():
    """Fetches all attendance records."""
    conn = get_db_connection()
    if not conn:
        return []

    try:
        cursor = conn.cursor()
        cursor.execute("""
        SELECT id, timestamp, recognized_name, is_attended
        FROM AttendanceRecords
        ORDER BY timestamp DESC
        """)
        rows = cursor.fetchall()
        return [{"id": r[0], "timestamp": str(r[1]), "recognized_name": r[2], "is_attended": bool(r[3])} for r in rows]
    except pyodbc.Error as e:
        print(f"Error fetching attendance records: {e}")
        return []
    finally:
        conn.close()


def add_student(name, face_encoding):
    """Registers a new student with their face encoding."""
    conn = get_db_connection()
    if not conn:
        return

    try:
        cursor = conn.cursor()
        face_enc_json = json.dumps(face_encoding) if face_encoding else None
        cursor.execute("INSERT INTO Students (student_name, face_encoding) VALUES (?, ?)", (name, face_enc_json))
        conn.commit()
    except pyodbc.Error as e:
        print(f"Error adding student: {e}")
    finally:
        conn.close()


def get_all_students():
    """Fetches all students and their face encodings."""
    conn = get_db_connection()
    if not conn:
        return []

    try:
        cursor = conn.cursor()
        cursor.execute("SELECT id, student_name, face_encoding FROM Students")
        rows = cursor.fetchall()
        return [{"id": r[0], "student_name": r[1], "face_encoding": json.loads(r[2]) if r[2] else None} for r in rows]
    except pyodbc.Error as e:
        print(f"Error fetching students: {e}")
        return []
    finally:
        conn.close()


def match_student_by_enc(face_enc):
    """Matches a detected face with stored student data."""
    students = get_all_students()
    best_match = {"id": None, "name": None, "distance": 0.6}

    for s in students:
        if not s["face_encoding"]:
            continue
        known_enc = np.array(s["face_encoding"], dtype=np.float32)
        dist = np.linalg.norm(face_enc - known_enc)
        if dist < best_match["distance"]:
            best_match.update({"id": s["id"], "name": s["student_name"], "distance": dist})

    return best_match["id"], best_match["name"]


def add_engagement_record(phone_detected, gaze, sleeping):
    """Logs engagement details for each student."""
    conn = get_db_connection()
    if not conn:
        return

    try:
        cursor = conn.cursor()
        cursor.execute("""
        INSERT INTO EngagementRecords (phone_detected, gaze, sleeping)
        VALUES (?, ?, ?)
        """, (1 if phone_detected else 0, gaze, 1 if sleeping else 0))
        conn.commit()
    except pyodbc.Error as e:
        print(f"Error inserting engagement record: {e}")
    finally:
        conn.close()


def get_engagement_records():
    """Retrieves all engagement records."""
    conn = get_db_connection()
    if not conn:
        return []

    try:
        cursor = conn.cursor()
        cursor.execute("""
        SELECT timestamp, phone_detected, gaze, sleeping
        FROM EngagementRecords
        ORDER BY timestamp DESC
        """)
        rows = cursor.fetchall()
        return [
            {
                "timestamp": str(r[0]),
                "phone_detected": bool(r[1]),
                "gaze": r[2],
                "sleeping": bool(r[3])
            } for r in rows
        ]
    except pyodbc.Error as e:
        print(f"Error fetching engagement records: {e}")
        return []
    finally:
        conn.close()
