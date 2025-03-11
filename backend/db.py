import os
import pyodbc
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get DB connection string from environment variables
DB_CONNECTION_STRING = os.getenv("DB_CONNECTION_STRING", "")

def get_db_connection():
    """Establish a connection to the database."""
    try:
        conn = pyodbc.connect(DB_CONNECTION_STRING)
        return conn
    except pyodbc.Error as e:
        logger.error(f"❌ Database connection error: {e}")
        raise Exception("Database connection failed.")

def create_tables():
    """
    Creates necessary tables if they do not exist or alters existing tables to match the expected schema.
    Required tables and columns:
      - Students: id, student_name, face_encoding, image_url
      - AttendanceRecords: id, timestamp, student_id, recognized_name, image_url, is_attended
      - EngagementRecords: id, timestamp, student_name, phone_detected, gaze, sleeping
      - SystemSettings: id, capture_active
    """
    conn = get_db_connection()
    if not conn:
        return

    try:
        cursor = conn.cursor()

        # ----- Students Table -----
        cursor.execute("SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'Students'")
        if not cursor.fetchone():
            cursor.execute("""
            CREATE TABLE Students (
                id INT IDENTITY(1,1) PRIMARY KEY,
                student_name VARCHAR(100) NOT NULL UNIQUE,
                face_encoding TEXT NULL,
                image_url VARCHAR(255) NULL
            )
            """)
            logger.info("✅ Students table created.")
        else:
            # Check for 'image_url' column
            cursor.execute("""
                SELECT * FROM INFORMATION_SCHEMA.COLUMNS 
                WHERE TABLE_NAME = 'Students' AND COLUMN_NAME = 'image_url'
            """)
            if not cursor.fetchone():
                cursor.execute("ALTER TABLE Students ADD image_url VARCHAR(255) NULL")
                logger.info("✅ Students table altered: 'image_url' column added.")

        # ----- AttendanceRecords Table -----
        cursor.execute("SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'AttendanceRecords'")
        if not cursor.fetchone():
            cursor.execute("""
            CREATE TABLE AttendanceRecords (
                id INT IDENTITY(1,1) PRIMARY KEY,
                timestamp DATETIME DEFAULT GETUTCDATE(),
                student_id INT NULL,
                recognized_name VARCHAR(100) NOT NULL,
                image_url VARCHAR(255) NULL,
                is_attended BIT NOT NULL
            )
            """)
            logger.info("✅ AttendanceRecords table created.")
        else:
            # Check for 'image_url' column in AttendanceRecords
            cursor.execute("""
                SELECT * FROM INFORMATION_SCHEMA.COLUMNS 
                WHERE TABLE_NAME = 'AttendanceRecords' AND COLUMN_NAME = 'image_url'
            """)
            if not cursor.fetchone():
                cursor.execute("ALTER TABLE AttendanceRecords ADD image_url VARCHAR(255) NULL")
                logger.info("✅ AttendanceRecords table altered: 'image_url' column added.")

        # ----- EngagementRecords Table -----
        cursor.execute("SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'EngagementRecords'")
        if not cursor.fetchone():
            cursor.execute("""
            CREATE TABLE EngagementRecords (
                id INT IDENTITY(1,1) PRIMARY KEY,
                timestamp DATETIME DEFAULT GETUTCDATE(),
                student_name VARCHAR(100) NOT NULL,
                phone_detected BIT NOT NULL,
                gaze VARCHAR(50) NOT NULL,
                sleeping BIT NOT NULL
            )
            """)
            logger.info("✅ EngagementRecords table created.")
        else:
            # Check for 'student_name' column in EngagementRecords
            cursor.execute("""
                SELECT * FROM INFORMATION_SCHEMA.COLUMNS 
                WHERE TABLE_NAME = 'EngagementRecords' AND COLUMN_NAME = 'student_name'
            """)
            if not cursor.fetchone():
                # Adding a NOT NULL column requires a default value. Here we add it with default 'Unknown'.
                cursor.execute("ALTER TABLE EngagementRecords ADD student_name VARCHAR(100) NOT NULL DEFAULT 'Unknown'")
                logger.info("✅ EngagementRecords table altered: 'student_name' column added.")

        # ----- SystemSettings Table -----
        cursor.execute("SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'SystemSettings'")
        if not cursor.fetchone():
            cursor.execute("""
            CREATE TABLE SystemSettings (
                id INT PRIMARY KEY,
                capture_active BIT NOT NULL DEFAULT 0
            )
            """)
            logger.info("✅ SystemSettings table created.")
            cursor.execute("INSERT INTO SystemSettings (id, capture_active) VALUES (1, 0)")
            logger.info("✅ SystemSettings default row inserted.")
        else:
            cursor.execute("SELECT * FROM SystemSettings WHERE id = 1")
            if not cursor.fetchone():
                cursor.execute("INSERT INTO SystemSettings (id, capture_active) VALUES (1, 0)")
                logger.info("✅ SystemSettings default row inserted.")

        conn.commit()
    except pyodbc.Error as e:
        logger.error(f"❌ Error creating or updating tables: {e}")
    finally:
        conn.close()

def add_student(name, face_encoding=None, image_url=None):
    """Registers a new student with their face encoding and image URL."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        face_enc_json = json.dumps(face_encoding) if face_encoding else None
        cursor.execute("INSERT INTO Students (student_name, face_encoding, image_url) VALUES (?, ?, ?)",
                       (name, face_enc_json, image_url))
        conn.commit()
        logger.info(f"✅ Student {name} added successfully.")
    except pyodbc.Error as e:
        logger.error(f"❌ Error adding student {name}: {e}")
    finally:
        conn.close()

def get_student_by_name(name):
    """Fetches a student by their name."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT id, student_name, face_encoding, image_url FROM Students WHERE student_name = ?", (name,))
        row = cursor.fetchone()
        if not row:
            return None
        return {
            "id": row[0],
            "student_name": row[1],
            "face_encoding": json.loads(row[2]) if row[2] else None,
            "image_url": row[3]
        }
    except pyodbc.Error as e:
        logger.error(f"❌ Error fetching student {name}: {e}")
        return None
    finally:
        conn.close()

def add_attendance_record(student_id, recognized_name, image_url, is_attended):
    """Records attendance with student ID, name, and image URL."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
        INSERT INTO AttendanceRecords (student_id, recognized_name, image_url, is_attended)
        VALUES (?, ?, ?, ?)
        """, (student_id, recognized_name, image_url, 1 if is_attended else 0))
        conn.commit()
        logger.info(f"✅ Attendance recorded for {recognized_name}.")
    except pyodbc.Error as e:
        logger.error(f"❌ Error inserting attendance record: {e}")
    finally:
        conn.close()

def get_attendance_records(start_date=None, end_date=None):
    """Fetches all attendance records with optional date filtering."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        query = """
        SELECT timestamp, recognized_name, image_url, is_attended
        FROM AttendanceRecords
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
                "timestamp": str(row[0]),
                "recognized_name": row[1],
                "image_url": row[2],
                "is_attended": bool(row[3])
            } for row in rows
        ]
    except pyodbc.Error as e:
        logger.error(f"❌ Error fetching attendance records: {e}")
        return []
    finally:
        conn.close()

def add_engagement_record(student_name, phone_detected, gaze, sleeping):
    """Adds an engagement record to the database."""
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
        logger.error(f"❌ Error adding engagement record for {student_name}: {e}")
    finally:
        conn.close()

def get_engagement_records(start_date=None, end_date=None):
    """Fetches all engagement records with optional date filtering."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        query = """
        SELECT timestamp, student_name, phone_detected, gaze, sleeping
        FROM EngagementRecords
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
                "timestamp": str(row[0]),
                "student_name": row[1],
                "phone_detected": bool(row[2]),
                "gaze": row[3],
                "sleeping": bool(row[4])
            } for row in rows
        ]
    except pyodbc.Error as e:
        logger.error(f"❌ Error fetching engagement records: {e}")
        return []
    finally:
        conn.close()

def update_capture_status(active: bool):
    """Updates system capture status (1 for active, 0 for inactive)."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("UPDATE SystemSettings SET capture_active=? WHERE id=1", (1 if active else 0,))
        conn.commit()
    except pyodbc.Error as e:
        logger.error(f"❌ Error updating capture status: {e}")
    finally:
        conn.close()

def get_capture_status():
    """Retrieves the current capture status (active or not)."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT capture_active FROM SystemSettings WHERE id=1")
        row = cursor.fetchone()
        if row is None:
            return {"capture_active": False}
        return {"capture_active": bool(row[0])}
    except pyodbc.Error as e:
        logger.error(f"❌ Error fetching capture status: {e}")
        return {"capture_active": False}
    finally:
        conn.close()
