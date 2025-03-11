from ..db import match_student_by_enc, add_attendance_record

def process_attendance(faces_info):
    """Matches faces with database for attendance"""
    recognized_students = []
    for face in faces_info:
        student_id, student_name = match_student_by_enc(face["encoding"])
        if student_id:
            recognized_students.append({"id": student_id, "name": student_name})
            add_attendance_record(student_id, student_name, True)

    return recognized_students
