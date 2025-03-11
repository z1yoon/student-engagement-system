from db import get_engagement_records, get_attendance_records

def get_analyze_results(start_date=None, end_date=None):
    """
    Fetches engagement and attendance records with optional date filtering and returns a summary.
    """
    engagement_records = get_engagement_records(start_date, end_date)
    attendance_records = get_attendance_records(start_date, end_date)

    summary = {}

    # Process Engagement Data
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

    # Merge Attendance Data
    for record in attendance_records:
        name = record["recognized_name"]
        if name in summary:
            summary[name]["attended"] = record["is_attended"]
            summary[name]["image_url"] = record["image_url"]

    return summary
