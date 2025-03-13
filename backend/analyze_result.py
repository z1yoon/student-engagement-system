from db import get_engagement_records, get_attendance_records

def get_analyze_results(start_date=None, end_date=None):
    """
    Returns a summary of engagement & attendance data.
    """
    engagement_records = get_engagement_records(start_date, end_date)
    attendance_records = get_attendance_records(start_date, end_date)

    summary = {}

    # Build engagement summary
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
        # Gaze
        if record["gaze"] == "focused":
            summary[name]["focused"] += 1
        else:
            summary[name]["distracted"] += 1
        # Phone usage
        if record["phone_detected"]:
            summary[name]["phone_usage"] += 1
        # Sleeping
        if record["sleeping"]:
            summary[name]["sleeping"] += 1

    # Merge attendance
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
        summary[name]["image_url"] = record["image_url"]

    return summary
