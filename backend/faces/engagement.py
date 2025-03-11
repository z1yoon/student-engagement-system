from db import get_engagement_records

def summarize_engagement():
    """Aggregates all engagement records into a final report."""
    records = get_engagement_records()
    student_focus_summary = {}

    for record in records:
        name = record["student_name"]
        if name not in student_focus_summary:
            student_focus_summary[name] = {"focused": 0, "distracted": 0, "phone_usage": 0}

        if record["gaze"] == "focused":
            student_focus_summary[name]["focused"] += 1
        else:
            student_focus_summary[name]["distracted"] += 1

        if record["phone_detected"]:
            student_focus_summary[name]["phone_usage"] += 1

    return student_focus_summary
