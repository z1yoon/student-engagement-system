from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

class Student(BaseModel):
    """
    Represents a student with optional facial embeddings for recognition.
    """
    name: str = Field(..., min_length=1, max_length=100, description="Full name of the student")
    face_embedding: Optional[List[float]] = Field(
        None, description="Numerical representation of the student's face for recognition"
    )

class EngagementRecord(BaseModel):
    """
    Represents engagement data including phone usage, gaze direction, sleep status, and talking.
    """
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Time when engagement was recorded")
    phone_detected: bool = Field(..., description="True if a phone is detected")
    gaze: str = Field(..., regex="^(focused|distracted)$", description="Student's gaze status")
    sleeping: bool = Field(..., description="True if the student appears to be sleeping")
    talking: bool = Field(..., description="True if the student appears to be talking")

class CaptureStatus(BaseModel):
    """
    Represents whether the capture process is currently active.
    """
    capture_active: bool = Field(..., description="True if image capture is active")
