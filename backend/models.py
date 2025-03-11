from pydantic import BaseModel
from typing import Optional

class Student(BaseModel):
    name: str
    face_encoding: Optional[str]

class EngagementRecord(BaseModel):
    phone_detected: bool
    gaze: str
    sleeping: bool
    talking: bool

class CaptureStatus(BaseModel):
    capture_active: bool
