# Student Engagement System

A comprehensive system to monitor and analyze student engagement in classrooms using computer vision techniques.

## Architecture Overview

The system consists of several components working together:

1. **IoT Device**: Captures images at regular intervals and sends them to Azure IoT Hub
2. **Azure Function App**: Receives images from IoT Hub and forwards them to the backend for processing
3. **Backend API (FastAPI)**: Analyzes images for student recognition and engagement metrics
4. **Frontend Application (React)**: Provides an interface for teachers to enroll students, control capture, and view analytics

## Key Features

- **Student Enrollment**: Capture student faces and store embeddings for later recognition
- **Automatic Attendance**: Recognize students in classroom images and mark attendance automatically
- **Engagement Analysis**: Detect various engagement metrics:
  - Gaze direction (focused vs. distracted)
  - Drowsiness/sleeping detection
  - Phone usage detection
- **Analytics Dashboard**: View historical engagement data and metrics

## Technology Stack

- **Backend**: Python FastAPI
- **Frontend**: React
- **Computer Vision**: 
  - MediaPipe for face mesh, pose detection, and engagement analysis
  - InsightFace for face recognition
- **Database**: SQL Database (Azure SQL)
- **Cloud**: Azure IoT Hub, Azure Functions, Azure Blob Storage

## MediaPipe Engagement Analysis

The system uses MediaPipe's suite of ML tools to analyze student engagement:

1. **Face Detection**: Detects and tracks student faces in the classroom
2. **Face Mesh**: Maps 468 facial landmarks for detailed facial analysis
3. **Gaze Detection**: Tracks eye movements to determine focus direction
4. **Eye Aspect Ratio (EAR)**: Measures eye openness to detect drowsiness
5. **Pose Detection**: Tracks body pose to detect phone usage

## Getting Started

### Prerequisites

- Docker and Docker Compose
- Azure subscription with IoT Hub, Functions, and Storage
- Environment variables configured in `.env` file

### Running the System

```bash
# Build and start all services
docker-compose up --build

# Or run individual components
docker-compose up backend
docker-compose up frontend
```

### Environment Configuration

Create a `.env` file in the root directory with the following variables:

```
# Azure credentials
VISION_API_ENDPOINT=
VISION_API_KEY=
IOT_HUB_CONNECTION_STRING=
STORAGE_CONNECTION_STRING=
STORAGE_CONTAINER_NAME=

# IoT Device settings
DEVICE_ID=
CAPTURE_INTERVAL=60

# Database settings
DB_SERVER=
DB_NAME=
DB_USER=
DB_PASSWORD=
```

## Usage

1. **Enroll Students**: Use the enrollment page to add students to the system
2. **Start Capture**: Begin the automated capture and analysis process
3. **View Analytics**: Check the analytics page for engagement metrics and trends

## Engagement Metrics Explanation

- **Focused**: Student's gaze is directed towards the front/center (likely paying attention)
- **Distracted**: Student's gaze is directed away from the center
- **Sleeping**: Student's eyes are closed for extended periods (detected by eye aspect ratio)
- **Phone Usage**: Student's hand is positioned near their face/ear (detected via pose estimation)
