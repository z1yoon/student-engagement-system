import React, { useState, useRef, useEffect } from "react";
import axios from "axios";

function EnrollStudent() {
  const [name, setName] = useState("");
  const [capturedImage, setCapturedImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [videoReady, setVideoReady] = useState(false);
  const [message, setMessage] = useState({ type: "", text: "" });

  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  useEffect(() => {
    async function startCamera() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 640, height: 480 }
        });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;

          videoRef.current.addEventListener("loadedmetadata", () => {
            console.log("Video metadata loaded. Video is ready!");
            setVideoReady(true);
          });
        }
      } catch (error) {
        console.error("Error accessing camera:", error);
        setMessage({
          type: "error",
          text: "Unable to access camera. Please check camera permissions."
        });
      }
    }
    startCamera();

    return () => {
      if (videoRef.current && videoRef.current.srcObject) {
        const tracks = videoRef.current.srcObject.getTracks();
        tracks.forEach((track) => track.stop());
      }
    };
  }, []);

  const captureImage = () => {
    if (!videoReady || !videoRef.current) {
      setMessage({
        type: "error",
        text: "Video is not ready for capture. Please wait."
      });
      return;
    }

    const video = videoRef.current;
    const canvas = canvasRef.current;

    if (video.videoWidth === 0 || video.videoHeight === 0) {
      setMessage({
        type: "error",
        text: "Video dimensions are invalid. Please reload the page."
      });
      return;
    }

    if (canvas) {
      const ctx = canvas.getContext("2d");
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      const dataUrl = canvas.toDataURL("image/jpeg", 0.7);
      setCapturedImage(dataUrl);
      setMessage({
        type: "success",
        text: "Image captured successfully!"
      });
    }
  };

  const handleEnroll = async () => {
    if (!name) {
      setMessage({
        type: "error",
        text: "Please enter a student name."
      });
      return;
    }
    
    if (!capturedImage) {
      setMessage({
        type: "error",
        text: "Please capture an image first."
      });
      return;
    }
    
    setLoading(true);
    try {
      const response = await axios.post("http://localhost:8000/api/enroll-student/", {
        student_name: name,
        image_data: capturedImage
      });
      
      setMessage({
        type: "success",
        text: "Student enrolled successfully!"
      });
      
      // Reset form
      setName("");
      setCapturedImage(null);
    } catch (error) {
      console.error("Error enrolling student:", error);
      setMessage({
        type: "error",
        text: error.response?.data?.detail || "Error enrolling student. Please try again."
      });
    }
    setLoading(false);
  };

  const resetForm = () => {
    setName("");
    setCapturedImage(null);
    setMessage({ type: "", text: "" });
  };

  return (
    <div className="enroll-container">
      <div className="card">
        <h2>Enroll New Student</h2>
        
        {message.text && (
          <div className={`message ${message.type}`}>
            {message.text}
          </div>
        )}
        
        <div className="form-group">
          <label htmlFor="studentName">Student Name</label>
          <input
            id="studentName"
            type="text"
            placeholder="Enter student's full name"
            value={name}
            onChange={(e) => setName(e.target.value)}
          />
        </div>
        
        <div className="camera-container">
          <div className="video-wrapper">
            <video
              ref={videoRef}
              autoPlay
              playsInline
              className="camera-feed"
            />
            <canvas ref={canvasRef} style={{ display: "none" }} />
            
            {!videoReady && (
              <div className="camera-overlay">
                <div className="loading-spinner"></div>
                <p>Initializing camera...</p>
              </div>
            )}
          </div>
          
          {capturedImage && (
            <div className="preview-wrapper">
              <img src={capturedImage} alt="Captured" className="preview-image" />
            </div>
          )}
        </div>
        
        <div className="button-group">
          <button
            className="btn-primary"
            onClick={captureImage}
            disabled={!videoReady || loading}
          >
            {!videoReady ? "Camera Loading..." : "Capture Image"}
          </button>
          
          <button
            className="btn-success"
            onClick={handleEnroll}
            disabled={loading || !capturedImage || !name}
          >
            {loading ? "Enrolling..." : "Enroll Student"}
          </button>
          
          <button
            className="btn-secondary"
            onClick={resetForm}
            disabled={loading}
          >
            Reset
          </button>
        </div>
      </div>
    </div>
  );
}

export default EnrollStudent;
