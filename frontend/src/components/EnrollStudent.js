import React, { useState, useRef, useEffect } from "react";
import axios from "axios";

function EnrollStudent() {
  const [name, setName] = useState("");
  const [capturedImages, setCapturedImages] = useState({
    center: null,
    left: null,
    right: null
  });
  const [currentPosition, setCurrentPosition] = useState("center");
  const [loading, setLoading] = useState(false);
  const [videoReady, setVideoReady] = useState(false);
  const [message, setMessage] = useState({ type: "", text: "" });
  const [enrollmentSuccess, setEnrollmentSuccess] = useState(false);

  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  // Effect to reset success message after delay
  useEffect(() => {
    let timer;
    if (enrollmentSuccess) {
      timer = setTimeout(() => {
        setEnrollmentSuccess(false);
      }, 10000); // Hide success message after 10 seconds
    }
    return () => clearTimeout(timer);
  }, [enrollmentSuccess]);

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
      
      setCapturedImages(prev => ({
        ...prev,
        [currentPosition]: dataUrl
      }));
      
      setMessage({
        type: "success",
        text: `${currentPosition.charAt(0).toUpperCase() + currentPosition.slice(1)} position image captured successfully!`
      });
      
      // Automatically move to next position if needed
      if (currentPosition === "center") {
        setCurrentPosition("left");
        setMessage({
          type: "info",
          text: "Now turn your head to the LEFT and capture again."
        });
      } else if (currentPosition === "left") {
        setCurrentPosition("right");
        setMessage({
          type: "info",
          text: "Now turn your head to the RIGHT and capture again."
        });
      }
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
    
    // Check if all positions are captured
    if (!capturedImages.center || !capturedImages.left || !capturedImages.right) {
      setMessage({
        type: "error",
        text: "Please capture images for all three head positions (center, left, and right)."
      });
      return;
    }
    
    setLoading(true);
    try {
      const response = await axios.post("http://localhost:8000/api/enroll-student/", {
        student_name: name,
        image_center: capturedImages.center,
        image_left: capturedImages.left,
        image_right: capturedImages.right
      });
      
      // Set the success message and show the prominent success notification
      setMessage({
        type: "success",
        text: "Student enrolled successfully with all three head positions!"
      });
      
      // Store student name for success message and activate success UI
      setEnrollmentSuccess(name);
      
      // Reset form
      resetForm();
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
    setCapturedImages({
      center: null,
      left: null,
      right: null
    });
    setCurrentPosition("center");
    setMessage({ 
      type: "info", 
      text: "Start by capturing the CENTER position (looking straight at the camera)." 
    });
  };
  
  const setPosition = (position) => {
    setCurrentPosition(position);
    setMessage({
      type: "info",
      text: `Now capturing ${position} position. ${
        position === "center" 
          ? "Look straight at the camera." 
          : position === "left" 
          ? "Turn your head to the left." 
          : "Turn your head to the right."
      }`
    });
  };
  
  const allPositionsCaptured = capturedImages.center && capturedImages.left && capturedImages.right;

  return (
    <div className="enroll-container">
      {/* Prominent success message when student is enrolled successfully */}
      {enrollmentSuccess && (
        <div className="enrollment-success-banner">
          <div className="success-icon">✅</div>
          <div className="success-message">
            <h3>Student Enrolled Successfully!</h3>
            <p>Student <strong>{enrollmentSuccess}</strong> has been enrolled with facial recognition.</p>
          </div>
        </div>
      )}
      
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
        
        <div className="position-selector">
          <div className="position-buttons">
            <button 
              className={`position-btn ${currentPosition === "center" ? "active" : ""}`}
              onClick={() => setPosition("center")}
            >
              Center {capturedImages.center && "✓"}
            </button>
            <button 
              className={`position-btn ${currentPosition === "left" ? "active" : ""}`}
              onClick={() => setPosition("left")}
            >
              Left {capturedImages.left && "✓"}
            </button>
            <button 
              className={`position-btn ${currentPosition === "right" ? "active" : ""}`}
              onClick={() => setPosition("right")}
            >
              Right {capturedImages.right && "✓"}
            </button>
          </div>
          <p className="position-hint">
            <strong>Current Position:</strong> {currentPosition.toUpperCase()}
          </p>
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
          
          <div className="preview-grid">
            {capturedImages.center && (
              <div className="preview-wrapper">
                <span className="position-label">Center</span>
                <img src={capturedImages.center} alt="Center" className="preview-image" />
              </div>
            )}
            {capturedImages.left && (
              <div className="preview-wrapper">
                <span className="position-label">Left</span>
                <img src={capturedImages.left} alt="Left" className="preview-image" />
              </div>
            )}
            {capturedImages.right && (
              <div className="preview-wrapper">
                <span className="position-label">Right</span>
                <img src={capturedImages.right} alt="Right" className="preview-image" />
              </div>
            )}
          </div>
        </div>
        
        <div className="button-group">
          <button
            className="btn-primary"
            onClick={captureImage}
            disabled={!videoReady || loading}
          >
            {!videoReady 
              ? "Camera Loading..." 
              : `Capture ${currentPosition.charAt(0).toUpperCase() + currentPosition.slice(1)} Position`}
          </button>
          
          <button
            className="btn-success"
            onClick={handleEnroll}
            disabled={loading || !allPositionsCaptured || !name}
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
