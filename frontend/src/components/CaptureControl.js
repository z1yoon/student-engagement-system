import React, { useState, useEffect } from "react";
import {
  startCapture,
  stopCapture,
  getCaptureStatus,
  getAnalyzeResults
} from "../api";

function CaptureControl() {
  const [isActive, setIsActive] = useState(false);
  const [annotatedImage, setAnnotatedImage] = useState("");
  const [captureStatus, setCaptureStatus] = useState(false);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState({ type: "", text: "" });

  useEffect(() => {
    // Initial data fetch
    fetchImageAndStatus();
    
    // Poll every ~minute to get the latest analyzed image
    const imageInterval = setInterval(fetchImageAndStatus, 60000);

    return () => {
      clearInterval(imageInterval);
    };
  }, []);
  
  const fetchImageAndStatus = async () => {
    try {
      // Get latest analyzed image
      const summary = await getAnalyzeResults();
      if (summary.latest_annotated_image) {
        setAnnotatedImage(summary.latest_annotated_image);
      }
      
      // Get capture status
      const status = await getCaptureStatus();
      setCaptureStatus(status.active);
      setIsActive(status.active);
    } catch (error) {
      console.error("Error fetching data:", error);
      setMessage({
        type: "error",
        text: "Failed to fetch the latest data. Please try again."
      });
    }
  };

  const handleStart = async () => {
    setLoading(true);
    try {
      const res = await startCapture();
      setIsActive(true);
      setMessage({
        type: "success",
        text: "Capture started successfully."
      });
    } catch (err) {
      setMessage({
        type: "error",
        text: `Error starting capture: ${err.message}`
      });
    } finally {
      setLoading(false);
    }
  };

  const handleStop = async () => {
    setLoading(true);
    try {
      const res = await stopCapture();
      setIsActive(false);
      setMessage({
        type: "success",
        text: "Capture stopped successfully."
      });
    } catch (err) {
      setMessage({
        type: "error",
        text: `Error stopping capture: ${err.message}`
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="capture-container">
      <div className="card">
        <h2>📸 Capture Control</h2>
        
        {message.text && (
          <div className={`message ${message.type}`}>
            {message.text}
          </div>
        )}
        
        <div className="status-indicator">
          <div className={`status-dot ${captureStatus ? 'active' : 'inactive'}`}></div>
          <p>Current Status: 
            <span className={captureStatus ? 'text-success' : 'text-danger'}>
              {captureStatus ? " Active" : " Inactive"}
            </span>
          </p>
        </div>
        
        <div className="button-group">
          <button
            className="btn-success"
            onClick={handleStart}
            disabled={isActive || loading}
          >
            {loading && !isActive ? "Starting..." : "Start Capture"}
          </button>
          
          <button
            className="btn-danger"
            onClick={handleStop}
            disabled={!isActive || loading}
          >
            {loading && isActive ? "Stopping..." : "Stop Capture"}
          </button>
        </div>
      </div>
      
      {annotatedImage && (
        <div className="card image-card">
          <h3>Latest Analyzed Image</h3>
          <div className="annotated-image-container">
            <img
              src={`data:image/jpeg;base64,${annotatedImage}`}
              alt="Annotated Result"
              className="annotated-image"
            />
          </div>
          <div className="image-info">
            <p>This is the most recent image processed by the system.</p>
            <button 
              className="btn-secondary"
              onClick={fetchImageAndStatus}
            >
              Refresh Data
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

export default CaptureControl;
