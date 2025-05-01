import React, { useState, useEffect } from "react";
import {
  startCapture,
  stopCapture,
  getCaptureStatus,
  getAnalyzeResults,
  getLatestAnalysis
} from "../api";
import axios from "axios";

const API_BASE_URL = "http://localhost:8000";

function CaptureControl() {
  const [isActive, setIsActive] = useState(false);
  const [annotatedImage, setAnnotatedImage] = useState("");
  const [captureStatus, setCaptureStatus] = useState(false);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState({ type: "", text: "" });
  const [latestAnalysis, setLatestAnalysis] = useState(null);

  useEffect(() => {
    // Initial data fetch
    fetchImageAndStatus();
    
    // Poll every 30 seconds to get the latest analyzed image and status
    const imageInterval = setInterval(fetchImageAndStatus, 30000);

    return () => {
      clearInterval(imageInterval);
    };
  }, []);
  
  const fetchImageAndStatus = async () => {
    try {
      setLoading(true);
      
      // Get capture status
      const status = await getCaptureStatus();
      setCaptureStatus(status.active);
      setIsActive(status.active);
      
      // Get latest analyzed data
      try {
        // Get latest analyze results which might contain the latest annotated image
        const summary = await getAnalyzeResults();
        if (summary && summary.latest_annotated_image) {
          setAnnotatedImage(summary.latest_annotated_image);
        }
        
        // Get the latest analysis result with student status details
        const analysis = await getLatestAnalysis();
        if (analysis && analysis.annotated_image) {
          setAnnotatedImage(analysis.annotated_image);
          setLatestAnalysis(analysis);
        }
      } catch (error) {
        console.error("Error fetching latest analysis:", error);
      }
      
      setLoading(false);
    } catch (error) {
      console.error("Error fetching data:", error);
      setMessage({
        type: "error",
        text: "Failed to fetch the latest data. Please try again."
      });
      setLoading(false);
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
      // Fetch latest status after starting capture
      setTimeout(fetchImageAndStatus, 2000);
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
          
          {latestAnalysis && (
            <div className="analysis-details">
              <h4>Detected Students</h4>
              {latestAnalysis.recognized_students && latestAnalysis.recognized_students.length > 0 ? (
                <ul className="student-status-list">
                  {latestAnalysis.student_status && Object.entries(latestAnalysis.student_status).map(([name, status], idx) => (
                    <li key={idx} className="student-status-item">
                      <div className="student-name">{name}</div>
                      <div className={`student-state ${status.is_sleeping ? 'sleeping' : (status.is_distracted ? 'distracted' : (status.using_phone ? 'phone' : 'focused'))}`}>
                        {status.is_sleeping ? '😴 Sleeping' : 
                          (status.is_distracted ? '👀 Distracted' : 
                            (status.using_phone ? '📱 Phone' : '✅ Focused'))}
                      </div>
                      <div className="student-gaze">Gaze: {status.gaze}</div>
                    </li>
                  ))}
                </ul>
              ) : (
                <p>No students detected in this frame.</p>
              )}
            </div>
          )}
          
          <div className="image-info">
            <p>This is the most recent image processed by the system.</p>
            <button 
              className="btn-secondary"
              onClick={fetchImageAndStatus}
              disabled={loading}
            >
              {loading ? "Refreshing..." : "Refresh Data"}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

export default CaptureControl;
