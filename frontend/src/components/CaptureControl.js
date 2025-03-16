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

  useEffect(() => {
    // Poll every ~minute to get the latest analyzed image
    const imageInterval = setInterval(async () => {
      try {
        const summary = await getAnalyzeResults();
        if (summary.latest_annotated_image) {
          setAnnotatedImage(summary.latest_annotated_image);
        }
      } catch (error) {
        console.error("Error fetching analyzed results:", error);
      }
    }, 60000); // 1 minute

    // Poll capture status every ~minute
    const statusInterval = setInterval(async () => {
      try {
        const status = await getCaptureStatus();
        setCaptureStatus(status.capture_active);
      } catch (error) {
        console.error("Error fetching capture status:", error);
      }
    }, 60000); // 1 minute

    return () => {
      clearInterval(imageInterval);
      clearInterval(statusInterval);
    };
  }, []);

  const handleStart = async () => {
    try {
      const res = await startCapture();
      setIsActive(true);
      alert(res.message);
    } catch (err) {
      alert("Error starting capture: " + err.message);
    }
  };

  const handleStop = async () => {
    try {
      const res = await stopCapture();
      setIsActive(false);
      alert(res.message);
    } catch (err) {
      alert("Error stopping capture: " + err.message);
    }
  };

  return (
    <div>
      <h2>Capture Control</h2>
      <div>
        <button onClick={handleStart} disabled={isActive}>
          Start Capture
        </button>
        <button onClick={handleStop} disabled={!isActive}>
          Stop Capture
        </button>
      </div>
      <div>
        <p>Current Capture Status: {captureStatus ? "Active" : "Inactive"}</p>
      </div>
      {annotatedImage && (
        <div>
          <h3>Latest Analyzed Image</h3>
          <img
            src={`data:image/jpeg;base64,${annotatedImage}`}
            alt="Annotated Result"
            style={{
              width: "100%",
              maxWidth: "600px",
              border: "2px solid black",
              borderRadius: "10px"
            }}
          />
        </div>
      )}
    </div>
  );
}

export default CaptureControl;
