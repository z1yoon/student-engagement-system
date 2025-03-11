import React, { useState } from "react";
import { startCapture, stopCapture } from "../api";

function CaptureControl() {
  const [isActive, setIsActive] = useState(false);

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
      <button onClick={handleStart} disabled={isActive}>
        Start
      </button>
      <button onClick={handleStop} disabled={!isActive}>
        Stop
      </button>
    </div>
  );
}

export default CaptureControl;
