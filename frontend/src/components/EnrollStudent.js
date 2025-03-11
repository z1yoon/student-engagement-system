import React, { useState, useRef, useEffect } from "react";
import axios from "axios";

function EnrollStudent() {
  const [name, setName] = useState("");
  const [capturedImage, setCapturedImage] = useState(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  useEffect(() => {
    // Start the webcam stream when the component loads
    const startCamera = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch (error) {
        console.error("Error accessing camera:", error);
      }
    };

    startCamera();
  }, []);

  const captureImage = () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;

    if (video && canvas) {
      const ctx = canvas.getContext("2d");
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      const dataUrl = canvas.toDataURL("image/jpeg");
      setCapturedImage(dataUrl);
    }
  };

  const handleEnroll = async () => {
    if (!name || !capturedImage) {
      alert("Please provide a name and capture an image.");
      return;
    }

    try {
      await axios.post("http://localhost:8000/api/enroll-student/", {
        student_name: name,
        image_data: capturedImage,
      });
      alert("Student enrolled successfully!");
    } catch (error) {
      alert("Error enrolling student.");
      console.error(error);
    }
  };

  return (
    <div>
      <h2>Enroll Student</h2>
      <input
        type="text"
        placeholder="Student Name"
        value={name}
        onChange={(e) => setName(e.target.value)}
      />
      <div>
        <video ref={videoRef} autoPlay playsInline style={{ width: "300px", border: "1px solid black" }} />
        <canvas ref={canvasRef} style={{ display: "none" }} />
      </div>
      <button onClick={captureImage}>Capture Image</button>
      {capturedImage && <img src={capturedImage} alt="Captured" width={200} />}
      <button onClick={handleEnroll}>Enroll</button>
    </div>
  );
}

export default EnrollStudent;
