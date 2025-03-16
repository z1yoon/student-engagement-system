import React, { useState, useRef, useEffect } from "react";
import axios from "axios";

function EnrollStudent() {
  const [name, setName] = useState("");
  const [capturedImage, setCapturedImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [videoReady, setVideoReady] = useState(false); // Track when video is ready

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

          // Listen for 'loadedmetadata' to confirm video dimensions are available
          videoRef.current.addEventListener("loadedmetadata", () => {
            console.log("Video metadata loaded. Video is ready!");
            setVideoReady(true);
          });
        }
      } catch (error) {
        console.error("Error accessing camera:", error);
      }
    }
    startCamera();

    // Cleanup: stop the stream if the component unmounts
    return () => {
      if (videoRef.current && videoRef.current.srcObject) {
        const tracks = videoRef.current.srcObject.getTracks();
        tracks.forEach((track) => track.stop());
      }
    };
  }, []);

  const captureImage = () => {
    if (!videoReady || !videoRef.current) {
      console.warn("Video is not ready for capture.");
      return;
    }

    const video = videoRef.current;
    const canvas = canvasRef.current;

    // Double-check that video dimensions are valid
    if (video.videoWidth === 0 || video.videoHeight === 0) {
      console.warn("Video width/height is 0, skipping capture.");
      return;
    }

    if (canvas) {
      const ctx = canvas.getContext("2d");
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      // Lower JPEG quality to reduce image size
      const dataUrl = canvas.toDataURL("image/jpeg", 0.7);
      setCapturedImage(dataUrl);
      console.log(
        "Captured image data (first 100 chars):",
        dataUrl.substring(0, 100)
      );
    }
  };

  const handleEnroll = async () => {
    if (!name || !capturedImage) {
      alert("Please provide a name and capture an image.");
      return;
    }
    setLoading(true);
    try {
      const response = await axios.post("http://localhost:8000/api/enroll-student/", {
        student_name: name,
        image_data: capturedImage
      });
      console.log("Enroll response:", response.data);
      alert("Student enrolled successfully!");
    } catch (error) {
      console.error("Error enrolling student:", error);
      alert("Error enrolling student.");
    }
    setLoading(false);
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
      <div style={{ marginTop: "10px" }}>
        <video
          ref={videoRef}
          autoPlay
          playsInline
          style={{ width: "300px", border: "1px solid black" }}
        />
        <canvas ref={canvasRef} style={{ display: "none" }} />
      </div>
      <div style={{ marginTop: "10px" }}>
        {/* Disable capture if video isn't ready */}
        <button onClick={captureImage} disabled={!videoReady}>
          {videoReady ? "Capture Image" : "Camera Not Ready"}
        </button>
      </div>
      {capturedImage && (
        <div style={{ marginTop: "10px" }}>
          <img src={capturedImage} alt="Captured" width={200} />
        </div>
      )}
      <div style={{ marginTop: "10px" }}>
        <button onClick={handleEnroll} disabled={loading || !capturedImage}>
          {loading ? "Enrolling..." : "Enroll"}
        </button>
      </div>
    </div>
  );
}

export default EnrollStudent;
