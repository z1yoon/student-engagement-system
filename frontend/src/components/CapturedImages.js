import React, { useState, useEffect } from 'react';
import { getRecentImages } from '../api';

const CapturedImages = () => {
  const [recentImages, setRecentImages] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Poll for new images every 10 seconds
  useEffect(() => {
    const fetchImages = async () => {
      try {
        setLoading(true);
        const data = await getRecentImages();
        setRecentImages(data.recent_images || []);
        setError(null);
      } catch (err) {
        console.error('Error fetching recent images:', err);
        setError('Failed to load recent images. Please try again.');
      } finally {
        setLoading(false);
      }
    };

    // Fetch immediately on mount
    fetchImages();
    
    // Then set up polling
    const interval = setInterval(fetchImages, 10000);
    return () => clearInterval(interval);
  }, []);

  // Simple time ago formatter
  const formatTimeAgo = (timestamp) => {
    const now = new Date();
    const date = new Date(timestamp);
    const seconds = Math.floor((now - date) / 1000);
    
    let interval = Math.floor(seconds / 31536000);
    if (interval > 1) return `${interval} years ago`;
    
    interval = Math.floor(seconds / 2592000);
    if (interval > 1) return `${interval} months ago`;
    
    interval = Math.floor(seconds / 86400);
    if (interval > 1) return `${interval} days ago`;
    
    interval = Math.floor(seconds / 3600);
    if (interval > 1) return `${interval} hours ago`;
    
    interval = Math.floor(seconds / 60);
    if (interval > 1) return `${interval} minutes ago`;
    
    return "just now";
  };

  if (loading && recentImages.length === 0) {
    return (
      <div className="loading-indicator">
        <div className="spinner"></div>
        <p>Loading images...</p>
      </div>
    );
  }

  if (error && recentImages.length === 0) {
    return (
      <div className="message error">
        {error}
      </div>
    );
  }

  if (recentImages.length === 0) {
    return (
      <div className="message info">
        No captured images yet. Start capturing to see images appear here.
      </div>
    );
  }

  return (
    <div className="captured-images-container">
      <div className="images-grid">
        {recentImages.slice().reverse().map((item, index) => (
          <div className="image-card" key={index}>
            <div className="image-container">
              <img
                src={`data:image/jpeg;base64,${item.image}`}
                alt={`Captured image ${index}`}
                className="captured-image"
              />
            </div>
            <div className="image-details">
              <p className="timestamp">{formatTimeAgo(item.timestamp)}</p>
              <p className="recognition-status">
                {item.face_detected ? (
                  <>Face detected: {item.recognized_students?.length || 0} student(s) recognized</>
                ) : (
                  <>No faces detected</>
                )}
              </p>
              {item.recognized_students?.length > 0 && (
                <p className="students-list">
                  Students: {item.recognized_students.join(', ')}
                </p>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default CapturedImages;