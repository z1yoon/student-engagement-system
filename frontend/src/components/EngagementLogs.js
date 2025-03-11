import React, { useState, useEffect } from "react";
import { getEngagement } from "../api";

function EngagementLogs() {
  const [logs, setLogs] = useState([]);

  useEffect(() => {
    getEngagement().then(setLogs);
  }, []);

  return (
    <div>
      <h2>Engagement Logs</h2>
      <ul>
        {logs.map((log, index) => (
          <li key={index}>
            {log.timestamp} - Gaze: {log.gaze} - Sleeping: {log.sleeping ? "Yes" : "No"} - Talking: {log.talking ? "Yes" : "No"} - Phone Usage: {log.phone_detected ? "Yes" : "No"}
          </li>
        ))}
      </ul>
    </div>
  );
}

export default EngagementLogs;
