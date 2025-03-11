import React, { useState, useEffect } from "react";
import axios from "axios";

function AttendanceView() {
  const [attendance, setAttendance] = useState([]);

  useEffect(() => {
    const fetchAttendance = async () => {
      try {
        const res = await axios.get("http://localhost:8000/api/attendance");
        setAttendance(res.data);
      } catch (error) {
        console.error("Error fetching attendance:", error);
      }
    };

    fetchAttendance();
  }, []);

  return (
    <div>
      <h2>Attendance Records</h2>
      <ul>
        {attendance.map((record, index) => (
          <li key={index}>
            {record.timestamp} - {record.recognized_name} - Attended:{" "}
            {record.is_attended ? "Yes" : "No"}
          </li>
        ))}
      </ul>
    </div>
  );
}

export default AttendanceView;
