import React, { useState, useEffect } from "react";
import "chart.js/auto";
import { Bar } from "react-chartjs-2";
import axios from "axios";

const API_BASE_URL = "http://localhost:8000";

function AnalyzeReport() {
  const [reportData, setReportData] = useState({});
  const [chartData, setChartData] = useState(null);
  const [searchTerm, setSearchTerm] = useState("");
  const [currentPage, setCurrentPage] = useState(1);
  const [startDate, setStartDate] = useState(""); // Date Range Start
  const [endDate, setEndDate] = useState(""); // Date Range End
  const studentsPerPage = 5;

  useEffect(() => {
    fetchAnalyzeReport();
    const interval = setInterval(fetchAnalyzeReport, 180000);
    return () => clearInterval(interval);
  }, [startDate, endDate]);

  const fetchAnalyzeReport = () => {
    axios
      .get(`${API_BASE_URL}/api/analyze_results`, {
        params: { start_date: startDate, end_date: endDate },
      })
      .then((response) => {
        setReportData(response.data);
        generateChartData(response.data);
      })
      .catch((error) => {
        console.error("Error fetching analyze report:", error);
      });
  };

  const generateChartData = (data) => {
    const labels = Object.keys(data);
    const focusedCounts = labels.map((name) => data[name].focused);
    const distractedCounts = labels.map((name) => data[name].distracted);
    const phoneUsageCounts = labels.map((name) => data[name].phone_usage);
    const sleepingCounts = labels.map((name) => data[name].sleeping);

    setChartData({
      labels,
      datasets: [
        { label: "Focused", data: focusedCounts, backgroundColor: "green" },
        { label: "Distracted", data: distractedCounts, backgroundColor: "red" },
        { label: "Phone Usage", data: phoneUsageCounts, backgroundColor: "blue" },
        { label: "Sleeping", data: sleepingCounts, backgroundColor: "purple" },
      ],
    });
  };

  // Filter students based on search input
  const filteredStudents = Object.entries(reportData).filter(([name]) =>
    name.toLowerCase().includes(searchTerm.toLowerCase())
  );

  // Pagination logic
  const indexOfLastStudent = currentPage * studentsPerPage;
  const indexOfFirstStudent = indexOfLastStudent - studentsPerPage;
  const currentStudents = filteredStudents.slice(indexOfFirstStudent, indexOfLastStudent);

  return (
    <div style={{ padding: "20px" }}>
      <h2>📊 Analyze Report</h2>

      {/* Search Bar */}
      <input
        type="text"
        placeholder="🔍 Search student..."
        value={searchTerm}
        onChange={(e) => setSearchTerm(e.target.value)}
        style={{
          padding: "8px",
          width: "100%",
          marginBottom: "15px",
          border: "1px solid #ddd",
          borderRadius: "5px",
        }}
      />

      {/* Date Range Filter */}
      <div style={{ display: "flex", gap: "10px", marginBottom: "15px" }}>
        <input
          type="date"
          value={startDate}
          onChange={(e) => setStartDate(e.target.value)}
          style={{ padding: "8px", border: "1px solid #ddd", borderRadius: "5px" }}
        />
        <input
          type="date"
          value={endDate}
          onChange={(e) => setEndDate(e.target.value)}
          style={{ padding: "8px", border: "1px solid #ddd", borderRadius: "5px" }}
        />
      </div>

      {/* Engagement Chart */}
      <h3>📊 Engagement Chart</h3>
      {chartData ? <Bar data={chartData} options={{ responsive: true }} /> : <p>Loading chart...</p>}

      {/* Engagement & Attendance Details */}
      <h3>📜 Student Details</h3>
      <ul style={{ listStyle: "none", padding: 0 }}>
        {currentStudents.map(([name, details], index) => (
          <li
            key={index}
            style={{
              display: "flex",
              alignItems: "center",
              gap: "15px",
              padding: "10px",
              borderBottom: "1px solid #ddd",
            }}
          >
            {details.image_url ? (
              <img
                src={details.image_url}
                alt={name}
                width={50}
                height={50}
                style={{ borderRadius: "50%", objectFit: "cover" }}
              />
            ) : (
              <div style={{ width: 50, height: 50, backgroundColor: "#ccc", borderRadius: "50%" }} />
            )}
            <div style={{ flex: 1 }}>
              <strong>{name}</strong>
              <div>
                <span>
                  ✅ Focused: {details.focused}, ❌ Distracted: {details.distracted}, 📱 Phone Usage: {details.phone_usage}, 💤 Sleeping: {details.sleeping}
                </span>
              </div>
              <div>
                {/* Attendance Indicator */}
                {details.attended ? (
                  <span style={{ color: "green", fontWeight: "bold" }}>✔ Attended</span>
                ) : (
                  <span style={{ color: "red", fontWeight: "bold" }}>✘ Absent</span>
                )}
              </div>
            </div>
          </li>
        ))}
      </ul>

      {/* Pagination Controls */}
      <div style={{ marginTop: "15px", textAlign: "center" }}>
        <button
          onClick={() => setCurrentPage(currentPage - 1)}
          disabled={currentPage === 1}
          style={{
            marginRight: "10px",
            padding: "5px 10px",
            cursor: "pointer",
            border: "1px solid #ddd",
            borderRadius: "5px",
            background: currentPage === 1 ? "#ccc" : "#007BFF",
            color: "white",
          }}
        >
          ◀ Prev
        </button>
        <span> Page {currentPage} </span>
        <button
          onClick={() => setCurrentPage(currentPage + 1)}
          disabled={indexOfLastStudent >= filteredStudents.length}
          style={{
            marginLeft: "10px",
            padding: "5px 10px",
            cursor: "pointer",
            border: "1px solid #ddd",
            borderRadius: "5px",
            background: indexOfLastStudent >= filteredStudents.length ? "#ccc" : "#007BFF",
            color: "white",
          }}
        >
          Next ▶
        </button>
      </div>
    </div>
  );
}

export default AnalyzeReport;
