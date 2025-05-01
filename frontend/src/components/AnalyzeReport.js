import React, { useState, useEffect } from "react";
import "chart.js/auto";
import { Bar } from "react-chartjs-2";
import axios from "axios";

const API_BASE_URL = "http://localhost:8000";

function AnalyzeReport() {
  // State variables
  const [reportData, setReportData] = useState({});
  const [chartData, setChartData] = useState(null);
  const [searchTerm, setSearchTerm] = useState("");
  const [currentPage, setCurrentPage] = useState(1);
  const [startDate, setStartDate] = useState("");
  const [endDate, setEndDate] = useState("");
  const [loading, setLoading] = useState(true);
  const studentsPerPage = 5;

  // Fetch data on component mount and when date range changes
  useEffect(() => {
    fetchAnalyzeReport();
    const interval = setInterval(fetchAnalyzeReport, 180000); // Refresh every 3 minutes
    return () => clearInterval(interval);
  }, [startDate, endDate]);

  // Fetch analyze report data from API
  const fetchAnalyzeReport = () => {
    setLoading(true);
    axios
      .get(`${API_BASE_URL}/api/analyze_results`, {
        params: { start_date: startDate, end_date: endDate },
      })
      .then((response) => {
        setReportData(response.data);
        generateChartData(response.data);
        setLoading(false);
      })
      .catch((error) => {
        console.error("Error fetching analyze report:", error);
        setLoading(false);
      });
  };

  // Generate chart data from API response
  const generateChartData = (data) => {
    // Filter out 'Latest Analyzed Image' entry
    const filteredData = Object.fromEntries(
      Object.entries(data).filter(([key]) => key !== 'Latest Analyzed Image')
    );
    
    const labels = Object.keys(filteredData);
    
    // Extract engagement metrics for chart datasets
    const datasets = [
      { 
        label: "Focused", 
        data: labels.map((name) => filteredData[name].focused), 
        backgroundColor: "rgba(34, 197, 94, 0.7)",
        borderColor: "rgb(34, 197, 94)",
        borderWidth: 1
      },
      { 
        label: "Distracted", 
        data: labels.map((name) => filteredData[name].distracted), 
        backgroundColor: "rgba(239, 68, 68, 0.7)",
        borderColor: "rgb(239, 68, 68)",
        borderWidth: 1
      },
      { 
        label: "Phone Usage", 
        data: labels.map((name) => filteredData[name].phone_usage), 
        backgroundColor: "rgba(59, 130, 246, 0.7)",
        borderColor: "rgb(59, 130, 246)",
        borderWidth: 1
      },
      { 
        label: "Sleeping", 
        data: labels.map((name) => filteredData[name].sleeping), 
        backgroundColor: "rgba(124, 58, 237, 0.7)",
        borderColor: "rgb(124, 58, 237)",
        borderWidth: 1
      }
    ];

    setChartData({ labels, datasets });
  };

  // Filtered students based on search
  const filteredStudents = Object.entries(reportData).filter(([name]) =>
    name.toLowerCase().includes(searchTerm.toLowerCase()) && 
    name !== 'Latest Analyzed Image'
  );

  // Pagination logic
  const indexOfLastStudent = currentPage * studentsPerPage;
  const indexOfFirstStudent = indexOfLastStudent - studentsPerPage;
  const currentStudents = filteredStudents.slice(indexOfFirstStudent, indexOfLastStudent);
  const totalPages = Math.ceil(filteredStudents.length / studentsPerPage);

  // Chart configuration
  const chartOptions = {
    responsive: true,
    plugins: {
      legend: { position: 'top' },
      title: {
        display: true,
        text: 'Student Engagement Summary'
      }
    },
    scales: {
      y: { beginAtZero: true }
    }
  };

  return (
    <div className="analyze-container">
      {/* Report Card */}
      <div className="card">
        <h2>📊 Analyze Report</h2>

        {/* Filters */}
        <div className="filter-section">
          {/* Search Bar */}
          <div className="search-wrapper">
            <input
              type="text"
              placeholder="🔍 Search student..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="search-input"
            />
          </div>

          {/* Date Range Filter */}
          <div className="date-filter">
            <div className="date-input-group">
              <label>Start Date</label>
              <input
                type="date"
                value={startDate}
                onChange={(e) => setStartDate(e.target.value)}
                className="date-input"
              />
            </div>
            <div className="date-input-group">
              <label>End Date</label>
              <input
                type="date"
                value={endDate}
                onChange={(e) => setEndDate(e.target.value)}
                className="date-input"
              />
            </div>
            <button onClick={fetchAnalyzeReport} className="filter-button">
              Apply Filters
            </button>
          </div>
        </div>

        {/* Engagement Chart */}
        <div className="chart-section">
          <h3>📊 Engagement Chart</h3>
          {loading ? (
            <div className="loading-spinner">Loading...</div>
          ) : chartData ? (
            <Bar data={chartData} options={chartOptions} />
          ) : (
            <p className="no-data">No data available for the selected filters</p>
          )}
        </div>
      </div>

      {/* Student Details Card */}
      <div className="card mt-4">
        <h3>📜 Student Details</h3>
        {loading ? (
          <div className="loading-spinner">Loading...</div>
        ) : currentStudents.length > 0 ? (
          <>
            <ul className="student-list">
              {currentStudents.map(([name, details], index) => (
                <li key={index} className="student-item">
                  <div className="student-info">
                    <h4 className="student-name">{name}</h4>
                    
                    {/* Metrics */}
                    <div className="metrics">
                      <span className="metric">
                        <span className="metric-label">Focused:</span>
                        <span className="metric-value">{details.focused}</span>
                      </span>
                      <span className="metric">
                        <span className="metric-label">Distracted:</span>
                        <span className="metric-value">{details.distracted}</span>
                      </span>
                      <span className="metric">
                        <span className="metric-label">Phone:</span>
                        <span className="metric-value">{details.phone_usage}</span>
                      </span>
                      <span className="metric">
                        <span className="metric-label">Sleeping:</span>
                        <span className="metric-value">{details.sleeping}</span>
                      </span>
                    </div>
                    
                    {/* Status indicators */}
                    <div className="status-indicators">
                      {details.distracted > 0 && (
                        <div className="status-detail">
                          <span className="status-icon distracted">👀</span>
                          <span className="status-text">Distracted when looking left/right for 3+ consecutive frames</span>
                        </div>
                      )}
                      {details.sleeping > 0 && (
                        <div className="status-detail">
                          <span className="status-icon sleeping">😴</span>
                          <span className="status-text">Sleeping detected when eyes closed or head down for 3+ consecutive frames</span>
                        </div>
                      )}
                      {details.phone_usage > 0 && (
                        <div className="status-detail">
                          <span className="status-icon phone">📱</span>
                          <span className="status-text">Phone usage detected via Azure Computer Vision</span>
                        </div>
                      )}
                    </div>
                    
                    {/* Attendance status */}
                    <div className="attendance">
                      {details.attended ? (
                        <span className="badge attended">✓ Attended</span>
                      ) : (
                        <span className="badge absent">✗ Absent</span>
                      )}
                    </div>
                  </div>
                </li>
              ))}
            </ul>

            {/* Pagination */}
            <div className="pagination">
              <button
                onClick={() => setCurrentPage(currentPage - 1)}
                disabled={currentPage === 1}
                className="page-button"
              >
                Previous
              </button>
              <span className="page-info">
                Page {currentPage} of {totalPages}
              </span>
              <button
                onClick={() => setCurrentPage(currentPage + 1)}
                disabled={indexOfLastStudent >= filteredStudents.length}
                className="page-button"
              >
                Next
              </button>
            </div>
          </>
        ) : (
          <p className="no-data">No students found matching your criteria</p>
        )}
      </div>

      {/* Detection Criteria Information */}
      <div className="card mt-4">
        <h3>💡 Engagement Detection Criteria</h3>
        <div className="info-box">
          <h4>Phone Detection</h4>
          <p>Uses Azure Computer Vision API to identify phones in the camera frame. When a phone is detected, it's associated with the closest student's face.</p>

          <h4>Sleeping Detection</h4>
          <p>A student is considered sleeping when <strong>either</strong> of these criteria are met for 3 consecutive frames:</p>
          <ul>
            <li>Eyes are closed (measured using eye aspect ratio)</li>
            <li>Head is tilted significantly downward</li>
          </ul>

          <h4>Distraction Detection</h4>
          <p>A student is marked as distracted when their head is turned significantly to the left or right for 3 consecutive frames.</p>
          
          <h4>Multiple Students</h4>
          <p>When multiple students are in frame, each face is analyzed separately. Status indicators and engagement metrics are tracked individually.</p>
        </div>
      </div>
    </div>
  );
}

export default AnalyzeReport;