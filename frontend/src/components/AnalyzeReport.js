import React, { useState, useEffect } from "react";
import "chart.js/auto";
import { Bar } from "react-chartjs-2";
import axios from "axios";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faEye } from "@fortawesome/free-solid-svg-icons";
import Modal from "react-modal";

const API_BASE_URL = "http://localhost:8000";

// Modal styles
const customStyles = {
  content: {
    top: "50%",
    left: "50%",
    right: "auto",
    bottom: "auto",
    marginRight: "-50%",
    transform: "translate(-50%, -50%)",
    maxWidth: "90%",
    maxHeight: "90%",
    overflow: "auto",
    borderRadius: "0.75rem",
    padding: "1.5rem",
    border: "none",
    boxShadow: "0 10px 15px -3px rgba(0, 0, 0, 0.1)"
  },
  overlay: {
    backgroundColor: "rgba(0, 0, 0, 0.75)"
  }
};

function AnalyzeReport() {
  const [reportData, setReportData] = useState({});
  const [chartData, setChartData] = useState(null);
  const [searchTerm, setSearchTerm] = useState("");
  const [currentPage, setCurrentPage] = useState(1);
  const [startDate, setStartDate] = useState("");
  const [endDate, setEndDate] = useState("");
  const [modalIsOpen, setModalIsOpen] = useState(false);
  const [selectedImage, setSelectedImage] = useState("");
  const [loading, setLoading] = useState(true);
  const studentsPerPage = 5;

  useEffect(() => {
    fetchAnalyzeReport();
    const interval = setInterval(fetchAnalyzeReport, 180000);
    return () => clearInterval(interval);
  }, [startDate, endDate]);

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

  const generateChartData = (data) => {
    const labels = Object.keys(data);
    const focusedCounts = labels.map((name) => data[name].focused);
    const distractedCounts = labels.map((name) => data[name].distracted);
    const phoneUsageCounts = labels.map((name) => data[name].phone_usage);
    const sleepingCounts = labels.map((name) => data[name].sleeping);

    setChartData({
      labels,
      datasets: [
        { 
          label: "Focused", 
          data: focusedCounts, 
          backgroundColor: "rgba(34, 197, 94, 0.7)",
          borderColor: "rgb(34, 197, 94)",
          borderWidth: 1
        },
        { 
          label: "Distracted", 
          data: distractedCounts, 
          backgroundColor: "rgba(239, 68, 68, 0.7)",
          borderColor: "rgb(239, 68, 68)",
          borderWidth: 1
        },
        { 
          label: "Phone Usage", 
          data: phoneUsageCounts, 
          backgroundColor: "rgba(59, 130, 246, 0.7)",
          borderColor: "rgb(59, 130, 246)",
          borderWidth: 1
        },
        { 
          label: "Sleeping", 
          data: sleepingCounts, 
          backgroundColor: "rgba(124, 58, 237, 0.7)",
          borderColor: "rgb(124, 58, 237)",
          borderWidth: 1
        },
      ],
    });
  };

  const openModal = (imageUrl) => {
    setSelectedImage(imageUrl);
    setModalIsOpen(true);
  };

  const closeModal = () => {
    setModalIsOpen(false);
  };

  // Filter students based on search input
  const filteredStudents = Object.entries(reportData).filter(([name]) =>
    name.toLowerCase().includes(searchTerm.toLowerCase())
  );

  // Pagination logic
  const indexOfLastStudent = currentPage * studentsPerPage;
  const indexOfFirstStudent = indexOfLastStudent - studentsPerPage;
  const currentStudents = filteredStudents.slice(indexOfFirstStudent, indexOfLastStudent);
  const totalPages = Math.ceil(filteredStudents.length / studentsPerPage);

  return (
    <div className="analyze-container">
      <div className="card">
        <h2>📊 Analyze Report</h2>

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
            <Bar 
              data={chartData} 
              options={{ 
                responsive: true,
                plugins: {
                  legend: {
                    position: 'top',
                  },
                  title: {
                    display: true,
                    text: 'Student Engagement Summary'
                  }
                },
                scales: {
                  y: {
                    beginAtZero: true
                  }
                }
              }} 
            />
          ) : (
            <p className="no-data">No data available for the selected filters</p>
          )}
        </div>
      </div>

      {/* Engagement & Attendance Details */}
      <div className="card mt-4">
        <h3>📜 Student Details</h3>
        {loading ? (
          <div className="loading-spinner">Loading...</div>
        ) : currentStudents.length > 0 ? (
          <>
            <ul className="student-list">
              {currentStudents.map(([name, details], index) => (
                <li key={index} className="student-item">
                  <div className="student-image">
                    {details.image_url ? (
                      <div className="image-viewer" onClick={() => openModal(details.image_url)}>
                        <FontAwesomeIcon icon={faEye} />
                      </div>
                    ) : (
                      <div className="no-image"></div>
                    )}
                  </div>
                  <div className="student-info">
                    <h4 className="student-name">{name}</h4>
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

            {/* Pagination Controls */}
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

      {/* Modal for displaying the image */}
      <Modal
        isOpen={modalIsOpen}
        onRequestClose={closeModal}
        style={customStyles}
        contentLabel="Enrolled Student Image"
      >
        <div className="modal-content">
          <img
            src={selectedImage}
            alt="Enrolled Student"
            className="modal-image"
          />
          <button
            onClick={closeModal}
            className="modal-close"
          >
            Close
          </button>
        </div>
      </Modal>
    </div>
  );
}

export default AnalyzeReport;