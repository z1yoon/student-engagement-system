import axios from "axios";

const API_BASE = "http://localhost:8000/api";

/**
 * Enrolls a student with their name and image.
 * @param {string} studentName - The name of the student.
 * @param {string} dataUrl - The base64 encoded image data.
 * @returns {Promise} API response data.
 */
export async function enrollStudent(studentName, dataUrl) {
  const payload = { student_name: studentName, image_data: dataUrl };
  return axios.post(`${API_BASE}/enroll-student/`, payload).then((res) => res.data);
}

/**
 * Starts real-time image capture and sends data to IoT Hub.
 * @returns {Promise} API response data.
 */
export async function startCapture() {
  return axios.post(`${API_BASE}/start-capture`).then((res) => res.data);
}

/**
 * Stops image capture while keeping engagement updates ongoing.
 * @returns {Promise} API response data.
 */
export async function stopCapture() {
  return axios.post(`${API_BASE}/stop-capture`).then((res) => res.data);
}

/**
 * Fetches the latest capture status (active/inactive).
 * @returns {Promise} API response data.
 */
export async function getCaptureStatus() {
  return axios.get(`${API_BASE}/capture-status`).then((res) => res.data);
}

/**
 * Fetches real-time engagement analysis results.
 * Supports optional date range filtering.
 * @param {string} [startDate] - Optional start date filter (YYYY-MM-DD).
 * @param {string} [endDate] - Optional end date filter (YYYY-MM-DD).
 * @returns {Promise} API response data.
 */
export async function getAnalyzeResults(startDate = "", endDate = "") {
  return axios.get(`${API_BASE}/analyze_results`, {
    params: { start_date: startDate, end_date: endDate },
  }).then((res) => res.data);
}
