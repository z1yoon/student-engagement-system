import axios from "axios";

const API_BASE = "http://localhost:8000/api";

export async function enrollStudent(studentName, dataUrl) {
  const payload = { student_name: studentName, image_data: dataUrl };
  return axios.post(`${API_BASE}/enroll-student/`, payload).then((res) => res.data);
}

export async function startCapture() {
  return axios.post(`${API_BASE}/start-capture`).then((res) => res.data);
}

export async function stopCapture() {
  return axios.post(`${API_BASE}/stop-capture`).then((res) => res.data);
}

export async function getCaptureStatus() {
  return axios.get(`${API_BASE}/capture-status`).then((res) => res.data);
}

export async function getAnalyzeResults(startDate = "", endDate = "") {
  return axios
    .get(`${API_BASE}/analyze_results`, {
      params: { start_date: startDate, end_date: endDate },
    })
    .then((res) => res.data);
}

export async function getLatestAnalysis() {
  return axios.get(`${API_BASE}/latest_analysis`).then((res) => res.data);
}
