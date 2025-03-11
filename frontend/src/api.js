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

export async function getAttendance() {
  return axios.get(`${API_BASE}/attendance`).then((res) => res.data);
}

export async function getEngagement() {
  return axios.get(`${API_BASE}/engagement`).then((res) => res.data);
}

export async function getEngagementSummary() {
  return axios.get(`${API_BASE}/summarize_engagement`).then((res) => res.data);
}
