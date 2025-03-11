import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Navbar from "./components/Navbar";
import EnrollStudent from "./components/EnrollStudent";
import CaptureControl from "./components/CaptureControl";
import AttendanceView from "./components/AttendanceView";
import EngagementLogs from "./components/EngagementLogs";
import EngagementSummary from "./components/EngagementSummary";

function App() {
  return (
    <Router>
      <Navbar />
      <Routes>
        <Route path="/" element={<EnrollStudent />} />
        <Route path="/capture" element={<CaptureControl />} />
        <Route path="/attendance" element={<AttendanceView />} />
        <Route path="/engagement" element={<EngagementLogs />} />
        <Route path="/summary" element={<EngagementSummary />} />
      </Routes>
    </Router>
  );
}

export default App;
