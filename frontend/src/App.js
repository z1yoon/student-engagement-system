import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Navbar from "./components/Navbar";
import EnrollStudent from "./components/EnrollStudent";
import CaptureControl from "./components/CaptureControl";
import AnalyzeReport from "./components/AnalyzeReport";

function App() {
  return (
    <Router>
      <Navbar />
      <Routes>
        <Route path="/" element={<EnrollStudent />} />
        <Route path="/capture" element={<CaptureControl />} />
        <Route path="/analyze" element={<AnalyzeReport />} />
      </Routes>
    </Router>
  );
}

export default App;
