import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Navbar from "./components/Navbar";
import EnrollStudent from "./components/EnrollStudent";
import CaptureControl from "./components/CaptureControl";
import AnalyzeReport from "./components/AnalyzeReport";
import "./styles/global.css";

function App() {
  return (
    <Router>
      <div className="app">
        <Navbar />
        <div className="container">
          <Routes>
            <Route path="/" element={<EnrollStudent />} />
            <Route path="/capture" element={<CaptureControl />} />
            <Route path="/analyze" element={<AnalyzeReport />} />
          </Routes>
        </div>
      </div>
    </Router>
  );
}

export default App;
