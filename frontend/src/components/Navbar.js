import React from "react";
import { Link } from "react-router-dom";

function Navbar() {
  return (
    <nav style={{ backgroundColor: "#007BFF", padding: "10px" }}>
      <ul style={{ display: "flex", listStyle: "none", gap: "15px", margin: 0, padding: 0 }}>
        <li><Link to="/" style={{ color: "white", textDecoration: "none" }}>🏠 Enroll Student</Link></li>
        <li><Link to="/capture" style={{ color: "white", textDecoration: "none" }}>📸 Capture Control</Link></li>
        <li><Link to="/analyze" style={{ color: "white", textDecoration: "none" }}>📊 Analyze Report</Link></li>
      </ul>
    </nav>
  );
}

export default Navbar;
