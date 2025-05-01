import React from "react";
import { Link, useLocation } from "react-router-dom";

function Navbar() {
  const location = useLocation();

  const isActive = (path) => {
    return location.pathname === path ? "active" : "";
  };

  return (
    <nav className="navbar">
      <div className="container">
        <div className="navbar-brand">
          <h1>Student Engagement</h1>
        </div>
        <ul className="navbar-menu">
          <li className={`navbar-item ${isActive("/")}`}>
            <Link to="/" className="navbar-link">
              <span className="navbar-icon">🏫</span>
              <span>Enroll</span>
            </Link>
          </li>
          <li className={`navbar-item ${isActive("/capture")}`}>
            <Link to="/capture" className="navbar-link">
              <span className="navbar-icon">📸</span>
              <span>Capture</span>
            </Link>
          </li>
          <li className={`navbar-item ${isActive("/analyze")}`}>
            <Link to="/analyze" className="navbar-link">
              <span className="navbar-icon">📊</span>
              <span>Analyze</span>
            </Link>
          </li>
        </ul>
      </div>
    </nav>
  );
}

export default Navbar;
