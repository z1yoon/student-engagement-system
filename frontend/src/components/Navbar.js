import React from "react";
import { Link } from "react-router-dom";

function Navbar() {
  return (
    <nav>
      <ul>
        <li><Link to="/">Enroll Student</Link></li>
        <li><Link to="/capture">Capture Control</Link></li>
        <li><Link to="/attendance">Attendance</Link></li>
        <li><Link to="/engagement">Engagement Logs</Link></li>
        <li><Link to="/summary">Engagement Summary</Link></li>
      </ul>
    </nav>
  );
}

export default Navbar;
