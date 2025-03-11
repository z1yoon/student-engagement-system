import React, { useState, useEffect } from "react";
import { getEngagementSummary } from "../api";
import { Bar } from "react-chartjs-2";

function EngagementSummary() {
  const [summary, setSummary] = useState({});

  useEffect(() => {
    getEngagementSummary().then(setSummary);
  }, []);

  const data = {
    labels: Object.keys(summary),
    datasets: [
      {
        label: "Focused",
        data: Object.values(summary).map((s) => s.focused),
        backgroundColor: "green",
      },
      {
        label: "Distracted",
        data: Object.values(summary).map((s) => s.distracted),
        backgroundColor: "red",
      },
    ],
  };

  return (
    <div>
      <h2>Engagement Summary</h2>
      <Bar data={data} />
    </div>
  );
}

export default EngagementSummary;
