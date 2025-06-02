"use client";

import React from "react";
import Plot from "react-plotly.js";
import PlotWrapper from "./PlotWrapper";

type HeatmapChartProps = {
  zData: number[][]; // matrix of counts (7 rows for days, 24 cols for hours)
  user: string;
};

const dayLabels = [
  "Monday",
  "Tuesday",
  "Wednesday",
  "Thursday",
  "Friday",
  "Saturday",
  "Sunday",
];

const hourLabels = Array.from({ length: 24 }, (_, i) => `${i}:00`);

const HeatmapChart: React.FC<HeatmapChartProps> = ({ zData, user }) => {
  if (!zData || zData.length === 0) return null;

  return (
    <PlotWrapper title={`Activity Heatmap: ${user}`}>
      <Plot
        data={[
          {
            z: zData,
            x: hourLabels,
            y: dayLabels,
            type: "heatmap",
            colorscale: "YlGnBu",
            reversescale: false,
          },
        ]}
        layout={{
          margin: { t: 30, b: 30 },
          xaxis: { title: "Hour of Day" },
          yaxis: { title: "Day of Week" },
          paper_bgcolor: "transparent",
          plot_bgcolor: "transparent",
          font: { color: "#1f2937" },
        }}
        config={{ responsive: true }}
        style={{ width: "100%", height: "100%" }}
      />
    </PlotWrapper>
  );
};

export default HeatmapChart;
