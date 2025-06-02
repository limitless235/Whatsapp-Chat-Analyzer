"use client";

import React from "react";
import Plot from "react-plotly.js";
import PlotWrapper from "./PlotWrapper";

type EmotionDataPoint = {
  date: string;
  [emotion: string]: string | number; // e.g., joy, anger, fear, etc.
};

type EmotionChartProps = {
  data: EmotionDataPoint[];
};

const EmotionChart: React.FC<EmotionChartProps> = ({ data }) => {
  if (data.length === 0) return null;

  const dates = data.map((d) => d.date);
  const emotionKeys = Object.keys(data[0]).filter((k) => k !== "date");

  const traces = emotionKeys.map((emotion) => ({
    x: dates,
    y: data.map((d) => d[emotion] as number),
    stackgroup: "one",
    name: emotion,
    type: "scatter",
    mode: "none",
  }));

  return (
    <PlotWrapper title="Emotion Drift Over Time">
      <Plot
        data={traces}
        layout={{
          autosize: true,
          margin: { t: 30 },
          xaxis: {
            title: "Date",
            type: "date",
          },
          yaxis: {
            title: "Emotion Intensity",
            range: [0, 1],
          },
          showlegend: true,
          plot_bgcolor: "transparent",
          paper_bgcolor: "transparent",
          font: { color: "#1f2937" },
        }}
        config={{ responsive: true }}
        style={{ width: "100%", height: "100%" }}
      />
    </PlotWrapper>
  );
};

export default EmotionChart;
