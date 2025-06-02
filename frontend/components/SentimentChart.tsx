"use client";

import React from "react";
import Plot from "react-plotly.js";
import PlotWrapper from "./PlotWrapper";

type SentimentDataPoint = {
  date: string;
  sentiment: number;
};

type SentimentChartProps = {
  data: SentimentDataPoint[];
};

const SentimentChart: React.FC<SentimentChartProps> = ({ data }) => {
  const dates = data.map((d) => d.date);
  const sentiments = data.map((d) => d.sentiment);

  return (
    <PlotWrapper title="Sentiment Over Time">
      <Plot
        data={[
          {
            x: dates,
            y: sentiments,
            type: "scatter",
            mode: "lines+markers",
            marker: { color: "blue" },
            name: "Sentiment Score",
          },
        ]}
        layout={{
          autosize: true,
          margin: { t: 30 },
          xaxis: {
            title: "Date",
            type: "date",
          },
          yaxis: {
            title: "Sentiment Score",
            range: [-1, 1],
          },
          plot_bgcolor: "transparent",
          paper_bgcolor: "transparent",
          font: { color: "#1f2937" }, // Tailwind's gray-800
        }}
        config={{ responsive: true }}
        style={{ width: "100%", height: "100%" }}
      />
    </PlotWrapper>
  );
};

export default SentimentChart;
