"use client";

import React from "react";
import Plot from "react-plotly.js";
import PlotWrapper from "./PlotWrapper";

type ToxicityPoint = {
  date: string;
  toxicity: number;
};

type ToxicityChartProps = {
  data: ToxicityPoint[];
};

const ToxicityChart: React.FC<ToxicityChartProps> = ({ data }) => {
  if (data.length === 0) return null;

  const dates = data.map((d) => d.date);
  const scores = data.map((d) => d.toxicity);

  return (
    <PlotWrapper title="Toxicity Over Time">
      <Plot
        data={[
          {
            x: dates,
            y: scores,
            type: "scatter",
            mode: "lines+markers",
            name: "Toxicity",
            line: { color: "#ef4444" },
            marker: { size: 6 },
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
            title: "Avg Toxicity Score",
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

export default ToxicityChart;
