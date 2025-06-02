"use client";

import React from "react";
import Plot from "react-plotly.js";
import PlotWrapper from "./PlotWrapper";

type TraitScore = {
  trait: string;
  score: number; // between 0 and 1
};

type RadarPersonalityProps = {
  data: TraitScore[];
  user: string;
};

const RadarPersonality: React.FC<RadarPersonalityProps> = ({ data, user }) => {
  if (!data || data.length === 0) return null;

  const traits = data.map((d) => d.trait);
  const scores = data.map((d) => d.score);

  // Close the radar loop
  traits.push(traits[0]);
  scores.push(scores[0]);

  return (
    <PlotWrapper title={`Big Five Personality: ${user}`}>
      <Plot
        data={[
          {
            type: "scatterpolar",
            r: scores,
            theta: traits,
            fill: "toself",
            name: user,
            line: { color: "#636EFA" },
            marker: { color: "#636EFA" },
          },
        ]}
        layout={{
          polar: {
            radialaxis: {
              visible: true,
              range: [0, 1],
              tickfont: { size: 10 },
            },
          },
          showlegend: false,
          margin: { t: 30, b: 30 },
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

export default RadarPersonality;
