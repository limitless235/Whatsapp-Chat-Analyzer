"use client";

import React from "react";
import Plot from "react-plotly.js";
import PlotWrapper from "./PlotWrapper";

type EmojiUsageChartProps = {
  emojiCounts: { emoji: string; count: number }[];
  user: string;
};

const EmojiUsageChart: React.FC<EmojiUsageChartProps> = ({ emojiCounts, user }) => {
  if (!emojiCounts || emojiCounts.length === 0) return null;

  // Sort descending and take top 10
  const topEmojis = emojiCounts
    .sort((a, b) => b.count - a.count)
    .slice(0, 10);

  return (
    <PlotWrapper title={`Top 10 Emojis Used by ${user}`}>
      <Plot
        data={[
          {
            x: topEmojis.map((e) => e.emoji),
            y: topEmojis.map((e) => e.count),
            type: "bar",
            marker: { color: "#636efa" },
          },
        ]}
        layout={{
          margin: { t: 30, b: 50 },
          xaxis: { title: "Emoji" },
          yaxis: { title: "Count" },
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

export default EmojiUsageChart;
