"use client";

import React from "react";
import Plot from "react-plotly.js";
import PlotWrapper from "./PlotWrapper";

type UMAPPoint = {
  x: number;
  y: number;
  cluster: number;
  user: string;
};

type UMAPClusterProps = {
  data: UMAPPoint[];
};

const clusterColors = [
  "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
  "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52",
];

const UMAPCluster: React.FC<UMAPClusterProps> = ({ data }) => {
  if (data.length === 0) return null;

  const clusters = Array.from(new Set(data.map((point) => point.cluster)));

  const traces = clusters.map((clusterId) => {
    const clusterPoints = data.filter((p) => p.cluster === clusterId);
    return {
      x: clusterPoints.map((p) => p.x),
      y: clusterPoints.map((p) => p.y),
      mode: "markers",
      type: "scattergl",
      name: `Cluster ${clusterId}`,
      text: clusterPoints.map((p) => p.user),
      marker: {
        color: clusterColors[clusterId % clusterColors.length],
        size: 6,
        line: { width: 0.5, color: "#1f2937" },
      },
    };
  });

  return (
    <PlotWrapper title="Style Clusters (UMAP Projection)">
      <Plot
        data={traces}
        layout={{
          autosize: true,
          margin: { t: 30 },
          xaxis: { title: "UMAP-1" },
          yaxis: { title: "UMAP-2" },
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

export default UMAPCluster;
