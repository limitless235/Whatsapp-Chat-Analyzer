"use client";

import React from "react";

type SummaryCardsProps = {
  totalMessages: number;
  activeDays: number;
  averageSentiment: number; // -1 to 1 range
};

const SummaryCards: React.FC<SummaryCardsProps> = ({
  totalMessages,
  activeDays,
  averageSentiment,
}) => {
  // Convert sentiment to label and color
  const sentimentLabel =
    averageSentiment > 0.2
      ? "Positive"
      : averageSentiment < -0.2
      ? "Negative"
      : "Neutral";

  const sentimentColor =
    averageSentiment > 0.2
      ? "text-green-600"
      : averageSentiment < -0.2
      ? "text-red-600"
      : "text-yellow-600";

  return (
    <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 my-4">
      <div className="bg-white shadow rounded p-4 text-center">
        <h3 className="text-lg font-semibold">Total Messages</h3>
        <p className="text-2xl">{totalMessages.toLocaleString()}</p>
      </div>
      <div className="bg-white shadow rounded p-4 text-center">
        <h3 className="text-lg font-semibold">Active Days</h3>
        <p className="text-2xl">{activeDays}</p>
      </div>
      <div className="bg-white shadow rounded p-4 text-center">
        <h3 className="text-lg font-semibold">Average Sentiment</h3>
        <p className={`text-2xl font-bold ${sentimentColor}`}>
          {sentimentLabel} ({averageSentiment.toFixed(2)})
        </p>
      </div>
    </div>
  );
};

export default SummaryCards;
