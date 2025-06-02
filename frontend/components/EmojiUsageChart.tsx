import React from "react";

export type EmojiUsageChartProps = {
  user: string;
  emojiData: any;
};

const EmojiUsageChart: React.FC<EmojiUsageChartProps> = ({ user, emojiData }) => {
  // Render logic here (placeholder for now)
  return (
    <div className="my-4">
      <h2 className="text-xl font-semibold">Emoji Usage for {user}</h2>
      <pre className="text-sm text-gray-700">{JSON.stringify(emojiData, null, 2)}</pre>
    </div>
  );
};

export default EmojiUsageChart;
