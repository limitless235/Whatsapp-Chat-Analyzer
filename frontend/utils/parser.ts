// Example types for parsed data, adjust to your actual API response
export interface SentimentDataPoint {
  date: string;
  sentiment_score: number;
}

export interface EmotionDataPoint {
  date: string;
  emotions: Record<string, number>;
}

export interface UserSummary {
  userId: string;
  username: string;
  totalMessages: number;
  avgSentiment: number;
  toxicityScore: number;
}

// Parsing sentiment response
export function parseSentimentResponse(data: any): SentimentDataPoint[] {
  // Assuming data.sentiment is an array of { date: string, score: number }
  if (!data || !Array.isArray(data.sentiment)) return [];
  return data.sentiment.map((item: any) => ({
    date: item.date,
    sentiment_score: item.score,
  }));
}

// Parsing emotion response
export function parseEmotionResponse(data: any): EmotionDataPoint[] {
  if (!data || !Array.isArray(data.emotions)) return [];
  return data.emotions.map((item: any) => ({
    date: item.date,
    emotions: item.values,
  }));
}

// Parsing user summary (example)
export function parseUserSummary(data: any): UserSummary[] {
  if (!data || !Array.isArray(data.users)) return [];
  return data.users.map((user: any) => ({
    userId: user.id,
    username: user.name,
    totalMessages: user.message_count,
    avgSentiment: user.avg_sentiment,
    toxicityScore: user.toxicity,
  }));
}
