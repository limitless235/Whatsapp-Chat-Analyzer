"use client";

import React, { useState } from "react";
import dynamic from "next/dynamic";
import FileUpload from "../components/FileUpload";
import UserDropdown from "@/components/UserDropdown";
import axios from "../utils/api";

const SentimentChart = dynamic(() => import("../components/SentimentChart"), { ssr: false });
const EmotionChart = dynamic(() => import("../components/EmotionChart"), { ssr: false });
const ToxicityChart = dynamic(() => import("../components/ToxicityChart"), { ssr: false });
const UMAPCluster = dynamic(() => import("../components/UMAPCluster"), { ssr: false });
const RadarPersonality = dynamic(() => import("../components/RadarPersonality"), { ssr: false });
const HeatmapChart = dynamic(() => import("../components/HeatmapChart"), { ssr: false });
const EmojiUsageChart = dynamic(() => import("../components/EmojiUsageChart"), { ssr: false });
const SummaryCards = dynamic(() => import("../components/SummaryCards"), { ssr: false });

type AnalysisResult = {
  users: string[];
  sentiment: Record<string, any>;
  emotion: Record<string, any>;
  toxicity: Record<string, any>;
  umapClusters: any;
  personality: Record<string, any>;
  heatmaps: Record<string, any>;
  emojiUsage: Record<string, any>;
  summary: {
    totalMessages: number;
    activeDays: number;
    averageSentiment: number;
  };
};

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [selectedUser, setSelectedUser] = useState<string | null>(null);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = (file: File) => {
    setFile(file);
    setSelectedUser(null);
    setAnalysisResult(null);
    setError(null);
  };

  const handleAnalyze = async () => {
    if (!file) {
      setError("Please upload a chat file before analyzing.");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const formData = new FormData();
      formData.append("file", file);
      const response = await axios.post("/analyze", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setAnalysisResult(response.data);
      if (response.data.users.length > 0) {
        setSelectedUser(response.data.users[0]);
      }
    } catch (err: any) {
      setError(err.response?.data?.detail || "Failed to analyze file.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="max-w-7xl mx-auto p-4">
      <h1 className="text-3xl font-bold mb-4 text-center">WhatsApp Chat Analyzer</h1>

      <FileUpload onFileSelect={handleFileChange} />

      <button
        onClick={handleAnalyze}
        disabled={loading || !file}
        className="mt-4 px-6 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50"
      >
        {loading ? "Analyzing..." : "Analyze Chat"}
      </button>

      {error && <p className="mt-4 text-red-600 text-center">{error}</p>}

      {analysisResult && (
        <>
          <UserDropdown
            users={analysisResult.users}
            selectedUser={selectedUser}
            onChange={setSelectedUser}
          />

          <SummaryCards {...analysisResult.summary} />

          {selectedUser && (
            <>
              <SentimentChart data={analysisResult.sentiment[selectedUser]} />
              <EmotionChart data={analysisResult.emotion[selectedUser]} />
              <ToxicityChart data={analysisResult.toxicity[selectedUser]} />
              <UMAPCluster data={analysisResult.umapClusters} />
              <RadarPersonality
                user={selectedUser}
                data={analysisResult.personality[selectedUser]}
              />
              <HeatmapChart
                user={selectedUser}
                zData={analysisResult.heatmaps[selectedUser]}
              />
              <EmojiUsageChart
                user={selectedUser}
                emojiData={analysisResult.emojiUsage[selectedUser]}
              />
            </>
          )}
        </>
      )}
    </main>
  );
}
