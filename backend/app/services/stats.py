# backend/app/services/stats.py

from typing import List, Dict, Any
import pandas as pd
import numpy as np
from collections import Counter
import re

class StatsService:
    def __init__(self):
        self.url_pattern = re.compile(r"https?://\S+")

    def _parse_dataframe(self, messages: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Convert messages to DataFrame and parse timestamp.
        """
        df = pd.DataFrame(messages)
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df.dropna(subset=["timestamp"], inplace=True)
        df["text"] = df["text"].fillna("")
        return df

    def compute_basic_stats(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute basic chat statistics.
        """
        df = self._parse_dataframe(messages)
        total_messages = len(df)
        total_words = df["text"].str.split().map(len).sum()
        total_links = df["text"].str.count(self.url_pattern).sum()
        total_chars = df["text"].str.len().sum()
        avg_words_per_message = total_words / total_messages if total_messages > 0 else 0

        return {
            "total_messages": total_messages,
            "total_words": int(total_words),
            "total_characters": int(total_chars),
            "total_links": int(total_links),
            "avg_words_per_message": round(avg_words_per_message, 2)
        }

    def per_user_stats(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Compute message and word count stats per user.
        """
        df = self._parse_dataframe(messages)
        df["word_count"] = df["text"].str.split().map(len)

        grouped = df.groupby("sender").agg(
            total_messages=("text", "count"),
            total_words=("word_count", "sum"),
            avg_words_per_message=("word_count", "mean")
        ).reset_index()

        grouped["avg_words_per_message"] = grouped["avg_words_per_message"].round(2)

        return grouped.to_dict(orient="records")

    def most_common_words(self, messages: List[Dict[str, Any]], top_n: int = 20) -> List[Dict[str, Any]]:
        """
        Return top N most common words (excluding URLs).
        """
        df = self._parse_dataframe(messages)
        all_words = df["text"].str.lower().str.cat(sep=" ").split()
        filtered_words = [w for w in all_words if not self.url_pattern.match(w) and len(w) > 1]

        word_freq = Counter(filtered_words).most_common(top_n)

        return [{"word": word, "count": count} for word, count in word_freq]

    def message_length_distribution(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Returns histogram bins for message lengths.
        """
        df = self._parse_dataframe(messages)
        lengths = df["text"].str.len()

        counts, bins = np.histogram(lengths, bins=20)

        return {
            "bin_edges": bins.tolist(),
            "frequencies": counts.tolist()
        }
