# app/services/toxicity.py

from typing import List, Dict
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class VADERToxicityAnalyzer:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def compute_toxicity(self, text: str) -> float:
        """Returns toxicity score if compound sentiment is highly negative."""
        if not isinstance(text, str) or not text.strip():
            return 0.0
        sentiment = self.analyzer.polarity_scores(text)
        if sentiment["compound"] < -0.5:
            return abs(sentiment["compound"])
        return 0.0

    def compute_scores(self, texts: List[str]) -> List[float]:
        """Applies compute_toxicity to a list of texts."""
        return [self.compute_toxicity(t) for t in texts]


def get_toxicity_over_time(df: pd.DataFrame) -> Dict[str, List[float]]:
    """
    Returns a dictionary mapping each user to a list of their toxicity scores over time.
    """
    analyzer = VADERToxicityAnalyzer()
    df["toxicity_score"] = analyzer.compute_scores(df["text"].fillna(""))
    grouped = df.groupby("sender_name")["toxicity_score"].apply(list).to_dict()
    return grouped


def toxicity_result(df: pd.DataFrame) -> Dict[str, List[float]]:
    """
    Wrapper function to align with other services' function names.
    Returns per-user toxicity scores as a dictionary.
    """
    return get_toxicity_over_time(df)
