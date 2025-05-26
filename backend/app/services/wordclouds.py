# backend/app/services/wordclouds.py

from typing import List, Dict, Any
import pandas as pd
from collections import Counter
import re
from wordcloud import WordCloud
from io import BytesIO
import base64

class WordCloudService:
    def __init__(self):
        self.url_pattern = re.compile(r"https?://\S+")

    def _parse_dataframe(self, messages: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Convert messages to a DataFrame and sanitize text.
        """
        df = pd.DataFrame(messages)
        df["text"] = df["text"].fillna("").astype(str).str.lower()
        return df

    def _clean_words(self, texts: List[str]) -> List[str]:
        """
        Tokenize, remove URLs and short words.
        """
        words = " ".join(texts).split()
        return [word for word in words if not self.url_pattern.match(word) and len(word) > 2]

    def get_word_frequencies(self, messages: List[Dict[str, Any]], top_n: int = 100) -> Dict[str, int]:
        """
        Return the most common words and their counts.
        """
        df = self._parse_dataframe(messages)
        words = self._clean_words(df["text"].tolist())
        freq = dict(Counter(words).most_common(top_n))
        return freq

    def generate_wordcloud_base64(self, messages: List[Dict[str, Any]]) -> str:
        """
        Generate a word cloud image as a base64-encoded string.
        """
        freqs = self.get_word_frequencies(messages, top_n=200)
        if not freqs:
            return ""

        wc = WordCloud(width=800, height=400, background_color='white', colormap='viridis')
        wc.generate_from_frequencies(freqs)

        # Encode image to base64
        image_stream = BytesIO()
        wc.to_image().save(image_stream, format="PNG")
        image_stream.seek(0)
        encoded = base64.b64encode(image_stream.read()).decode("utf-8")

        return f"data:image/png;base64,{encoded}"
