import pandas as pd
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from typing import Union, Tuple, IO


class ProfanityAnalyzer:
    def __init__(self, profanity_csv_path: str):
        """
        Initialize ProfanityAnalyzer by loading custom profanity words and
        setting up VADER sentiment analyzer.
        """
        # Load custom profanity words once and lowercase them
        profanity_words_raw = pd.read_csv(profanity_csv_path, header=None)[0].dropna()
        self.custom_profanities = set(profanity_words_raw.astype(str).str.lower().tolist())

        # Initialize VADER sentiment analyzer once
        self.vader = SentimentIntensityAnalyzer()

        # Compile WhatsApp chat message regex pattern
        self.chat_line_pattern = re.compile(
            r'^(\d{1,2}/\d{1,2}/\d{2,4}),\s(\d{1,2}:\d{2}(?:\s?[APMapm]{2})?)\s-\s([^:]+):\s(.*)'
        )

    def parse_whatsapp_chat(self, file: Union[str, IO]) -> pd.DataFrame:
        """
        Parse a WhatsApp exported chat file.
        Accepts a file path or file-like object.
        Returns a DataFrame with columns: date, time, user, text.
        """
        if isinstance(file, str):
            with open(file, encoding="utf-8") as f:
                lines = f.readlines()
        else:
            # Assume file-like object (e.g., BytesIO)
            lines = file.read().decode("utf-8").splitlines()

        messages = []
        for line in lines:
            match = self.chat_line_pattern.match(line)
            if match:
                date, time, user, message = match.groups()
                messages.append([date, time, user, message])
            elif messages:
                # Append multiline message continuation to the last message
                messages[-1][3] += ' ' + line.strip()

        return pd.DataFrame(messages, columns=['date', 'time', 'user', 'text'])

    def analyze_message(self, text: str) -> dict:
        """
        Analyze a single message for sentiment (VADER) and profanity (custom list).
        Returns a dict with VADER scores and profanity flag.
        """
        text_lower = text.lower()
        vader_scores = self.vader.polarity_scores(text)
        is_profane = any(word in text_lower for word in self.custom_profanities)

        return {
            'vader_neg': vader_scores['neg'],
            'vader_neu': vader_scores['neu'],
            'vader_pos': vader_scores['pos'],
            'vader_compound': vader_scores['compound'],
            'is_profane': is_profane
        }

    def analyze_chat(self, file: Union[str, IO]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Parse and analyze the entire WhatsApp chat export file.
        Returns a tuple of:
          - messages DataFrame with sentiment and profanity columns added
          - user summary DataFrame grouped by user with aggregated stats
        """
        chat_df = self.parse_whatsapp_chat(file)
        analysis_results = chat_df['text'].apply(self.analyze_message)
        analysis_df = pd.DataFrame(list(analysis_results))
        chat_df = pd.concat([chat_df, analysis_df], axis=1)

        user_summary = chat_df.groupby('user').agg({
            'vader_neg': 'mean',
            'vader_neu': 'mean',
            'vader_pos': 'mean',
            'vader_compound': 'mean',
            'is_profane': 'sum',
            'text': 'count'
        }).rename(columns={'text': 'total_messages', 'is_profane': 'profanities_detected'})

        return chat_df, user_summary

    def contains_profanity(self, text: str) -> bool:
        """
        Check if the text contains any profane words.
        """
        text_lower = text.lower()
        return any(word in text_lower for word in self.custom_profanities)

    def count_profanities(self, text: str) -> int:
        """
        Count how many profane words are present in the text.
        """
        text_lower = text.lower()
        return sum(word in text_lower for word in self.custom_profanities)
