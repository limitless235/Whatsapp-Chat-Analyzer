# app/services/emoji_usage.py

from typing import Dict
from collections import Counter
import pandas as pd
import emoji
import re


def extract_emojis(text: str, emoji_regex: re.Pattern) -> list:
    """Extract all emojis from a single string using regex."""
    if not isinstance(text, str):
        return []
    return emoji_regex.findall(text)


def emojiUsage(df: pd.DataFrame) -> Dict[str, list]:
    """
    Analyzes emoji usage across all messages.
    Returns a dictionary with the top 10 emojis and their counts + descriptions.
    """
    # Precompile emoji regex from emoji.EMOJI_DATA
    emojis_list = map(lambda x: ''.join(x.split()), emoji.EMOJI_DATA.keys())
    emoji_regex = re.compile('|'.join(re.escape(e) for e in emojis_list))

    emoji_ctr = Counter()

    for msg in df["text"].fillna(""):
        emojis_found = extract_emojis(msg, emoji_regex)
        emoji_ctr.update(emojis_found)

    top_emojis = emoji_ctr.most_common(10)

    return {
        "emoji": [e[0] for e in top_emojis],
        "emoji_count": [e[1] for e in top_emojis],
        "emoji_description": [emoji.demojize(e[0])[1:-1] for e in top_emojis],
    }
