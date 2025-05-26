# backend/app/utils/cleaning.py

import re
import pandas as pd

class ChatCleaner:
    def __init__(self, remove_system_messages: bool = True):
        self.remove_system_messages = remove_system_messages
        self.system_msg_patterns = [
            r"Messages to this chat and calls are now secured with end-to-end encryption.",
            r"You changed this group's icon",
            r"You changed this group's subject",
            r".*deleted this message.*",
            r".*joined using this group's invite link.*",
            r".*left the group.*",
            r".*created group .*",
            r".*removed .*",
        ]
        self.system_msg_regex = re.compile("|".join(self.system_msg_patterns), flags=re.IGNORECASE)

        self.url_regex = re.compile(r"https?://\S+|www\.\S+")
        self.emoji_pattern = re.compile(
            "[" 
            "\U0001F600-\U0001F64F"
            "\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF"
            "\U0001F1E0-\U0001F1FF"
            "]+", flags=re.UNICODE
        )

    def clean_message(self, text: str) -> str:
        text = self.url_regex.sub("", text)
        text = self.emoji_pattern.sub("", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def filter_system_message(self, text: str) -> bool:
        if not self.remove_system_messages:
            return False
        return bool(self.system_msg_regex.search(text))

    def clean_chat_df(self, chat_df: pd.DataFrame) -> pd.DataFrame:
        if self.remove_system_messages:
            is_system = chat_df['text'].apply(self.filter_system_message)
            chat_df = chat_df.loc[~is_system].copy()

        chat_df['text'] = chat_df['text'].apply(self.clean_message)
        chat_df = chat_df[chat_df['text'].str.strip() != ''].copy()

        return chat_df
