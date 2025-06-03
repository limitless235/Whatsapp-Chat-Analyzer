import pandas as pd
import re
from datetime import datetime

SPLIT_FORMATS = {
    '12hr': r'\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s?[APap][mM]\s-\s',
    '24hr': r'\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s-\s'
}

DATETIME_FORMATS = {
    '12hr': '%d/%m/%Y, %I:%M %p - ',
    '24hr': '%d/%m/%Y, %H:%M - '
}

def parse_from_string(chat_str: str, format_key: str = '24hr') -> pd.DataFrame:
    """
    Parses raw WhatsApp chat string and returns a DataFrame with columns:
    ['date_time', 'message', 'sender_name']
    """
    user_msg = re.split(SPLIT_FORMATS[format_key], chat_str)[1:]
    date_time = re.findall(SPLIT_FORMATS[format_key], chat_str)

    df = pd.DataFrame({'date_time': date_time, 'user_msg': user_msg})
    df['date_time'] = pd.to_datetime(df['date_time'], format=DATETIME_FORMATS[format_key])

    sender_names = []
    messages = []

    for msg in df['user_msg']:
        match = re.match(r'^(.*?):\s(.*)', msg)
        if match:
            sender_names.append(match.group(1))
            messages.append(match.group(2))
        else:
            sender_names.append("group_notification")
            messages.append(msg)

    df['message'] = messages
    df['sender_name'] = sender_names
    df.drop(columns=['user_msg'], inplace=True)

    return df
