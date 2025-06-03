# backend/app/services/sentiment.py
import logging
from typing import List, Dict, Union

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np
import nltk
import pandas as pd

try:
    nltk.data.find("sentiment/vader_lexicon")
except LookupError:
    nltk.download("vader_lexicon")

from ..utils.model_loader import load_transformer_model_and_tokenizer
from ..utils.cleaning import clean_text

log = logging.getLogger(__name__)

# Initialize VADER analyzer once
vader_analyzer = SentimentIntensityAnalyzer()

# Globals for RoBERTa model & tokenizer, lazy-loaded
roberta_model = None
roberta_tokenizer = None

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
LABELS = ['negative', 'neutral', 'positive']

def load_roberta():
    global roberta_model, roberta_tokenizer
    if roberta_model is None or roberta_tokenizer is None:
        roberta_model, roberta_tokenizer = load_transformer_model_and_tokenizer(MODEL_NAME)
        roberta_model.eval()
        roberta_model.to('cuda' if torch.cuda.is_available() else 'cpu')
        log.info(f"Loaded transformer model and tokenizer: {MODEL_NAME}")

def vader_sentiment_batch(texts: List[str]) -> List[Dict[str, Union[str, float]]]:
    """
    Efficiently compute VADER sentiment for a list of texts.
    Returns list of dicts with compound score and simplified label.
    """
    results = []
    for text in texts:
        cleaned_text = clean_text(text)
        scores = vader_analyzer.polarity_scores(cleaned_text)
        compound = scores['compound']

        if compound >= 0.05:
            label = 'positive'
        elif compound <= -0.05:
            label = 'negative'
        else:
            label = 'neutral'

        results.append({
            "label": label,
            "compound": compound,
            "scores": scores
        })
    return results

def roberta_sentiment_batch(texts: List[str]) -> List[Dict]:
    """
    Batch RoBERTa sentiment for a list of texts.
    Returns a list of dicts with label and confidence scores.
    """
    load_roberta()

    device = next(roberta_model.parameters()).device
    cleaned_texts = [clean_text(message) for text in texts]

    inputs = roberta_tokenizer(cleaned_texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = roberta_model(**inputs)
        logits = outputs.logits

    probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()

    results = []
    for prob in probs:
        max_idx = np.argmax(prob)
        label = LABELS[max_idx]
        results.append({
            "label": label,
            "confidence_scores": {
                "negative": float(prob[0]),
                "neutral": float(prob[1]),
                "positive": float(prob[2]),
            }
        })

    return results

def combined_sentiment(texts: List[str]) -> List[Dict]:
    """
    Run both VADER and RoBERTa sentiment on a list of texts.
    Returns list of combined sentiment dicts per text.
    """
    vader_results = vader_sentiment_batch(texts)
    roberta_results = roberta_sentiment_batch(texts)

    combined = []
    for text, vader_res, roberta_res in zip(texts, vader_results, roberta_results):
        combined.append({
            "text": text,
            "vader": vader_res,
            "roberta": roberta_res
        })

    return combined

def get_sentiment_over_time(df: pd.DataFrame, date_col: str = "date", text_col: str = "text", model: str = "vader") -> pd.DataFrame:
    """
    Compute sentiment over time (grouped by date) using VADER or RoBERTa.
    Returns a DataFrame with sentiment values aggregated per date.
    """
    # Validate required columns exist
    missing_cols = [col for col in [text_col, date_col] if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required column(s) in DataFrame: {missing_cols}")

    # Extract text column
    texts = df[text_col].fillna("").astype(str).tolist()

    # Convert and validate date column
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    df["date_only"] = df[date_col].dt.date

    if model == "vader":
        vader_results = vader_sentiment_batch(texts)
        df["compound"] = [res["compound"] for res in vader_results]
        return df.groupby("date_only")["compound"].mean().reset_index(name="avg_compound")

    elif model == "roberta":
        roberta_results = roberta_sentiment_batch(texts)
        df["positive"] = [res["confidence_scores"]["positive"] for res in roberta_results]
        df["neutral"] = [res["confidence_scores"]["neutral"] for res in roberta_results]
        df["negative"] = [res["confidence_scores"]["negative"] for res in roberta_results]
        return df.groupby("date_only")[["positive", "neutral", "negative"]].mean().reset_index()

    else:
        raise ValueError(f"Unsupported model type: {model}")
