# backend/app/services/sentiment.py
import logging
from typing import List, Dict, Union

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np

from ..utils.model_loader import load_transformer_model_and_tokenizer

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
        scores = vader_analyzer.polarity_scores(text)
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
    inputs = roberta_tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
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
