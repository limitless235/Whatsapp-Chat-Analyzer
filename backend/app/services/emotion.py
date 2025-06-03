from typing import List, Dict
import torch
import torch.nn.functional as F
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from nrclex import NRCLex

from ..utils.model_loader import load_transformer_model_and_tokenizer
from ..utils.cleaning import clean_text

class EmotionAnalyzer:
    def __init__(self, device: str = "cpu", batch_size: int = 32):
        self.device = device
        self.batch_size = batch_size
        
        # Use lazy loader for model/tokenizer
        model_name = "cardiffnlp/twitter-roberta-base-emotion"
        self.tokenizer, self.model = load_transformer_model_and_tokenizer(model_name, classification=True)
        self.model.to(self.device)
        self.model.eval()
        
        self.emotion_labels = ['anger', 'joy', 'optimism', 'sadness']

    def analyze_nrclex(self, texts: List[str]) -> List[Dict[str, int]]:
        """Analyze emotions lexicon-based for each text (NRCLex)."""
        results = []
        for text in texts:
            cleaned = clean_text(text)
            doc = NRCLex(cleaned)
            results.append(dict(doc.raw_emotion_scores))
        return results

    def analyze_roberta_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        """Batch analyze emotions using RoBERTa model."""
        all_probs = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = [clean_text(t) for t in texts[i:i + self.batch_size]]
            inputs = self.tokenizer(batch_texts, return_tensors="pt", truncation=True, padding=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = F.softmax(logits, dim=-1).cpu().numpy()
            
            for prob in probs:
                all_probs.append(dict(zip(self.emotion_labels, prob.tolist())))
        return all_probs

    def analyze(self, texts: List[str]) -> List[Dict[str, Dict]]:
        """
        Run NRCLex and RoBERTa analysis on a list of texts.
        Returns list of dicts with keys 'nrclex' and 'roberta'.
        """
        nrclex_results = self.analyze_nrclex(texts)
        roberta_results = self.analyze_roberta_batch(texts)
        
        combined = []
        for nrc, rob in zip(nrclex_results, roberta_results):
            combined.append({"nrclex": nrc, "roberta": rob})
        return combined


# ✅ Add this to fix the missing function error
def get_emotion_over_time(df: pd.DataFrame, device: str = "cpu", batch_size: int = 32) -> dict:
    """
    Run emotion analysis and compute per-user emotion distribution.
    Returns a dict of structure: {username: {emotion: average_score}}
    """
    analyzer = EmotionAnalyzer(device=device, batch_size=batch_size)
    emotion_scores = analyzer.analyze_roberta_batch(df["text"].tolist())
    
    df = df.copy()
    df["sender"] = df["sender_name"]
    df["emotion_scores"] = emotion_scores

    # Flatten emotion scores to columns
    emotion_df = pd.json_normalize(df["emotion_scores"])
    emotion_df["sender"] = df["sender"]

    # Group by sender and take mean of each emotion score
    grouped = emotion_df.groupby("sender").mean(numeric_only=True)

    return grouped.to_dict(orient="index")
