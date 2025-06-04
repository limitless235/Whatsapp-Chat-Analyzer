from typing import List, Dict
import numpy as np
import torch
from sklearn.decomposition import PCA

from ..utils.model_loader import load_transformer_model_and_tokenizer


class PersonalityAnalyzer:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cpu"):
        self.device = device
        self.model_name = model_name
        self.model, self.tokenizer = load_transformer_model_and_tokenizer(self.model_name)
        self.model.to(self.device)
        self.model.eval()

    def compare_users(self, users: Dict[str, List[str]]) -> Dict[str, Dict[str, float]]:
        results = {}
        for user, msgs in users.items():
            results[user] = self.analyze_user_personality(msgs)
        return results


    def _embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Batch encode texts into embeddings using the transformer model.
        """
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=128)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                last_hidden = outputs.last_hidden_state
                attention_mask = inputs["attention_mask"].unsqueeze(-1)
                summed = (last_hidden * attention_mask).sum(dim=1)
                count = attention_mask.sum(dim=1)
                embeddings = (summed / count).cpu().numpy()
                all_embeddings.append(embeddings)

        return np.vstack(all_embeddings)

    def analyze_user_personality(self, user_messages: List[str]) -> Dict[str, float]:
        if not user_messages:
            return {}

        embeddings = self._embed_texts(user_messages)
        n_samples, n_features = embeddings.shape

        n_components = min(5, n_samples, n_features)
        if n_components == 0:
            # Not enough data to perform PCA
            return {}

        pca = PCA(n_components=n_components)
        reduced = pca.fit_transform(embeddings)
        trait_vector = reduced.mean(axis=0)

        # Create a fixed-size trait vector of length 5, fill missing with 0
        traits = np.zeros(5, dtype=float)
        traits[:len(trait_vector)] = trait_vector

        return {
            "expressiveness": float(traits[0]),
            "formality": float(traits[1]),
            "assertiveness": float(traits[2]),
            "emotional_variability": float(traits[3]),
            "cognitive_complexity": float(traits[4]),
        }

    def compare_users(self, users: Dict[str, List[str]]) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple users' personalities.
        Returns a dict mapping username to their trait vector.
        """
        results = {}
        for user, msgs in users.items():
            results[user] = self.analyze_user_personality(msgs)
        return results
import pandas as pd
from typing import Dict
from .personality import PersonalityAnalyzer
import logging

logger = logging.getLogger(__name__)

def generate_profiles(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Generates personality profiles per user from a DataFrame.
    Expects 'sender_name' and 'message' columns.
    """
    logger.info(f"DataFrame columns: {df.columns.tolist()}")

    # Normalize column names (optional but helpful)
    df.columns = df.columns.str.strip().str.lower()
    
    if "sender_name" not in df.columns or "text" not in df.columns:
        raise ValueError("Missing required columns 'sender_name' and/or 'message' in DataFrame.")

    messages_by_user = (
        df.dropna(subset=["sender_name", "text"])
          .groupby("sender_name")["text"]
          .apply(list)
          .to_dict()
    )

    analyzer = PersonalityAnalyzer()
    return analyzer.compare_users(messages_by_user)
