from typing import List, Dict
import numpy as np
import torch
from sklearn.decomposition import PCA

from ..utils.model_loader import load_transformer_model_and_tokenizer


class PersonalityAnalyzer:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cpu"):
        self.device = device
        self.model_name = model_name
        self.tokenizer, self.model = load_transformer_model_and_tokenizer(self.model_name)
        self.model.to(self.device)
        self.model.eval()

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
        """
        Analyze a single user's messages and return a personality feature vector.
        Uses PCA on sentence embeddings to simulate stylistic traits.
        """
        if not user_messages:
            return {}

        embeddings = self._embed_texts(user_messages)
        pca = PCA(n_components=5)
        reduced = pca.fit_transform(embeddings)
        trait_vector = reduced.mean(axis=0)

        # Named traits (customizable or mapped to Big Five)
        return {
            "expressiveness": float(trait_vector[0]),
            "formality": float(trait_vector[1]),
            "assertiveness": float(trait_vector[2]),
            "emotional_variability": float(trait_vector[3]),
            "cognitive_complexity": float(trait_vector[4]),
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
