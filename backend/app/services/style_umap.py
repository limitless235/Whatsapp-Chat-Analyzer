# backend/app/services/style_umap.py

from typing import List, Tuple
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from umap import UMAP

from ..utils.model_loader import load_transformer_model_and_tokenizer

class StyleUMAP:
    def __init__(self, device: str = "cpu"):
        self.device = device

        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.tokenizer, self.model = None, None
        self._load_embedding_model()

        self.tfidf_vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        self.umap_model = None

    def _load_embedding_model(self):
        self.tokenizer, self.model = load_transformer_model_and_tokenizer(self.model_name)
        self.model.to(self.device)
        self.model.eval()

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = self.tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                last_hidden_state = outputs.last_hidden_state
                attention_mask = inputs['attention_mask'].unsqueeze(-1)
                masked_embeddings = last_hidden_state * attention_mask
                summed = masked_embeddings.sum(dim=1)
                counts = attention_mask.sum(dim=1)
                batch_embeddings = (summed / counts).cpu().numpy()
                embeddings.append(batch_embeddings)

        return np.vstack(embeddings)

    def combine_features(self, texts: List[str]) -> np.ndarray:
        tfidf = self.tfidf_vectorizer.fit_transform(texts)
        embeddings = self.embed_texts(texts)
        return np.hstack([tfidf.toarray(), embeddings])

    def reduce(self, texts: List[str], n_components: int = 2, n_neighbors: int = 15, min_dist: float = 0.1) -> np.ndarray:
        """
        Perform UMAP reduction to n_components dimensions (2D or 3D).
        Returns array of shape (n_samples, n_components).
        """
        combined_features = self.combine_features(texts)

        self.umap_model = UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
        reduced = self.umap_model.fit_transform(combined_features)

        return reduced

    def transform_new_texts(self, texts: List[str]) -> np.ndarray:
        """
        Transform new texts using already-fitted UMAP model.
        Returns reduced vectors if model is fitted.
        """
        if self.umap_model is None:
            raise RuntimeError("UMAP model not fitted yet. Call reduce() first.")

        tfidf = self.tfidf_vectorizer.transform(texts)
        embeddings = self.embed_texts(texts)
        combined_features = np.hstack([tfidf.toarray(), embeddings])
        return self.umap_model.transform(combined_features)
