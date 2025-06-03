# backend/app/services/clustering.py
from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import torch

from ..utils.model_loader import load_transformer_model_and_tokenizer
from ..utils.cleaning import clean_text

class ClusteringService:
    def __init__(self, device: str = "cpu"):
        self.device = device
        
        # Model name
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"

        # Load model and tokenizer
        self.tokenizer, self.model = load_transformer_model_and_tokenizer(self.model_name)
        self.model.to(self.device)
        self.model.eval()

        # TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)

        self.kmeans_model = None

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Batch embed cleaned texts using the transformer model."""
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
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

    def fit_clusters(self, texts: List[str], n_clusters: int = 8) -> Tuple[np.ndarray, np.ndarray]:
        """Fit KMeans on combined TF-IDF and embedding features."""
        cleaned_texts = [clean_text(t) for t in texts]
        tfidf_features = self.tfidf_vectorizer.fit_transform(cleaned_texts)
        embeddings = self.embed_texts(cleaned_texts)
        combined_features = np.hstack([tfidf_features.toarray(), embeddings])
        self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = self.kmeans_model.fit_predict(combined_features)
        return cluster_labels, combined_features

    def predict_clusters(self, texts: List[str]) -> np.ndarray:
        if self.kmeans_model is None:
            raise RuntimeError("KMeans model not fitted yet.")
        cleaned_texts = [clean_text(t) for t in texts]
        tfidf_features = self.tfidf_vectorizer.transform(cleaned_texts)
        embeddings = self.embed_texts(cleaned_texts)
        combined_features = np.hstack([tfidf_features.toarray(), embeddings])
        return self.kmeans_model.predict(combined_features)


# ✅ Global instance and function used by analyze.py
_service = ClusteringService()

def perform_clustering(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze clustering on the provided DataFrame.
    Expects a 'clean_text' column in the DataFrame.
    """
    texts = df['clean_text'].tolist()
    labels, _ = _service.fit_clusters(texts)
    df['cluster'] = labels

    # Optional: summarize cluster counts
    cluster_counts = df['cluster'].value_counts().sort_index().to_dict()

    return {
        "cluster_labels": labels.tolist(),
        "cluster_counts": cluster_counts,
    }
