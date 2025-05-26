# backend/app/services/clustering.py
from typing import List, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import torch

from ..utils.model_loader import get_transformer_model_and_tokenizer
from ..utils.cleaning import clean_text  # Import text cleaning function

class ClusteringService:
    def __init__(self, device: str = "cpu"):
        self.device = device
        
        # Model name
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"

        # Lazy load transformer model + tokenizer using model_loader.py
        self.tokenizer, self.model = get_transformer_model_and_tokenizer(self.model_name)
        self.model.to(self.device)
        self.model.eval()

        # TF-IDF vectorizer - fit on input texts later
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)

        self.kmeans_model = None

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Batch embed cleaned texts with transformer model."""
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            inputs = self.tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                last_hidden_state = outputs.last_hidden_state  # (batch_size, seq_len, hidden_dim)
                attention_mask = inputs['attention_mask'].unsqueeze(-1)
                masked_embeddings = last_hidden_state * attention_mask
                summed = masked_embeddings.sum(dim=1)
                counts = attention_mask.sum(dim=1)
                batch_embeddings = (summed / counts).cpu().numpy()
                embeddings.append(batch_embeddings)
        
        return np.vstack(embeddings)

    def fit_clusters(self, texts: List[str], n_clusters: int = 8) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit KMeans clusters on combined TF-IDF + embedding features.
        Returns tuple of (cluster_labels, feature_vectors)
        """
        # Step 0: Clean all texts before feature extraction
        cleaned_texts = [clean_text(t) for t in texts]

        # Step 1: Fit and transform TF-IDF on cleaned texts
        tfidf_features = self.tfidf_vectorizer.fit_transform(cleaned_texts)  # sparse matrix

        # Step 2: Embed all cleaned texts
        embeddings = self.embed_texts(cleaned_texts)  # numpy array

        # Step 3: Combine TF-IDF (dense) and embeddings
        combined_features = np.hstack([tfidf_features.toarray(), embeddings])

        # Step 4: Fit KMeans clustering
        self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = self.kmeans_model.fit_predict(combined_features)

        return cluster_labels, combined_features

    def predict_clusters(self, texts: List[str]) -> np.ndarray:
        """
        Predict clusters for new texts using existing KMeans model.
        """
        if self.kmeans_model is None:
            raise RuntimeError("KMeans model not fitted yet. Call fit_clusters() first.")

        # Clean texts before prediction
        cleaned_texts = [clean_text(t) for t in texts]

        tfidf_features = self.tfidf_vectorizer.transform(cleaned_texts)
        embeddings = self.embed_texts(cleaned_texts)

        combined_features = np.hstack([tfidf_features.toarray(), embeddings])

        return self.kmeans_model.predict(combined_features)
