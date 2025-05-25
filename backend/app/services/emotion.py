from typing import List, Dict
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from nrclex import NRCLex
import torch.nn.functional as F

class EmotionAnalyzer:
    def __init__(self, device: str = "cpu", batch_size: int = 32):
        self.device = device
        self.batch_size = batch_size
        
        # Load RoBERTa emotion model + tokenizer from cardiffnlp
        model_name = "cardiffnlp/twitter-roberta-base-emotion"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        self.emotion_labels = ['anger', 'joy', 'optimism', 'sadness']

    def analyze_nrclex(self, texts: List[str]) -> List[Dict[str, int]]:
        """Analyze emotions lexicon-based for each text (NRCLex)."""
        results = []
        for text in texts:
            doc = NRCLex(text)
            results.append(dict(doc.raw_emotion_scores))
        return results

    def analyze_roberta_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        """Batch analyze emotions using RoBERTa model."""
        all_probs = []
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            inputs = self.tokenizer(batch_texts, return_tensors="pt", truncation=True, padding=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = F.softmax(logits, dim=-1).cpu().numpy()
            
            # Map probabilities to labels
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


# Example usage:
# analyzer = EmotionAnalyzer(device="cuda" if torch.cuda.is_available() else "cpu", batch_size=64)
# texts = ["I feel great!", "This is bad.", "Let's do it!", ...]  # list of messages
# results = analyzer.analyze(texts)
# print(results[0])  # Output for first message
