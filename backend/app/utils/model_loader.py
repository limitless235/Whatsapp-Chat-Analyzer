# backend/app/utils/model_loader.py

import logging
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
import torch

log = logging.getLogger(__name__)

# Cache dicts to avoid reloading
_model_cache = {}
_tokenizer_cache = {}

def load_transformer_model_and_tokenizer(model_name: str, 
                                         classification: bool = False,
                                         device: str = None):
    """
    Load a transformer model and tokenizer with caching.
    
    Args:
        model_name (str): HuggingFace model name.
        classification (bool): If True, loads AutoModelForSequenceClassification else AutoModel.
        device (str): Device to place model on ('cpu' or 'cuda'). Default auto.

    Returns:
        tokenizer, model
    """
    global _model_cache, _tokenizer_cache

    if model_name in _model_cache and model_name in _tokenizer_cache:
        model = _model_cache[model_name]
        tokenizer = _tokenizer_cache[model_name]
        log.info(f"Loaded cached model & tokenizer: {model_name}")
    else:
        log.info(f"Loading model & tokenizer: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if classification:
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
        else:
            model = AutoModel.from_pretrained(model_name)  # ✅ CORRECTED HERE
        _model_cache[model_name] = model
        _tokenizer_cache[model_name] = tokenizer
        log.info(f"Loaded model & tokenizer: {model_name}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)
    model.eval()
    return tokenizer, model
