import logging
from typing import Union, IO, Any
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
import datetime
from fastapi.encoders import jsonable_encoder  # ✅

from app.services import (
    file_parser,
    profanity,
    sentiment,
    emotion,
    clustering,
    style_umap,
    graph_analysis,
    heatmaps,
    stats,
    wordclouds,
    personality,
    toxicity
)
from ..utils import cleaning

log = logging.getLogger(__name__)
router = APIRouter()


class ChatAnalyzer:
    def __init__(self, profanities_csv_path: Union[str, IO]):
        self.profanity_analyzer = profanity.ProfanityAnalyzer(profanities_csv_path)

    def analyze(self, chat_str: str) -> dict:
        log.info("Parsing WhatsApp chat...")
        df = file_parser.parse_from_string(chat_str)
        log.info(f"Parsed {len(df)} messages.")

        # ✅ Standardize expected column names
        df = df.rename(columns={"message": "text", "date_time": "date"})

        df['clean_text'] = df['text'].apply(cleaning.clean_text)
        df['is_profane'] = df['clean_text'].apply(self.profanity_analyzer.contains_profanity)
        df['profanity_count'] = df['clean_text'].apply(self.profanity_analyzer.count_profanities)

        log.info("Running sentiment analysis...")
        sentiment_result = sentiment.get_sentiment_over_time(df)

        log.info("Running emotion analysis...")
        emotion_result = emotion.get_emotion_over_time(df)

        log.info("Running clustering analysis...")
        clustering_result = clustering.perform_clustering(df)

        log.info("Generating style UMAP projection...")
        texts = df["text"].dropna().astype(str).tolist()
        umap_result = style_umap.generate_umap_projection(texts)

        log.info("Generating interaction graph analysis...")
        graph_result = graph_analysis.generate_interaction_graph(df)

        log.info("Generating activity heatmaps...")
        heatmaps_result = heatmaps.generate_heatmaps(df)

        log.info("Calculating message statistics...")
        stats_result = stats.compute_stats(df)

        log.info("Generating word clouds...")
        wordcloud_result = wordclouds.generate_wordclouds(df)

        log.info("Generating personality profiles...")
        personality_result = personality.generate_profiles(df)

        log.info("Running toxicity detection...")
        toxicity_result = toxicity.toxicity_result(df)

        from app.services import emoji_usage

        emoji_data = emoji_usage.emojiUsage(df)


        # ✅ Explicitly extract user list for frontend
        user_list = list(personality_result.keys())

        analysis_results = {
            "sentiment": sentiment_result,
            "emotion": emotion_result,
            "toxicity": toxicity_result,  # ✅ ADD THIS
            "clustering": clustering_result,
            "umap": umap_result,
            "interaction_graph": graph_result,
            "heatmaps": heatmaps_result,
            "stats": stats_result,
            "wordclouds": wordcloud_result,
            "personality": personality_result,
            "emojiUsage": emoji_data,
            "users": user_list,  # ✅ Add this
            "users": df["sender_name"].dropna().unique().tolist(),
        }

        log.info("Chat analysis complete.")
        return self._make_json_serializable(analysis_results)

    def _make_json_serializable(self, obj: Any) -> Any:
        """ Recursively convert DataFrames and other non-serializable types into JSON-safe structures. """
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")
        elif isinstance(obj, (pd.Series, pd.Index)):
            return obj.tolist()
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, (pd.Timestamp, pd.Timedelta, datetime.datetime, datetime.date)):
            return str(obj)
        elif isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        else:
            return obj


# ✅ Create analyzer instance with path to profanities.csv
from pathlib import Path
csv_path = Path(__file__).resolve().parent.parent / "data" / "profanities.csv"
chat_analyzer = ChatAnalyzer(profanities_csv_path=csv_path)


@router.post("/analyze")
async def analyze_endpoint(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        chat_text = contents.decode("utf-8")
        result = chat_analyzer.analyze(chat_text)

        # ✅ Use FastAPI’s encoder to ensure JSON serialization
        json_safe_result = jsonable_encoder(result)
        return JSONResponse(content=json_safe_result)

    except Exception as e:
        log.error(f"Error analyzing chat: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})
