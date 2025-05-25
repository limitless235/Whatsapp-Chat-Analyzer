# app/routers/analyze.py
from fastapi import APIRouter, UploadFile, File
from services import file_parser, sentiment, emotion, style_umap, personality
from models.schemas import AnalysisResponse

import pandas as pd
import io

router = APIRouter(tags=["Analyze"])

@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_chat(chat: UploadFile = File(...), profanities: UploadFile = File(...)):
    chat_bytes = await chat.read()
    csv_bytes = await profanities.read()

    # Decode and parse
    chat_str = chat_bytes.decode('utf-8')
    df = file_parser.parse_from_string(chat_str)

    # Run analytics
    sentiment_data = sentiment.get_sentiment_over_time(df)
    emotion_data = emotion.get_emotion_over_time(df)
    umap_data = style_umap.generate_umap_projection(df)
    personality_data = personality.generate_profiles(df)

    return {
        "sentiment": sentiment_data,
        "emotions": emotion_data,
        "umap": umap_data,
        "personality": personality_data
    }
