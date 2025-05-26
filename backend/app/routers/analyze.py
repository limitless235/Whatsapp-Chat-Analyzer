import logging
from typing import Union, IO
import pandas as pd

from . import (
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
)
from ..utils import cleaning

log = logging.getLogger(__name__)

class ChatAnalyzer:
    def __init__(self, profanities_csv_path: Union[str, IO]):
        # Initialize profanity analyzer with CSV file or file-like object
        self.profanity_analyzer = profanity.ProfanityAnalyzer(profanities_csv_path)

    def analyze(self, chat_str: str) -> dict:
        log.info("Parsing WhatsApp chat...")
        df = file_parser.parse_from_string(chat_str)
        log.info(f"Parsed {len(df)} messages.")

        # Clean messages for profanity and other analyses
        df['clean_text'] = df['text'].apply(cleaning.clean_message_text)

        # Profanity detection flags & counts
        df['is_profane'] = df['clean_text'].apply(self.profanity_analyzer.contains_profanity)
        df['profanity_count'] = df['clean_text'].apply(self.profanity_analyzer.count_profanities)

        # Sentiment analysis
        log.info("Running sentiment analysis...")
        sentiment_result = sentiment.get_sentiment_over_time(df)

        # Emotion analysis
        log.info("Running emotion analysis...")
        emotion_result = emotion.get_emotion_over_time(df)

        # Clustering (TF-IDF + MiniLM KMeans)
        log.info("Running clustering analysis...")
        clustering_result = clustering.perform_clustering(df)

        # Style UMAP projection
        log.info("Generating style UMAP projection...")
        umap_result = style_umap.generate_umap_projection(df)

        # Interaction graph analysis (NetworkX)
        log.info("Generating interaction graph analysis...")
        graph_result = graph_analysis.generate_interaction_graph(df)

        # Activity heatmaps (daily, hourly, monthly)
        log.info("Generating activity heatmaps...")
        heatmaps_result = heatmaps.generate_heatmaps(df)

        # Message stats (counts, activity stats)
        log.info("Calculating message statistics...")
        stats_result = stats.compute_stats(df)

        # Word cloud generation
        log.info("Generating word clouds...")
        wordcloud_result = wordclouds.generate_wordclouds(df)

        # Personality profiles (Big Five radar)
        log.info("Generating personality profiles...")
        personality_result = personality.generate_profiles(df)

        # Combine all analysis results into one dictionary
        analysis_results = {
            "sentiment": sentiment_result,
            "emotions": emotion_result,
            "clustering": clustering_result,
            "umap": umap_result,
            "interaction_graph": graph_result,
            "heatmaps": heatmaps_result,
            "stats": stats_result,
            "wordclouds": wordcloud_result,
            "personality": personality_result,
        }

        log.info("Chat analysis complete.")
        return analysis_results
