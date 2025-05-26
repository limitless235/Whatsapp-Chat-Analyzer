from typing import List, Optional, Dict
from pydantic import BaseModel

# === Request ===

class ChatUploadRequest(BaseModel):
    chat_text: str


# === Personality ===

class PersonalityTraits(BaseModel):
    openness: float
    conscientiousness: float
    extraversion: float
    agreeableness: float
    neuroticism: float


# === Sentiment & Emotion ===

class SentimentScores(BaseModel):
    vader: float
    roberta: float
    average: float

class EmotionScores(BaseModel):
    joy: float
    sadness: float
    anger: float
    fear: float
    surprise: float
    disgust: float
    trust: float
    anticipation: float


# === Heatmaps ===

class ActivityHeatmap(BaseModel):
    daily: Dict[str, int]         # e.g. {"Monday": 300}
    hourly: Dict[int, int]        # e.g. {0: 45, 13: 120}
    monthly: Dict[str, int]       # e.g. {"Jan": 400}


# === Wordcloud ===

class WordCloudData(BaseModel):
    top_words: List[str]
    frequencies: Dict[str, int]


# === User Stats ===

class UserStats(BaseModel):
    total_messages: int
    total_words: int
    avg_words_per_message: float
    media_messages: int


# === Clustering & Projection ===

class ProjectionPoint(BaseModel):
    x: float
    y: float
    cluster_label: int


# === Interaction Graph ===

class GraphEdge(BaseModel):
    source: str
    target: str
    weight: float

class GraphData(BaseModel):
    edges: List[GraphEdge]
    nodes: List[str]


# === Named Entity Recognition ===

class NamedEntity(BaseModel):
    text: str
    label: str
    start_char: int
    end_char: int

class UserNERResult(BaseModel):
    user: str
    entities: List[NamedEntity]


# === Toxicity Classification for Chat Chunks ===

class ChatChunkToxicity(BaseModel):
    chunk_id: int
    text: str
    toxicity_score: float
    toxic: bool


# === Mood Drift Over Time ===

class TimeSeriesPoint(BaseModel):
    timestamp: str     # ISO 8601 datetime string
    value: float

class MoodDriftSeries(BaseModel):
    sentiment: List[TimeSeriesPoint]
    emotions: Dict[str, List[TimeSeriesPoint]]  # key = emotion name, e.g. 'joy'


# === User Result ===

class UserAnalysisResult(BaseModel):
    name: str
    sentiment: Optional[SentimentScores]
    emotions: Optional[EmotionScores]
    toxicity_score: Optional[float]
    personality: Optional[PersonalityTraits]
    cluster_label: Optional[int]
    umap_projection: Optional[ProjectionPoint]
    activity_heatmap: Optional[ActivityHeatmap]
    wordcloud: Optional[WordCloudData]
    stats: Optional[UserStats]
    ner: Optional[UserNERResult]
    mood_drift: Optional[MoodDriftSeries]
    chunk_toxicity: Optional[List[ChatChunkToxicity]]


# === Global Result ===

class GlobalAnalysisResult(BaseModel):
    users: List[UserAnalysisResult]
    interaction_graph: Optional[GraphData]
    clusters: Optional[Dict[int, List[str]]]  # e.g. {0: ['Alice', 'Bob'], 1: ['Charlie']}
    overall_sentiment: Optional[SentimentScores]


# === Response ===

class AnalysisResponse(BaseModel):
    summary: GlobalAnalysisResult
