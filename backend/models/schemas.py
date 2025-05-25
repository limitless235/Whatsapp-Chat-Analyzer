# app/models/schemas.py
from typing import List, Dict, Optional
from pydantic import BaseModel

class MessageData(BaseModel):
    user: str
    week: str
    sentiment: float

class EmotionData(BaseModel):
    user: str
    week: str
    emotions: Dict[str, float]

class UMAPPoint(BaseModel):
    user: str
    x: float
    y: float
    cluster: int

class PersonalityProfile(BaseModel):
    user: str
    traits: Dict[str, float]

class AnalysisResponse(BaseModel):
    sentiment: List[MessageData]
    emotions: List[EmotionData]
    umap: List[UMAPPoint]
    personality: List[PersonalityProfile]
