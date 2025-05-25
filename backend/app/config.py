# app/config.py
from pydantic import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str = "WhatsApp Chat Analyzer"
    DEBUG: bool = True

    class Config:
        env_file = ".env"

settings = Settings()
