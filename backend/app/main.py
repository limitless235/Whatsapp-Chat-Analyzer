from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import analyze    # <-- add 'app.' prefix here
import logging
logging.basicConfig(level=logging.INFO)


app = FastAPI(
    title="WhatsApp Chat Analyzer",
    description="API for processing and analyzing WhatsApp chat exports.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analyze.router, prefix="/api")
