# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import analyze

app = FastAPI(
    title="WhatsApp Chat Analyzer",
    description="API for processing and analyzing WhatsApp chat exports.",
    version="1.0.0"
)

# Allow frontend on localhost:3000 or deployed origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(analyze.router, prefix="/api")
