"""
AI Archaeologist - Industrial Incident Investigation Platform
FastAPI Backend Main Application.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.core.config import settings
from app.database.connection import init_db
from app.api.v1.incidents import router as incidents_router
from app.api.v1.timeline import router as timeline_router
from app.api.v1.reconstruction import router as reconstruction_router
from app.api.v1.evidence import router as evidence_router
from app.api.v1.graph import router as graph_router
from app.api.v1.model import router as model_router
from app.api.v1.sensor import router as sensor_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize Database Tables
    init_db()
    print("AI Archaeologist Database initialized successfully.")
    yield
    # Shutdown

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="AI Archaeologist: Multimodal Industrial Incident Reconstruction & Forensic Reasoning Engine",
    lifespan=lifespan
)

# CORS Configuration for React Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API Routers
app.include_router(incidents_router, prefix=settings.API_V1_STR)
app.include_router(timeline_router, prefix=settings.API_V1_STR)
app.include_router(reconstruction_router, prefix=settings.API_V1_STR)
app.include_router(evidence_router, prefix=settings.API_V1_STR)
app.include_router(graph_router, prefix=settings.API_V1_STR)
app.include_router(model_router, prefix=settings.API_V1_STR)
app.include_router(sensor_router, prefix=settings.API_V1_STR)

@app.get("/")
def root():
    return {
        "name": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "status": "operational",
        "docs_url": "/docs"
    }
