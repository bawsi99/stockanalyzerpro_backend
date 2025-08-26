#!/usr/bin/env python3
"""
consolidated_service.py

Consolidated FastAPI service that combines both Data Service and Analysis Service
into a single application for deployment on Render.

This service mounts both services under different path prefixes:
- Data Service: /data/*
- Analysis Service: /analysis/*
- Root endpoints: / (health, etc.)
"""

import os
import sys
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Add the backend directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Create the main FastAPI app
app = FastAPI(
    title="Stock Analyzer Pro - Consolidated Service",
    description="Combined Data and Analysis Services for Stock Analysis",
    version="1.0.0"
)

# Load CORS origins from environment variable
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:8080,http://localhost:5173,http://127.0.0.1:3000,http://127.0.0.1:8080,http://127.0.0.1:5173,https://stock-analyzer-pro.vercel.app,https://stock-analyzer-pro-git-prototype-aaryan-manawats-projects.vercel.app,https://stock-analyzer-cl9o3tivx-aaryan-manawats-projects.vercel.app").split(",")
CORS_ORIGINS = [origin.strip() for origin in CORS_ORIGINS if origin.strip()]

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import both service apps
try:
    from data_service import app as data_app
    from analysis_service import app as analysis_app
    print("✅ Successfully imported both Data and Analysis services")
except ImportError as e:
    print(f"❌ Error importing services: {e}")
    raise

# Mount the Data Service under /data prefix
app.mount("/data", data_app)

# Mount the Analysis Service under /analysis prefix
app.mount("/analysis", analysis_app)

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint for the Consolidated Service."""
    return {
        "service": "Stock Analyzer Pro - Consolidated Service",
        "version": "1.0.0",
        "status": "running",
        "services": {
            "data_service": "/data",
            "analysis_service": "/analysis"
        },
        "endpoints": {
            "health": "/health",
            "data_service_health": "/data/health",
            "analysis_service_health": "/analysis/health",
            "stock_data": "/data/stock/{symbol}/history",
            "stock_analysis": "/analysis/analyze",
            "sector_list": "/analysis/sector/list",
            "websocket": "/data/ws/stream"
        },
        "timestamp": "2025-08-26T20:20:00Z"
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check for the consolidated service."""
    return {
        "status": "healthy",
        "service": "consolidated",
        "timestamp": "2025-08-26T20:20:00Z"
    }

# Redirect legacy endpoints to their new locations
@app.get("/stock/{symbol}/history")
async def redirect_stock_history(symbol: str, request: Request):
    """Redirect stock history requests to data service."""
    query_params = str(request.query_params)
    redirect_url = f"/data/stock/{symbol}/history{query_params}"
    return JSONResponse(
        status_code=307,
        content={"detail": f"Redirecting to {redirect_url}"},
        headers={"Location": redirect_url}
    )

@app.get("/sector/list")
async def redirect_sector_list():
    """Redirect sector list requests to analysis service."""
    return JSONResponse(
        status_code=307,
        content={"detail": "Redirecting to /analysis/sector/list"},
        headers={"Location": "/analysis/sector/list"}
    )

@app.get("/analyses/user/{user_id}")
async def redirect_user_analyses(user_id: str, request: Request):
    """Redirect user analyses requests to analysis service."""
    query_params = str(request.query_params)
    redirect_url = f"/analysis/analyses/user/{user_id}{query_params}"
    return JSONResponse(
        status_code=307,
        content={"detail": f"Redirecting to {redirect_url}"},
        headers={"Location": redirect_url}
    )

@app.get("/analyze")
async def redirect_analyze():
    """Redirect analyze requests to analysis service."""
    return JSONResponse(
        status_code=307,
        content={"detail": "Redirecting to /analysis/analyze"},
        headers={"Location": "/analysis/analyze"}
    )

# WebSocket endpoint (needs special handling)
@app.websocket("/ws/stream")
async def websocket_endpoint(websocket):
    """WebSocket endpoint for real-time data."""
    # Import and use the WebSocket handler from data service
    from data_service import websocket_endpoint as data_websocket
    await data_websocket(websocket)

if __name__ == "__main__":
    # Get port from environment or default to 8000
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    print(f"🚀 Starting Consolidated Service on {host}:{port}")
    print(f"📊 Data Service mounted at /data")
    print(f"🔍 Analysis Service mounted at /analysis")
    print(f"🌐 CORS Origins: {CORS_ORIGINS}")
    
    uvicorn.run(
        "consolidated_service:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )
