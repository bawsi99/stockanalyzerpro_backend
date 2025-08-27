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

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Loaded environment variables from .env file")
except ImportError:
    print("⚠️  python-dotenv not available, using system environment variables")
except Exception as e:
    print(f"⚠️  Error loading .env file: {e}")

# Create the main FastAPI app
app = FastAPI(
    title="Stock Analyzer Pro - Consolidated Service",
    description="Combined Data and Analysis Services for Stock Analysis",
    version="1.0.0"
)

# Load CORS origins from environment variable
DEFAULT_CORS_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:8080",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8080",
    "http://127.0.0.1:5173",
    "https://stock-analyzer-pro.vercel.app",
    "https://stock-analyzer-pro-git-prototype-aaryan-manawats-projects.vercel.app",
    "https://stock-analyzer-cl9o3tivx-aaryan-manawats-projects.vercel.app",
    "https://stockanalyzer-pro.vercel.app"
]

CORS_ORIGINS_STR = os.getenv("CORS_ORIGINS", ",".join(DEFAULT_CORS_ORIGINS))
CORS_ORIGINS = [origin.strip() for origin in CORS_ORIGINS_STR.split(",") if origin.strip()]

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Check and log environment variables
print("🔧 Environment Check:")
print(f"   ZERODHA_API_KEY: {'✅ Set' if os.getenv('ZERODHA_API_KEY') else '❌ Not Set'}")
print(f"   ZERODHA_ACCESS_TOKEN: {'✅ Set' if os.getenv('ZERODHA_ACCESS_TOKEN') else '❌ Not Set'}")
print(f"   GEMINI_API_KEY: {'✅ Set' if os.getenv('GEMINI_API_KEY') else '❌ Not Set'}")
print(f"   SUPABASE_URL: {'✅ Set' if os.getenv('SUPABASE_URL') else '❌ Not Set'}")
print(f"   CORS_ORIGINS: {CORS_ORIGINS}")

# Import both service apps
try:
    from data_service import app as data_app
    from analysis_service import app as analysis_app
    print("✅ Successfully imported both Data and Analysis services")
except ImportError as e:
    print(f"❌ Error importing services: {e}")
    raise

# Override CORS_ORIGINS in data_service to match consolidated service
import data_service
data_service.CORS_ORIGINS = CORS_ORIGINS
print(f"🔧 Updated data_service CORS_ORIGINS: {data_service.CORS_ORIGINS}")

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
        "environment": {
            "zerodha_configured": bool(os.getenv("ZERODHA_API_KEY")),
            "gemini_configured": bool(os.getenv("GEMINI_API_KEY")),
            "supabase_configured": bool(os.getenv("SUPABASE_URL")),
            "cors_origins_count": len(CORS_ORIGINS)
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
        "environment": {
            "zerodha_configured": bool(os.getenv("ZERODHA_API_KEY")),
            "gemini_configured": bool(os.getenv("GEMINI_API_KEY")),
            "supabase_configured": bool(os.getenv("SUPABASE_URL"))
        },
        "timestamp": "2025-08-26T20:20:00Z"
    }

# Redirect legacy endpoints to their new locations
@app.get("/stock/{symbol}/history")
async def redirect_stock_history(symbol: str, request: Request):
    """Redirect stock history requests to data service."""
    # Properly handle query parameters
    query_string = str(request.query_params)
    query_suffix = f"?{query_string}" if query_string else ""
    redirect_url = f"/data/stock/{symbol}/history{query_suffix}"
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
    # Properly handle query parameters
    query_string = str(request.query_params)
    query_suffix = f"?{query_string}" if query_string else ""
    redirect_url = f"/analysis/analyses/user/{user_id}{query_suffix}"
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

@app.get("/stock/{symbol}/sector")
async def redirect_stock_sector(symbol: str, request: Request):
    """Redirect stock sector requests to analysis service."""
    # Properly handle query parameters
    query_string = str(request.query_params)
    query_suffix = f"?{query_string}" if query_string else ""
    redirect_url = f"/analysis/stock/{symbol}/sector{query_suffix}"
    return JSONResponse(
        status_code=307,
        content={"detail": f"Redirecting to {redirect_url}"},
        headers={"Location": redirect_url}
    )

@app.get("/auth/verify")
async def redirect_auth_verify(request: Request):
    """Redirect auth verify requests to data service."""
    # Properly handle query parameters
    query_string = str(request.query_params)
    query_suffix = f"?{query_string}" if query_string else ""
    redirect_url = f"/data/auth/verify{query_suffix}"
    return JSONResponse(
        status_code=307,
        content={"detail": f"Redirecting to {redirect_url}"},
        headers={"Location": redirect_url}
    )

@app.post("/auth/token")
async def redirect_auth_token(request: Request):
    """Redirect auth token requests to data service."""
    # Properly handle query parameters
    query_string = str(request.query_params)
    query_suffix = f"?{query_string}" if query_string else ""
    redirect_url = f"/data/auth/token{query_suffix}"
    return JSONResponse(
        status_code=307,
        content={"detail": f"Redirecting to {redirect_url}"},
        headers={"Location": redirect_url}
    )

@app.options("/auth/token")
async def redirect_auth_token_options(request: Request):
    """Redirect auth token OPTIONS requests to data service."""
    # Properly handle query parameters
    query_string = str(request.query_params)
    query_suffix = f"?{query_string}" if query_string else ""
    redirect_url = f"/data/auth/token{query_suffix}"
    return JSONResponse(
        status_code=307,
        content={"detail": f"Redirecting to {redirect_url}"},
        headers={"Location": redirect_url}
    )

# WebSocket endpoint is handled by the mounted data service at /data/ws/stream

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
