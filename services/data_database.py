#!/usr/bin/env python3
"""
Unified FastAPI Application - StockAnalyzer Pro Backend
Combines Database Service and Data Service into a single deployment for Render.

This application provides:
- Database operations (storing/retrieving analysis data)
- Real-time data streaming via WebSocket
- Historical data retrieval
- Market data optimization
"""

import os
import sys
import asyncio
import uvicorn
from datetime import datetime

# Add the parent directories to Python path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.dirname(__file__))

# Environment setup
import dotenv
dotenv.load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', 'config', '.env'))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Import both service applications from the same directory
from database_service import app as database_app, db_manager
from data_service import app as data_app

# Create main application
app = FastAPI(
    title="StockAnalyzer Pro Backend",
    description="Unified backend service providing database operations and real-time data streaming",
    version="1.0.0"
)

# Configure CORS for the main app
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "")
CORS_ORIGINS = CORS_ORIGINS.split(",") if CORS_ORIGINS else []
CORS_ORIGINS = [origin.strip() for origin in CORS_ORIGINS if origin.strip()]

print(f"🔧 Main App CORS_ORIGINS: {CORS_ORIGINS}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount sub-applications
app.mount("/database", database_app)
app.mount("/data", data_app)

# Root endpoint for the unified service
@app.get("/")
async def root():
    """Root endpoint for the unified StockAnalyzer Pro Backend."""
    return {
        "service": "StockAnalyzer Pro Backend",
        "version": "1.0.0",
        "status": "running",
        "description": "Unified backend service providing database operations and real-time data streaming",
        "services": {
            "database": {
                "mounted_at": "/database",
                "description": "Database operations for analysis storage and retrieval",
                "health_check": "/database/health"
            },
            "data": {
                "mounted_at": "/data",
                "description": "Real-time data streaming and historical data retrieval",
                "health_check": "/data/health",
                "websocket": "/data/ws/stream"
            }
        },
        "endpoints": {
            "health": "/health",
            "database_service": "/database/",
            "data_service": "/data/",
            "websocket_streaming": "/data/ws/stream"
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint for the unified service."""
    try:
        # Check both sub-services
        database_status = "healthy"
        data_status = "healthy"
        
        # Test database connection
        try:
            # Simple test to verify database manager is working
            if not db_manager.supabase:
                database_status = "unhealthy - no database connection"
        except Exception as e:
            database_status = f"unhealthy - {str(e)}"
        
        # Overall status
        overall_status = "healthy" if database_status == "healthy" and data_status == "healthy" else "degraded"
        
        return {
            "status": overall_status,
            "services": {
                "database": database_status,
                "data": data_status
            },
            "timestamp": datetime.now().isoformat(),
            "uptime": "running"
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )

@app.on_event("startup")
async def startup_event():
    """Startup event for the unified application."""
    print("🚀 Starting StockAnalyzer Pro Unified Backend...")
    print("📊 Database Service mounted at /database")
    print("🔌 Data Service mounted at /data")
    print("🌐 WebSocket streaming available at /data/ws/stream")
    print("✅ Unified backend ready!")

if __name__ == "__main__":
    # Get port from environment (Render provides PORT env var)
    port = int(os.getenv("PORT", os.getenv("SERVICE_PORT", 8000)))
    host = os.getenv("HOST", "0.0.0.0")
    
    print(f"🚀 Starting StockAnalyzer Pro Backend on {host}:{port}")
    print(f"🔧 Environment: {os.getenv('RENDER', 'local')}")
    
    # Run with uvicorn
    uvicorn.run(
        "data_database:app",
        host=host,
        port=port,
        reload=False,  # Disable reload in production
        access_log=True,
        log_level="info"
    )
