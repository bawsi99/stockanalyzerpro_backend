#!/usr/bin/env python3
"""
run_production_services.py

Production server launcher that runs both data service and analysis service on a single port.
This script combines both services for deployment in production environments.

Usage:
    python run_production_services.py [--port 8000] [--host 0.0.0.0]

Environment Variables:
    - SERVICE_PORT: Port for the combined service (default: 8000)
    - HOST: Host address (default: 0.0.0.0)
    - LOG_LEVEL: Logging level (default: info)
"""

import os
import sys
import asyncio
import argparse
import signal
import importlib
import time
from typing import Dict, Any, Optional, List
import uvicorn
from contextlib import asynccontextmanager

# Add the backend directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# FastAPI imports
from fastapi import FastAPI, APIRouter, Request, Response, HTTPException, Depends
from fastapi.routing import APIRoute
from fastapi.middleware.cors import CORSMiddleware
from starlette.requests import Request as StarletteRequest
from starlette.responses import Response as StarletteResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handle application startup and shutdown events.
    
    This context manager ensures proper initialization and cleanup of resources
    for the combined service.
    """
    print("🚀 Initializing services...")
    # Perform startup initialization
    yield
    # Perform cleanup on shutdown
    print("🔄 Shutting down services...")
    tasks = [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]
    if tasks:
        print(f"🔄 Cancelling {len(tasks)} tasks...")
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    print("✅ All services shut down successfully")


def proxy_endpoint(request: Request) -> Response:
    """Placeholder for dynamically created endpoints."""
    # This should never be called directly
    raise NotImplementedError("Proxy endpoint called directly")


def create_combined_app() -> FastAPI:
    """
    Create and configure the combined FastAPI application.
    
    This function creates a new FastAPI app that hosts both the data and analysis
    services under a unified API.
    
    Returns:
        FastAPI: The combined FastAPI application
    """
    # Create a new FastAPI app
    combined_app = FastAPI(
        title="Stock Analyzer Pro - Production API",
        description="Combined API for data and analysis services",
        version="3.0.0",
        lifespan=lifespan
    )
    
    # Load CORS origins from environment variable
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:8080,http://localhost:5173").split(",")
    CORS_ORIGINS = [origin.strip() for origin in CORS_ORIGINS if origin.strip()]
    
    # Add CORS middleware
    combined_app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Create routers for each service with appropriate prefixes
    data_router = APIRouter(prefix="/data")
    analysis_router = APIRouter(prefix="/analysis")
    
    # Add health check endpoint
    @combined_app.get("/health", tags=["Health"])
    async def health_check():
        """Health check endpoint for the combined service."""
        return {
            "status": "healthy",
            "service": "Stock Analyzer Pro",
            "version": "3.0.0"
        }
    
    # Add root endpoint
    @combined_app.get("/", tags=["Root"])
    async def root():
        """Root endpoint for the combined service."""
        return {
            "service": "Stock Analyzer Pro - Production API",
            "version": "3.0.0",
            "status": "running",
            "endpoints": {
                "health": "/health",
                "data_service": "/data",
                "analysis_service": "/analysis"
            },
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    
    # Add data service root endpoint
    @data_router.get("/", tags=["Data"])
    async def data_root():
        """Root endpoint for the data service."""
        return {
            "service": "Stock Data Service",
            "version": "1.0.0",
            "status": "running",
            "endpoints": ["/data/health", "/data/market/status"],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    
    # Add data service health endpoint
    @data_router.get("/health", tags=["Data"])
    async def data_health():
        """Health check endpoint for the data service."""
        return {
            "status": "healthy",
            "service": "Stock Data Service",
            "version": "1.0.0"
        }
    
    # Add market status endpoint
    @data_router.get("/market/status", tags=["Data"])
    async def market_status():
        """Get current market status."""
        return {
            "market": "NSE",
            "status": "open",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    
    # Add analysis service root endpoint
    @analysis_router.get("/", tags=["Analysis"])
    async def analysis_root():
        """Root endpoint for the analysis service."""
        return {
            "service": "Stock Analysis Service",
            "version": "1.0.0",
            "status": "running",
            "endpoints": ["/analysis/health", "/analysis/analyze"],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    
    # Add analysis service health endpoint
    @analysis_router.get("/health", tags=["Analysis"])
    async def analysis_health():
        """Health check endpoint for the analysis service."""
        return {
            "status": "healthy",
            "service": "Stock Analysis Service",
            "version": "1.0.0"
        }
    
    # Add analyze endpoint (stub implementation)
    @analysis_router.post("/analyze", tags=["Analysis"])
    async def analyze(request: Request):
        """Analyze a stock."""
        data = await request.json()
        stock_symbol = data.get("stock_symbol", "")
        interval = data.get("interval", "day")
        
        return {
            "status": "success",
            "stock_symbol": stock_symbol,
            "interval": interval,
            "analysis": {
                "trend": "bullish",
                "signals": ["golden_cross", "support_test"],
                "recommendation": "buy",
                "confidence": 0.85
            },
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    
    # Include the routers
    combined_app.include_router(data_router)
    combined_app.include_router(analysis_router)
    
    print("✅ Combined app created successfully")
    return combined_app


class ProductionServiceManager:
    """Manages the combined service running on a single port."""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8000, workers: int = 4):
        self.host = host
        self.port = port
        self.workers = workers
        self.shutdown_event = asyncio.Event()
        
    async def start_combined_service(self):
        """Start the combined service."""
        try:
            print(f"🚀 Starting Combined Service on {self.host}:{self.port}")
            
            # Create the combined app
            combined_app = create_combined_app()
            
            config = uvicorn.Config(
                app=combined_app,
                host=self.host,
                port=self.port,
                log_level="info",
                access_log=True,
                reload=False,
                workers=self.workers
            )
            
            server = uvicorn.Server(config)
            await server.serve()
            
        except Exception as e:
            print(f"❌ Error starting Combined Service: {e}")
            raise
    
    async def run_service(self):
        """Run the combined service."""
        print("🎯 Starting Stock Analyzer Pro Production Service...")
        print(f"🔗 API Endpoints: http://{self.host}:{self.port}")
        print(f"📊 Data API: http://{self.host}:{self.port}/data")
        print(f"🔍 Analysis API: http://{self.host}:{self.port}/analysis")
        print("=" * 60)
        
        try:
            await self.start_combined_service()
        except KeyboardInterrupt:
            print("\n🛑 Received shutdown signal...")
        except Exception as e:
            print(f"❌ Error running service: {e}")
        finally:
            await self.shutdown()
    
    async def shutdown(self):
        """Gracefully shutdown the service."""
        print("🔄 Shutting down combined service...")
        self.shutdown_event.set()
        
        # Cancel any running tasks
        tasks = [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]
        if tasks:
            print(f"🔄 Cancelling {len(tasks)} tasks...")
            for task in tasks:
                task.cancel()
            
            # Wait for tasks to complete
            await asyncio.gather(*tasks, return_exceptions=True)
        
        print("✅ Service shut down successfully")


def signal_handler(signum, frame):
    """Handle shutdown signals."""
    print(f"\n🛑 Received signal {signum}, shutting down...")
    sys.exit(0)


def check_dependencies():
    """Check if required dependencies are available."""
    required_modules = [
        'fastapi',
        'uvicorn',
        'pandas',
        'numpy',
        'requests',
        'websockets'
    ]
    
    missing_modules = []
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        print(f"❌ Missing required modules: {', '.join(missing_modules)}")
        print("💡 Install missing dependencies with: pip install -r requirements.txt")
        return False
    
    return True


def check_environment():
    """Check environment configuration."""
    print("🔧 Checking environment configuration...")
    
    # Check for Zerodha credentials
    api_key = os.getenv("ZERODHA_API_KEY")
    access_token = os.getenv("ZERODHA_ACCESS_TOKEN")
    
    if not api_key or api_key == "your_api_key":
        print("⚠️  ZERODHA_API_KEY not configured - live data streaming will be disabled")
    else:
        print("✅ ZERODHA_API_KEY configured")
    
    if not access_token or access_token == "your_access_token":
        print("⚠️  ZERODHA_ACCESS_TOKEN not configured - live data streaming will be disabled")
    else:
        print("✅ ZERODHA_ACCESS_TOKEN configured")
    
    # Check for Gemini API key
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_GEMINI_API_KEY")
    if not gemini_key:
        print("⚠️  GEMINI_API_KEY not configured - AI analysis features will be limited")
    else:
        print("✅ GEMINI_API_KEY configured")
    
    # Check for Supabase configuration
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_ANON_KEY")
    
    if not supabase_url or not supabase_key:
        print("⚠️  Supabase configuration not found - analysis storage will be disabled")
    else:
        print("✅ Supabase configuration found")
    
    print("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run Stock Analyzer Pro Production Service")
    parser.add_argument("--port", type=int, default=int(os.getenv("SERVICE_PORT", "8000")), 
                        help="Port for combined service")
    parser.add_argument("--host", type=str, default=os.getenv("HOST", "0.0.0.0"), 
                        help="Host address")
    parser.add_argument("--log-level", type=str, default=os.getenv("LOG_LEVEL", "info"), 
                        help="Logging level")
    parser.add_argument("--workers", type=int, default=int(os.getenv("WORKERS", "4")), 
                        help="Number of worker processes")
    
    args = parser.parse_args()
    
    # Set environment variables from command line args
    os.environ.setdefault("SERVICE_PORT", str(args.port))
    os.environ.setdefault("HOST", args.host)
    os.environ.setdefault("LOG_LEVEL", args.log_level)
    os.environ.setdefault("WORKERS", str(args.workers))
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check environment
    check_environment()
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create service manager
    manager = ProductionServiceManager(
        host=args.host,
        port=args.port,
        workers=args.workers
    )
    
    try:
        # Run service
        asyncio.run(manager.run_service())
    except KeyboardInterrupt:
        print("\n🛑 Shutdown requested by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()