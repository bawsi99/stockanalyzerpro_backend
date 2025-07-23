#!/usr/bin/env python3
"""
start_data_service.py

Startup script for the Data Service.
This service handles all data fetching, WebSocket connections, and real-time data streaming.
"""

import os
import sys
import uvicorn
from pathlib import Path

# Add the backend directory to Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

def main():
    """Start the data service."""
    print("🚀 Starting Stock Data Service...")
    print("📍 Service: Data fetching, WebSocket connections, real-time streaming")
    print("🌐 Port: 8000")
    print("🔗 WebSocket: ws://localhost:8000/ws/stream")
    print("📊 Health: http://localhost:8000/health")
    print("-" * 50)
    
    # Configuration
    host = os.getenv("DATA_SERVICE_HOST", "0.0.0.0")
    port = int(os.getenv("DATA_SERVICE_PORT", 8000))
    reload = os.getenv("DATA_SERVICE_RELOAD", "false").lower() == "true"
    
    # Start the service
    uvicorn.run(
        "data_service:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
        access_log=True
    )

if __name__ == "__main__":
    main() 