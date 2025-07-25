#!/usr/bin/env python3
"""
start_websocket_service.py

Startup script for the WebSocket Stream Service.
This service handles real-time data streaming via WebSocket on port 8081.
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
    """Start the WebSocket stream service."""
    print("🚀 Starting WebSocket Stream Service...")
    print("📍 Service: Real-time data streaming via WebSocket")
    print("🌐 Port: 8081")
    print("🔗 WebSocket: ws://localhost:8081/ws/stream")
    print("📊 Health: http://localhost:8081/health")
    print("-" * 50)
    
    # Configuration from environment variables
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000)) + 81  # Use port 8081 (8000 + 81)
    cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:8080,http://localhost:8081").split(",")
    
    print(f"🌍 CORS Origins: {cors_origins}")
    
    # Start the service
    uvicorn.run(
        "websocket_stream_service:app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
        access_log=True
    )

if __name__ == "__main__":
    main() 