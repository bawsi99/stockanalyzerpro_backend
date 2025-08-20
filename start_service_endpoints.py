#!/usr/bin/env python3
"""
start_service_endpoints.py

Startup script for the Service Endpoints.
This service provides individual endpoints for testing each component.
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
    """Start the service endpoints."""
    print("🚀 Starting Service Endpoints...")
    print("📍 Service: Individual component testing endpoints")
    print("🌐 Port: 8002")
    print("📊 Health: http://localhost:8002/health")
    print("📋 Status: http://localhost:8002/status")
    print("-" * 50)
    
    # Configuration
    host = os.getenv("SERVICE_ENDPOINTS_HOST", "0.0.0.0")
    port = int(os.getenv("SERVICE_ENDPOINTS_PORT", 8002))
    reload = os.getenv("SERVICE_ENDPOINTS_RELOAD", "false").lower() == "true"
    
    # Start the service
    uvicorn.run(
        "service_endpoints:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
        access_log=True
    )

if __name__ == "__main__":
    main()
