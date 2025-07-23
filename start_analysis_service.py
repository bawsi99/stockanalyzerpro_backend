#!/usr/bin/env python3
"""
start_analysis_service.py

Startup script for the Analysis Service.
This service handles all analysis, AI processing, and chart generation.
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
    """Start the analysis service."""
    print("🧠 Starting Stock Analysis Service...")
    print("📍 Service: AI analysis, technical indicators, chart generation")
    print("🌐 Port: 8001")
    print("📊 Health: http://localhost:8001/health")
    print("🔍 Analysis: http://localhost:8001/analyze")
    print("📈 Enhanced: http://localhost:8001/analyze/enhanced")
    print("-" * 50)
    
    # Configuration
    host = os.getenv("ANALYSIS_SERVICE_HOST", "0.0.0.0")
    port = int(os.getenv("ANALYSIS_SERVICE_PORT", 8001))
    reload = os.getenv("ANALYSIS_SERVICE_RELOAD", "false").lower() == "true"
    
    # Start the service
    uvicorn.run(
        "analysis_service:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
        access_log=True
    )

if __name__ == "__main__":
    main() 