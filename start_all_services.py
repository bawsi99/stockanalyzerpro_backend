#!/usr/bin/env python3
"""
start_all_services.py

Startup script for all main services.
This script starts both the data service and analysis service.
"""

import os
import sys
import subprocess
import time
import signal
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
    """Start all main services."""
    print("🚀 Starting All Stock Analysis Services...")
    print("📍 Data Service: Port 8000 (Data fetching, WebSocket, real-time streaming)")
    print("📍 Analysis Service: Port 8001 (AI analysis, indicators, charts)")
    print("🌐 Frontend should connect to:")
    print("   - Data Service: http://localhost:8000")
    print("   - Analysis Service: http://localhost:8001")
    print("📊 Health checks:")
    print("   - Data Service: http://localhost:8000/health")
    print("   - Analysis Service: http://localhost:8001/health")
    print("-" * 60)
    
    # Configuration
    data_host = os.getenv("DATA_SERVICE_HOST", "0.0.0.0")
    data_port = int(os.getenv("DATA_SERVICE_PORT", 8000))
    data_reload = os.getenv("DATA_SERVICE_RELOAD", "false").lower() == "true"
    
    analysis_host = os.getenv("ANALYSIS_SERVICE_HOST", "0.0.0.0")
    analysis_port = int(os.getenv("ANALYSIS_SERVICE_PORT", 8001))
    analysis_reload = os.getenv("ANALYSIS_SERVICE_RELOAD", "false").lower() == "true"
    
    # Start data service
    print("📡 Starting Data Service...")
    data_process = subprocess.Popen([
        sys.executable, "start_data_service.py"
    ], cwd=backend_dir)
    
    # Wait a moment for data service to start
    time.sleep(3)
    
    # Start analysis service
    print("🧠 Starting Analysis Service...")
    analysis_process = subprocess.Popen([
        sys.executable, "start_analysis_service.py"
    ], cwd=backend_dir)
    
    print("✅ All services started!")
    print("Press Ctrl+C to stop all services")
    
    try:
        # Wait for both processes
        data_process.wait()
        analysis_process.wait()
    except KeyboardInterrupt:
        print("\n🛑 Stopping all services...")
        data_process.terminate()
        analysis_process.terminate()
        
        # Wait for graceful shutdown
        try:
            data_process.wait(timeout=5)
            analysis_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print("⚠️  Force killing services...")
            data_process.kill()
            analysis_process.kill()
        
        print("✅ All services stopped")

if __name__ == "__main__":
    main()
