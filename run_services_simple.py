#!/usr/bin/env python3
"""
run_services_simple.py

Simple server launcher that runs both data service and analysis service together.
This script starts both services on different ports using multiprocessing.

Usage:
    python run_services_simple.py [--data-port 8000] [--analysis-port 8001] [--host 0.0.0.0]

Environment Variables:
    - DATA_SERVICE_PORT: Port for data service (default: 8000)
    - ANALYSIS_SERVICE_PORT: Port for analysis service (default: 8001)
    - HOST: Host address (default: 0.0.0.0)
"""

import os
import sys
import time
import signal
import argparse
import multiprocessing
from multiprocessing import Process
import uvicorn

# Add the backend directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def run_data_service(host: str, port: int):
    """Run the data service in a separate process."""
    try:
        print(f"🚀 Starting Data Service on {host}:{port}")
        
        # Import the data service app
        from data_service import app as data_app
        
        uvicorn.run(
            data_app,
            host=host,
            port=port,
            log_level="info",
            access_log=True,
            reload=False
        )
    except Exception as e:
        print(f"❌ Error in Data Service: {e}")

def run_analysis_service(host: str, port: int):
    """Run the analysis service in a separate process."""
    try:
        print(f"🚀 Starting Analysis Service on {host}:{port}")
        
        # Import the analysis service app
        from analysis_service import app as analysis_app
        
        uvicorn.run(
            analysis_app,
            host=host,
            port=port,
            log_level="info",
            access_log=True,
            reload=False
        )
    except Exception as e:
        print(f"❌ Error in Analysis Service: {e}")

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

def signal_handler(signum, frame):
    """Handle shutdown signals."""
    print(f"\n🛑 Received signal {signum}, shutting down...")
    sys.exit(0)

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run Stock Analyzer Pro Services")
    parser.add_argument("--data-port", type=int, default=8000, help="Port for data service")
    parser.add_argument("--analysis-port", type=int, default=8001, help="Port for analysis service")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    
    args = parser.parse_args()
    
    # Set environment variables from command line args
    os.environ.setdefault("DATA_SERVICE_PORT", str(args.data_port))
    os.environ.setdefault("ANALYSIS_SERVICE_PORT", str(args.analysis_port))
    os.environ.setdefault("HOST", args.host)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check environment
    check_environment()
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("🎯 Starting Stock Analyzer Pro Services...")
    print(f"📊 Data Service: http://{args.host}:{args.data_port}")
    print(f"🔍 Analysis Service: http://{args.host}:{args.analysis_port}")
    print("=" * 60)
    
    # Create processes for each service
    data_process = Process(
        target=run_data_service,
        args=(args.host, args.data_port),
        name="DataService"
    )
    
    analysis_process = Process(
        target=run_analysis_service,
        args=(args.host, args.analysis_port),
        name="AnalysisService"
    )
    
    try:
        # Start both processes
        data_process.start()
        time.sleep(2)  # Give data service time to start
        
        analysis_process.start()
        time.sleep(2)  # Give analysis service time to start
        
        print("✅ Both services started successfully!")
        print("🔄 Services are running. Press Ctrl+C to stop.")
        
        # Wait for processes to complete (they won't unless there's an error)
        data_process.join()
        analysis_process.join()
        
    except KeyboardInterrupt:
        print("\n🛑 Shutdown requested by user...")
    except Exception as e:
        print(f"❌ Error starting services: {e}")
    finally:
        # Cleanup processes
        print("🔄 Shutting down services...")
        
        if data_process.is_alive():
            print("🔄 Terminating Data Service...")
            data_process.terminate()
            data_process.join(timeout=5)
            if data_process.is_alive():
                print("🔄 Force killing Data Service...")
                data_process.kill()
        
        if analysis_process.is_alive():
            print("🔄 Terminating Analysis Service...")
            analysis_process.terminate()
            analysis_process.join(timeout=5)
            if analysis_process.is_alive():
                print("🔄 Force killing Analysis Service...")
                analysis_process.kill()
        
        print("✅ All services shut down successfully")

if __name__ == "__main__":
    # Set multiprocessing start method
    multiprocessing.set_start_method('spawn', force=True)
    main()

