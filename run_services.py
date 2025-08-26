#!/usr/bin/env python3
"""
run_services.py

Server launcher that runs both data service and analysis service together.
This script starts both services on different ports in the same terminal.

Usage:
    python run_services.py [--data-port 8000] [--analysis-port 8001] [--host 0.0.0.0]

Environment Variables:
    - DATA_SERVICE_PORT: Port for data service (default: 8000)
    - ANALYSIS_SERVICE_PORT: Port for analysis service (default: 8001)
    - HOST: Host address (default: 0.0.0.0)
    - LOG_LEVEL: Logging level (default: info)
"""

import os
import sys
import asyncio
import argparse
import signal
import time
from typing import Optional
import uvicorn
from contextlib import asynccontextmanager

# Add the backend directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class ServiceManager:
    """Manages multiple FastAPI services running concurrently."""
    
    def __init__(self, host: str = "0.0.0.0", data_port: int = 8000, analysis_port: int = 8001):
        self.host = host
        self.data_port = data_port
        self.analysis_port = analysis_port
        self.services = []
        self.shutdown_event = asyncio.Event()
        
    async def start_data_service(self):
        """Start the data service."""
        try:
            print(f"🚀 Starting Data Service on {self.host}:{self.data_port}")
            
            # Import the data service app
            from data_service import app as data_app
            
            config = uvicorn.Config(
                app=data_app,
                host=self.host,
                port=self.data_port,
                log_level="info",
                access_log=True,
                reload=False
            )
            
            server = uvicorn.Server(config)
            await server.serve()
            
        except Exception as e:
            print(f"❌ Error starting Data Service: {e}")
            raise
    
    async def start_analysis_service(self):
        """Start the analysis service."""
        try:
            print(f"🚀 Starting Analysis Service on {self.host}:{self.analysis_port}")
            
            # Import the analysis service app
            from analysis_service import app as analysis_app
            
            config = uvicorn.Config(
                app=analysis_app,
                host=self.host,
                port=self.analysis_port,
                log_level="info",
                access_log=True,
                reload=False
            )
            
            server = uvicorn.Server(config)
            await server.serve()
            
        except Exception as e:
            print(f"❌ Error starting Analysis Service: {e}")
            raise
    
    async def run_services(self):
        """Run both services concurrently."""
        print("🎯 Starting Stock Analyzer Pro Services...")
        print(f"📊 Data Service: http://{self.host}:{self.data_port}")
        print(f"🔍 Analysis Service: http://{self.host}:{self.analysis_port}")
        print("=" * 60)
        
        try:
            # Start both services concurrently
            await asyncio.gather(
                self.start_data_service(),
                self.start_analysis_service(),
                return_exceptions=True
            )
        except KeyboardInterrupt:
            print("\n🛑 Received shutdown signal...")
        except Exception as e:
            print(f"❌ Error running services: {e}")
        finally:
            await self.shutdown()
    
    async def shutdown(self):
        """Gracefully shutdown all services."""
        print("🔄 Shutting down services...")
        self.shutdown_event.set()
        
        # Cancel any running tasks
        tasks = [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]
        if tasks:
            print(f"🔄 Cancelling {len(tasks)} tasks...")
            for task in tasks:
                task.cancel()
            
            # Wait for tasks to complete
            await asyncio.gather(*tasks, return_exceptions=True)
        
        print("✅ All services shut down successfully")

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
    parser = argparse.ArgumentParser(description="Run Stock Analyzer Pro Services")
    parser.add_argument("--data-port", type=int, default=8000, help="Port for data service")
    parser.add_argument("--analysis-port", type=int, default=8001, help="Port for analysis service")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    parser.add_argument("--log-level", type=str, default="info", help="Logging level")
    
    args = parser.parse_args()
    
    # Set environment variables from command line args
    os.environ.setdefault("DATA_SERVICE_PORT", str(args.data_port))
    os.environ.setdefault("ANALYSIS_SERVICE_PORT", str(args.analysis_port))
    os.environ.setdefault("HOST", args.host)
    os.environ.setdefault("LOG_LEVEL", args.log_level)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check environment
    check_environment()
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create service manager
    manager = ServiceManager(
        host=args.host,
        data_port=args.data_port,
        analysis_port=args.analysis_port
    )
    
    try:
        # Run services
        asyncio.run(manager.run_services())
    except KeyboardInterrupt:
        print("\n🛑 Shutdown requested by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

