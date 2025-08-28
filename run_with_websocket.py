#!/usr/bin/env python3
"""
Run Consolidated Service with WebSocket

This script starts both the Zerodha WebSocket client and the consolidated service.
It ensures that the WebSocket client is running before starting the consolidated service.
"""

import os
import sys
import time
import subprocess
import signal
import atexit

# Try to load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Loaded environment variables from .env file")
except ImportError:
    print("⚠️ python-dotenv not available, using system environment variables")

# Process management
processes = []

def cleanup():
    """Clean up all child processes on exit."""
    print("\n🧹 Cleaning up processes...")
    for p in processes:
        if p.poll() is None:  # If process is still running
            print(f"🛑 Terminating process {p.pid}")
            try:
                p.terminate()
                p.wait(timeout=5)
            except:
                try:
                    p.kill()
                except:
                    pass
    print("👋 Goodbye!")

# Register cleanup handler
atexit.register(cleanup)

# Handle SIGINT (Ctrl+C)
def signal_handler(sig, frame):
    print("\n🛑 Received interrupt signal")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def main():
    """Main function to start all services."""
    print("🚀 Starting StockAnalyzer Pro Backend Services")
    print("=" * 50)
    
    # Check if Zerodha credentials are set
    api_key = os.getenv("ZERODHA_API_KEY")
    access_token = os.getenv("ZERODHA_ACCESS_TOKEN")
    
    if not api_key or not access_token or api_key == 'your_api_key' or access_token == 'your_access_token':
        print("⚠️ Zerodha credentials not configured or invalid")
        print("📊 Historical data will be available via REST API")
        print("🔴 Live data streaming will be disabled")
        print("💡 To enable live data, set ZERODHA_API_KEY and ZERODHA_ACCESS_TOKEN in .env file")
        print("🚀 Starting consolidated service without WebSocket client...")
        
        # Start consolidated service directly
        consolidated_cmd = [sys.executable, "consolidated_service.py"]
        consolidated_process = subprocess.Popen(consolidated_cmd)
        processes.append(consolidated_process)
        
        print(f"✅ Consolidated service started (PID: {consolidated_process.pid})")
        
    else:
        print("✅ Zerodha credentials found")
        print("🚀 Starting Zerodha WebSocket client...")
        
        # Start WebSocket client
        ws_cmd = [sys.executable, "start_zerodha_ws.py"]
        ws_process = subprocess.Popen(ws_cmd)
        processes.append(ws_process)
        
        print(f"✅ WebSocket client started (PID: {ws_process.pid})")
        print("⏳ Waiting for WebSocket client to initialize...")
        time.sleep(5)  # Give WebSocket client time to connect
        
        # Start consolidated service
        print("🚀 Starting consolidated service...")
        consolidated_cmd = [sys.executable, "consolidated_service.py"]
        consolidated_process = subprocess.Popen(consolidated_cmd)
        processes.append(consolidated_process)
        
        print(f"✅ Consolidated service started (PID: {consolidated_process.pid})")
    
    print("=" * 50)
    print("📊 All services are now running")
    print("💡 Press Ctrl+C to stop all services")
    
    # Wait for all processes to complete (which they won't unless terminated)
    try:
        for p in processes:
            p.wait()
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
