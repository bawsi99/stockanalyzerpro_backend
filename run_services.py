#!/usr/bin/env python3
"""
run_services.py

Script to run both data service and analysis service simultaneously.
This is a convenience script for development and testing.
"""

import os
import sys
import time
import signal
import subprocess
from pathlib import Path

# Add the backend directory to Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

def print_banner():
    """Print startup banner."""
    print("=" * 60)
    print("🚀 TRADER PRO - Split Backend Services")
    print("=" * 60)
    print("📊 Data Service:     http://localhost:8000")
    print("🧠 Analysis Service: http://localhost:8001")
    print("🔗 WebSocket Stream: ws://localhost:8081/ws/stream")
    print("=" * 60)
    print("Press Ctrl+C to stop all services")
    print("=" * 60)

def check_ports():
    """Check if ports are available."""
    import socket
    
    ports_to_check = [8000, 8001, 8081]
    unavailable_ports = []
    
    for port in ports_to_check:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('localhost', port))
        sock.close()
        if result == 0:
            unavailable_ports.append(port)
    
    if unavailable_ports:
        print(f"❌ Ports {unavailable_ports} are already in use!")
        print("Please stop any existing services or use different ports.")
        return False
    
    return True

def start_service(service_name, script_path, port):
    """Start a service using subprocess."""
    try:
        # Change to backend directory for proper module resolution
        process = subprocess.Popen(
            [sys.executable, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Redirect stderr to stdout
            text=True,
            bufsize=1,
            universal_newlines=True,
            cwd=backend_dir  # Set working directory to backend
        )
        print(f"✅ {service_name} started (PID: {process.pid})")
        return process
    except Exception as e:
        print(f"❌ Failed to start {service_name}: {e}")
        return None

def monitor_processes(processes):
    """Monitor running processes and handle output."""
    try:
        while True:
            # Check if any process has died
            for name, process in processes.items():
                if process.poll() is not None:
                    print(f"❌ {name} has stopped unexpectedly (exit code: {process.returncode})")
                    return False
            
            # Print any available output from all processes
            for name, process in processes.items():
                if process.stdout:
                    try:
                        # Non-blocking read
                        output = process.stdout.readline()
                        if output:
                            print(f"[{name}] {output.strip()}")
                    except Exception as e:
                        print(f"Error reading output from {name}: {e}")
            
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n🛑 Stopping all services...")
        return True

def stop_processes(processes):
    """Stop all running processes."""
    for name, process in processes.items():
        if process and process.poll() is None:
            print(f"🛑 Stopping {name}...")
            try:
                process.terminate()
                process.wait(timeout=5)
                print(f"✅ {name} stopped")
            except subprocess.TimeoutExpired:
                print(f"⚠️  Force killing {name}...")
                process.kill()
                process.wait()
            except Exception as e:
                print(f"Error stopping {name}: {e}")

def main():
    """Main function to run both services."""
    print_banner()
    
    # Check if ports are available
    if not check_ports():
        return 1
    
    # Define services
    services = {
        "Data Service": "start_data_service.py",
        "Analysis Service": "start_analysis_service.py",
        "WebSocket Stream Service": "start_websocket_service.py",
        "Service Endpoints": "start_service_endpoints.py"
    }
    
    # Start services
    processes = {}
    for service_name, script_name in services.items():
        script_path = backend_dir / script_name
        if not script_path.exists():
            print(f"❌ Script not found: {script_path}")
            return 1
        
        # Determine port based on service name
        if "Data" in service_name:
            port = 8000
        elif "Analysis" in service_name:
            port = 8001
        elif "WebSocket" in service_name:
            port = 8081
        elif "Service Endpoints" in service_name:
            port = 8002
        else:
            port = 8000
            
        process = start_service(service_name, script_path, port)
        if process:
            processes[service_name] = process
        else:
            # Stop already started processes
            stop_processes(processes)
            return 1
    
    # Wait a moment for services to start
    print("⏳ Waiting for services to start...")
    time.sleep(3)
    
    # Check if services are running
    for name, process in processes.items():
        if process.poll() is not None:
            print(f"❌ {name} failed to start (exit code: {process.returncode})")
            # Print any error output
            if process.stdout:
                output = process.stdout.read()
                if output:
                    print(f"Error output from {name}:")
                    print(output)
            stop_processes(processes)
            return 1
    
    print("✅ All services started successfully!")
    print("🌐 Data Service:     http://localhost:8000/health")
    print("🧠 Analysis Service: http://localhost:8001/health")
    print("🔗 WebSocket Stream: http://localhost:8081/health")
    print("🔧 Service Endpoints: http://localhost:8002/health")
    print("\n📝 Logs will appear below:")
    print("-" * 60)
    
    # Monitor processes
    try:
        monitor_processes(processes)
    except KeyboardInterrupt:
        pass
    finally:
        stop_processes(processes)
        print("👋 All services stopped. Goodbye!")
    
    return 0

if __name__ == "__main__":
    exit(main()) 