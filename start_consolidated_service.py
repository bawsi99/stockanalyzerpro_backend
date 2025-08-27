#!/usr/bin/env python3
"""
start_consolidated_service.py

Startup script for the consolidated service with proper environment setup.
This script ensures all dependencies are loaded and the service starts correctly.
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
    print("✅ Loaded environment variables from .env file")
except ImportError:
    print("⚠️  python-dotenv not available, using system environment variables")
except Exception as e:
    print(f"⚠️  Error loading .env file: {e}")

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
    
    # Check CORS configuration
    cors_origins = os.getenv("CORS_ORIGINS", "")
    if cors_origins:
        print(f"✅ CORS_ORIGINS configured: {cors_origins}")
    else:
        print("⚠️  CORS_ORIGINS not configured - using defaults")
    
    print("=" * 60)

def main():
    """Main entry point."""
    print("🚀 Starting Stock Analyzer Pro - Consolidated Service")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check environment
    check_environment()
    
    # Get configuration from environment
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    log_level = os.getenv("LOG_LEVEL", "info")
    
    print(f"🌐 Service Configuration:")
    print(f"   Host: {host}")
    print(f"   Port: {port}")
    print(f"   Log Level: {log_level}")
    print(f"   Service URL: http://{host}:{port}")
    print(f"   WebSocket URL: ws://{host}:{port}/ws/stream")
    print("=" * 60)
    
    try:
        # Import and start the consolidated service
        from consolidated_service import app
        
        print("✅ Consolidated service imported successfully")
        print("🚀 Starting server...")
        
        uvicorn.run(
            "consolidated_service:app",
            host=host,
            port=port,
            log_level=log_level,
            reload=False,
            access_log=True
        )
        
    except ImportError as e:
        print(f"❌ Failed to import consolidated service: {e}")
        print("💡 Make sure all dependencies are installed and the service files are present")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Failed to start consolidated service: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
