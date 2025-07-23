#!/usr/bin/env python3
"""
Setup script for live charts implementation.
This script validates the environment, dependencies, and configuration.
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not available, environment variables may not be loaded")

def check_python_version() -> bool:
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def check_dependencies() -> bool:
    """Check if required dependencies are installed."""
    package_imports = {
        'fastapi': 'fastapi',
        'uvicorn': 'uvicorn', 
        'kiteconnect': 'kiteconnect',
        'python-dotenv': 'dotenv',
        'PyJWT': 'jwt',
        'pandas': 'pandas',
        'numpy': 'numpy'
    }
    
    missing_packages = []
    
    for package, import_name in package_imports.items():
        try:
            __import__(import_name)
            print(f"✅ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} is missing")
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Install them with: pip install " + " ".join(missing_packages))
        return False
    
    return True

def check_env_file() -> bool:
    """Check if .env file exists and has required variables."""
    env_file = Path('.env')
    
    if not env_file.exists():
        print("❌ .env file not found")
        print("Creating .env file with template...")
        create_env_template()
        return False
    
    print("✅ .env file found")
    
    # Check required variables
    required_vars = ['ZERODHA_API_KEY', 'ZERODHA_ACCESS_TOKEN', 'JWT_SECRET']
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        print("Please update your .env file with the required values")
        return False
    
    print("✅ All required environment variables are set")
    return True

def create_env_template() -> None:
    """Create a template .env file."""
    template = """# Zerodha API Configuration
ZERODHA_API_KEY=your_zerodha_api_key_here
ZERODHA_ACCESS_TOKEN=your_zerodha_access_token_here

# JWT Authentication
JWT_SECRET=your-super-secret-jwt-key-change-this-in-production
REQUIRE_AUTH=true
API_KEYS=test-api-key-1,test-api-key-2

# Optional: Redis for caching
REDIS_URL=redis://localhost:6379/0

# Environment
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=INFO

# Server Configuration
HOST=0.0.0.0
PORT=8000
CORS_ORIGINS=http://localhost:3000,http://localhost:8080,http://127.0.0.1:3000,http://127.0.0.1:8080
"""
    
    with open('.env', 'w') as f:
        f.write(template)
    
    print("✅ Created .env template file")
    print("Please update the .env file with your actual credentials")

def test_imports() -> bool:
    """Test if all modules can be imported."""
    try:
        from zerodha_ws_client import zerodha_ws_client, candle_aggregator
        print("✅ zerodha_ws_client imported successfully")
    except ImportError as e:
        print(f"❌ Error importing zerodha_ws_client: {e}")
        return False
    
    try:
        from api import app
        print("✅ API module imported successfully")
    except ImportError as e:
        print(f"❌ Error importing API module: {e}")
        return False
    
    return True

def validate_zerodha_credentials() -> bool:
    """Validate Zerodha credentials format."""
    api_key = os.getenv('ZERODHA_API_KEY', '')
    access_token = os.getenv('ZERODHA_ACCESS_TOKEN', '')
    
    if api_key == 'your_zerodha_api_key_here' or access_token == 'your_zerodha_access_token_here':
        print("❌ Please update your .env file with actual Zerodha credentials")
        return False
    
    if len(api_key) < 10:
        print("❌ API key seems too short")
        return False
    
    if len(access_token) < 10:
        print("❌ Access token seems too short")
        return False
    
    print("✅ Zerodha credentials format looks valid")
    return True

def check_directory_structure() -> bool:
    """Check if required files and directories exist."""
    required_files = [
        'zerodha_ws_client.py',
        'api.py',
        'requirements.txt'
    ]
    
    for file in required_files:
        if not Path(file).exists():
            print(f"❌ Required file not found: {file}")
            return False
    
    print("✅ All required files found")
    return True

def run_quick_test() -> bool:
    """Run a quick test of the WebSocket client."""
    try:
        from test_websocket_connection import test_candle_aggregation
        test_candle_aggregation()
        print("✅ Candle aggregation test passed")
        return True
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        return False

def main() -> None:
    """Main setup function."""
    print("🔧 Live Charts Setup and Validation")
    print("=" * 50)
    
    checks = [
        ("Python Version", check_python_version),
        ("Directory Structure", check_directory_structure),
        ("Dependencies", check_dependencies),
        ("Environment File", check_env_file),
        ("Module Imports", test_imports),
        ("Zerodha Credentials", validate_zerodha_credentials),
        ("Quick Test", run_quick_test)
    ]
    
    passed = 0
    total = len(checks)
    
    for check_name, check_func in checks:
        print(f"\n📋 {check_name}:")
        try:
            if check_func():
                passed += 1
            else:
                print(f"   ⚠️  {check_name} check failed")
        except Exception as e:
            print(f"   ❌ {check_name} check error: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Setup Results: {passed}/{total} checks passed")
    
    if passed == total:
        print("🎉 All checks passed! Your live charts setup is ready.")
        print("\nNext steps:")
        print("1. Start the backend: python -m uvicorn api:app --reload --host 0.0.0.0 --port 8000")
        print("2. Test WebSocket: python test_websocket_connection.py")
        print("3. Start frontend and test live charts")
    else:
        print("⚠️  Some checks failed. Please fix the issues above before proceeding.")
        sys.exit(1)

if __name__ == "__main__":
    main() 