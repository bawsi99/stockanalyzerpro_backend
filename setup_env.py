#!/usr/bin/env python3
"""
Environment Setup Script

This script helps users set up their .env file with the necessary configuration
for Redis image storage and other required environment variables.
"""

import os
import sys
from pathlib import Path

def create_env_file():
    """Create a .env file with default configuration."""
    
    env_content = """# Zerodha API Configuration
ZERODHA_API_KEY=your_api_key_here
ZERODHA_ACCESS_TOKEN=your_access_token_here
ZERODHA_REQUEST_TOKEN=your_request_token_here

# Redis Configuration for Image Storage
REDIS_URL=redis://localhost:6379/0

# Redis Image Manager Configuration
REDIS_IMAGE_MAX_AGE_HOURS=24
REDIS_IMAGE_MAX_SIZE_MB=1000
REDIS_IMAGE_CLEANUP_INTERVAL_MINUTES=60
REDIS_IMAGE_ENABLE_CLEANUP=true
REDIS_IMAGE_QUALITY=85
REDIS_IMAGE_FORMAT=PNG

# Redis Cache Manager Configuration
REDIS_CACHE_ENABLE_COMPRESSION=true
REDIS_CACHE_ENABLE_LOCAL_FALLBACK=true
REDIS_CACHE_LOCAL_SIZE=1000
REDIS_CACHE_CLEANUP_INTERVAL_MINUTES=60

# Chart Manager Configuration (file-based fallback)
CHART_MAX_AGE_HOURS=24
CHART_MAX_SIZE_MB=1000
CHART_CLEANUP_INTERVAL_MINUTES=60
CHART_OUTPUT_DIR=./output/charts
CHART_ENABLE_CLEANUP=true

# Environment Configuration
ENVIRONMENT=development

# Database Configuration
DATABASE_URL=your_database_url_here

# JWT Configuration
JWT_SECRET=your_jwt_secret_here

# Gemini AI Configuration
GEMINI_API_KEY=your_gemini_api_key_here

# Service Configuration
HOST=0.0.0.0
DATA_SERVICE_PORT=8000
ANALYSIS_SERVICE_PORT=8001
LOG_LEVEL=INFO

# Scheduled Tasks
ENABLE_SCHEDULED_CALIBRATION=0
"""
    
    env_file = Path('.env')
    
    if env_file.exists():
        print("⚠️  .env file already exists!")
        response = input("Do you want to overwrite it? (y/N): ")
        if response.lower() != 'y':
            print("❌ Setup cancelled.")
            return False
    
    try:
        with open(env_file, 'w') as f:
            f.write(env_content)
        print("✅ .env file created successfully!")
        print("\n📝 Next steps:")
        print("1. Update the .env file with your actual API keys and credentials")
        print("2. Install Redis if not already installed")
        print("3. Run: python test_redis_image_manager.py")
        print("4. Run: python test_redis_cache_manager.py")
        return True
    except Exception as e:
        print(f"❌ Error creating .env file: {e}")
        return False

def check_redis_installation():
    """Check if Redis is installed and running."""
    print("🔍 Checking Redis installation...")
    
    # Check if redis-cli is available
    import subprocess
    try:
        result = subprocess.run(['redis-cli', 'ping'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0 and 'PONG' in result.stdout:
            print("✅ Redis is installed and running")
            return True
        else:
            print("⚠️  Redis is installed but not responding")
            return False
    except FileNotFoundError:
        print("❌ Redis is not installed")
        print("\n💡 To install Redis:")
        print("  macOS: brew install redis && brew services start redis")
        print("  Ubuntu: sudo apt install redis-server && sudo systemctl start redis-server")
        print("  Docker: docker run -d --name redis -p 6379:6379 redis:alpine")
        return False
    except subprocess.TimeoutExpired:
        print("⚠️  Redis connection timed out")
        return False

def main():
    """Main setup function."""
    print("🚀 Environment Setup for StockAnalyzer Pro")
    print("=" * 50)
    
    # Check if we're in the backend directory
    if not Path('requirements.txt').exists():
        print("❌ Please run this script from the backend directory")
        return False
    
    # Create .env file
    if not create_env_file():
        return False
    
    # Check Redis installation
    redis_ok = check_redis_installation()
    
    print("\n📋 Setup Summary:")
    print(f"  ✅ .env file created")
    print(f"  {'✅' if redis_ok else '❌'} Redis installation")
    
    if not redis_ok:
        print("\n⚠️  Please install and start Redis before proceeding")
        print("   See REDIS_IMAGE_STORAGE_SETUP.md for detailed instructions")
    
    print("\n🎉 Setup completed!")
    print("\n📖 Next steps:")
    print("1. Edit .env file with your API keys")
    print("2. Install Redis if needed")
    print("3. Test image storage: python test_redis_image_manager.py")
    print("4. Test cache system: python test_redis_cache_manager.py")
    print("5. Start services: python start_all_services.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
