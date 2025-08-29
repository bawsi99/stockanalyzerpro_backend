#!/bin/bash

# 🚀 StockAnalyzer Pro v3.0 - Cloud Redis Startup Script
# This script ensures proper environment loading and cloud Redis usage

set -e  # Exit on any error

echo "🌐 StockAnalyzer Pro v3.0 - Cloud Redis Startup"
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Check if .env file exists
if [ ! -f ".env" ]; then
    print_status $RED "❌ .env file not found!"
    echo "Please ensure you have a .env file with your Redis configuration."
    exit 1
fi

# Load environment variables
print_status $BLUE "🔧 Loading environment variables..."
source .env

# Check if REDIS_URL is set
if [ -z "$REDIS_URL" ]; then
    print_status $RED "❌ REDIS_URL environment variable not set!"
    echo "Please check your .env file."
    exit 1
fi

print_status $GREEN "✅ Environment variables loaded"
print_status $BLUE "🔌 Redis URL: $REDIS_URL"

# Check if local Redis is running
print_status $BLUE "🏠 Checking local Redis status..."
if pgrep -x "redis-server" > /dev/null; then
    print_status $YELLOW "⚠️  Local Redis is running!"
    echo "This might interfere with cloud Redis usage."
    echo "Consider stopping it with: brew services stop redis"
else
    print_status $GREEN "✅ Local Redis is not running"
fi

# Test cloud Redis connection
print_status $BLUE "🧪 Testing cloud Redis connection..."
if python test_cloud_redis_connection.py > /dev/null 2>&1; then
    print_status $GREEN "✅ Cloud Redis connection successful"
else
    print_status $RED "❌ Cloud Redis connection failed!"
    echo "Please check your network connection and Redis credentials."
    exit 1
fi

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    print_status $YELLOW "⚠️  Virtual environment not activated"
    echo "Activating virtual environment..."
    
    # Try to find and activate virtual environment
    if [ -d ".venv" ]; then
        source .venv/bin/activate
        print_status $GREEN "✅ Virtual environment activated (.venv)"
    elif [ -d "myenv" ]; then
        source myenv/bin/activate
        print_status $GREEN "✅ Virtual environment activated (myenv)"
    else
        print_status $YELLOW "⚠️  No virtual environment found"
        echo "Consider creating one with: python -m venv .venv"
    fi
else
    print_status $GREEN "✅ Virtual environment already activated: $VIRTUAL_ENV"
fi

# Display startup options
echo ""
print_status $BLUE "🚀 Startup Options:"
echo "1. Quick validation (recommended)"
echo "2. Full system validation"
echo "3. Start data service"
echo "4. Start analysis service"
echo "5. Start both services"
echo "6. Exit"

read -p "Choose an option (1-6): " choice

case $choice in
    1)
        print_status $BLUE "🔍 Running quick validation..."
        python quick_validation.py
        ;;
    2)
        print_status $BLUE "🔍 Running full system validation..."
        python validate_redis_unified_cache.py
        ;;
    3)
        print_status $BLUE "🚀 Starting data service..."
        python data_service.py
        ;;
    4)
        print_status $BLUE "🚀 Starting analysis service..."
        python analysis_service.py
        ;;
    5)
        print_status $BLUE "🚀 Starting both services..."
        echo "Starting data service in background..."
        python data_service.py &
        DATA_PID=$!
        echo "Data service PID: $DATA_PID"
        
        echo "Starting analysis service..."
        python analysis_service.py &
        ANALYSIS_PID=$!
        echo "Analysis service PID: $ANALYSIS_PID"
        
        echo ""
        echo "Both services started!"
        echo "Data service PID: $DATA_PID"
        echo "Analysis service PID: $ANALYSIS_PID"
        echo ""
        echo "To stop services:"
        echo "kill $DATA_PID $ANALYSIS_PID"
        ;;
    6)
        print_status $GREEN "👋 Goodbye!"
        exit 0
        ;;
    *)
        print_status $RED "❌ Invalid option"
        exit 1
        ;;
esac

echo ""
print_status $GREEN "✅ Startup complete!"
echo ""
echo "📋 Useful commands:"
echo "  • Check Redis status: python test_cloud_redis_connection.py"
echo "  • Quick validation: python quick_validation.py"
echo "  • System health: python system_health_check.py"
echo "  • Stop local Redis: brew services stop redis"
echo ""
echo "🌐 Your system is now using cloud Redis!"
