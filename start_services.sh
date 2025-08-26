#!/bin/bash

# Stock Analyzer Pro - Service Launcher Script
# This script starts both data service and analysis service on different ports

set -e  # Exit on any error

# Default configuration
DATA_PORT=${DATA_SERVICE_PORT:-8000}
ANALYSIS_PORT=${ANALYSIS_SERVICE_PORT:-8001}
HOST=${HOST:-"0.0.0.0"}
LOG_LEVEL=${LOG_LEVEL:-"info"}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}$1${NC}"
}

# Function to check if a port is available
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 0  # Port is in use
    else
        return 1  # Port is available
    fi
}

# Function to check dependencies
check_dependencies() {
    print_status "Checking dependencies..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed"
        exit 1
    fi
    
    # Check pip
    if ! command -v pip3 &> /dev/null; then
        print_error "pip3 is not installed"
        exit 1
    fi
    
    # Check if requirements.txt exists
    if [ ! -f "requirements.txt" ]; then
        print_warning "requirements.txt not found in current directory"
    fi
    
    print_status "Dependencies check completed"
}

# Function to check environment
check_environment() {
    print_status "Checking environment configuration..."
    
    # Check for .env file
    if [ -f ".env" ]; then
        print_status "Found .env file"
        export $(cat .env | grep -v '^#' | xargs)
    else
        print_warning "No .env file found"
    fi
    
    # Check Zerodha credentials
    if [ -z "$ZERODHA_API_KEY" ] || [ "$ZERODHA_API_KEY" = "your_api_key" ]; then
        print_warning "ZERODHA_API_KEY not configured - live data streaming will be disabled"
    else
        print_status "ZERODHA_API_KEY configured"
    fi
    
    if [ -z "$ZERODHA_ACCESS_TOKEN" ] || [ "$ZERODHA_ACCESS_TOKEN" = "your_access_token" ]; then
        print_warning "ZERODHA_ACCESS_TOKEN not configured - live data streaming will be disabled"
    else
        print_status "ZERODHA_ACCESS_TOKEN configured"
    fi
    
    # Check Gemini API key
    if [ -z "$GEMINI_API_KEY" ] && [ -z "$GOOGLE_GEMINI_API_KEY" ]; then
        print_warning "GEMINI_API_KEY not configured - AI analysis features will be limited"
    else
        print_status "GEMINI_API_KEY configured"
    fi
    
    # Check Supabase configuration
    if [ -z "$SUPABASE_URL" ] || [ -z "$SUPABASE_ANON_KEY" ]; then
        print_warning "Supabase configuration not found - analysis storage will be disabled"
    else
        print_status "Supabase configuration found"
    fi
}

# Function to start data service
start_data_service() {
    print_status "Starting Data Service on $HOST:$DATA_PORT"
    
    # Check if port is available
    if check_port $DATA_PORT; then
        print_error "Port $DATA_PORT is already in use"
        exit 1
    fi
    
    # Start the service in background
    python3 -c "
import sys
sys.path.insert(0, '.')
from data_service import app
import uvicorn
uvicorn.run(app, host='$HOST', port=$DATA_PORT, log_level='$LOG_LEVEL', access_log=True, reload=False)
" > data_service.log 2>&1 &
    
    DATA_PID=$!
    echo $DATA_PID > data_service.pid
    print_status "Data Service started with PID: $DATA_PID"
}

# Function to start analysis service
start_analysis_service() {
    print_status "Starting Analysis Service on $HOST:$ANALYSIS_PORT"
    
    # Check if port is available
    if check_port $ANALYSIS_PORT; then
        print_error "Port $ANALYSIS_PORT is already in use"
        exit 1
    fi
    
    # Start the service in background
    python3 -c "
import sys
sys.path.insert(0, '.')
from analysis_service import app
import uvicorn
uvicorn.run(app, host='$HOST', port=$ANALYSIS_PORT, log_level='$LOG_LEVEL', access_log=True, reload=False)
" > analysis_service.log 2>&1 &
    
    ANALYSIS_PID=$!
    echo $ANALYSIS_PID > analysis_service.pid
    print_status "Analysis Service started with PID: $ANALYSIS_PID"
}

# Function to stop services
stop_services() {
    print_status "Stopping services..."
    
    # Stop data service
    if [ -f "data_service.pid" ]; then
        DATA_PID=$(cat data_service.pid)
        if kill -0 $DATA_PID 2>/dev/null; then
            print_status "Stopping Data Service (PID: $DATA_PID)"
            kill $DATA_PID
            rm -f data_service.pid
        fi
    fi
    
    # Stop analysis service
    if [ -f "analysis_service.pid" ]; then
        ANALYSIS_PID=$(cat analysis_service.pid)
        if kill -0 $ANALYSIS_PID 2>/dev/null; then
            print_status "Stopping Analysis Service (PID: $ANALYSIS_PID)"
            kill $ANALYSIS_PID
            rm -f analysis_service.pid
        fi
    fi
    
    print_status "Services stopped"
}

# Function to show status
show_status() {
    print_status "Service Status:"
    
    # Check data service
    if [ -f "data_service.pid" ]; then
        DATA_PID=$(cat data_service.pid)
        if kill -0 $DATA_PID 2>/dev/null; then
            print_status "Data Service: Running (PID: $DATA_PID) on port $DATA_PORT"
        else
            print_error "Data Service: Not running (stale PID file)"
            rm -f data_service.pid
        fi
    else
        print_error "Data Service: Not running"
    fi
    
    # Check analysis service
    if [ -f "analysis_service.pid" ]; then
        ANALYSIS_PID=$(cat analysis_service.pid)
        if kill -0 $ANALYSIS_PID 2>/dev/null; then
            print_status "Analysis Service: Running (PID: $ANALYSIS_PID) on port $ANALYSIS_PORT"
        else
            print_error "Analysis Service: Not running (stale PID file)"
            rm -f analysis_service.pid
        fi
    else
        print_error "Analysis Service: Not running"
    fi
}

# Function to show logs
show_logs() {
    local service=$1
    
    case $service in
        "data")
            if [ -f "data_service.log" ]; then
                tail -f data_service.log
            else
                print_error "Data service log file not found"
            fi
            ;;
        "analysis")
            if [ -f "analysis_service.log" ]; then
                tail -f analysis_service.log
            else
                print_error "Analysis service log file not found"
            fi
            ;;
        *)
            print_error "Invalid service. Use 'data' or 'analysis'"
            exit 1
            ;;
    esac
}

# Main script logic
case "${1:-start}" in
    "start")
        print_header "🎯 Starting Stock Analyzer Pro Services"
        print_status "Data Service: http://$HOST:$DATA_PORT"
        print_status "Analysis Service: http://$HOST:$ANALYSIS_PORT"
        echo "============================================================"
        
        check_dependencies
        check_environment
        
        # Start services
        start_data_service
        sleep 2
        start_analysis_service
        sleep 2
        
        print_status "✅ Both services started successfully!"
        print_status "Use './start_services.sh status' to check service status"
        print_status "Use './start_services.sh stop' to stop services"
        print_status "Use './start_services.sh logs data' or './start_services.sh logs analysis' to view logs"
        ;;
    
    "stop")
        print_header "🛑 Stopping Stock Analyzer Pro Services"
        stop_services
        ;;
    
    "restart")
        print_header "🔄 Restarting Stock Analyzer Pro Services"
        stop_services
        sleep 2
        $0 start
        ;;
    
    "status")
        show_status
        ;;
    
    "logs")
        if [ -z "$2" ]; then
            print_error "Please specify service: 'data' or 'analysis'"
            exit 1
        fi
        show_logs "$2"
        ;;
    
    "help"|"-h"|"--help")
        print_header "Stock Analyzer Pro Service Launcher"
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  start     Start both services (default)"
        echo "  stop      Stop both services"
        echo "  restart   Restart both services"
        echo "  status    Show service status"
        echo "  logs <service>  Show logs for specified service (data|analysis)"
        echo "  help      Show this help message"
        echo ""
        echo "Environment Variables:"
        echo "  DATA_SERVICE_PORT     Port for data service (default: 8000)"
        echo "  ANALYSIS_SERVICE_PORT Port for analysis service (default: 8001)"
        echo "  HOST                  Host address (default: 0.0.0.0)"
        echo "  LOG_LEVEL             Logging level (default: info)"
        ;;
    
    *)
        print_error "Unknown command: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac

