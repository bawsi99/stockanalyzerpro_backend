#!/bin/bash
# Script to run production services and test endpoints

echo "🚀 Starting Stock Analyzer Pro Production Services..."

# Start the production service in the background
python run_production_services.py --port 8000 &
SERVICE_PID=$!

echo "⏳ Waiting for service to start (5 seconds)..."
sleep 5

# Test the endpoints
echo "🧪 Running endpoint tests..."
python test_production_endpoints.py --host localhost --port 8000

# Ask if user wants to keep the service running
read -p "Keep service running? (y/n): " KEEP_RUNNING

if [[ "$KEEP_RUNNING" != "y" ]]; then
    echo "🛑 Stopping service (PID: $SERVICE_PID)..."
    kill $SERVICE_PID
    echo "✅ Service stopped"
else
    echo "✅ Service running on http://localhost:8000"
    echo "   - To stop the service, run: kill $SERVICE_PID"
fi
