#!/bin/bash

# Memory Analysis Runner Script for Stock Analyzer Pro
# This script runs comprehensive memory analysis on the service

echo "🚀 Stock Analyzer Pro - Memory Analysis Runner"
echo "=============================================="

# Check if we're in the right directory
if [ ! -f "start_with_cors_fix.py" ]; then
    echo "❌ Error: Please run this script from the backend directory"
    echo "   Current directory: $(pwd)"
    echo "   Expected files: start_with_cors_fix.py"
    exit 1
fi

# Check Python version
python_version=$(python3 --version 2>&1)
echo "🐍 Python version: $python_version"

# Check if required packages are installed
echo "📦 Checking required packages..."
python3 -c "import psutil, aiohttp, requests" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Missing required packages. Installing..."
    pip3 install -r memory_analysis_requirements.txt
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install required packages"
        exit 1
    fi
    echo "✅ Required packages installed"
else
    echo "✅ Required packages are available"
fi

# Create output directory
mkdir -p memory_analysis_results
cd memory_analysis_results

echo ""
echo "🔧 Configuration:"
echo "   Service URL: http://localhost:8000"
echo "   Baseline monitoring: 30 seconds"
echo "   Load test duration: 120 seconds (2 minutes)"
echo "   Cooldown monitoring: 30 seconds"
echo "   Concurrent users: 20"
echo "   Requests per user: 15"
echo "   Total test duration: ~3 minutes"
echo ""

# Ask for confirmation
read -p "Do you want to proceed with the analysis? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Analysis cancelled"
    exit 0
fi

echo ""
echo "🚀 Starting comprehensive memory analysis..."
echo "   This will take approximately 3 minutes"
echo "   Press Ctrl+C to stop early"
echo ""

# Run the comprehensive analysis
python3 ../memory_analysis_runner.py \
    --url "http://localhost:8000" \
    --baseline 30 \
    --load-test 120 \
    --cooldown 30 \
    --users 20 \
    --requests 15 \
    --interval 0.5

echo ""
echo "✅ Memory analysis completed!"
echo "📁 Results saved in: $(pwd)"
echo ""
echo "📊 Generated files:"
echo "   - comprehensive_memory_analysis.json (main analysis)"
echo "   - load_test_during_analysis.json (load test results)"
echo "   - memory_analysis.json (raw memory data)"
echo ""
echo "💡 Next steps:"
echo "   1. Review the analysis summary above"
echo "   2. Check the JSON files for detailed data"
echo "   3. Use the results to plan cloud deployment"
echo "   4. Consider optimization based on recommendations"
