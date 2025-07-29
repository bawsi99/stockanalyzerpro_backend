#!/usr/bin/env python3
"""
Demonstration script for Gemini API debugging system.
This script shows how to enable debugging and see detailed API call information.
"""

import os
import sys
import asyncio
from pathlib import Path

# Add the backend directory to the Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

def main():
    """Main demonstration function"""
    print("🔍 Gemini API Debugging Demonstration")
    print("=" * 50)
    
    # Check if API key is available
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY environment variable not set")
        print("\nTo run this demonstration:")
        print("1. Get a Gemini API key from: https://makersuite.google.com/app/apikey")
        print("2. Set the environment variable:")
        print("   export GEMINI_API_KEY='your-api-key-here'")
        print("3. Run this script again")
        return
    
    print("✅ API key found")
    print("\n🚀 Starting demonstration...")
    
    # Import the debug configuration
    from gemini.debug_config import show_gemini_debug_status, enable_gemini_debug
    
    # Show current debug status
    print("\n📊 Current Debug Configuration:")
    show_gemini_debug_status()
    
    # Enable debugging if not already enabled
    enable_gemini_debug()
    
    # Run the demonstration
    asyncio.run(run_demo())

async def run_demo():
    """Run the actual demonstration"""
    try:
        from gemini.gemini_client import GeminiClient
        
        print("\n🔧 Initializing Gemini Client...")
        client = GeminiClient()
        print("✅ Client initialized")
        
        # Demo 1: Simple text request
        print("\n" + "="*60)
        print("📝 DEMO 1: Simple Text Request")
        print("="*60)
        
        print("Sending a simple prompt to analyze stock indicators...")
        
        response = await client.build_indicators_summary(
            symbol="AAPL",
            indicators={
                "rsi": 65.5,
                "macd": {"macd": 2.5, "signal": 1.8, "histogram": 0.7},
                "bollinger_bands": {"upper": 155.0, "middle": 150.0, "lower": 145.0},
                "volume": 1500000,
                "price": 150.25
            },
            period=14,
            interval="1d"
        )
        
        print(f"✅ Response received: {len(str(response))} characters")
        
        # Demo 2: Code execution request
        print("\n" + "="*60)
        print("💻 DEMO 2: Code Execution Request")
        print("="*60)
        
        print("Sending a request with code execution enabled...")
        
        response = await client.analyze_stock_with_enhanced_calculations(
            symbol="TSLA",
            indicators={
                "prices": [100, 102, 98, 105, 103, 107, 104, 108, 106, 110, 112, 109, 115, 113],
                "volumes": [1000000, 1200000, 800000, 1500000, 1100000, 1800000, 900000, 2000000, 1300000, 1600000, 1400000, 1200000, 2200000, 1900000]
            },
            chart_paths={},
            period=14,
            interval="1d"
        )
        
        print(f"✅ Enhanced analysis completed: {len(str(response))} characters")
        
        print("\n" + "="*60)
        print("🎉 DEMONSTRATION COMPLETED!")
        print("="*60)
        
        print("\n📋 What you should see in the terminal:")
        print("   • Detailed API request logs with prompts")
        print("   • API response logs with timing and structure")
        print("   • Code execution results and outputs")
        print("   • JSON parsing attempts and fixes")
        print("   • Any errors with full context")
        
        print("\n📁 Log files have been saved to:")
        print("   backend/logs/gemini_debug_YYYYMMDD_HHMMSS.log")
        
        print("\n🔧 To manage debugging:")
        print("   python gemini_debug_cli.py status    # Check status")
        print("   python gemini_debug_cli.py disable   # Disable debugging")
        print("   python gemini_debug_cli.py level DEBUG  # Set log level")
        
    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 