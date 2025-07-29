#!/usr/bin/env python3
"""
Test script to demonstrate Gemini API debugging functionality.
This script will make a simple API call and show detailed logging output.
"""

import os
import sys
import asyncio
from pathlib import Path

# Add the backend directory to the Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

from gemini.gemini_client import GeminiClient
from gemini.debug_logger import debug_logger

async def test_gemini_debug():
    """Test the Gemini debugging functionality"""
    
    print("🧪 Testing Gemini API Debugging")
    print("=" * 50)
    
    # Check if API key is available
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY environment variable not set")
        print("Please set your Gemini API key:")
        print("export GEMINI_API_KEY='your-api-key-here'")
        return
    
    print("✅ API key found")
    
    try:
        # Initialize the client
        client = GeminiClient(api_key)
        print("✅ Gemini client initialized")
        
        # Test 1: Simple text prompt
        print("\n📝 Test 1: Simple text prompt")
        print("-" * 30)
        
        simple_prompt = """
        Analyze the following stock data and provide a brief summary:
        
        Stock: AAPL
        Current Price: $150.00
        Volume: 1,000,000
        RSI: 65
        
        Please provide a simple analysis in JSON format.
        """
        
        print("Sending simple prompt to Gemini...")
        response = await client.build_indicators_summary(
            symbol="AAPL",
            indicators={"rsi": 65, "price": 150.00, "volume": 1000000},
            period=14,
            interval="1d"
        )
        
        print(f"✅ Response received: {len(response) if response else 0} characters")
        
        # Test 2: Code execution prompt
        print("\n💻 Test 2: Code execution prompt")
        print("-" * 30)
        
        code_prompt = """
        Calculate the following technical indicators for a stock:
        
        Price data: [100, 102, 98, 105, 103, 107, 104, 108, 106, 110]
        
        Please calculate:
        1. Simple Moving Average (5-period)
        2. RSI (14-period)
        3. Price change percentage
        
        Return the results in JSON format.
        """
        
        print("Sending code execution prompt to Gemini...")
        
        # This will trigger the code execution method
        response = await client.analyze_stock_with_enhanced_calculations(
            symbol="TEST",
            indicators={"prices": [100, 102, 98, 105, 103, 107, 104, 108, 106, 110]},
            chart_paths={},
            period=14,
            interval="1d"
        )
        
        print(f"✅ Enhanced analysis completed: {len(str(response)) if response else 0} characters")
        
        print("\n🎉 All tests completed!")
        print("\n📋 Debug logs have been saved to:")
        print("   - Console output (above)")
        print("   - Log files in backend/logs/ directory")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main function"""
    print("🚀 Starting Gemini Debug Test")
    print("This will demonstrate detailed logging of Gemini API calls")
    print("=" * 60)
    
    # Run the async test
    asyncio.run(test_gemini_debug())

if __name__ == "__main__":
    main() 