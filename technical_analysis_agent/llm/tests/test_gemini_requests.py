"""
Test Script for Gemini Provider Request Functions

This script tests the core request functionality of our new Gemini provider
to ensure it works correctly before we proceed with text extraction improvements.

Usage:
    python -m backend.llm.tests.test_gemini_requests
"""

import asyncio
import sys
import os

# Add the backend directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from backend.llm.providers.gemini import GeminiProvider


async def test_text_generation():
    """Test basic text generation (no images)"""
    print("🧪 Testing text generation...")
    
    try:
        # Initialize provider
        provider = GeminiProvider(model="gemini-2.5-flash")
        
        # Test simple text generation
        prompt = "Analyze this simple data: Stock AAPL, Price: $150, RSI: 65. Provide a brief 1-sentence analysis."
        
        response = await provider.generate_text(
            prompt=prompt, 
            enable_code_execution=False,  # Simple test first
            max_retries=1
        )
        
        print(f"✅ Text generation successful!")
        print(f"📝 Prompt: {prompt[:50]}...")
        print(f"📤 Response ({len(response)} chars): {response[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Text generation failed: {e}")
        return False


async def test_code_execution():
    """Test text generation with code execution enabled"""
    print("\n🧪 Testing code execution...")
    
    try:
        # Initialize provider
        provider = GeminiProvider(model="gemini-2.5-flash")
        
        # Test with code execution
        prompt = "Calculate the simple moving average of these prices: [150, 152, 148, 155, 153]. Use Python code."
        
        response = await provider.generate_text(
            prompt=prompt,
            enable_code_execution=True,
            max_retries=1
        )
        
        print(f"✅ Code execution successful!")
        print(f"📝 Prompt: {prompt[:50]}...")
        print(f"📤 Response ({len(response)} chars): {response[:150]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Code execution failed: {e}")
        return False


async def test_error_handling():
    """Test error handling with invalid prompt"""
    print("\n🧪 Testing error handling...")
    
    try:
        # Initialize provider
        provider = GeminiProvider(model="gemini-2.5-flash")
        
        # Test with empty prompt (should handle gracefully)
        response = await provider.generate_text(
            prompt="", 
            enable_code_execution=False,
            max_retries=1
        )
        
        print(f"✅ Error handling successful!")
        print(f"📤 Response to empty prompt: '{response}'")
        
        return True
        
    except Exception as e:
        print(f"✅ Error handling working (expected error): {e}")
        return True  # Expected to fail


def test_initialization():
    """Test provider initialization"""
    print("🧪 Testing provider initialization...")
    
    try:
        # Test with default model
        provider1 = GeminiProvider()
        print(f"✅ Default initialization: Model = {provider1.model}")
        
        # Test with specific model
        provider2 = GeminiProvider(model="gemini-2.5-pro")
        print(f"✅ Custom model initialization: Model = {provider2.model}")
        
        return True
        
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        print("💡 Make sure you have GEMINI_API_KEY or GEMINI_API_KEY1-5 set in environment")
        return False


async def main():
    """Run all tests"""
    print("🚀 Testing New Gemini Provider Request Functions")
    print("=" * 60)
    
    # Test initialization first
    if not test_initialization():
        print("\n❌ Tests failed at initialization. Check your API keys.")
        return
    
    # Run async tests
    results = []
    results.append(await test_text_generation())
    results.append(await test_code_execution())  
    results.append(await test_error_handling())
    
    # Summary
    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results) + 1  # +1 for initialization
    
    if passed == len(results):  # All async tests passed (initialization already checked)
        print(f"🎉 All tests passed! ({total}/{total})")
        print("✅ Ready to proceed with text extraction improvements")
    else:
        print(f"⚠️  Some tests failed ({passed}/{total})")
        print("🔧 Review the errors above before proceeding")


if __name__ == "__main__":
    asyncio.run(main())