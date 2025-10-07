#!/usr/bin/env python3
"""
Test Gemini LLM with Real API Key

This script loads the GEMINI_API_KEY from backend/config/.env and tests
the Gemini provider with a simple prompt to verify it's working correctly.

Usage:
    cd backend/llm/tests
    python test_gemini_live.py
"""

import sys
import os
import asyncio
from pathlib import Path

# Add the project root to path
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent  # Go up to version3.0/3.0
sys.path.insert(0, str(project_root))

# Load environment variables from backend/config/.env
from dotenv import load_dotenv
env_path = project_root / "backend" / "config" / ".env"
load_dotenv(env_path)

from backend.llm import get_llm_client


async def test_gemini_basic():
    """Test basic Gemini text generation"""
    print("🧪 Testing Gemini Basic Text Generation")
    print("=" * 50)
    
    try:
        # Create client for indicator agent (uses gemini-2.5-flash)
        client = get_llm_client("indicator_agent")
        print(f"✅ Client created: {client.get_provider_info()}")
        
        # Test prompt
        prompt = "Explain AI in 50 words"
        print(f"📝 Sending prompt: {prompt}")
        
        # Make the call
        response = await client.generate_text(prompt)
        
        print(f"✅ Response received ({len(response)} chars):")
        print("-" * 50)
        print(response)
        print("-" * 50)
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_gemini_code_execution():
    """Test Gemini with code execution enabled"""
    print("\n🧪 Testing Gemini with Code Execution")
    print("=" * 50)
    
    try:
        # Create client with code execution
        client = get_llm_client("indicator_agent")
        print(f"✅ Client created: {client.get_provider_info()}")
        
        # Test prompt that requires calculation
        prompt = "Calculate the average of these numbers: 10, 20, 30, 40, 50. Show your work using Python code."
        print(f"📝 Sending prompt: {prompt}")
        
        # Make the call with code execution
        response = await client.generate(prompt, enable_code_execution=True)
        
        print(f"✅ Response received ({len(response)} chars):")
        print("-" * 50)
        print(response)
        print("-" * 50)
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_different_models():
    """Test different Gemini models"""
    print("\n🧪 Testing Different Gemini Models")
    print("=" * 50)
    
    models_to_test = [
        ("gemini-2.5-flash", "Fast model test"),
        ("gemini-2.5-pro", "Pro model test")
    ]
    
    results = []
    
    for model, description in models_to_test:
        try:
            print(f"\n🔄 Testing {model}...")
            
            # Create client with specific model
            client = get_llm_client(provider="gemini", model=model)
            print(f"   Client: {client.get_provider_info()}")
            
            # Simple prompt
            prompt = "What is machine learning? Answer in 30 words."
            response = await client.generate_text(prompt)
            
            print(f"   ✅ {description} - Response ({len(response)} chars):")
            print(f"   📤 {response[:100]}...")
            
            results.append(True)
            
        except Exception as e:
            print(f"   ❌ {description} failed: {e}")
            results.append(False)
    
    return all(results)


def check_environment():
    """Check that API keys are loaded"""
    print("🔧 Checking Environment")
    print("=" * 50)
    
    # Check for API keys
    api_keys = []
    for i in range(1, 6):
        key = os.getenv(f"GEMINI_API_KEY{i}")
        if key:
            api_keys.append(f"GEMINI_API_KEY{i}: ...{key[-8:]}")
    
    main_key = os.getenv("GEMINI_API_KEY")
    if main_key:
        api_keys.append(f"GEMINI_API_KEY: ...{main_key[-8:]}")
    
    if api_keys:
        print("✅ Found API keys:")
        for key_info in api_keys:
            print(f"   {key_info}")
        return True
    else:
        print("❌ No API keys found!")
        print("   Make sure backend/config/.env contains GEMINI_API_KEY values")
        return False


async def main():
    """Run all tests"""
    print("🚀 Gemini LLM Live Test Suite")
    print("=" * 60)
    
    # Check environment first
    if not check_environment():
        return False
    
    # Run tests
    tests = [
        test_gemini_basic,
        test_gemini_code_execution,
        test_different_models
    ]
    
    results = []
    for test in tests:
        try:
            result = await test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test function failed: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"🎉 All tests passed! ({passed}/{total})")
        print("✅ Gemini LLM is working correctly")
        return True
    else:
        print(f"⚠️  Some tests failed ({passed}/{total})")
        print("🔧 Check the errors above")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)