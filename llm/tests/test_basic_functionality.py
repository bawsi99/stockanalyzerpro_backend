"""
Basic LLM System Test

Tests the configuration and client setup without requiring actual API keys.
This ensures the system is properly wired together.
"""

import sys
import os
import asyncio

# Add backend to path for testing
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from backend.llm.config.config import LLMConfig, get_llm_config
from backend.llm.client import LLMClientFactory
from backend.llm import get_llm_client
from backend.llm.utils import validate_api_keys, format_provider_model_name, estimate_tokens


def test_configuration():
    """Test configuration loading and validation."""
    print("🧪 Testing Configuration System...")
    
    try:
        # Test config loading
        config = LLMConfig()
        print(f"   ✅ Config loaded from: {config.config_file}")
        print(f"   ✅ Environment: {config.environment}")
        
        # Test validation
        is_valid = config.validate_configuration()
        print(f"   ✅ Configuration validation: {'PASSED' if is_valid else 'FAILED'}")
        
        # Test agent configs
        agents = config.list_agents()
        print(f"   ✅ Configured agents: {agents}")
        
        # Test specific agent config
        indicator_config = config.get_agent_config("indicator_agent")
        print(f"   ✅ Indicator agent config: {indicator_config['provider']}:{indicator_config['model']}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Configuration test failed: {e}")
        return False


def test_client_factory():
    """Test client factory without requiring API keys."""
    print("\n🧪 Testing Client Factory...")
    
    try:
        # Create factory
        factory = LLMClientFactory()
        print(f"   ✅ Factory created")
        
        # List configured agents
        agents = factory.list_configured_agents()
        print(f"   ✅ Factory lists agents: {agents}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Client factory test failed: {e}")
        return False


def test_utilities():
    """Test utility functions."""
    print("\n🧪 Testing Utility Functions...")
    
    try:
        # Test API key validation (will show no keys, but shouldn't crash)
        keys = validate_api_keys()
        print(f"   ✅ API key validation completed: {len(keys)} providers checked")
        
        # Test formatting
        formatted = format_provider_model_name("gemini", "gemini-2.5-flash")
        print(f"   ✅ Provider formatting: {formatted}")
        
        # Test token estimation
        tokens = estimate_tokens("This is a test prompt for token estimation")
        print(f"   ✅ Token estimation: {tokens} tokens")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Utilities test failed: {e}")
        return False


def test_import_system():
    """Test that all imports work correctly."""
    print("\n🧪 Testing Import System...")
    
    try:
        # Test main imports
        from backend.llm import LLMClient, LLMConfig, get_llm_client, get_llm_config
        print("   ✅ Main module imports work")
        
        # Test provider imports
        from backend.llm.providers.gemini import GeminiProvider
        from backend.llm.providers.base import BaseLLMProvider
        print("   ✅ Provider imports work")
        
        # Test utils
        from backend.llm.utils import SimpleTimer
        print("   ✅ Utils imports work")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_client_creation():
    """Test client creation (will fail at provider init due to missing API key, but structure should work)."""
    print("\n🧪 Testing Client Creation (Structure Only)...")
    
    try:
        # This will work up until provider initialization
        config = get_llm_config()
        print("   ✅ Global config accessible")
        
        # Test that we get expected error for missing API key
        try:
            client = get_llm_client("indicator_agent")
            print("   ⚠️  Client created without API key (unexpected)")
            return False
        except ValueError as e:
            if "API key is required" in str(e):
                print("   ✅ Correct error for missing API key")
                return True
            else:
                print(f"   ❌ Unexpected error: {e}")
                return False
                
    except Exception as e:
        print(f"   ❌ Client creation test failed: {e}")
        return False


async def main():
    """Run all basic tests."""
    print("🚀 LLM System Basic Functionality Test")
    print("=" * 50)
    
    tests = [
        test_import_system,
        test_configuration,
        test_utilities,
        test_client_factory,
        test_client_creation
    ]
    
    results = []
    for test in tests:
        result = test()
        results.append(result)
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("\n" + "=" * 50)
    if passed == total:
        print(f"🎉 All basic tests passed! ({passed}/{total})")
        print("✅ The LLM system is properly structured and ready for API key configuration")
        print("\n💡 To use with real API calls:")
        print("   1. Set environment variable: GEMINI_API_KEY=your_key")
        print("   2. Or use numbered keys: GEMINI_API_KEY1, GEMINI_API_KEY2, etc.")
        print("   3. Run the full integration tests")
    else:
        print(f"⚠️  Some tests failed ({passed}/{total})")
        print("🔧 Check the errors above and fix before proceeding")
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)