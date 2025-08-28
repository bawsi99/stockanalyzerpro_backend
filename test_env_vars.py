#!/usr/bin/env python3
"""
Test script to verify environment variables are being read correctly.
Run this to debug environment variable issues in deployment.
"""

import os
from dotenv import load_dotenv

def test_environment_variables():
    """Test reading environment variables from both .env file and system environment."""
    
    print("🔍 Environment Variable Test")
    print("=" * 50)
    
    # Test loading .env file
    print("\n📁 Testing .env file loading:")
    if os.path.exists('.env'):
        print("✅ .env file exists")
        load_dotenv()
        print("✅ dotenv.load_dotenv() called")
    else:
        print("❌ .env file does not exist (this is normal in production)")
    
    # Test Zerodha variables
    print("\n🔑 Testing Zerodha Environment Variables:")
    zerodha_vars = [
        'ZERODHA_API_KEY',
        'ZERODHA_API_SECRET', 
        'ZERODHA_ACCESS_TOKEN',
        'ZERODHA_REQUEST_TOKEN'
    ]
    
    for var in zerodha_vars:
        value = os.getenv(var)
        if value:
            # Show first 8 characters for security
            display_value = value[:8] + "..." if len(value) > 8 else value
            print(f"✅ {var}: {display_value}")
        else:
            print(f"❌ {var}: Not set")
    
    # Test other important variables
    print("\n🌐 Testing Other Environment Variables:")
    other_vars = [
        'GEMINI_API_KEY',
        'SUPABASE_URL',
        'SUPABASE_KEY',
        'JWT_SECRET',
        'REQUIRE_AUTH',
        'CORS_ORIGINS'
    ]
    
    for var in other_vars:
        value = os.getenv(var)
        if value:
            if 'SECRET' in var or 'KEY' in var:
                display_value = value[:8] + "..." if len(value) > 8 else value
            else:
                display_value = value
            print(f"✅ {var}: {display_value}")
        else:
            print(f"❌ {var}: Not set")
    
    # Test the custom get_env_value function
    print("\n🔧 Testing Custom get_env_value Function:")
    try:
        from zerodha_client import get_env_value
        
        for var in zerodha_vars:
            value = get_env_value(var)
            if value:
                display_value = value[:8] + "..." if len(value) > 8 else value
                print(f"✅ get_env_value({var}): {display_value}")
            else:
                print(f"❌ get_env_value({var}): Not set")
                
    except ImportError as e:
        print(f"❌ Could not import get_env_value: {e}")
    
    print("\n" + "=" * 50)
    print("🏁 Environment variable test completed!")

if __name__ == "__main__":
    test_environment_variables()
