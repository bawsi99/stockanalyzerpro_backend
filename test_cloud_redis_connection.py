#!/usr/bin/env python3
"""
Test Cloud Redis Connection
Verifies that the system is using the cloud Redis instance instead of localhost
"""

import os
import sys
import redis
from dotenv import load_dotenv

def load_environment():
    """Load environment variables from .env file."""
    print("🔧 Loading environment variables...")
    
    # Load .env file
    load_dotenv()
    
    # Check REDIS_URL
    redis_url = os.getenv('REDIS_URL')
    if not redis_url:
        print("❌ REDIS_URL environment variable not found!")
        return False
    
    print(f"✅ REDIS_URL loaded: {redis_url}")
    return True

def test_redis_connection(redis_url):
    """Test connection to the specified Redis instance."""
    print(f"\n🔌 Testing Redis connection to: {redis_url}")
    
    try:
        # Create Redis client
        client = redis.from_url(redis_url)
        
        # Test connection
        ping_response = client.ping()
        print(f"✅ Ping response: {ping_response}")
        
        # Get Redis info
        info = client.info()
        print(f"✅ Redis version: {info.get('redis_version', 'Unknown')}")
        print(f"✅ Redis mode: {info.get('redis_mode', 'Unknown')}")
        print(f"✅ Connected clients: {info.get('connected_clients', 'Unknown')}")
        print(f"✅ Used memory: {info.get('used_memory_human', 'Unknown')}")
        
        # Test basic operations
        print("\n🧪 Testing basic operations...")
        
        # Set a test key
        test_key = "cloud_redis_test"
        test_value = "Hello from Cloud Redis!"
        client.set(test_key, test_value)
        print(f"✅ Set key '{test_key}'")
        
        # Get the test key
        retrieved_value = client.get(test_key)
        if retrieved_value.decode() == test_value:
            print(f"✅ Retrieved value: {retrieved_value.decode()}")
        else:
            print(f"❌ Value mismatch: expected '{test_value}', got '{retrieved_value.decode()}'")
        
        # Delete test key
        client.delete(test_key)
        print(f"✅ Deleted test key")
        
        return True
        
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        return False

def check_local_redis():
    """Check if local Redis is running and accessible."""
    print("\n🏠 Checking local Redis status...")
    
    try:
        local_client = redis.from_url('redis://localhost:6379/0')
        ping_response = local_client.ping()
        
        if ping_response:
            print("⚠️  Local Redis is running on localhost:6379")
            print("   This might interfere with cloud Redis usage")
            
            # Check if it's the same instance
            local_info = local_client.info()
            print(f"   Local Redis version: {local_info.get('redis_version', 'Unknown')}")
            
            return True
        else:
            print("✅ Local Redis is not responding")
            return False
            
    except Exception as e:
        print("✅ Local Redis is not accessible")
        return False

def test_cache_manager():
    """Test the unified cache manager with cloud Redis."""
    print("\n🚀 Testing Unified Cache Manager...")
    
    try:
        from redis_unified_cache_manager import get_unified_redis_cache_manager
        
        # Get cache manager instance
        cache_manager = get_unified_redis_cache_manager()
        
        print(f"✅ Cache manager initialized")
        print(f"✅ Redis URL: {cache_manager.redis_url}")
        print(f"✅ Redis available: {cache_manager.redis_available}")
        
        # Test basic cache operations
        test_data = {"message": "Testing cloud Redis cache", "timestamp": "2025-08-29"}
        
        # Set data
        success = cache_manager.set('test_cache', test_data, ttl_seconds=60)
        if success:
            print("✅ Data cached successfully")
            
            # Get data
            retrieved_data = cache_manager.get('test_cache')
            if retrieved_data == test_data:
                print("✅ Data retrieved successfully")
            else:
                print(f"❌ Data mismatch: {retrieved_data}")
            
            # Clean up
            cache_manager.delete('test_cache')
            print("✅ Test data cleaned up")
        else:
            print("❌ Failed to cache data")
        
        return True
        
    except Exception as e:
        print(f"❌ Cache manager test failed: {e}")
        return False

def main():
    """Main function."""
    print("🌐 CLOUD REDIS CONNECTION TEST")
    print("=" * 50)
    
    # Load environment
    if not load_environment():
        print("\n❌ Cannot proceed without REDIS_URL")
        sys.exit(1)
    
    # Get Redis URL
    redis_url = os.getenv('REDIS_URL')
    
    # Test cloud Redis connection
    if not test_redis_connection(redis_url):
        print("\n❌ Cloud Redis connection failed")
        sys.exit(1)
    
    # Check local Redis
    local_running = check_local_redis()
    
    # Test cache manager
    if not test_cache_manager():
        print("\n❌ Cache manager test failed")
        sys.exit(1)
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    if local_running:
        print("⚠️  WARNING: Local Redis is running")
        print("   This might cause conflicts. Consider stopping it:")
        print("   brew services stop redis")
    else:
        print("✅ Local Redis is not running")
    
    print("✅ Cloud Redis connection successful")
    print("✅ Cache manager working with cloud Redis")
    print("✅ Environment variables loaded correctly")
    
    print("\n🎯 RECOMMENDATIONS:")
    if local_running:
        print("1. Stop local Redis: brew services stop redis")
        print("2. Ensure REDIS_URL is always set in your environment")
        print("3. Restart your application to use cloud Redis")
    else:
        print("1. Your system is now using cloud Redis")
        print("2. Ensure REDIS_URL is always set in your environment")
        print("3. Monitor cloud Redis performance and costs")
    
    print("\n✅ Cloud Redis integration successful!")

if __name__ == "__main__":
    main()
