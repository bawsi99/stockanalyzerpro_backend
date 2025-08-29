#!/usr/bin/env python3
"""
Quick Validation Script

This script provides a quick way to validate that the Redis unified cache system
is working correctly after migration. It's designed to be fast and informative.

Usage:
    python quick_validation.py [--verbose]
"""

import os
import sys
import time
import logging
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from redis_unified_cache_manager import get_unified_redis_cache_manager
    print("✅ Redis Unified Cache Manager imported successfully")
except ImportError as e:
    print(f"❌ Failed to import Redis Unified Cache Manager: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def quick_redis_test():
    """Quick Redis connection test."""
    print("\n🔍 Testing Redis Connection...")
    
    try:
        cache = get_unified_redis_cache_manager()
        
        if not cache.redis_available:
            print("❌ Redis is not available")
            return False
        
        # Test ping
        start_time = time.time()
        cache.redis_client.ping()
        ping_time = (time.time() - start_time) * 1000
        
        print(f"✅ Redis connection successful (ping: {ping_time:.2f}ms)")
        return True
        
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        return False

def quick_cache_test():
    """Quick cache functionality test."""
    print("\n🔍 Testing Cache Functionality...")
    
    try:
        cache = get_unified_redis_cache_manager()
        
        # Test data
        test_data = {
            "test": "quick_validation",
            "timestamp": datetime.now().isoformat(),
            "numbers": [1, 2, 3, 4, 5],
            "nested": {"key": "value", "deep": {"data": "test"}}
        }
        
        # Test set
        print("  Testing cache set...")
        success = cache.set('quick_test', test_data, ttl_seconds=60)
        if not success:
            print("❌ Cache set failed")
            return False
        print("✅ Cache set successful")
        
        # Test get
        print("  Testing cache get...")
        retrieved_data = cache.get('quick_test')
        if retrieved_data != test_data:
            print("❌ Cache get failed - data mismatch")
            return False
        print("✅ Cache get successful")
        
        # Test delete
        print("  Testing cache delete...")
        success = cache.delete('quick_test')
        if not success:
            print("⚠️  Cache delete failed")
        else:
            print("✅ Cache delete successful")
        
        # Verify deletion
        retrieved_data = cache.get('quick_test')
        if retrieved_data is not None:
            print("❌ Data still exists after deletion")
            return False
        
        print("✅ Cache functionality test passed")
        return True
        
    except Exception as e:
        print(f"❌ Cache functionality test failed: {e}")
        return False

def quick_data_types_test():
    """Quick test of different data types."""
    print("\n🔍 Testing Data Types...")
    
    try:
        cache = get_unified_redis_cache_manager()
        
        # Test pandas DataFrame
        try:
            import pandas as pd
            test_df = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})
            
            success = cache.set('test_dataframe', test_df, ttl_seconds=60)
            if success:
                retrieved_df = cache.get('test_dataframe')
                if retrieved_df is not None and test_df.equals(retrieved_df):
                    print("✅ Pandas DataFrame caching working")
                else:
                    print("❌ Pandas DataFrame caching failed")
                cache.delete('test_dataframe')
            else:
                print("❌ Failed to cache DataFrame")
        except ImportError:
            print("⚠️  Pandas not available, skipping DataFrame test")
        
        # Test numpy array
        try:
            import numpy as np
            test_array = np.array([1, 2, 3, 4, 5])
            
            success = cache.set('test_array', test_array, ttl_seconds=60)
            if success:
                retrieved_array = cache.get('test_array')
                if retrieved_array is not None and np.array_equal(test_array, retrieved_array):
                    print("✅ Numpy array caching working")
                else:
                    print("❌ Numpy array caching failed")
                cache.delete('test_array')
            else:
                print("❌ Failed to cache numpy array")
        except ImportError:
            print("⚠️  Numpy not available, skipping array test")
        
        return True
        
    except Exception as e:
        print(f"❌ Data types test failed: {e}")
        return False

def quick_performance_test():
    """Quick performance test."""
    print("\n🔍 Testing Performance...")
    
    try:
        cache = get_unified_redis_cache_manager()
        
        # Small data performance test
        small_data = {"small": "data", "id": 123}
        
        # Test set performance
        start_time = time.time()
        for i in range(10):
            cache.set(f'perf_test_small_{i}', small_data, ttl_seconds=60)
        set_time = (time.time() - start_time) / 10 * 1000
        
        # Test get performance
        start_time = time.time()
        for i in range(10):
            cache.get(f'perf_test_small_{i}')
        get_time = (time.time() - start_time) / 10 * 1000
        
        # Clean up
        for i in range(10):
            cache.delete(f'perf_test_small_{i}')
        
        print(f"✅ Small data performance: set={set_time:.2f}ms, get={get_time:.2f}ms")
        
        # Performance thresholds
        if set_time < 20 and get_time < 20:
            print("✅ Performance is excellent")
        elif set_time < 50 and get_time < 50:
            print("✅ Performance is good")
        elif set_time < 100 and get_time < 100:
            print("⚠️  Performance is acceptable but could be improved")
        else:
            print("❌ Performance is below acceptable thresholds")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False

def quick_stats_test():
    """Quick test of cache statistics."""
    print("\n🔍 Testing Cache Statistics...")
    
    try:
        cache = get_unified_redis_cache_manager()
        
        # Get statistics
        stats = cache.get_stats()
        
        # Check required fields
        required_fields = [
            'redis_hits', 'redis_misses', 'compression_savings', 
            'errors', 'redis_available', 'redis_url'
        ]
        
        missing_fields = [field for field in required_fields if field not in stats]
        
        if missing_fields:
            print(f"❌ Missing statistics fields: {missing_fields}")
            return False
        
        print("✅ Cache statistics available:")
        print(f"  - Redis hits: {stats['redis_hits']}")
        print(f"  - Redis misses: {stats['redis_misses']}")
        print(f"  - Compression savings: {stats['compression_savings']:,} bytes")
        print(f"  - Errors: {stats['errors']}")
        print(f"  - Redis available: {stats['redis_available']}")
        print(f"  - Redis URL: {stats['redis_url']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Cache statistics test failed: {e}")
        return False

def quick_migration_check():
    """Quick check for migration status."""
    print("\n🔍 Checking Migration Status...")
    
    try:
        # Check if old cache files exist
        old_cache_dirs = ['cache', 'output/charts', 'ml/quant_system/cache']
        
        existing_old_caches = []
        for cache_dir in old_cache_dirs:
            if os.path.exists(cache_dir):
                existing_old_caches.append(cache_dir)
        
        if existing_old_caches:
            print(f"⚠️  Old cache directories still exist: {', '.join(existing_old_caches)}")
            print("   Consider running: python migrate_to_redis_cache.py --cleanup-local")
        else:
            print("✅ No old cache directories found - migration appears complete")
        
        # Check Redis keys
        cache = get_unified_redis_cache_manager()
        redis_client = cache.redis_client
        
        # Count cache keys
        cache_keys = redis_client.keys("cache:*")
        print(f"✅ Redis cache keys found: {len(cache_keys)}")
        
        # Check LRU tracking
        lru_keys = redis_client.keys("lru:*")
        if lru_keys:
            print(f"✅ LRU tracking active: {len(lru_keys)} LRU sets")
        else:
            print("ℹ️  No LRU tracking found (normal if no historical data cached)")
        
        return True
        
    except Exception as e:
        print(f"❌ Migration status check failed: {e}")
        return False

def main():
    """Main validation function."""
    print("🚀 QUICK VALIDATION - Redis Unified Cache System")
    print("=" * 60)
    
    start_time = datetime.now()
    
    # Run all quick tests
    tests = [
        ("Redis Connection", quick_redis_test),
        ("Cache Functionality", quick_cache_test),
        ("Data Types", quick_data_types_test),
        ("Performance", quick_performance_test),
        ("Cache Statistics", quick_stats_test),
        ("Migration Status", quick_migration_check)
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed_tests += 1
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
    
    # Calculate results
    end_time = datetime.now()
    duration = end_time - start_time
    success_rate = (passed_tests / total_tests) * 100
    
    # Print summary
    print("\n" + "=" * 60)
    print("QUICK VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Total Tests: {total_tests}")
    print(f"✅ Passed: {passed_tests}")
    print(f"❌ Failed: {total_tests - passed_tests}")
    print(f"Success Rate: {success_rate:.1f}%")
    print(f"Duration: {duration}")
    print("=" * 60)
    
    if success_rate >= 90:
        print("\n🎉 EXCELLENT: System is working well!")
        print("   Your Redis unified cache system is ready for production use.")
    elif success_rate >= 80:
        print("\n✅ GOOD: System is mostly working")
        print("   Minor issues detected, review failed tests above.")
    elif success_rate >= 70:
        print("\n⚠️  FAIR: System has some issues")
        print("   Several tests failed, review and fix issues before production use.")
    else:
        print("\n❌ POOR: System has significant issues")
        print("   Many tests failed, system needs attention before use.")
    
    # Exit with appropriate code
    if success_rate >= 80:
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
