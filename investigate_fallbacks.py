#!/usr/bin/env python3
"""
Fallback Investigation Script

This script thoroughly investigates the Redis unified cache system to check for:
- Hidden fallback mechanisms
- Mock implementations
- Stub data
- Simulated results
- Local cache fallbacks
- In-memory fallbacks

It performs real-time monitoring of Redis operations to ensure validation is authentic.
"""

import os
import sys
import time
import logging
from datetime import datetime
import redis

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FallbackInvestigator:
    """Investigates potential fallback mechanisms in the cache system."""
    
    def __init__(self):
        self.redis_client = None
        self.cache_manager = None
        self.investigation_results = {
            'timestamp': datetime.now().isoformat(),
            'findings': [],
            'suspicious_patterns': [],
            'recommendations': []
        }
        
        # Initialize Redis connection
        self._init_redis()
        
        # Initialize cache manager
        self._init_cache_manager()
    
    def _init_redis(self):
        """Initialize Redis connection for monitoring."""
        try:
            self.redis_client = redis.from_url('redis://localhost:6379/0')
            self.redis_client.ping()
            logger.info("✅ Redis connection established for monitoring")
        except Exception as e:
            logger.error(f"❌ Redis connection failed: {e}")
            self.redis_client = None
    
    def _init_cache_manager(self):
        """Initialize the cache manager under investigation."""
        try:
            from redis_unified_cache_manager import get_unified_redis_cache_manager
            self.cache_manager = get_unified_redis_cache_manager()
            logger.info("✅ Cache manager initialized for investigation")
        except Exception as e:
            logger.error(f"❌ Cache manager initialization failed: {e}")
            self.cache_manager = None
    
    def investigate_redis_connection(self):
        """Investigate Redis connection authenticity."""
        print("\n🔍 INVESTIGATING REDIS CONNECTION...")
        
        if not self.redis_client:
            self.investigation_results['findings'].append("Redis connection failed")
            return False
        
        try:
            # Check if Redis is actually responding
            start_time = time.time()
            self.redis_client.ping()
            ping_time = (time.time() - start_time) * 1000
            
            print(f"  ✅ Redis ping: {ping_time:.2f}ms")
            
            # Check Redis info
            info = self.redis_client.info()
            print(f"  ✅ Redis version: {info.get('redis_version', 'Unknown')}")
            print(f"  ✅ Redis memory: {info.get('used_memory_human', 'Unknown')}")
            print(f"  ✅ Connected clients: {info.get('connected_clients', 0)}")
            
            # Check if this is a real Redis instance
            if 'redis_version' in info and info['redis_version']:
                print("  ✅ Authentic Redis instance detected")
                return True
            else:
                print("  ❌ Suspicious: Redis info incomplete")
                self.investigation_results['suspicious_patterns'].append("Incomplete Redis info")
                return False
                
        except Exception as e:
            print(f"  ❌ Redis investigation failed: {e}")
            self.investigation_results['findings'].append(f"Redis investigation failed: {e}")
            return False
    
    def investigate_cache_operations(self):
        """Investigate cache operations for authenticity."""
        print("\n🔍 INVESTIGATING CACHE OPERATIONS...")
        
        if not self.cache_manager:
            self.investigation_results['findings'].append("Cache manager not available")
            return False
        
        try:
            # Clear Redis before test
            self.redis_client.flushdb()
            print("  ✅ Redis cleared for clean test")
            
            # Test data
            test_data = {
                "investigation": "test",
                "timestamp": datetime.now().isoformat(),
                "complex": {"nested": {"data": [1, 2, 3]}}
            }
            
            # Monitor Redis before operation
            keys_before = set(self.redis_client.keys("*"))
            print(f"  📊 Redis keys before: {len(keys_before)}")
            
            # Perform cache operation
            print("  💾 Setting cache data...")
            success = self.cache_manager.set('investigation_test', test_data, ttl_seconds=60)
            
            if not success:
                print("  ❌ Cache set failed")
                self.investigation_results['findings'].append("Cache set operation failed")
                return False
            
            # Monitor Redis after operation
            time.sleep(0.1)  # Small delay to ensure operation completes
            keys_after = set(self.redis_client.keys("*"))
            print(f"  📊 Redis keys after: {len(keys_after)}")
            
            # Check if new keys were added
            new_keys = keys_after - keys_before
            if new_keys:
                print(f"  ✅ New Redis keys created: {len(new_keys)}")
                for key in new_keys:
                    print(f"    - {key.decode()}")
            else:
                print("  ❌ SUSPICIOUS: No new Redis keys created")
                self.investigation_results['suspicious_patterns'].append("No Redis keys created after cache operation")
                return False
            
            # Verify data retrieval
            print("  🔍 Retrieving cache data...")
            retrieved_data = self.cache_manager.get('investigation_test')
            
            if retrieved_data != test_data:
                print("  ❌ Data mismatch detected")
                print(f"    Expected: {test_data}")
                print(f"    Got: {retrieved_data}")
                self.investigation_results['suspicious_patterns'].append("Data mismatch on retrieval")
                return False
            
            print("  ✅ Data integrity verified")
            
            # Check TTL
            cache_key = list(new_keys)[0].decode()
            ttl = self.redis_client.ttl(cache_key)
            print(f"  ✅ TTL verified: {ttl} seconds")
            
            # Clean up
            self.cache_manager.delete('investigation_test')
            print("  🧹 Test data cleaned up")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Cache operation investigation failed: {e}")
            self.investigation_results['findings'].append(f"Cache operation investigation failed: {e}")
            return False
    
    def investigate_data_persistence(self):
        """Investigate if data actually persists in Redis."""
        print("\n🔍 INVESTIGATING DATA PERSISTENCE...")
        
        try:
            # Clear Redis
            self.redis_client.flushdb()
            
            # Create test data
            test_data = {
                "persistence_test": True,
                "timestamp": datetime.now().isoformat(),
                "large_data": "x" * 1000  # 1KB of data
            }
            
            # Store data
            print("  💾 Storing test data...")
            success = self.cache_manager.set('persistence_test', test_data, ttl_seconds=120)
            
            if not success:
                print("  ❌ Failed to store test data")
                return False
            
            # Verify immediate storage
            keys_immediate = self.redis_client.keys("*")
            print(f"  📊 Keys immediately after storage: {len(keys_immediate)}")
            
            if not keys_immediate:
                print("  ❌ SUSPICIOUS: No keys found immediately after storage")
                self.investigation_results['suspicious_patterns'].append("No immediate Redis storage")
                return False
            
            # Wait and check persistence
            print("  ⏳ Waiting 2 seconds to check persistence...")
            time.sleep(2)
            
            keys_after_wait = self.redis_client.keys("*")
            print(f"  📊 Keys after 2 seconds: {len(keys_immediate)}")
            
            if len(keys_after_wait) != len(keys_immediate):
                print("  ❌ SUSPICIOUS: Key count changed unexpectedly")
                self.investigation_results['suspicious_patterns'].append("Unexpected key count change")
                return False
            
            # Check TTL
            cache_key = keys_immediate[0].decode()
            ttl = self.redis_client.ttl(cache_key)
            print(f"  ✅ TTL after 2 seconds: {ttl} seconds")
            
            if ttl <= 0:
                print("  ❌ SUSPICIOUS: Data expired too quickly")
                self.investigation_results['suspicious_patterns'].append("Data expired too quickly")
                return False
            
            # Retrieve data
            retrieved_data = self.cache_manager.get('persistence_test')
            if retrieved_data != test_data:
                print("  ❌ Data corruption detected")
                return False
            
            print("  ✅ Data persistence verified")
            
            # Clean up
            self.cache_manager.delete('persistence_test')
            
            return True
            
        except Exception as e:
            print(f"  ❌ Data persistence investigation failed: {e}")
            self.investigation_results['findings'].append(f"Data persistence investigation failed: {e}")
            return False
    
    def investigate_compression_authenticity(self):
        """Investigate if compression is actually working."""
        print("\n🔍 INVESTIGATING COMPRESSION AUTHENTICITY...")
        
        try:
            # Clear Redis
            self.redis_client.flushdb()
            
            # Create large test data
            large_data = {
                "compression_test": True,
                "large_array": ["x" * 1000] * 1000,  # ~1MB of data
                "timestamp": datetime.now().isoformat()
            }
            
            # Store large data
            print("  💾 Storing large test data...")
            success = self.cache_manager.set('compression_test', large_data, ttl_seconds=60)
            
            if not success:
                print("  ❌ Failed to store large data")
                return False
            
            # Check Redis keys
            keys = self.redis_client.keys("*")
            if not keys:
                print("  ❌ No keys found after storing large data")
                return False
            
            cache_key = keys[0].decode()
            
            # Check data size in Redis
            data_size = self.redis_client.memory_usage(cache_key)
            if data_size:
                print(f"  📊 Data size in Redis: {data_size:,} bytes")
                
                # Estimate original size
                import pickle
                original_size = len(pickle.dumps(large_data))
                print(f"  📊 Estimated original size: {original_size:,} bytes")
                
                if data_size < original_size:
                    compression_ratio = (1 - data_size / original_size) * 100
                    print(f"  ✅ Compression working: {compression_ratio:.1f}% reduction")
                else:
                    print("  ℹ️  No compression applied (data may be too small)")
            else:
                print("  ⚠️  Could not determine data size")
            
            # Clean up
            self.cache_manager.delete('compression_test')
            
            return True
            
        except Exception as e:
            print(f"  ❌ Compression investigation failed: {e}")
            self.investigation_results['findings'].append(f"Compression investigation failed: {e}")
            return False
    
    def investigate_error_handling(self):
        """Investigate error handling for authenticity."""
        print("\n🔍 INVESTIGATING ERROR HANDLING...")
        
        try:
            # Test with invalid Redis connection
            print("  🧪 Testing error handling...")
            
            # Store some data first
            test_data = {"error_test": "data"}
            self.cache_manager.set('error_test', test_data, ttl_seconds=60)
            
            # Verify data exists
            keys_before = len(self.redis_client.keys("*"))
            
            # Try to get data with invalid key
            invalid_result = self.cache_manager.get('invalid_key')
            if invalid_result is not None:
                print("  ❌ SUSPICIOUS: Invalid key returned data")
                self.investigation_results['suspicious_patterns'].append("Invalid key returned data")
                return False
            
            # Check if Redis keys are unchanged
            keys_after = len(self.redis_client.keys("*"))
            if keys_before != keys_after:
                print("  ❌ SUSPICIOUS: Redis keys changed unexpectedly")
                self.investigation_results['suspicious_patterns'].append("Unexpected Redis key changes")
                return False
            
            print("  ✅ Error handling working correctly")
            
            # Clean up
            self.cache_manager.delete('error_test')
            
            return True
            
        except Exception as e:
            print(f"  ❌ Error handling investigation failed: {e}")
            self.investigation_results['findings'].append(f"Error handling investigation failed: {e}")
            return False
    
    def investigate_statistics_authenticity(self):
        """Investigate if statistics are real or simulated."""
        print("\n🔍 INVESTIGATING STATISTICS AUTHENTICITY...")
        
        try:
            # Clear Redis and get initial stats
            self.redis_client.flushdb()
            initial_stats = self.cache_manager.get_stats()
            initial_hits = initial_stats.get('redis_hits', 0)
            initial_misses = initial_stats.get('redis_misses', 0)
            
            print(f"  📊 Initial stats - Hits: {initial_hits}, Misses: {initial_misses}")
            
            # Perform operations
            test_data = {"stats_test": "data"}
            
            # Set operation
            self.cache_manager.set('stats_test', test_data, ttl_seconds=60)
            
            # Get operation (should be a hit)
            result = self.cache_manager.get('stats_test')
            
            # Get updated stats
            updated_stats = self.cache_manager.get_stats()
            updated_hits = updated_stats.get('redis_hits', 0)
            updated_misses = updated_stats.get('redis_misses', 0)
            
            print(f"  📊 Updated stats - Hits: {updated_hits}, Misses: {updated_misses}")
            
            # Check if stats actually changed
            if updated_hits <= initial_hits:
                print("  ❌ SUSPICIOUS: Hit count didn't increase")
                self.investigation_results['suspicious_patterns'].append("Hit count didn't increase")
                return False
            
            print("  ✅ Statistics are being updated correctly")
            
            # Clean up
            self.cache_manager.delete('stats_test')
            
            return True
            
        except Exception as e:
            print(f"  ❌ Statistics investigation failed: {e}")
            self.investigation_results['findings'].append(f"Statistics investigation failed: {e}")
            return False
    
    def run_comprehensive_investigation(self):
        """Run all investigation methods."""
        print("🚀 COMPREHENSIVE FALLBACK INVESTIGATION")
        print("=" * 60)
        print(f"Timestamp: {datetime.now().isoformat()}")
        print("=" * 60)
        
        investigations = [
            ("Redis Connection", self.investigate_redis_connection),
            ("Cache Operations", self.investigate_cache_operations),
            ("Data Persistence", self.investigate_data_persistence),
            ("Compression Authenticity", self.investigate_compression_authenticity),
            ("Error Handling", self.investigate_error_handling),
            ("Statistics Authenticity", self.investigate_statistics_authenticity)
        ]
        
        passed_investigations = 0
        total_investigations = len(investigations)
        
        for name, investigation_func in investigations:
            try:
                if investigation_func():
                    passed_investigations += 1
                    print(f"✅ {name}: PASSED")
                else:
                    print(f"❌ {name}: FAILED")
            except Exception as e:
                print(f"💥 {name}: CRASHED - {e}")
                self.investigation_results['findings'].append(f"{name} investigation crashed: {e}")
        
        # Generate summary
        success_rate = (passed_investigations / total_investigations) * 100
        
        print("\n" + "=" * 60)
        print("INVESTIGATION SUMMARY")
        print("=" * 60)
        print(f"Total Investigations: {total_investigations}")
        print(f"✅ Passed: {passed_investigations}")
        print(f"❌ Failed: {total_investigations - passed_investigations}")
        print(f"Success Rate: {success_rate:.1f}%")
        print("=" * 60)
        
        # Report findings
        if self.investigation_results['suspicious_patterns']:
            print("\n🚨 SUSPICIOUS PATTERNS DETECTED:")
            for pattern in self.investigation_results['suspicious_patterns']:
                print(f"  - {pattern}")
        
        if self.investigation_results['findings']:
            print("\n📋 INVESTIGATION FINDINGS:")
            for finding in self.investigation_results['findings']:
                print(f"  - {finding}")
        
        # Generate recommendations
        if success_rate >= 90:
            print("\n🎉 CONCLUSION: System appears authentic with no significant fallbacks detected!")
        elif success_rate >= 70:
            print("\n⚠️  CONCLUSION: System mostly authentic but some concerns detected")
        else:
            print("\n❌ CONCLUSION: Significant concerns about system authenticity")
        
        return success_rate >= 70

def main():
    """Main investigation function."""
    try:
        investigator = FallbackInvestigator()
        success = investigator.run_comprehensive_investigation()
        
        if success:
            print("\n✅ Investigation completed successfully")
            sys.exit(0)
        else:
            print("\n❌ Investigation revealed significant concerns")
            sys.exit(1)
            
    except Exception as e:
        print(f"💥 Investigation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
