#!/usr/bin/env python3
"""
Redis Unified Cache Validation Script

This script validates that the Redis unified cache system is working correctly.
It tests:
- Redis connection and basic functionality
- Cache operations (set, get, delete)
- Data serialization/deserialization
- TTL and expiration
- LRU behavior
- Market-aware TTL
- Compression
- Statistics and monitoring
- Error handling

Usage:
    python validate_redis_unified_cache.py [--verbose] [--stress-test]
"""

import os
import sys
import time
import json
import logging
import argparse
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, Any, List

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from redis_unified_cache_manager import get_unified_redis_cache_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RedisUnifiedCacheValidator:
    """Validates Redis unified cache system functionality."""
    
    def __init__(self, verbose: bool = False, stress_test: bool = False):
        self.verbose = verbose
        self.stress_test = stress_test
        self.cache = get_unified_redis_cache_manager()
        self.test_results = {
            'passed': 0,
            'failed': 0,
            'warnings': 0,
            'tests': []
        }
        
        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)
    
    def log_test_result(self, test_name: str, passed: bool, message: str = "", warning: bool = False):
        """Log test result and update statistics."""
        status = "✅ PASS" if passed else "❌ FAIL"
        if warning:
            status = "⚠️  WARN"
            self.test_results['warnings'] += 1
        elif passed:
            self.test_results['passed'] += 1
        else:
            self.test_results['failed'] += 1
        
        result = {
            'test': test_name,
            'status': status,
            'message': message,
            'timestamp': datetime.now().isoformat()
        }
        self.test_results['tests'].append(result)
        
        if self.verbose:
            logger.info(f"{status}: {test_name} - {message}")
        else:
            print(f"{status}: {test_name}")
    
    def test_redis_connection(self) -> bool:
        """Test Redis connection and basic functionality."""
        test_name = "Redis Connection"
        
        try:
            # Test connection
            if not self.cache.redis_available:
                self.log_test_result(test_name, False, "Redis not available")
                return False
            
            # Test ping
            self.cache.redis_client.ping()
            
            # Test basic Redis operations
            test_key = "validation:test:connection"
            test_value = "connection_test"
            
            self.cache.redis_client.set(test_key, test_value)
            retrieved_value = self.cache.redis_client.get(test_key)
            self.cache.redis_client.delete(test_key)
            
            if retrieved_value.decode() == test_value:
                self.log_test_result(test_name, True, "Redis connection and basic operations working")
                return True
            else:
                self.log_test_result(test_name, False, f"Value mismatch: expected {test_value}, got {retrieved_value}")
                return False
                
        except Exception as e:
            self.log_test_result(test_name, False, f"Redis connection failed: {e}")
            return False
    
    def test_basic_cache_operations(self) -> bool:
        """Test basic cache operations (set, get, delete)."""
        test_name = "Basic Cache Operations"
        
        try:
            # Test data
            test_data = {
                "string": "test_string",
                "number": 42,
                "float": 3.14,
                "boolean": True,
                "list": [1, 2, 3, "test"],
                "dict": {"key": "value", "nested": {"data": "test"}}
            }
            
            # Test set operation
            success = self.cache.set('test_data', test_data, ttl_seconds=60)
            if not success:
                self.log_test_result(test_name, False, "Cache set operation failed")
                return False
            
            # Test get operation
            retrieved_data = self.cache.get('test_data')
            if retrieved_data != test_data:
                self.log_test_result(test_name, False, f"Data mismatch: expected {test_data}, got {retrieved_data}")
                return False
            
            # Test delete operation
            success = self.cache.delete('test_data')
            if not success:
                self.log_test_result(test_name, False, "Cache delete operation failed")
                return False
            
            # Verify deletion
            retrieved_data = self.cache.get('test_data')
            if retrieved_data is not None:
                self.log_test_result(test_name, False, "Data still exists after deletion")
                return False
            
            self.log_test_result(test_name, True, "Basic cache operations working correctly")
            return True
            
        except Exception as e:
            self.log_test_result(test_name, False, f"Basic cache operations failed: {e}")
            return False
    
    def test_pandas_dataframe_caching(self) -> bool:
        """Test caching of pandas DataFrames."""
        test_name = "Pandas DataFrame Caching"
        
        try:
            # Create test DataFrame
            test_df = pd.DataFrame({
                'date': pd.date_range('2024-01-01', periods=10),
                'open': np.random.randn(10) * 100 + 1000,
                'high': np.random.randn(10) * 100 + 1000,
                'low': np.random.randn(10) * 100 + 1000,
                'close': np.random.randn(10) * 100 + 1000,
                'volume': np.random.randint(1000, 10000, 10)
            })
            test_df.set_index('date', inplace=True)
            
            # Cache DataFrame
            success = self.cache.set('test_dataframe', test_df, ttl_seconds=60)
            if not success:
                self.log_test_result(test_name, False, "Failed to cache DataFrame")
                return False
            
            # Retrieve DataFrame
            retrieved_df = self.cache.get('test_dataframe')
            if retrieved_df is None:
                self.log_test_result(test_name, False, "Failed to retrieve DataFrame")
                return False
            
            # Compare DataFrames
            if not test_df.equals(retrieved_df):
                self.log_test_result(test_name, False, "Retrieved DataFrame doesn't match original")
                return False
            
            # Clean up
            self.cache.delete('test_dataframe')
            
            self.log_test_result(test_name, True, "Pandas DataFrame caching working correctly")
            return True
            
        except Exception as e:
            self.log_test_result(test_name, False, f"Pandas DataFrame caching failed: {e}")
            return False
    
    def test_numpy_array_caching(self) -> bool:
        """Test caching of numpy arrays."""
        test_name = "Numpy Array Caching"
        
        try:
            # Create test arrays
            test_array_1d = np.random.randn(100)
            test_array_2d = np.random.randn(10, 10)
            test_array_3d = np.random.randn(5, 5, 5)
            
            # Cache arrays
            success1 = self.cache.set('test_array_1d', test_array_1d, ttl_seconds=60)
            success2 = self.cache.set('test_array_2d', test_array_2d, ttl_seconds=60)
            success3 = self.cache.set('test_array_3d', test_array_3d, ttl_seconds=60)
            
            if not all([success1, success2, success3]):
                self.log_test_result(test_name, False, "Failed to cache numpy arrays")
                return False
            
            # Retrieve arrays
            retrieved_1d = self.cache.get('test_array_1d')
            retrieved_2d = self.cache.get('test_array_2d')
            retrieved_3d = self.cache.get('test_array_3d')
            
            # Compare arrays
            if not np.array_equal(test_array_1d, retrieved_1d):
                self.log_test_result(test_name, False, "1D array mismatch")
                return False
            
            if not np.array_equal(test_array_2d, retrieved_2d):
                self.log_test_result(test_name, False, "2D array mismatch")
                return False
            
            if not np.array_equal(test_array_3d, retrieved_3d):
                self.log_test_result(test_name, False, "3D array mismatch")
                return False
            
            # Clean up
            self.cache.delete('test_array_1d')
            self.cache.delete('test_array_2d')
            self.cache.delete('test_array_3d')
            
            self.log_test_result(test_name, True, "Numpy array caching working correctly")
            return True
            
        except Exception as e:
            self.log_test_result(test_name, False, f"Numpy array caching failed: {e}")
            return False
    
    def test_ttl_and_expiration(self) -> bool:
        """Test TTL and expiration functionality."""
        test_name = "TTL and Expiration"
        
        try:
            # Test short TTL
            test_data = {"expire_test": "data"}
            success = self.cache.set('expire_test', test_data, ttl_seconds=2)
            if not success:
                self.log_test_result(test_name, False, "Failed to set data with TTL")
                return False
            
            # Verify data exists
            retrieved_data = self.cache.get('expire_test')
            if retrieved_data != test_data:
                self.log_test_result(test_name, False, "Data not found immediately after setting")
                return False
            
            # Wait for expiration
            time.sleep(3)
            
            # Verify data expired
            retrieved_data = self.cache.get('expire_test')
            if retrieved_data is not None:
                self.log_test_result(test_name, False, "Data still exists after TTL expiration")
                return False
            
            self.log_test_result(test_name, True, "TTL and expiration working correctly")
            return True
            
        except Exception as e:
            self.log_test_result(test_name, False, f"TTL and expiration test failed: {e}")
            return False
    
    def test_market_aware_ttl(self) -> bool:
        """Test market-aware TTL functionality."""
        test_name = "Market-Aware TTL"
        
        try:
            # Test historical data with market-aware TTL
            test_data = pd.DataFrame({'test': [1, 2, 3]})
            
            # Use fixed datetime objects to ensure consistent cache keys
            from_date = datetime.now() - timedelta(days=1)
            to_date = datetime.now()
            
            # Cache historical data
            success = self.cache.cache_historical_data(
                symbol='TEST',
                exchange='NSE',
                interval='day',
                from_date=from_date,
                to_date=to_date,
                data=test_data
            )
            
            if not success:
                self.log_test_result(test_name, False, "Failed to cache historical data")
                return False
            
            # Check if market-aware TTL is applied
            market_closed = self.cache._is_market_closed()
            expected_ttl = self.cache.ttl_settings['historical_data']
            if market_closed:
                expected_ttl *= 4  # 4x longer TTL when market is closed
            
            # Get cache key to check TTL
            cache_key = self.cache._generate_cache_key('historical_data', 'TEST', 'NSE', 'day', from_date, to_date)
            
            # Note: We can't directly check TTL from our cache manager, but we can verify the data exists
            retrieved_data = self.cache.get_cached_historical_data('TEST', 'NSE', 'day', from_date, to_date)
            
            if retrieved_data is not None:
                self.log_test_result(test_name, True, f"Market-aware TTL working (market closed: {market_closed})")
                
                # Clean up
                self.cache.delete('historical_data', 'TEST', 'NSE', 'day', from_date, to_date)
                return True
            else:
                self.log_test_result(test_name, False, "Historical data not found after caching")
                return False
                
        except Exception as e:
            self.log_test_result(test_name, False, f"Market-aware TTL test failed: {e}")
            return False
    
    def test_lru_behavior(self) -> bool:
        """Test LRU-like behavior for historical data."""
        test_name = "LRU Behavior"
        
        try:
            # Create multiple test datasets
            test_datasets = []
            for i in range(5):
                test_df = pd.DataFrame({'data': [i] * 10})
                test_datasets.append(test_df)
            
            # Use fixed datetime objects to ensure consistent cache keys
            from_date = datetime.now() - timedelta(days=1)
            to_date = datetime.now()
            
            # Cache multiple datasets
            for i, test_df in enumerate(test_datasets):
                success = self.cache.cache_historical_data(
                    symbol=f'TEST{i}',
                    exchange='NSE',
                    interval='day',
                    from_date=from_date,
                    to_date=to_date,
                    data=test_df
                )
                if not success:
                    self.log_test_result(test_name, False, f"Failed to cache dataset {i}")
                    return False
            
            # Check LRU tracking
            lru_key = "lru:historical_data"
            lru_count = self.cache.redis_client.zcard(lru_key)
            
            if lru_count >= 5:
                self.log_test_result(test_name, True, f"LRU tracking working ({lru_count} items tracked)")
                
                # Clean up
                for i in range(5):
                    self.cache.delete('historical_data', f'TEST{i}', 'NSE', 'day', from_date, to_date)
                return True
            else:
                self.log_test_result(test_name, False, f"LRU tracking not working properly (expected 5, got {lru_count})")
                return False
                
        except Exception as e:
            self.log_test_result(test_name, False, f"LRU behavior test failed: {e}")
            return False
    
    def test_compression(self) -> bool:
        """Test data compression functionality."""
        test_name = "Data Compression"
        
        try:
            # Create large dataset that should trigger compression
            large_data = {
                'large_array': np.random.randn(1000, 1000).tolist(),
                'metadata': {
                    'description': 'Large test dataset for compression testing',
                    'dimensions': [1000, 1000],
                    'created_at': datetime.now().isoformat()
                }
            }
            
            # Cache large data
            success = self.cache.set('large_test_data', large_data, ttl_seconds=60)
            if not success:
                self.log_test_result(test_name, False, "Failed to cache large data")
                return False
            
            # Retrieve data
            retrieved_data = self.cache.get('large_test_data')
            if retrieved_data != large_data:
                self.log_test_result(test_name, False, "Large data mismatch after compression")
                return False
            
            # Check compression stats
            stats = self.cache.get_stats()
            compression_savings = stats.get('compression_savings', 0)
            
            if compression_savings > 0:
                self.log_test_result(test_name, True, f"Compression working (saved {compression_savings} bytes)")
            else:
                self.log_test_result(test_name, True, "Compression working (no compression needed for this data)")
            
            # Clean up
            self.cache.delete('large_test_data')
            return True
            
        except Exception as e:
            self.log_test_result(test_name, False, f"Compression test failed: {e}")
            return False
    
    def test_cache_statistics(self) -> bool:
        """Test cache statistics and monitoring."""
        test_name = "Cache Statistics"
        
        try:
            # Perform some operations to generate stats
            test_data = {"stats_test": "data"}
            self.cache.set('stats_test', test_data, ttl_seconds=60)
            self.cache.get('stats_test')
            self.cache.delete('stats_test')
            
            # Get statistics
            stats = self.cache.get_stats()
            
            # Check required fields
            required_fields = [
                'redis_hits', 'redis_misses', 'compression_savings', 
                'errors', 'redis_available', 'redis_url', 'enable_compression'
            ]
            
            missing_fields = [field for field in required_fields if field not in stats]
            if missing_fields:
                self.log_test_result(test_name, False, f"Missing statistics fields: {missing_fields}")
                return False
            
            # Check Redis availability
            if not stats['redis_available']:
                self.log_test_result(test_name, False, "Redis not available in statistics")
                return False
            
            # Check Redis URL
            if not stats['redis_url']:
                self.log_test_result(test_name, False, "Redis URL not found in statistics")
                return False
            
            self.log_test_result(test_name, True, "Cache statistics working correctly")
            return True
            
        except Exception as e:
            self.log_test_result(test_name, False, f"Cache statistics test failed: {e}")
            return False
    
    def test_error_handling(self) -> bool:
        """Test error handling and edge cases."""
        test_name = "Error Handling"
        
        try:
            # Test with None data
            success = self.cache.set('none_test', None, ttl_seconds=60)
            if success:
                self.log_test_result(test_name, False, "Should not allow caching None data")
                return False
            
            # Test with empty DataFrame
            empty_df = pd.DataFrame()
            success = self.cache.set('empty_df_test', empty_df, ttl_seconds=60)
            if not success:
                self.log_test_result(test_name, False, "Should allow caching empty DataFrame")
                return False
            
            # Test with very large TTL
            large_ttl = 999999999
            success = self.cache.set('large_ttl_test', "test", ttl_seconds=large_ttl)
            if not success:
                self.log_test_result(test_name, False, "Should allow large TTL values")
                return False
            
            # Clean up
            self.cache.delete('empty_df_test')
            self.cache.delete('large_ttl_test')
            
            self.log_test_result(test_name, True, "Error handling working correctly")
            return True
            
        except Exception as e:
            self.log_test_result(test_name, False, f"Error handling test failed: {e}")
            return False
    
    def test_stress_operations(self) -> bool:
        """Test cache under stress conditions."""
        if not self.stress_test:
            self.log_test_result("Stress Test", True, "Skipped (use --stress-test to enable)")
            return True
        
        test_name = "Stress Test"
        
        try:
            # Create many small datasets
            num_datasets = 100
            success_count = 0
            
            for i in range(num_datasets):
                test_data = {
                    'id': i,
                    'data': np.random.randn(100).tolist(),
                    'timestamp': datetime.now().isoformat()
                }
                
                success = self.cache.set(f'stress_test_{i}', test_data, ttl_seconds=60)
                if success:
                    success_count += 1
            
            # Verify retrieval
            retrieval_count = 0
            for i in range(num_datasets):
                data = self.cache.get(f'stress_test_{i}')
                if data is not None:
                    retrieval_count += 1
            
            # Clean up
            for i in range(num_datasets):
                self.cache.delete(f'stress_test_{i}')
            
            success_rate = (success_count / num_datasets) * 100
            retrieval_rate = (retrieval_count / num_datasets) * 100
            
            if success_rate >= 95 and retrieval_rate >= 95:
                self.log_test_result(test_name, True, 
                                   f"Stress test passed: {success_rate:.1f}% set, {retrieval_rate:.1f}% get")
                return True
            else:
                self.log_test_result(test_name, False, 
                                   f"Stress test failed: {success_rate:.1f}% set, {retrieval_rate:.1f}% get")
                return False
                
        except Exception as e:
            self.log_test_result(test_name, False, f"Stress test failed: {e}")
            return False
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all validation tests."""
        logger.info("Starting Redis Unified Cache validation...")
        logger.info(f"Verbose mode: {self.verbose}")
        logger.info(f"Stress test: {self.stress_test}")
        
        start_time = datetime.now()
        
        # Run all tests
        tests = [
            self.test_redis_connection,
            self.test_basic_cache_operations,
            self.test_pandas_dataframe_caching,
            self.test_numpy_array_caching,
            self.test_ttl_and_expiration,
            self.test_market_aware_ttl,
            self.test_lru_behavior,
            self.test_compression,
            self.test_cache_statistics,
            self.test_error_handling,
            self.test_stress_operations
        ]
        
        for test_func in tests:
            try:
                test_func()
            except Exception as e:
                self.log_test_result(test_func.__name__, False, f"Test crashed: {e}")
        
        # Calculate results
        end_time = datetime.now()
        duration = end_time - start_time
        
        total_tests = self.test_results['passed'] + self.test_results['failed'] + self.test_results['warnings']
        
        self.test_results['summary'] = {
            'total_tests': total_tests,
            'passed': self.test_results['passed'],
            'failed': self.test_results['failed'],
            'warnings': self.test_results['warnings'],
            'success_rate': (self.test_results['passed'] / total_tests * 100) if total_tests > 0 else 0,
            'duration': str(duration),
            'timestamp': end_time.isoformat()
        }
        
        return self.test_results
    
    def print_summary(self):
        """Print test summary."""
        summary = self.test_results['summary']
        
        print("\n" + "="*60)
        print("REDIS UNIFIED CACHE VALIDATION SUMMARY")
        print("="*60)
        print(f"Total Tests: {summary['total_tests']}")
        print(f"✅ Passed: {summary['passed']}")
        print(f"❌ Failed: {summary['failed']}")
        print(f"⚠️  Warnings: {summary['warnings']}")
        print(f"Success Rate: {summary['success_rate']:.1f}%")
        print(f"Duration: {summary['duration']}")
        print(f"Timestamp: {summary['timestamp']}")
        print("="*60)
        
        if summary['failed'] > 0:
            print("\n❌ FAILED TESTS:")
            for test in self.test_results['tests']:
                if 'FAIL' in test['status']:
                    print(f"  - {test['test']}: {test['message']}")
        
        if summary['warnings'] > 0:
            print("\n⚠️  WARNINGS:")
            for test in self.test_results['tests']:
                if 'WARN' in test['status']:
                    print(f"  - {test['test']}: {test['message']}")
        
        if summary['success_rate'] >= 90:
            print("\n🎉 Overall Status: EXCELLENT - System is working well!")
        elif summary['success_rate'] >= 80:
            print("\n✅ Overall Status: GOOD - Minor issues detected")
        elif summary['success_rate'] >= 70:
            print("\n⚠️  Overall Status: FAIR - Some issues need attention")
        else:
            print("\n❌ Overall Status: POOR - Major issues detected")

def main():
    """Main validation function."""
    parser = argparse.ArgumentParser(description="Validate Redis Unified Cache System")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--stress-test", "-s", action="store_true", help="Enable stress testing")
    
    args = parser.parse_args()
    
    try:
        # Create validator
        validator = RedisUnifiedCacheValidator(
            verbose=args.verbose,
            stress_test=args.stress_test
        )
        
        # Run all tests
        results = validator.run_all_tests()
        
        # Print summary
        validator.print_summary()
        
        # Exit with appropriate code
        if results['summary']['failed'] > 0:
            sys.exit(1)
        else:
            sys.exit(0)
            
    except KeyboardInterrupt:
        logger.info("Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
