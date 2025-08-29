#!/usr/bin/env python3
"""
System Health Check Script

This script performs a comprehensive health check of the entire Redis caching system.
It validates:
- Redis server health and performance
- Cache manager functionality
- Data integrity
- Performance metrics
- System configuration
- Integration points

Usage:
    python system_health_check.py [--detailed] [--performance] [--export-results]
"""

import os
import sys
import json
import time
import logging
import argparse
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from redis_unified_cache_manager import get_unified_redis_cache_manager
    # Redis image manager removed - charts are now generated in-memory
except ImportError as e:
    print(f"Warning: Could not import cache managers: {e}")

try:
    from deployment_config import DeploymentConfig
except ImportError as e:
    print(f"Warning: Could not import deployment config: {e}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SystemHealthChecker:
    """Comprehensive system health checker for Redis caching system."""
    
    def __init__(self, detailed: bool = False, performance: bool = False, export_results: bool = False):
        self.detailed = detailed
        self.performance = performance
        self.export_results = export_results
        self.health_results = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'UNKNOWN',
            'checks': [],
            'summary': {},
            'recommendations': []
        }
        
        # Initialize components
        self.redis_cache = None
        self.deployment_config = None
        
        try:
            self.redis_cache = get_unified_redis_cache_manager()
        except Exception as e:
            logger.warning(f"Could not initialize Redis cache manager: {e}")
        
        try:
            self.deployment_config = DeploymentConfig
        except Exception as e:
            logger.warning(f"Could not import deployment config: {e}")
    
    def log_check_result(self, check_name: str, status: str, message: str = "", details: Dict = None):
        """Log health check result."""
        result = {
            'check': check_name,
            'status': status,
            'message': message,
            'timestamp': datetime.now().isoformat()
        }
        
        if details:
            result['details'] = details
        
        self.health_results['checks'].append(result)
        
        # Print status
        status_icon = {
            'PASS': '✅',
            'FAIL': '❌',
            'WARN': '⚠️',
            'INFO': 'ℹ️'
        }
        
        icon = status_icon.get(status, '❓')
        print(f"{icon} {check_name}: {message}")
        
        if self.detailed and details:
            for key, value in details.items():
                print(f"    {key}: {value}")
    
    def check_redis_server_health(self) -> bool:
        """Check Redis server health and basic functionality."""
        check_name = "Redis Server Health"
        
        try:
            if not self.redis_cache or not self.redis_cache.redis_available:
                self.log_check_result(check_name, 'FAIL', "Redis cache manager not available")
                return False
            
            # Test basic Redis operations
            redis_client = self.redis_cache.redis_client
            
            # Ping test
            start_time = time.time()
            redis_client.ping()
            ping_time = (time.time() - start_time) * 1000
            
            # Get Redis info
            info = redis_client.info()
            
            # Check memory usage
            used_memory = info.get('used_memory_human', 'Unknown')
            max_memory = info.get('maxmemory_human', 'No limit')
            memory_policy = info.get('maxmemory_policy', 'Unknown')
            
            # Check client connections
            connected_clients = info.get('connected_clients', 0)
            max_clients = info.get('maxclients', 'Unknown')
            
            # Check performance metrics
            total_commands = info.get('total_commands_processed', 0)
            keyspace_hits = info.get('keyspace_hits', 0)
            keyspace_misses = info.get('keyspace_misses', 0)
            
            # Calculate hit rate
            hit_rate = 0
            if keyspace_hits + keyspace_misses > 0:
                hit_rate = (keyspace_hits / (keyspace_hits + keyspace_misses)) * 100
            
            details = {
                'ping_time_ms': f"{ping_time:.2f}",
                'used_memory': used_memory,
                'max_memory': max_memory,
                'memory_policy': memory_policy,
                'connected_clients': connected_clients,
                'max_clients': max_clients,
                'total_commands': total_commands,
                'hit_rate_percent': f"{hit_rate:.2f}%",
                'redis_version': info.get('redis_version', 'Unknown')
            }
            
            # Determine status
            if ping_time < 10 and hit_rate > 80:
                status = 'PASS'
                message = "Redis server is healthy and performing well"
            elif ping_time < 50:
                status = 'WARN'
                message = "Redis server is healthy but response time is elevated"
            else:
                status = 'FAIL'
                message = "Redis server has performance issues"
            
            self.log_check_result(check_name, status, message, details)
            return status != 'FAIL'
            
        except Exception as e:
            self.log_check_result(check_name, 'FAIL', f"Redis server health check failed: {e}")
            return False
    
    def check_cache_manager_functionality(self) -> bool:
        """Check cache manager functionality."""
        check_name = "Cache Manager Functionality"
        
        try:
            if not self.redis_cache:
                self.log_check_result(check_name, 'FAIL', "Redis cache manager not available")
                return False
            
            # Test basic operations
            test_data = {"health_check": "test", "timestamp": datetime.now().isoformat()}
            
            # Test set
            start_time = time.time()
            success = self.redis_cache.set('health_check_test', test_data, ttl_seconds=60)
            set_time = (time.time() - start_time) * 1000
            
            if not success:
                self.log_check_result(check_name, 'FAIL', "Cache set operation failed")
                return False
            
            # Test get
            start_time = time.time()
            retrieved_data = self.redis_cache.get('health_check_test')
            get_time = (time.time() - start_time) * 1000
            
            if retrieved_data != test_data:
                self.log_check_result(check_name, 'FAIL', "Cache get operation failed - data mismatch")
                return False
            
            # Test delete
            success = self.redis_cache.delete('health_check_test')
            if not success:
                self.log_check_result(check_name, 'WARN', "Cache delete operation failed")
            
            # Get cache statistics
            stats = self.redis_cache.get_stats()
            
            details = {
                'set_time_ms': f"{set_time:.2f}",
                'get_time_ms': f"{get_time:.2f}",
                'redis_hits': stats.get('redis_hits', 0),
                'redis_misses': stats.get('redis_misses', 0),
                'compression_savings': f"{stats.get('compression_savings', 0):,} bytes",
                'errors': stats.get('errors', 0)
            }
            
            # Determine status
            if set_time < 10 and get_time < 10 and stats.get('errors', 0) == 0:
                status = 'PASS'
                message = "Cache manager is functioning correctly"
            elif set_time < 50 and get_time < 50:
                status = 'WARN'
                message = "Cache manager is functional but performance is degraded"
            else:
                status = 'FAIL'
                message = "Cache manager has significant performance issues"
            
            self.log_check_result(check_name, status, message, details)
            return status != 'FAIL'
            
        except Exception as e:
            self.log_check_result(check_name, 'FAIL', f"Cache manager functionality check failed: {e}")
            return False
    
    def check_data_integrity(self) -> bool:
        """Check data integrity and serialization."""
        check_name = "Data Integrity"
        
        try:
            if not self.redis_cache:
                self.log_check_result(check_name, 'FAIL', "Redis cache manager not available")
                return False
            
            # Test different data types
            test_cases = [
                ("string", "test_string"),
                ("integer", 42),
                ("float", 3.14159),
                ("boolean", True),
                ("list", [1, 2, 3, "test"]),
                ("dict", {"key": "value", "nested": {"data": "test"}}),
                ("dataframe", pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})),
                ("numpy_array", np.array([1, 2, 3, 4, 5]))
            ]
            
            passed_tests = 0
            total_tests = len(test_cases)
            
            for data_type, test_data in test_cases:
                try:
                    # Cache data
                    success = self.redis_cache.set(f'integrity_test_{data_type}', test_data, ttl_seconds=60)
                    if not success:
                        continue
                    
                    # Retrieve data
                    retrieved_data = self.redis_cache.get(f'integrity_test_{data_type}')
                    
                    # Compare data
                    if isinstance(test_data, pd.DataFrame):
                        data_matches = test_data.equals(retrieved_data)
                    elif isinstance(test_data, np.ndarray):
                        data_matches = np.array_equal(test_data, retrieved_data)
                    else:
                        data_matches = test_data == retrieved_data
                    
                    if data_matches:
                        passed_tests += 1
                    
                    # Clean up
                    self.redis_cache.delete(f'integrity_test_{data_type}')
                    
                except Exception as e:
                    logger.debug(f"Data integrity test failed for {data_type}: {e}")
            
            integrity_rate = (passed_tests / total_tests) * 100
            
            details = {
                'passed_tests': passed_tests,
                'total_tests': total_tests,
                'integrity_rate_percent': f"{integrity_rate:.1f}%"
            }
            
            # Determine status
            if integrity_rate >= 95:
                status = 'PASS'
                message = "Data integrity is excellent"
            elif integrity_rate >= 80:
                status = 'WARN'
                message = "Data integrity is good with minor issues"
            else:
                status = 'FAIL'
                message = "Data integrity has significant issues"
            
            self.log_check_result(check_name, status, message, details)
            return status != 'FAIL'
            
        except Exception as e:
            self.log_check_result(check_name, 'FAIL', f"Data integrity check failed: {e}")
            return False
    
    def check_performance_metrics(self) -> bool:
        """Check performance metrics and benchmarks."""
        if not self.performance:
            self.log_check_result("Performance Metrics", 'INFO', "Skipped (use --performance to enable)")
            return True
        
        check_name = "Performance Metrics"
        
        try:
            if not self.redis_cache:
                self.log_check_result(check_name, 'FAIL', "Redis cache manager not available")
                return False
            
            # Performance benchmarks
            benchmarks = []
            
            # Small data benchmark
            small_data = {"small": "data"}
            start_time = time.time()
            for _ in range(100):
                self.redis_cache.set(f'small_bench_{_}', small_data, ttl_seconds=60)
            small_set_time = (time.time() - start_time) / 100 * 1000
            
            start_time = time.time()
            for _ in range(100):
                self.redis_cache.get(f'small_bench_{_}')
            small_get_time = (time.time() - start_time) / 100 * 1000
            
            # Clean up
            for _ in range(100):
                self.redis_cache.delete(f'small_bench_{_}')
            
            benchmarks.append(('small_data', small_set_time, small_get_time))
            
            # Large data benchmark
            large_data = {
                'large_array': np.random.randn(100, 100).tolist(),
                'metadata': {'size': '100x100', 'type': 'numpy_array'}
            }
            
            start_time = time.time()
            for _ in range(10):
                self.redis_cache.set(f'large_bench_{_}', large_data, ttl_seconds=60)
            large_set_time = (time.time() - start_time) / 10 * 1000
            
            start_time = time.time()
            for _ in range(10):
                self.redis_cache.get(f'large_bench_{_}')
            large_get_time = (time.time() - start_time) / 10 * 1000
            
            # Clean up
            for _ in range(10):
                self.redis_cache.delete(f'large_bench_{_}')
            
            benchmarks.append(('large_data', large_set_time, large_get_time))
            
            # Analyze results
            details = {
                'small_data_set_ms': f"{small_set_time:.2f}",
                'small_data_get_ms': f"{small_get_time:.2f}",
                'large_data_set_ms': f"{large_set_time:.2f}",
                'large_data_get_ms': f"{large_get_time:.2f}"
            }
            
            # Determine status based on performance thresholds
            small_threshold = 5  # ms
            large_threshold = 50  # ms
            
            if (small_set_time < small_threshold and small_get_time < small_threshold and
                large_set_time < large_threshold and large_get_time < large_threshold):
                status = 'PASS'
                message = "Performance is excellent"
            elif (small_set_time < small_threshold * 2 and small_get_time < small_threshold * 2 and
                  large_set_time < large_threshold * 2 and large_get_time < large_threshold * 2):
                status = 'WARN'
                message = "Performance is acceptable but could be improved"
            else:
                status = 'FAIL'
                message = "Performance is below acceptable thresholds"
            
            self.log_check_result(check_name, status, message, details)
            return status != 'FAIL'
            
        except Exception as e:
            self.log_check_result(check_name, 'FAIL', f"Performance metrics check failed: {e}")
            return False
    
    def check_system_configuration(self) -> bool:
        """Check system configuration and settings."""
        check_name = "System Configuration"
        
        try:
            config_checks = []
            
            # Check Redis cache configuration
            if self.redis_cache:
                config_checks.append(('redis_url', self.redis_cache.redis_url))
                config_checks.append(('enable_compression', self.redis_cache.enable_compression))
                config_checks.append(('lru_capacity', self.redis_cache.lru_capacity))
                config_checks.append(('cleanup_interval_minutes', self.redis_cache.cleanup_interval_minutes))
            
            # Check deployment configuration
            if self.deployment_config:
                try:
                    env = self.deployment_config.get_environment()
                    config_checks.append(('environment', env))
                    
                    if hasattr(self.deployment_config, 'get_unified_redis_cache_config'):
                        cache_config = self.deployment_config.get_unified_redis_cache_config()
                        config_checks.append(('cache_config_available', True))
                        config_checks.append(('cache_ttl_stock_data', cache_config.get('ttl_settings', {}).get('stock_data', 'Not found')))
                    else:
                        config_checks.append(('cache_config_available', False))
                        
                except Exception as e:
                    config_checks.append(('deployment_config_error', str(e)))
            
            # Check environment variables
            env_vars = [
                'REDIS_URL',
                'REDIS_CACHE_ENABLE_COMPRESSION',
                'REDIS_CACHE_CLEANUP_INTERVAL_MINUTES',
                'REDIS_CACHE_LRU_CAPACITY'
            ]
            
            for env_var in env_vars:
                value = os.getenv(env_var, 'Not set')
                config_checks.append((f'env_{env_var}', value))
            
            details = dict(config_checks)
            
            # Determine status
            missing_configs = [k for k, v in config_checks if v == 'Not set' or v == 'Not found']
            
            if not missing_configs:
                status = 'PASS'
                message = "System configuration is complete"
            elif len(missing_configs) <= 2:
                status = 'WARN'
                message = f"System configuration is mostly complete, missing: {', '.join(missing_configs)}"
            else:
                status = 'FAIL'
                message = f"System configuration is incomplete, missing: {', '.join(missing_configs)}"
            
            self.log_check_result(check_name, status, message, details)
            return status != 'FAIL'
            
        except Exception as e:
            self.log_check_result(check_name, 'FAIL', f"System configuration check failed: {e}")
            return False
    
    def check_integration_points(self) -> bool:
        """Check integration points with other system components."""
        check_name = "Integration Points"
        
        try:
            integration_checks = []
            
            # Check if cache managers can be imported
            try:
                from redis_cache_manager import get_redis_cache_manager
                integration_checks.append(('legacy_cache_manager', 'Available'))
            except ImportError:
                integration_checks.append(('legacy_cache_manager', 'Not available'))
            
            # Redis image manager removed - charts are now generated in-memory
            integration_checks.append(('redis_image_manager', 'Removed - charts are generated in-memory'))
            
            # Check if enhanced data service can use cache
            try:
                from enhanced_data_service import EnhancedDataService
                integration_checks.append(('enhanced_data_service', 'Available'))
            except ImportError:
                integration_checks.append(('enhanced_data_service', 'Not available'))
            
            # Check if ML pipeline can use cache
            try:
                from ml.quant_system.enhanced_data_pipeline import EnhancedDataPipeline
                integration_checks.append(('ml_pipeline', 'Available'))
            except ImportError:
                integration_checks.append(('ml_pipeline', 'Not available'))
            
            details = dict(integration_checks)
            
            # Determine status
            available_components = [k for k, v in integration_checks if v == 'Available']
            
            if len(available_components) >= 3:
                status = 'PASS'
                message = "Integration points are well established"
            elif len(available_components) >= 2:
                status = 'WARN'
                message = "Integration points are partially established"
            else:
                status = 'FAIL'
                message = "Integration points are limited"
            
            self.log_check_result(check_name, status, message, details)
            return status != 'FAIL'
            
        except Exception as e:
            self.log_check_result(check_name, 'FAIL', f"Integration points check failed: {e}")
            return False
    
    def generate_recommendations(self):
        """Generate recommendations based on health check results."""
        recommendations = []
        
        # Analyze results
        failed_checks = [check for check in self.health_results['checks'] if check['status'] == 'FAIL']
        warning_checks = [check for check in self.health_results['checks'] if check['status'] == 'WARN']
        
        if failed_checks:
            recommendations.append("🔴 CRITICAL: Address failed health checks immediately")
            for check in failed_checks:
                recommendations.append(f"   - Fix {check['check']}: {check['message']}")
        
        if warning_checks:
            recommendations.append("🟡 WARNING: Monitor and address warning conditions")
            for check in warning_checks:
                recommendations.append(f"   - Monitor {check['check']}: {check['message']}")
        
        # Performance recommendations
        if self.performance:
            performance_checks = [check for check in self.health_results['checks'] 
                               if 'Performance' in check['check'] or 'performance' in check['message'].lower()]
            
            if any(check['status'] == 'FAIL' for check in performance_checks):
                recommendations.append("⚡ PERFORMANCE: Optimize cache performance")
                recommendations.append("   - Review TTL settings")
                recommendations.append("   - Enable compression for large datasets")
                recommendations.append("   - Monitor Redis memory usage")
        
        # Configuration recommendations
        config_checks = [check for check in self.health_results['checks'] 
                        if 'Configuration' in check['check']]
        
        if any(check['status'] == 'WARN' for check in config_checks):
            recommendations.append("⚙️  CONFIGURATION: Review system configuration")
            recommendations.append("   - Set missing environment variables")
            recommendations.append("   - Verify deployment configuration")
        
        if not recommendations:
            recommendations.append("✅ EXCELLENT: System is healthy and well-configured")
            recommendations.append("   - Continue monitoring performance")
            recommendations.append("   - Consider enabling stress testing")
        
        self.health_results['recommendations'] = recommendations
    
    def run_health_check(self) -> Dict[str, Any]:
        """Run comprehensive health check."""
        logger.info("Starting system health check...")
        logger.info(f"Detailed mode: {self.detailed}")
        logger.info(f"Performance mode: {self.performance}")
        
        start_time = datetime.now()
        
        # Run all health checks
        checks = [
            self.check_redis_server_health,
            self.check_cache_manager_functionality,
            self.check_data_integrity,
            self.check_performance_metrics,
            self.check_system_configuration,
            self.check_integration_points
        ]
        
        passed_checks = 0
        total_checks = len(checks)
        
        for check_func in checks:
            try:
                if check_func():
                    passed_checks += 1
            except Exception as e:
                logger.error(f"Health check {check_func.__name__} crashed: {e}")
                self.log_check_result(check_func.__name__, 'FAIL', f"Check crashed: {e}")
        
        # Calculate overall status
        end_time = datetime.now()
        duration = end_time - start_time
        
        success_rate = (passed_checks / total_checks) * 100
        
        if success_rate >= 90:
            overall_status = 'HEALTHY'
        elif success_rate >= 70:
            overall_status = 'DEGRADED'
        else:
            overall_status = 'UNHEALTHY'
        
        self.health_results['overall_status'] = overall_status
        self.health_results['summary'] = {
            'total_checks': total_checks,
            'passed_checks': passed_checks,
            'success_rate_percent': f"{success_rate:.1f}%",
            'duration': str(duration),
            'timestamp': end_time.isoformat()
        }
        
        # Generate recommendations
        self.generate_recommendations()
        
        return self.health_results
    
    def print_health_summary(self):
        """Print health check summary."""
        summary = self.health_results['summary']
        overall_status = self.health_results['overall_status']
        
        print("\n" + "="*70)
        print("SYSTEM HEALTH CHECK SUMMARY")
        print("="*70)
        print(f"Overall Status: {overall_status}")
        print(f"Total Checks: {summary['total_checks']}")
        print(f"Passed Checks: {summary['passed_checks']}")
        print(f"Success Rate: {summary['success_rate_percent']}")
        print(f"Duration: {summary['duration']}")
        print(f"Timestamp: {summary['timestamp']}")
        print("="*70)
        
        # Print recommendations
        if self.health_results['recommendations']:
            print("\n📋 RECOMMENDATIONS:")
            for recommendation in self.health_results['recommendations']:
                print(f"  {recommendation}")
        
        # Print detailed results if requested
        if self.detailed:
            print("\n📊 DETAILED RESULTS:")
            for check in self.health_results['checks']:
                print(f"\n{check['check']}:")
                print(f"  Status: {check['status']}")
                print(f"  Message: {check['message']}")
                if 'details' in check:
                    print("  Details:")
                    for key, value in check['details'].items():
                        print(f"    {key}: {value}")
    
    def export_results(self, filename: str = None):
        """Export health check results to file."""
        if not self.export_results:
            return
        
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"system_health_check_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(self.health_results, f, indent=2, default=str)
            
            print(f"\n💾 Results exported to: {filename}")
            
        except Exception as e:
            logger.error(f"Failed to export results: {e}")

def main():
    """Main health check function."""
    parser = argparse.ArgumentParser(description="System Health Check for Redis Caching System")
    parser.add_argument("--detailed", "-d", action="store_true", help="Show detailed results")
    parser.add_argument("--performance", "-p", action="store_true", help="Enable performance testing")
    parser.add_argument("--export-results", "-e", action="store_true", help="Export results to JSON file")
    
    args = parser.parse_args()
    
    try:
        # Create health checker
        health_checker = SystemHealthChecker(
            detailed=args.detailed,
            performance=args.performance,
            export_results=args.export_results
        )
        
        # Run health check
        results = health_checker.run_health_check()
        
        # Print summary
        health_checker.print_health_summary()
        
        # Export results if requested
        if args.export_results:
            health_checker.export_results()
        
        # Exit with appropriate code
        if results['overall_status'] == 'UNHEALTHY':
            sys.exit(1)
        elif results['overall_status'] == 'DEGRADED':
            sys.exit(2)
        else:
            sys.exit(0)
            
    except KeyboardInterrupt:
        logger.info("Health check interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
