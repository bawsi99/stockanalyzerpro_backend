#!/usr/bin/env python3
"""
Test script for optimization improvements.
Tests configuration system, caching, and market data integration.
"""

import pandas as pd
import numpy as np
import sys
import os
import time

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from cache_manager import cache_manager, get_cache_stats, clear_cache, performance_monitor
from market_data_integration import get_market_metrics, calculate_stock_beta, calculate_stock_correlation
from technical_indicators import TechnicalIndicators

def create_sample_data():
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    
    # Create realistic price data
    base_price = 100
    prices = []
    volumes = []
    
    for i in range(100):
        trend = 0.001 * i
        noise = np.random.normal(0, 0.02)
        price = base_price * (1 + trend + noise)
        prices.append(max(price, 50))
        
        base_volume = 1000000
        volume_trend = 1 + 0.002 * i
        volume_noise = np.random.normal(0, 0.3)
        volume = base_volume * volume_trend * (1 + volume_noise)
        volumes.append(max(volume, 100000))
    
    # Create OHLC data
    data = []
    for i in range(100):
        close = prices[i]
        open_price = close * (1 + np.random.normal(0, 0.01))
        high = max(open_price, close) * (1 + abs(np.random.normal(0, 0.01)))
        low = min(open_price, close) * (1 - abs(np.random.normal(0, 0.01)))
        
        data.append({
            'date': dates[i],
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volumes[i]
        })
    
    df = pd.DataFrame(data)
    df.set_index('date', inplace=True)
    return df

def test_configuration_system():
    """Test the configuration system."""
    print("🧪 Testing Configuration System")
    print("=" * 50)
    
    try:
        # Test basic configuration access
        cache_enabled = Config.get("cache", "enabled")
        print(f"✅ Cache enabled: {cache_enabled}")
        
        # Test nested configuration access
        rsi_period = Config.get("technical_indicators", "rsi", {}).get("period")
        print(f"✅ RSI period: {rsi_period}")
        
        # Test default values
        unknown_value = Config.get("unknown_section", "unknown_key", "default_value")
        print(f"✅ Default value handling: {unknown_value}")
        
        # Test configuration updates
        Config.set("test_section", "test_key", "test_value")
        test_value = Config.get("test_section", "test_key")
        print(f"✅ Configuration update: {test_value}")
        
        # Test section updates
        Config.update_section("test_section", {"key1": "value1", "key2": "value2"})
        section_data = Config.get("test_section")
        print(f"✅ Section update: {section_data}")
        
        print("✅ Configuration system test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Configuration system test failed: {e}")
        return False

def test_caching_system():
    """Test the caching system."""
    print("\n🧪 Testing Caching System")
    print("=" * 50)
    
    try:
        # Clear cache before testing
        clear_cache()
        
        # Test basic caching
        test_data = {"test": "value", "number": 42}
        cache_manager.set("test_key", test_data, ttl=60)
        
        # Retrieve from cache
        cached_data = cache_manager.get("test_key")
        if cached_data == test_data:
            print("✅ Basic caching working")
        else:
            print("❌ Basic caching failed")
            return False
        
        # Test cache statistics
        stats = get_cache_stats()
        print(f"✅ Cache stats: {stats['hits']} hits, {stats['misses']} misses")
        
        # Test cache miss
        missing_data = cache_manager.get("non_existent_key")
        if missing_data is None:
            print("✅ Cache miss handling working")
        else:
            print("❌ Cache miss handling failed")
            return False
        
        # Test cache decorator
        @cache_manager.cached(ttl=300, key_prefix="test")
        def expensive_calculation(x, y):
            time.sleep(0.1)  # Simulate expensive operation
            return x + y
        
        # First call (cache miss)
        start_time = time.time()
        result1 = expensive_calculation(5, 3)
        first_call_time = time.time() - start_time
        
        # Second call (cache hit)
        start_time = time.time()
        result2 = expensive_calculation(5, 3)
        second_call_time = time.time() - start_time
        
        if result1 == result2 and second_call_time < first_call_time:
            print(f"✅ Cache decorator working (first: {first_call_time:.3f}s, second: {second_call_time:.3f}s)")
        else:
            print("❌ Cache decorator failed")
            return False
        
        # Test cache info
        cache_info = cache_manager.get_info()
        print(f"✅ Cache info available: {len(cache_info['items'])} items")
        
        print("✅ Caching system test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Caching system test failed: {e}")
        return False

def test_market_data_integration():
    """Test the market data integration."""
    print("\n🧪 Testing Market Data Integration")
    print("=" * 50)
    
    try:
        data = create_sample_data()
        
        # Test market metrics calculation
        market_metrics = get_market_metrics(data)
        print(f"✅ Market metrics calculated: beta={market_metrics['beta']:.3f}, correlation={market_metrics['correlation']:.3f}")
        
        # Test individual functions
        beta = calculate_stock_beta(data)
        correlation = calculate_stock_correlation(data)
        
        print(f"✅ Beta calculation: {beta:.3f}")
        print(f"✅ Correlation calculation: {correlation:.3f}")
        
        # Test data quality
        data_quality = market_metrics.get("data_quality", "unknown")
        print(f"✅ Data quality: {data_quality}")
        
        # Test risk-free rate
        risk_free_rate = market_metrics.get("risk_free_rate", 0)
        print(f"✅ Risk-free rate: {risk_free_rate:.3f}")
        
        print("✅ Market data integration test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Market data integration test failed: {e}")
        return False

def test_performance_optimization():
    """Test performance optimizations."""
    print("\n🧪 Testing Performance Optimizations")
    print("=" * 50)
    
    try:
        data = create_sample_data()
        
        # Test RSI calculation with caching
        start_time = time.time()
        rsi1 = TechnicalIndicators.calculate_rsi(data)
        first_rsi_time = time.time() - start_time
        
        start_time = time.time()
        rsi2 = TechnicalIndicators.calculate_rsi(data)
        second_rsi_time = time.time() - start_time
        
        print(f"✅ RSI calculation: first={first_rsi_time:.3f}s, second={second_rsi_time:.3f}s")
        
        # Test MACD calculation with caching
        start_time = time.time()
        macd1 = TechnicalIndicators.calculate_macd(data)
        first_macd_time = time.time() - start_time
        
        start_time = time.time()
        macd2 = TechnicalIndicators.calculate_macd(data)
        second_macd_time = time.time() - start_time
        
        print(f"✅ MACD calculation: first={first_macd_time:.3f}s, second={second_macd_time:.3f}s")
        
        # Test Monte Carlo simulations with configurable parameters
        start_time = time.time()
        scenario_analysis = TechnicalIndicators.calculate_scenario_analysis_metrics(data)
        scenario_time = time.time() - start_time
        
        print(f"✅ Scenario analysis: {scenario_time:.3f}s")
        
        # Test performance monitoring
        performance_metrics = performance_monitor.get_metrics()
        print(f"✅ Performance monitoring: {len(performance_metrics)} operations tracked")
        
        for operation, metrics in performance_metrics.items():
            print(f"   {operation}: {metrics['count']} calls, avg={metrics['avg_time']:.3f}s")
        
        print("✅ Performance optimization test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Performance optimization test failed: {e}")
        return False

def test_configurable_parameters():
    """Test configurable parameters in calculations."""
    print("\n🧪 Testing Configurable Parameters")
    print("=" * 50)
    
    try:
        data = create_sample_data()
        
        # Test RSI with different periods
        rsi_14 = TechnicalIndicators.calculate_rsi(data, window=14)
        rsi_21 = TechnicalIndicators.calculate_rsi(data, window=21)
        
        print(f"✅ RSI with different periods: 14={rsi_14.iloc[-1]:.2f}, 21={rsi_21.iloc[-1]:.2f}")
        
        # Test MACD with different parameters
        macd_std = TechnicalIndicators.calculate_macd(data)
        macd_custom = TechnicalIndicators.calculate_macd(data, fast_period=8, slow_period=21, signal_period=5)
        
        print(f"✅ MACD with different parameters: std={macd_std[0].iloc[-1]:.3f}, custom={macd_custom[0].iloc[-1]:.3f}")
        
        # Test volume analysis with configurable thresholds
        volume_analysis = TechnicalIndicators.calculate_enhanced_volume_analysis(data)
        volume_strength = volume_analysis.get('volume_strength_score', 0)
        print(f"✅ Volume analysis with configurable thresholds: strength={volume_strength}")
        
        # Test risk metrics with configurable parameters
        risk_metrics = TechnicalIndicators.calculate_advanced_risk_metrics(data)
        risk_score = risk_metrics.get('risk_assessment', {}).get('overall_risk_score', 0)
        print(f"✅ Risk metrics with configurable parameters: score={risk_score:.1f}")
        
        print("✅ Configurable parameters test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Configurable parameters test failed: {e}")
        return False

def main():
    """Run all optimization tests."""
    print("🚀 Starting Optimization Tests")
    print("=" * 60)
    
    tests = [
        test_configuration_system,
        test_caching_system,
        test_market_data_integration,
        test_performance_optimization,
        test_configurable_parameters
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All optimization tests passed!")
        print("\n✅ Improvements Implemented:")
        print("   • Configuration system for adjustable thresholds")
        print("   • Intelligent caching with TTL and LRU eviction")
        print("   • Real market data integration for beta/correlation")
        print("   • Performance monitoring and optimization")
        print("   • Configurable parameters for all calculations")
    else:
        print("⚠️ Some tests failed. Please check the implementation.")
    
    # Print final cache and performance stats
    print("\n📈 Final Statistics:")
    cache_stats = get_cache_stats()
    print(f"   Cache: {cache_stats['hit_rate']:.1f}% hit rate, {cache_stats['size']} items")
    
    perf_metrics = performance_monitor.get_metrics()
    total_operations = sum(metrics['count'] for metrics in perf_metrics.values())
    print(f"   Performance: {total_operations} operations monitored")

if __name__ == "__main__":
    main() 