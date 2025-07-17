#!/usr/bin/env python3
"""
Test script for enhanced volume analysis implementation.
This script tests the comprehensive volume analysis features.
"""

import pandas as pd
import numpy as np
import sys
import os

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from technical_indicators import TechnicalIndicators

def create_sample_data():
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    
    # Create realistic price data with some trends and patterns
    base_price = 100
    prices = []
    volumes = []
    
    for i in range(100):
        # Add some trend and volatility
        trend = 0.001 * i  # Slight uptrend
        noise = np.random.normal(0, 0.02)
        price = base_price * (1 + trend + noise)
        prices.append(max(price, 50))  # Ensure positive prices
        
        # Create realistic volume patterns
        base_volume = 1000000
        volume_trend = 1 + 0.002 * i  # Slight volume uptrend
        volume_noise = np.random.normal(0, 0.3)
        
        # Add some volume spikes
        if i in [20, 45, 70]:  # Volume anomalies
            volume = base_volume * volume_trend * (1 + volume_noise) * 3  # 3x normal volume
        else:
            volume = base_volume * volume_trend * (1 + volume_noise)
        
        volumes.append(max(volume, 100000))  # Ensure positive volume
    
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

def test_enhanced_volume_analysis():
    """Test the enhanced volume analysis functionality."""
    print("🧪 Testing Enhanced Volume Analysis")
    print("=" * 50)
    
    # Create sample data
    data = create_sample_data()
    print(f"✅ Created sample data with {len(data)} records")
    print(f"   Date range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
    print(f"   Current price: ₹{data['close'].iloc[-1]:.2f}")
    print(f"   Current volume: {data['volume'].iloc[-1]:,.0f}")
    print()
    
    # Test enhanced volume analysis
    try:
        volume_analysis = TechnicalIndicators.calculate_enhanced_volume_analysis(data)
        
        if "error" in volume_analysis:
            print(f"❌ Error in volume analysis: {volume_analysis['error']}")
            return False
        
        print("📊 Enhanced Volume Analysis Results:")
        print("-" * 30)
        
        # Daily metrics
        daily_metrics = volume_analysis['daily_metrics']
        print(f"📈 Current Volume: {daily_metrics['current_volume']:,.0f}")
        print(f"💰 Volume/Price Ratio: {daily_metrics['volume_price_ratio']:.2f}")
        print()
        
        # Volume ratios
        ratios = volume_analysis['volume_ratios']
        print("📊 Volume Ratios (vs Moving Averages):")
        print(f"   5-day:  {ratios['ratio_5d']:.2f}x")
        print(f"   10-day: {ratios['ratio_10d']:.2f}x")
        print(f"   20-day: {ratios['ratio_20d']:.2f}x")
        print(f"   50-day: {ratios['ratio_50d']:.2f}x")
        print(f"   Primary: {ratios['primary_ratio']:.2f}x")
        print()
        
        # Volume trends
        trends = volume_analysis['volume_trends']
        print("📈 Volume Trends:")
        print(f"   5-day:  {trends['trend_5d']}")
        print(f"   10-day: {trends['trend_10d']}")
        print(f"   20-day: {trends['trend_20d']}")
        print(f"   50-day: {trends['trend_50d']}")
        print()
        
        # Volume strength
        print(f"💪 Volume Strength Score: {volume_analysis['volume_strength_score']}/100")
        print()
        
        # Price-volume correlation
        correlation = volume_analysis['price_volume_correlation']
        print("🔗 Price-Volume Correlation:")
        print(f"   20-day: {correlation['correlation_20d']:.3f}")
        print(f"   50-day: {correlation['correlation_50d']:.3f}")
        print(f"   Strength: {correlation['correlation_strength']}")
        print()
        
        # Volume confirmation
        confirmation = volume_analysis['volume_confirmation']
        print("✅ Volume Confirmation:")
        print(f"   Price Trend: {confirmation['price_trend']}")
        print(f"   Volume Trend: {confirmation['volume_trend']}")
        print(f"   Status: {confirmation['confirmation_status']}")
        print(f"   Strength: {confirmation['strength']}")
        print()
        
        # Volume anomalies
        anomalies = volume_analysis['volume_anomalies']
        print(f"🚨 Volume Anomalies: {anomalies['total_anomalies']} total, {anomalies['recent_anomalies']} recent")
        if anomalies['anomaly_list']:
            print("   Recent anomalies:")
            for i, anomaly in enumerate(anomalies['anomaly_list'][:3]):
                print(f"     {i+1}. {anomaly['date']}: {anomaly['volume_ratio']:.2f}x volume ({anomaly['anomaly_strength']})")
        print()
        
        # Advanced indicators
        advanced = volume_analysis['advanced_indicators']
        print("🔬 Advanced Volume Indicators:")
        print(f"   MFI: {advanced['mfi']:.1f} ({advanced['mfi_status']})")
        print(f"   OBV Trend: {advanced['obv_trend']}")
        print(f"   VWAP: ₹{advanced['vwap']:.2f}")
        print(f"   Price vs VWAP: {advanced['price_vs_vwap_pct']:+.2f}%")
        print()
        
        # Volume volatility
        volatility = volume_analysis['volume_volatility']
        print("📊 Volume Volatility:")
        print(f"   Volatility Ratio: {volatility['volatility_ratio']:.3f}")
        print(f"   Regime: {volatility['volatility_regime']}")
        print()
        
        # Volume quality
        quality = volume_analysis['volume_quality']
        print("🎯 Volume Quality:")
        print(f"   Data Quality: {quality['data_quality']}")
        print(f"   Reliability: {quality['reliability']}")
        print()
        
        print("✅ Enhanced volume analysis test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing enhanced volume analysis: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integration_with_main_indicators():
    """Test integration with main indicators calculation."""
    print("\n🔗 Testing Integration with Main Indicators")
    print("=" * 50)
    
    try:
        data = create_sample_data()
        
        # Calculate all indicators
        indicators = TechnicalIndicators.calculate_all_indicators(data)
        
        # Check if enhanced volume analysis is included
        if 'enhanced_volume' in indicators:
            enhanced_vol = indicators['enhanced_volume']
            if 'comprehensive_analysis' in enhanced_vol:
                print("✅ Enhanced volume analysis successfully integrated!")
                print(f"   Volume strength score: {enhanced_vol['comprehensive_analysis']['volume_strength_score']}/100")
                print(f"   Primary volume ratio: {enhanced_vol['comprehensive_analysis']['volume_ratios']['primary_ratio']:.2f}x")
                return True
            else:
                print("❌ Comprehensive analysis not found in enhanced_volume")
                return False
        else:
            print("❌ Enhanced volume not found in indicators")
            return False
            
    except Exception as e:
        print(f"❌ Error testing integration: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🚀 Enhanced Volume Analysis Test Suite")
    print("=" * 60)
    
    # Test 1: Enhanced volume analysis
    test1_passed = test_enhanced_volume_analysis()
    
    # Test 2: Integration with main indicators
    test2_passed = test_integration_with_main_indicators()
    
    # Summary
    print("\n📋 Test Summary")
    print("=" * 30)
    print(f"Enhanced Volume Analysis: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"Integration Test: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 All tests passed! Enhanced volume analysis is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Please check the implementation.")
    
    return test1_passed and test2_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 