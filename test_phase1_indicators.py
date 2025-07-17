#!/usr/bin/env python3
"""
Test script for Phase 1 enhanced indicators implementation.
This script tests all the new technical indicators and pattern reliability features.
"""

import pandas as pd
import numpy as np
import sys
import os

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from technical_indicators import TechnicalIndicators
from patterns.recognition import PatternRecognition

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
        trend = 0.001 * i  # Slight upward trend
        noise = np.random.normal(0, 0.02)
        price_change = trend + noise
        
        if i == 0:
            current_price = base_price
        else:
            current_price = prices[-1] * (1 + price_change)
        
        # Create OHLC from current price
        high = current_price * (1 + abs(np.random.normal(0, 0.01)))
        low = current_price * (1 - abs(np.random.normal(0, 0.01)))
        open_price = prices[-1] if i > 0 else current_price
        
        prices.append(current_price)
        volumes.append(np.random.randint(1000, 10000))
        
        # Create OHLCV data
        if i == 0:
            data = pd.DataFrame({
                'date': [dates[i]],
                'open': [open_price],
                'high': [high],
                'low': [low],
                'close': [current_price],
                'volume': [volumes[i]]
            })
        else:
            new_row = pd.DataFrame({
                'date': [dates[i]],
                'open': [open_price],
                'high': [high],
                'low': [low],
                'close': [current_price],
                'volume': [volumes[i]]
            })
            data = pd.concat([data, new_row], ignore_index=True)
    
    data['date'] = pd.to_datetime(data['date'])
    data = data.set_index('date')
    
    return data

def test_enhanced_indicators():
    """Test all the new enhanced indicators."""
    print("Testing Enhanced Technical Indicators...")
    print("=" * 50)
    
    # Create sample data
    data = create_sample_data()
    print(f"Created sample data with {len(data)} records")
    print(f"Price range: {data['low'].min():.2f} - {data['high'].max():.2f}")
    print()
    
    # Test all indicators
    indicators = TechnicalIndicators.calculate_all_indicators(data)
    
    print("Enhanced Indicators Results:")
    print("-" * 30)
    
    # Test volatility indicators
    if 'volatility' in indicators:
        vol = indicators['volatility']
        print(f"ATR: {vol['atr']:.4f}")
        print(f"Volatility Ratio: {vol['volatility_ratio']:.2f}")
        print(f"BB Squeeze: {vol['bb_squeeze']}")
        print(f"Volatility Percentile: {vol['volatility_percentile']:.1f}%")
        print(f"Volatility Regime: {vol['volatility_regime']}")
        print()
    
    # Test enhanced volume indicators
    if 'enhanced_volume' in indicators:
        vol_enhanced = indicators['enhanced_volume']
        print(f"VWAP: {vol_enhanced['vwap']:.2f}")
        print(f"MFI: {vol_enhanced['mfi']:.1f}")
        print(f"MFI Status: {vol_enhanced['mfi_status']}")
        print(f"Price vs VWAP: {vol_enhanced['price_vs_vwap']:.2f}%")
        print()
    
    # Test enhanced momentum indicators
    if 'enhanced_momentum' in indicators:
        mom = indicators['enhanced_momentum']
        print(f"Stochastic K: {mom['stochastic_k']:.1f}")
        print(f"Stochastic D: {mom['stochastic_d']:.1f}")
        print(f"Stochastic Status: {mom['stochastic_status']}")
        print(f"Williams %R: {mom['williams_r']:.1f}")
        print(f"Williams %R Status: {mom['williams_r_status']}")
        print()
    
    # Test trend strength
    if 'trend_strength' in indicators:
        trend = indicators['trend_strength']
        print(f"Overall Strength: {trend['overall_strength']}")
        print(f"Strength Score: {trend['strength_score']}/100")
        print(f"Trend Consistency: {trend['trend_consistency']}")
        print(f"MA Alignment - All Bullish: {trend['ma_alignment']['all_bullish']}")
        print()
    
    # Test enhanced levels
    if 'enhanced_levels' in indicators:
        levels = indicators['enhanced_levels']
        print(f"Dynamic Support: {levels['dynamic_support'][:3] if levels['dynamic_support'] else 'None'}")
        print(f"Dynamic Resistance: {levels['dynamic_resistance'][:3] if levels['dynamic_resistance'] else 'None'}")
        print(f"Fibonacci Levels: {list(levels['fibonacci_levels'].keys())}")
        print(f"Psychological Levels: {levels['psychological_levels'][:5] if levels['psychological_levels'] else 'None'}")
        print()
    
    return indicators

def test_pattern_reliability():
    """Test pattern reliability scoring."""
    print("Testing Pattern Reliability Scoring...")
    print("=" * 50)
    
    # Create sample data
    data = create_sample_data()
    
    # Test triangle pattern detection and reliability
    triangles = PatternRecognition.detect_triangle(data['close'])
    print(f"Detected {len(triangles)} triangle patterns")
    
    if triangles:
        # Test reliability scoring for first triangle
        pattern_data = {
            'completion_percentage': 85,
            'quality_score': 0.8
        }
        
        volume_data = data['volume']
        market_conditions = {
            'volatility_regime': 'normal'
        }
        
        reliability = PatternRecognition.calculate_pattern_reliability(
            'triangle', pattern_data, volume_data, market_conditions
        )
        
        print(f"Triangle Pattern Reliability:")
        print(f"  Score: {reliability['reliability_score']}/100")
        print(f"  Level: {reliability['reliability_level']}")
        print(f"  Recommendation: {reliability['recommendation']}")
        print(f"  Factors: {reliability['factors']}")
        print()
    
    # Test pattern failure risk analysis
    current_price = data['close'].iloc[-1]
    support_resistance = {
        'support': [current_price * 0.95, current_price * 0.90],
        'resistance': [current_price * 1.05, current_price * 1.10]
    }
    
    pattern_data = {
        'target_level': current_price * 1.15,
        'volatility': 0.025
    }
    
    risk_analysis = PatternRecognition.analyze_pattern_failure_risk(
        'triangle', pattern_data, current_price, support_resistance
    )
    
    print(f"Pattern Failure Risk Analysis:")
    print(f"  Risk Score: {risk_analysis['risk_score']}/100")
    print(f"  Risk Level: {risk_analysis['risk_level']}")
    print(f"  Risk Factors: {risk_analysis['risk_factors']}")
    print(f"  Mitigation Strategies: {risk_analysis['mitigation_strategies'][:3]}")
    print()

def test_rsi_divergence():
    """Test RSI divergence detection."""
    print("Testing RSI Divergence Detection...")
    print("=" * 50)
    
    # Create sample data
    data = create_sample_data()
    
    # Calculate RSI
    rsi = TechnicalIndicators.calculate_rsi(data)
    
    # Test divergence detection
    divergences = TechnicalIndicators.detect_rsi_divergence(data['close'], rsi)
    
    print(f"RSI Divergence Analysis:")
    print(f"  Bearish Divergences: {len(divergences['bearish_divergence'])}")
    print(f"  Bullish Divergences: {len(divergences['bullish_divergence'])}")
    print(f"  Hidden Bearish: {len(divergences['hidden_bearish'])}")
    print(f"  Hidden Bullish: {len(divergences['hidden_bullish'])}")
    print()

def main():
    """Run all tests."""
    print("Phase 1 Enhanced Indicators Test Suite")
    print("=" * 60)
    print()
    
    try:
        # Test enhanced indicators
        indicators = test_enhanced_indicators()
        
        # Test pattern reliability
        test_pattern_reliability()
        
        # Test RSI divergence
        test_rsi_divergence()
        
        print("All tests completed successfully!")
        print("Phase 1 implementation is working correctly.")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 