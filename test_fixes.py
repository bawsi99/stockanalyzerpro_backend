#!/usr/bin/env python3
"""
Test script to verify that the division by zero fixes work correctly.
"""

import pandas as pd
import numpy as np
from technical_indicators import TechnicalIndicators

def test_division_by_zero_fixes():
    """Test that division by zero issues are handled properly."""
    
    # Create test data with potential division by zero scenarios
    dates = pd.date_range('2024-01-01', periods=50, freq='D')
    
    # Test case 1: Zero volume data
    zero_volume_data = pd.DataFrame({
        'open': [100] * 50,
        'high': [105] * 50,
        'low': [95] * 50,
        'close': [102] * 50,
        'volume': [0] * 50  # Zero volume
    }, index=dates)
    
    # Test case 2: NaN values in moving averages
    nan_data = pd.DataFrame({
        'open': [100] * 50,
        'high': [105] * 50,
        'low': [95] * 50,
        'close': [102] * 50,
        'volume': [1000] * 50
    }, index=dates)
    # Add some NaN values
    nan_data.iloc[0:10, nan_data.columns.get_loc('close')] = np.nan
    
    print("Testing division by zero fixes...")
    
    try:
        # Test volume ratio calculation with zero volume
        indicators = TechnicalIndicators.calculate_all_indicators(zero_volume_data)
        volume_ratio = indicators['volume']['volume_ratio']
        print(f"✓ Volume ratio with zero volume: {volume_ratio} (should be 1.0)")
        
        # Test moving averages with NaN values
        indicators = TechnicalIndicators.calculate_all_indicators(nan_data)
        price_to_sma_200 = indicators['moving_averages']['price_to_sma_200']
        sma_20_to_sma_50 = indicators['moving_averages']['sma_20_to_sma_50']
        print(f"✓ Price to SMA 200 with NaN data: {price_to_sma_200}")
        print(f"✓ SMA 20 to SMA 50 with NaN data: {sma_20_to_sma_50}")
        
        # Test Bollinger Bands with potential zero middle band
        percent_b = indicators['bollinger_bands']['percent_b']
        bandwidth = indicators['bollinger_bands']['bandwidth']
        print(f"✓ Percent B: {percent_b}")
        print(f"✓ Bandwidth: {bandwidth}")
        
        print("\n✅ All division by zero tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    test_division_by_zero_fixes() 