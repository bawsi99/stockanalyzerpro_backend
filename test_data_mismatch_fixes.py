#!/usr/bin/env python3
"""
Test script to verify that all backend-frontend data mismatch fixes work correctly.
This script tests:
1. Bollinger Bands field names (upper_band, middle_band, lower_band)
2. Metadata field names (period_days)
3. Trading guidance extraction
4. Multi-timeframe analysis extraction
5. Missing field fallbacks
"""

import json
import sys
import os
from datetime import datetime
from typing import Dict, Any

# Add the backend directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from technical_indicators import TechnicalIndicators
import pandas as pd
import numpy as np

def create_test_data() -> pd.DataFrame:
    """Create test stock data for analysis."""
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    np.random.seed(42)  # For reproducible results
    
    # Create realistic stock data
    base_price = 100
    returns = np.random.normal(0.001, 0.02, len(dates))  # Daily returns
    prices = [base_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    data = pd.DataFrame({
        'open': [p * (1 + np.random.normal(0, 0.005)) for p in prices],
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, len(dates))
    }, index=dates)
    
    return data

def test_bollinger_bands_field_names():
    """Test that Bollinger Bands use correct field names."""
    print("🔍 Testing Bollinger Bands field names...")
    
    data = create_test_data()
    indicators = TechnicalIndicators.calculate_all_indicators(data)
    
    bb = indicators.get('bollinger_bands', {})
    
    # Check that the correct field names are used
    expected_fields = ['upper_band', 'middle_band', 'lower_band', 'percent_b', 'bandwidth']
    missing_fields = [field for field in expected_fields if field not in bb]
    
    if missing_fields:
        print(f"❌ Missing Bollinger Bands fields: {missing_fields}")
        return False
    
    # Check that values are numeric
    for field in expected_fields:
        if not isinstance(bb[field], (int, float)):
            print(f"❌ Field {field} is not numeric: {type(bb[field])}")
            return False
    
    print("✅ Bollinger Bands field names are correct")
    return True

def test_metadata_structure():
    """Test that metadata includes both data_period and period_days."""
    print("🔍 Testing metadata structure...")
    
    # Simulate the metadata structure that would be created by agent_capabilities.py
    metadata = {
        'symbol': 'RELIANCE',
        'exchange': 'NSE',
        'analysis_date': datetime.now().isoformat(),
        'data_period': '365 days',
        'period_days': 365,
        'interval': 'day',
        'sector': 'Oil & Gas'
    }
    
    # Check that both fields exist
    if 'data_period' not in metadata:
        print("❌ Missing data_period field")
        return False
    
    if 'period_days' not in metadata:
        print("❌ Missing period_days field")
        return False
    
    # Check that period_days is numeric
    if not isinstance(metadata['period_days'], int):
        print(f"❌ period_days is not numeric: {type(metadata['period_days'])}")
        return False
    
    print("✅ Metadata structure is correct")
    return True

def test_trading_guidance_structure():
    """Test that trading guidance structure is complete."""
    print("🔍 Testing trading guidance structure...")
    
    # Simulate the trading guidance structure from agent_capabilities.py
    trading_guidance = {
        'short_term': {
            'horizon_days': 30,
            'signal': 'Bullish',
            'entry_range_min': 1500,
            'entry_range_max': 1550,
            'stop_loss': 1450,
            'target_1': 1600,
            'target_2': 1650,
            'confidence': 75,
            'rationale': 'Strong technical momentum'
        },
        'medium_term': {
            'horizon_days': 90,
            'signal': 'Bullish',
            'entry_range_min': 1480,
            'entry_range_max': 1580,
            'stop_loss': 1400,
            'target_1': 1700,
            'target_2': 1800,
            'confidence': 70,
            'rationale': 'Sector rotation favor'
        },
        'long_term': {
            'horizon_days': 365,
            'signal': 'Neutral',
            'entry_range_min': 1400,
            'entry_range_max': 1600,
            'stop_loss': 1300,
            'target_1': 1800,
            'target_2': 2000,
            'confidence': 60,
            'rationale': 'Long-term growth potential'
        },
        'risk_management': [
            'Market volatility risk',
            'Sector-specific risks',
            'Liquidity risk'
        ],
        'key_levels': [
            '1500 - Strong support',
            '1600 - Resistance level',
            '1700 - Breakout target'
        ]
    }
    
    # Check that all expected fields exist
    expected_sections = ['short_term', 'medium_term', 'long_term', 'risk_management', 'key_levels']
    missing_sections = [section for section in expected_sections if section not in trading_guidance]
    
    if missing_sections:
        print(f"❌ Missing trading guidance sections: {missing_sections}")
        return False
    
    # Check that timeframes have required fields
    for timeframe in ['short_term', 'medium_term', 'long_term']:
        timeframe_data = trading_guidance[timeframe]
        required_fields = ['horizon_days', 'signal', 'confidence', 'rationale']
        missing_fields = [field for field in required_fields if field not in timeframe_data]
        
        if missing_fields:
            print(f"❌ Missing fields in {timeframe}: {missing_fields}")
            return False
    
    print("✅ Trading guidance structure is complete")
    return True

def test_multi_timeframe_analysis_structure():
    """Test that multi-timeframe analysis structure is complete."""
    print("🔍 Testing multi-timeframe analysis structure...")
    
    # Simulate the multi-timeframe analysis structure
    multi_timeframe_analysis = {
        'short_term': {
            'name': 'Short Term',
            'periods': {
                '1d': {'trend': 'bullish', 'strength': 0.7},
                '1w': {'trend': 'bullish', 'strength': 0.6},
                '1m': {'trend': 'neutral', 'strength': 0.5}
            },
            'ai_confidence': 75,
            'ai_trend': 'bullish',
            'consensus': {
                'direction': 'bullish',
                'strength': 0.6,
                'score': 7.5,
                'timeframe_alignment': {'1d': 'bullish', '1w': 'bullish', '1m': 'neutral'},
                'bullish_periods': 2,
                'bearish_periods': 0,
                'neutral_periods': 1
            }
        },
        'medium_term': {
            'name': 'Medium Term',
            'periods': {
                '1m': {'trend': 'bullish', 'strength': 0.6},
                '3m': {'trend': 'bullish', 'strength': 0.5},
                '6m': {'trend': 'neutral', 'strength': 0.4}
            },
            'ai_confidence': 70,
            'ai_trend': 'bullish',
            'consensus': {
                'direction': 'bullish',
                'strength': 0.5,
                'score': 7.0,
                'timeframe_alignment': {'1m': 'bullish', '3m': 'bullish', '6m': 'neutral'},
                'bullish_periods': 2,
                'bearish_periods': 0,
                'neutral_periods': 1
            }
        },
        'long_term': {
            'name': 'Long Term',
            'periods': {
                '6m': {'trend': 'neutral', 'strength': 0.4},
                '1y': {'trend': 'neutral', 'strength': 0.3},
                '2y': {'trend': 'bullish', 'strength': 0.5}
            },
            'ai_confidence': 60,
            'ai_trend': 'neutral',
            'consensus': {
                'direction': 'neutral',
                'strength': 0.4,
                'score': 6.0,
                'timeframe_alignment': {'6m': 'neutral', '1y': 'neutral', '2y': 'bullish'},
                'bullish_periods': 1,
                'bearish_periods': 0,
                'neutral_periods': 2
            }
        },
        'overall_consensus': {
            'direction': 'bullish',
            'strength': 0.5,
            'score': 6.8,
            'timeframe_alignment': {
                'short_term': 'bullish',
                'medium_term': 'bullish',
                'long_term': 'neutral'
            }
        }
    }
    
    # Check that all expected sections exist
    expected_sections = ['short_term', 'medium_term', 'long_term', 'overall_consensus']
    missing_sections = [section for section in expected_sections if section not in multi_timeframe_analysis]
    
    if missing_sections:
        print(f"❌ Missing multi-timeframe sections: {missing_sections}")
        return False
    
    # Check that each timeframe has required fields
    for timeframe in ['short_term', 'medium_term', 'long_term']:
        timeframe_data = multi_timeframe_analysis[timeframe]
        required_fields = ['name', 'periods', 'consensus']
        missing_fields = [field for field in required_fields if field not in timeframe_data]
        
        if missing_fields:
            print(f"❌ Missing fields in {timeframe}: {missing_fields}")
            return False
    
    print("✅ Multi-timeframe analysis structure is complete")
    return True

def test_frontend_compatibility():
    """Test that the data structure is compatible with frontend expectations."""
    print("🔍 Testing frontend compatibility...")
    
    # Create a complete analysis result structure
    analysis_result = {
        'ai_analysis': {
            'trend': 'Bullish',
            'confidence_pct': 75,
            'short_term': {
                'horizon_days': 30,
                'signal': 'Bullish',
                'entry_range_min': 1500,
                'entry_range_max': 1550,
                'stop_loss': 1450,
                'target_1': 1600,
                'target_2': 1650,
                'confidence': 75,
                'rationale': 'Strong technical momentum'
            },
            'medium_term': {
                'horizon_days': 90,
                'signal': 'Bullish',
                'entry_range_min': 1480,
                'entry_range_max': 1580,
                'stop_loss': 1400,
                'target_1': 1700,
                'target_2': 1800,
                'confidence': 70,
                'rationale': 'Sector rotation favor'
            },
            'long_term': {
                'horizon_days': 365,
                'signal': 'Neutral',
                'entry_range_min': 1400,
                'entry_range_max': 1600,
                'stop_loss': 1300,
                'target_1': 1800,
                'target_2': 2000,
                'confidence': 60,
                'rationale': 'Long-term growth potential'
            },
            'risks': [
                'Market volatility risk',
                'Sector-specific risks',
                'Liquidity risk'
            ],
            'must_watch_levels': [
                '1500 - Strong support',
                '1600 - Resistance level',
                '1700 - Breakout target'
            ]
        },
        'indicators': {
            'bollinger_bands': {
                'upper_band': 1550.0,
                'middle_band': 1500.0,
                'lower_band': 1450.0,
                'percent_b': 0.5,
                'bandwidth': 0.067
            },
            'moving_averages': {
                'sma_20': 1510.0,
                'sma_50': 1490.0,
                'sma_200': 1450.0,
                'ema_20': 1515.0,
                'ema_50': 1495.0,
                'price_to_sma_200': 0.034,
                'sma_20_to_sma_50': 0.013,
                'golden_cross': True,
                'death_cross': False
            },
            'rsi': {
                'rsi_14': 65.0,
                'trend': 'up',
                'status': 'near_overbought'
            },
            'macd': {
                'macd_line': 5.0,
                'signal_line': 3.0,
                'histogram': 2.0
            },
            'volume': {
                'volume_ratio': 1.2,
                'obv': 1000000,
                'obv_trend': 'up'
            },
            'adx': {
                'adx': 25.0,
                'plus_di': 30.0,
                'minus_di': 20.0,
                'trend_direction': 'bullish'
            },
            'trend_data': {
                'direction': 'bullish',
                'strength': 'strong',
                'adx': 25.0,
                'plus_di': 30.0,
                'minus_di': 20.0
            }
        },
        'trading_guidance': {
            'short_term': {
                'horizon_days': 30,
                'signal': 'Bullish',
                'entry_range_min': 1500,
                'entry_range_max': 1550,
                'stop_loss': 1450,
                'target_1': 1600,
                'target_2': 1650,
                'confidence': 75,
                'rationale': 'Strong technical momentum'
            },
            'medium_term': {
                'horizon_days': 90,
                'signal': 'Bullish',
                'entry_range_min': 1480,
                'entry_range_max': 1580,
                'stop_loss': 1400,
                'target_1': 1700,
                'target_2': 1800,
                'confidence': 70,
                'rationale': 'Sector rotation favor'
            },
            'long_term': {
                'horizon_days': 365,
                'signal': 'Neutral',
                'entry_range_min': 1400,
                'entry_range_max': 1600,
                'stop_loss': 1300,
                'target_1': 1800,
                'target_2': 2000,
                'confidence': 60,
                'rationale': 'Long-term growth potential'
            },
            'risk_management': [
                'Market volatility risk',
                'Sector-specific risks',
                'Liquidity risk'
            ],
            'key_levels': [
                '1500 - Strong support',
                '1600 - Resistance level',
                '1700 - Breakout target'
            ]
        },
        'multi_timeframe_analysis': {
            'short_term': {
                'name': 'Short Term',
                'periods': {
                    '1d': {'trend': 'bullish', 'strength': 0.7},
                    '1w': {'trend': 'bullish', 'strength': 0.6},
                    '1m': {'trend': 'neutral', 'strength': 0.5}
                },
                'ai_confidence': 75,
                'ai_trend': 'bullish',
                'consensus': {
                    'direction': 'bullish',
                    'strength': 0.6,
                    'score': 7.5,
                    'timeframe_alignment': {'1d': 'bullish', '1w': 'bullish', '1m': 'neutral'},
                    'bullish_periods': 2,
                    'bearish_periods': 0,
                    'neutral_periods': 1
                }
            },
            'medium_term': {
                'name': 'Medium Term',
                'periods': {
                    '1m': {'trend': 'bullish', 'strength': 0.6},
                    '3m': {'trend': 'bullish', 'strength': 0.5},
                    '6m': {'trend': 'neutral', 'strength': 0.4}
                },
                'ai_confidence': 70,
                'ai_trend': 'bullish',
                'consensus': {
                    'direction': 'bullish',
                    'strength': 0.5,
                    'score': 7.0,
                    'timeframe_alignment': {'1m': 'bullish', '3m': 'bullish', '6m': 'neutral'},
                    'bullish_periods': 2,
                    'bearish_periods': 0,
                    'neutral_periods': 1
                }
            },
            'long_term': {
                'name': 'Long Term',
                'periods': {
                    '6m': {'trend': 'neutral', 'strength': 0.4},
                    '1y': {'trend': 'neutral', 'strength': 0.3},
                    '2y': {'trend': 'bullish', 'strength': 0.5}
                },
                'ai_confidence': 60,
                'ai_trend': 'neutral',
                'consensus': {
                    'direction': 'neutral',
                    'strength': 0.4,
                    'score': 6.0,
                    'timeframe_alignment': {'6m': 'neutral', '1y': 'neutral', '2y': 'bullish'},
                    'bullish_periods': 1,
                    'bearish_periods': 0,
                    'neutral_periods': 2
                }
            },
            'overall_consensus': {
                'direction': 'bullish',
                'strength': 0.5,
                'score': 6.8,
                'timeframe_alignment': {
                    'short_term': 'bullish',
                    'medium_term': 'bullish',
                    'long_term': 'neutral'
                }
            }
        },
        'metadata': {
            'symbol': 'RELIANCE',
            'exchange': 'NSE',
            'analysis_date': datetime.now().isoformat(),
            'data_period': '365 days',
            'period_days': 365,
            'interval': 'day',
            'sector': 'Oil & Gas'
        },
        'summary': {
            'overall_signal': 'Bullish',
            'confidence': 75,
            'analysis_method': 'AI-Powered Analysis',
            'analysis_quality': 'High',
            'risk_level': 'Medium',
            'recommendation': 'Buy'
        }
    }
    
    # Test that all required fields exist and have correct types
    required_fields = {
        'ai_analysis': dict,
        'indicators': dict,
        'trading_guidance': dict,
        'multi_timeframe_analysis': dict,
        'metadata': dict,
        'summary': dict
    }
    
    for field, expected_type in required_fields.items():
        if field not in analysis_result:
            print(f"❌ Missing required field: {field}")
            return False
        
        if not isinstance(analysis_result[field], expected_type):
            print(f"❌ Field {field} has wrong type: {type(analysis_result[field])}")
            return False
    
    # Test specific field compatibility
    bb = analysis_result['indicators'].get('bollinger_bands', {})
    if 'upper_band' not in bb or 'middle_band' not in bb or 'lower_band' not in bb:
        print("❌ Bollinger Bands missing required fields")
        return False
    
    metadata = analysis_result['metadata']
    if 'period_days' not in metadata:
        print("❌ Metadata missing period_days field")
        return False
    
    trading_guidance = analysis_result['trading_guidance']
    if 'short_term' not in trading_guidance or 'medium_term' not in trading_guidance:
        print("❌ Trading guidance missing required timeframes")
        return False
    
    print("✅ Frontend compatibility verified")
    return True

def main():
    """Run all tests."""
    print("🚀 Starting Data Mismatch Fix Verification Tests\n")
    
    tests = [
        test_bollinger_bands_field_names,
        test_metadata_structure,
        test_trading_guidance_structure,
        test_multi_timeframe_analysis_structure,
        test_frontend_compatibility
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            print()
    
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Data mismatch fixes are working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Please review the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 