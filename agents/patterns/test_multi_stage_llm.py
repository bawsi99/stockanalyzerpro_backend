#!/usr/bin/env python3
"""
Test script for Multi-Stage LLM Pattern Processing

Tests the new multi-stage LLM processing capability in the pattern recognition agent.
Simulates LLM client to verify the architecture works without requiring actual LLM calls.
"""

import asyncio
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Add the backend directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pattern_recognition.multi_stage_llm_processor import MultiStageLLMProcessor
from pattern_recognition.processor import PatternRecognitionProcessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MockLLMClient:
    """Mock LLM client for testing multi-stage processing"""
    
    def __init__(self):
        self.call_count = 0
        
    async def generate(self, prompt: str, enable_code_execution: bool = False) -> str:
        """Mock LLM response generation"""
        self.call_count += 1
        
        # Simulate different responses based on prompt content
        if "market structure" in prompt.lower():
            return f"""Market Structure Analysis for the given stock:

1. Structure Quality: High reliability based on clear swing points
2. Key Swing Points: Multiple significant highs and lows forming clear structure  
3. BOS/CHOCH: Break of Structure detected at recent levels, indicating potential trend change
4. Trend Strength: Medium to strong based on price action consistency
5. Support/Resistance: Key levels identified at 1350 and 1420 price zones

The market structure shows a well-defined pattern with clear institutional activity."""

        elif "pattern detection" in prompt.lower():
            return f"""Pattern Detection Analysis:

1. Chart Patterns: Double bottom formation near completion (85% complete)
2. Volume Confirmation: Volume increasing on recent bounce, confirming pattern
3. Momentum Signals: RSI showing bullish divergence, MACD crossing above signal
4. Pattern Reliability: High confidence (8/10) based on multiple confirmations
5. Price Targets: Target 1: $1450, Target 2: $1500, Invalidation: $1340

Key patterns show strong bullish bias with good risk/reward setup."""

        elif "cross-validate" in prompt.lower() or "validation" in prompt.lower():
            return f"""Cross-Validation Analysis:

1. Confirmations: Market structure aligns with pattern analysis - both show bullish setup
2. Conflicts: Minimal conflicts detected, slight divergence in timing expectations
3. Reliable Signals: Double bottom + BOS confirmation provides high reliability (8.5/10)
4. Risk Factors: Broader market volatility could affect individual pattern completion
5. Coherence: Overall analysis is highly coherent with multiple confirmations

Strong agreement between market structure and pattern analysis."""

        elif "trading insights" in prompt.lower():
            return f"""Trading Insights and Recommendations:

1. Primary Bias: BULLISH with high conviction (8/10)
2. Entry Points: $1380-1390 on pullbacks, breakout above $1420
3. Stop Loss: Below $1340 (pattern invalidation level)
4. Targets: T1: $1450 (R:R 2.8:1), T2: $1500 (R:R 4.2:1)
5. Position Size: Risk 1-2% of capital due to clear setup
6. Key Levels: Watch $1420 breakout and $1350 support hold

High probability bullish setup with excellent risk management parameters."""

        elif "final synthesis" in prompt.lower():
            return f"""Final Comprehensive Analysis:

1. Market Outlook: BULLISH - Strong technical setup with multiple confirmations
2. Key Insights: Double bottom + BOS + volume confirmation = high probability trade
3. Primary Recommendation: BUY on pullback to $1380-1390 or breakout above $1420
4. Risk Management: Stop below $1340, targets at $1450 and $1500
5. Confidence Level: 8.5/10 - Exceptional setup with clear risk parameters
6. Monitor: $1420 resistance break and broader market conditions

This is a high-conviction bullish setup with excellent risk/reward characteristics."""

        else:
            return f"Mock LLM response #{self.call_count} for prompt analysis."

def create_test_data():
    """Create sample stock data for testing"""
    dates = pd.date_range(start='2025-08-01', periods=50, freq='D')
    
    # Create realistic OHLCV data around $1380 level
    base_price = 1380
    prices = []
    volumes = []
    
    np.random.seed(42)  # For reproducible results
    
    for i in range(50):
        # Create some trend and volatility
        trend = 0.2 * np.sin(i / 10) + 0.1 * (i / 50)
        noise = np.random.normal(0, 10)
        price = base_price + trend * 50 + noise
        prices.append(max(price, 1300))  # Floor price
        volumes.append(int(np.random.normal(1000000, 200000)))
    
    # Create OHLC from close prices
    opens = [prices[0]] + prices[:-1]
    closes = prices
    highs = [max(o, c) + np.random.uniform(0, 5) for o, c in zip(opens, closes)]
    lows = [min(o, c) - np.random.uniform(0, 5) for o, c in zip(opens, closes)]
    
    return pd.DataFrame({
        'date': dates,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes
    })

def create_test_indicators():
    """Create sample technical indicators"""
    return {
        'rsi': {'values': np.random.uniform(30, 70, 50)},
        'macd': {'values': np.random.uniform(-5, 5, 50)},
        'macd_signal': {'values': np.random.uniform(-4, 4, 50)},
        'stoch_k': {'values': np.random.uniform(20, 80, 50)},
        'stoch_d': {'values': np.random.uniform(25, 75, 50)},
        'bb_upper': np.random.uniform(1400, 1450, 50),
        'bb_lower': np.random.uniform(1350, 1380, 50),
        'ema_20': np.random.uniform(1370, 1390, 50)
    }

async def test_multi_stage_processor():
    """Test the multi-stage LLM processor directly"""
    print("🧪 Testing Multi-Stage LLM Processor")
    print("=" * 50)
    
    # Create mock LLM client
    mock_llm = MockLLMClient()
    
    # Initialize processor
    processor = MultiStageLLMProcessor(llm_client=mock_llm)
    
    # Create test data
    technical_analysis = {
        'market_structure': {
            'swing_points': {'swing_highs': [1450, 1430], 'swing_lows': [1340, 1360]},
            'bos_choch_analysis': {'recent_bos': True, 'trend_change': 'bullish'},
            'trend_analysis': {'direction': 'bullish', 'strength': 0.75}
        },
        'price_patterns': {'chart_patterns': [{'type': 'double_bottom', 'completion': 0.85}]},
        'volume_patterns': {'volume_trend': {'trend_direction': 'increasing'}},
        'momentum_patterns': {'oscillator_patterns': {'rsi': {'current_level': 55}}}
    }
    
    market_structure = technical_analysis['market_structure']
    
    # Test multi-stage processing
    result = await processor.process_multi_stage_analysis(
        symbol="RELIANCE",
        technical_analysis=technical_analysis,
        market_structure=market_structure,
        current_price=1381.70,
        context="Test multi-stage analysis"
    )
    
    # Display results
    print(f"✅ Multi-stage processing completed")
    print(f"📊 Success: {result.get('success', False)}")
    print(f"⏱️  Processing time: {result.get('processing_time', 0):.2f}s")
    print(f"🎯 Stage summary: {result.get('stage_summary', {})}")
    print(f"📈 Overall confidence: {result.get('confidence_score', 0):.3f}")
    print(f"🔧 LLM calls made: {mock_llm.call_count}")
    
    # Show stage results
    stage_results = result.get('stage_results', {}).get('stages', {})
    print(f"\n📋 Stage Results:")
    for stage, stage_result in stage_results.items():
        status = "✅" if stage_result.get('success', False) else "❌"
        print(f"  {status} {stage}: {stage_result.get('error', 'Success')}")
    
    return result

async def test_pattern_recognition_with_llm():
    """Test pattern recognition processor with multi-stage LLM"""
    print("\n🔍 Testing Pattern Recognition with Multi-Stage LLM")
    print("=" * 55)
    
    # Create mock LLM client
    mock_llm = MockLLMClient()
    
    # Initialize pattern recognition processor with LLM
    processor = PatternRecognitionProcessor(llm_client=mock_llm)
    
    # Create test data
    stock_data = create_test_data()
    indicators = create_test_indicators()
    
    # Run pattern analysis
    result = await processor.analyze_async(
        stock_data=stock_data,
        indicators=indicators,
        context="RELIANCE NSE multi-stage test",
        chart_image=None
    )
    
    # Display results
    print(f"✅ Pattern recognition completed")
    print(f"🔧 LLM Enhanced: {result.get('llm_enhanced', False)}")
    print(f"🎯 Technical confidence: {result.get('confidence_score', 0):.3f}")
    print(f"📈 Final confidence: {result.get('final_confidence_score', 0):.3f}")
    print(f"⏱️  Processing time: {result.get('processing_time', 0):.2f}s")
    print(f"🔧 LLM calls made: {mock_llm.call_count}")
    
    # Show multi-stage results if available
    multi_stage = result.get('multi_stage_llm_analysis', {})
    if multi_stage:
        print(f"\n🚀 Multi-Stage LLM Results:")
        print(f"  📊 Success: {multi_stage.get('success', False)}")
        print(f"  📈 LLM Confidence: {multi_stage.get('confidence_score', 0):.3f}")
        
        stage_summary = multi_stage.get('stage_summary', {})
        if stage_summary:
            successful = stage_summary.get('successful_stages', 0)
            total = stage_summary.get('total_stages', 0)
            print(f"  🎯 Stages completed: {successful}/{total}")
    
    return result

async def test_comparison():
    """Compare standard vs multi-stage LLM processing"""
    print("\n⚖️  Testing Comparison: Standard vs Multi-Stage LLM")
    print("=" * 55)
    
    # Test standard processing (no LLM)
    print("🔧 Standard Processing (No LLM):")
    standard_processor = PatternRecognitionProcessor(llm_client=None)
    
    stock_data = create_test_data()
    indicators = create_test_indicators()
    
    standard_result = await standard_processor.analyze_async(
        stock_data=stock_data,
        indicators=indicators,
        context="RELIANCE NSE standard test"
    )
    
    print(f"  ⏱️  Time: {standard_result.get('processing_time', 0):.2f}s")
    print(f"  🎯 Confidence: {standard_result.get('confidence_score', 0):.3f}")
    print(f"  🔧 LLM Enhanced: {standard_result.get('llm_enhanced', False)}")
    
    # Test multi-stage processing (with LLM)
    print(f"\n🚀 Multi-Stage LLM Processing:")
    mock_llm = MockLLMClient()
    llm_processor = PatternRecognitionProcessor(llm_client=mock_llm)
    
    llm_result = await llm_processor.analyze_async(
        stock_data=stock_data,
        indicators=indicators,
        context="RELIANCE NSE llm test"
    )
    
    print(f"  ⏱️  Time: {llm_result.get('processing_time', 0):.2f}s")
    print(f"  🎯 Technical Confidence: {llm_result.get('confidence_score', 0):.3f}")
    print(f"  📈 Final Confidence: {llm_result.get('final_confidence_score', 0):.3f}")
    print(f"  🔧 LLM Enhanced: {llm_result.get('llm_enhanced', False)}")
    print(f"  🔧 LLM Calls: {mock_llm.call_count}")
    
    # Compare results
    print(f"\n📊 Comparison:")
    standard_conf = standard_result.get('confidence_score', 0)
    llm_conf = llm_result.get('final_confidence_score', 0)
    improvement = ((llm_conf - standard_conf) / standard_conf * 100) if standard_conf > 0 else 0
    
    print(f"  📈 Confidence Improvement: {improvement:+.1f}%")
    print(f"  ⏱️  Time Overhead: +{llm_result.get('processing_time', 0) - standard_result.get('processing_time', 0):.2f}s")

async def main():
    """Main test execution"""
    print("🚀 Multi-Stage LLM Pattern Processing Test Suite")
    print("=" * 60)
    
    try:
        # Test 1: Multi-stage processor directly
        await test_multi_stage_processor()
        
        # Test 2: Pattern recognition with multi-stage LLM
        await test_pattern_recognition_with_llm()
        
        # Test 3: Comparison between approaches
        await test_comparison()
        
        print(f"\n🎉 All tests completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())