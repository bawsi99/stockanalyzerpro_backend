#!/usr/bin/env python3
"""
Test Script for Pattern Analysis Agents

This script tests all the implemented pattern analysis agents:
- Reversal Patterns Agent
- Continuation Patterns Agent  
- Technical Overview Agent
- Pattern Agents Orchestrator

Usage: python test_pattern_agents.py
"""

import sys
import os
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import traceback

# Add the backend directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_test_data(periods=100):
    """Generate realistic test stock data with indicators"""
    
    # Generate dates
    dates = pd.date_range(start=datetime.now() - timedelta(days=periods), periods=periods, freq='D')
    
    # Generate price data with trend and volatility
    np.random.seed(42)  # For reproducible results
    
    # Starting price
    start_price = 100.0
    
    # Generate returns with some trend and volatility
    trend = 0.001  # Small upward trend
    volatility = 0.02
    
    returns = np.random.normal(trend, volatility, periods)
    
    # Create price series
    prices = [start_price]
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # Generate OHLC data
    closes = np.array(prices)
    
    # Generate highs and lows around close prices
    highs = closes * (1 + np.random.uniform(0, 0.03, periods))  # Up to 3% above close
    lows = closes * (1 - np.random.uniform(0, 0.03, periods))   # Up to 3% below close
    opens = closes * (1 + np.random.uniform(-0.02, 0.02, periods))  # +/- 2% from close
    
    # Ensure OHLC relationship is valid
    for i in range(periods):
        highs[i] = max(highs[i], opens[i], closes[i])
        lows[i] = min(lows[i], opens[i], closes[i])
    
    # Generate volume data
    base_volume = 1000000  # 1M base volume
    volume = np.random.lognormal(np.log(base_volume), 0.5, periods).astype(int)
    
    # Create DataFrame
    stock_data = pd.DataFrame({
        'open': opens,
        'high': highs, 
        'low': lows,
        'close': closes,
        'volume': volume
    }, index=dates)
    
    # Generate technical indicators
    indicators = generate_test_indicators(stock_data)
    
    return stock_data, indicators

def generate_test_indicators(stock_data):
    """Generate test technical indicators"""
    
    close_prices = stock_data['close']
    high_prices = stock_data['high']
    low_prices = stock_data['low']
    volume = stock_data['volume']
    
    indicators = {}
    
    # Simple Moving Averages
    indicators['sma_20'] = close_prices.rolling(window=20, min_periods=1).mean().values
    indicators['sma_50'] = close_prices.rolling(window=50, min_periods=1).mean().values
    
    # Exponential Moving Average
    indicators['ema_12'] = close_prices.ewm(span=12).mean().values
    
    # RSI
    delta = close_prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
    rs = gain / loss
    indicators['rsi'] = (100 - (100 / (1 + rs))).values
    
    # MACD
    ema_12 = close_prices.ewm(span=12).mean()
    ema_26 = close_prices.ewm(span=26).mean()
    macd = ema_12 - ema_26
    macd_signal = macd.ewm(span=9).mean()
    macd_histogram = macd - macd_signal
    
    indicators['macd'] = macd.values
    indicators['macd_signal'] = macd_signal.values
    indicators['macd_histogram'] = macd_histogram.values
    
    # Bollinger Bands
    bb_middle = close_prices.rolling(window=20, min_periods=1).mean()
    bb_std = close_prices.rolling(window=20, min_periods=1).std()
    indicators['bb_upper'] = (bb_middle + (bb_std * 2)).values
    indicators['bb_middle'] = bb_middle.values
    indicators['bb_lower'] = (bb_middle - (bb_std * 2)).values
    
    # Stochastic
    high_14 = high_prices.rolling(window=14, min_periods=1).max()
    low_14 = low_prices.rolling(window=14, min_periods=1).min()
    stoch_k = ((close_prices - low_14) / (high_14 - low_14) * 100).fillna(50)
    indicators['stoch_k'] = stoch_k.values
    indicators['stoch_d'] = stoch_k.rolling(window=3, min_periods=1).mean().values
    
    # Simple ADX approximation (using volatility)
    returns = close_prices.pct_change().fillna(0)
    volatility = returns.rolling(window=14, min_periods=1).std()
    indicators['adx'] = (volatility * 100).clip(0, 100).fillna(25).values
    
    return indicators

async def test_reversal_patterns_agent():
    """Test the Reversal Patterns Agent"""
    logger.info("Testing Reversal Patterns Agent...")
    
    try:
        from agents.patterns.reversal import ReversalPatternsProcessor, ReversalPatternsCharts
        
        # Generate test data
        stock_data, indicators = generate_test_data(100)
        
        # Test processor
        processor = ReversalPatternsProcessor()
        logger.info(f"Created {processor.name}: {processor.description}")
        
        # Run analysis
        result = await processor.analyze_async(stock_data, indicators, "Test context")
        
        # Verify results
        assert result['agent_name'] == 'reversal_patterns'
        assert 'reversal_patterns' in result
        assert 'confidence_score' in result
        assert 'primary_signal' in result
        
        logger.info(f"✅ Reversal Patterns Processor: {result['primary_signal']} (confidence: {result['confidence_score']:.2f})")
        logger.info(f"   Patterns detected: {len(result['reversal_patterns']['divergences'])} divergences, "
                   f"{len(result['reversal_patterns']['double_patterns'])} double patterns")
        
        # Test charts
        charts = ReversalPatternsCharts()
        chart_data = await charts.create_chart(stock_data, indicators)
        
        assert isinstance(chart_data, bytes)
        assert len(chart_data) > 0
        
        logger.info(f"✅ Reversal Patterns Charts: Generated {len(chart_data)} bytes")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Reversal Patterns Agent failed: {str(e)}")
        traceback.print_exc()
        return False

async def test_continuation_patterns_agent():
    """Test the Continuation Patterns Agent"""
    logger.info("Testing Continuation Patterns Agent...")
    
    try:
        from agents.patterns.continuation import ContinuationPatternsProcessor, ContinuationPatternsCharts
        
        # Generate test data
        stock_data, indicators = generate_test_data(100)
        
        # Test processor
        processor = ContinuationPatternsProcessor()
        logger.info(f"Created {processor.name}: {processor.description}")
        
        # Run analysis
        result = await processor.analyze_async(stock_data, indicators, "Test context")
        
        # Verify results
        assert result['agent_name'] == 'continuation_patterns'
        assert 'continuation_patterns' in result
        assert 'key_levels' in result
        assert 'confidence_score' in result
        
        logger.info(f"✅ Continuation Patterns Processor: {result['primary_signal']} (confidence: {result['confidence_score']:.2f})")
        logger.info(f"   Support levels: {len(result['key_levels']['support_levels'])}, "
                   f"Resistance levels: {len(result['key_levels']['resistance_levels'])}")
        
        # Test charts
        charts = ContinuationPatternsCharts()
        chart_data = await charts.create_chart(stock_data, indicators)
        
        assert isinstance(chart_data, bytes)
        assert len(chart_data) > 0
        
        logger.info(f"✅ Continuation Patterns Charts: Generated {len(chart_data)} bytes")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Continuation Patterns Agent failed: {str(e)}")
        traceback.print_exc()
        return False

async def test_technical_overview_agent():
    """Test the Technical Overview Agent"""
    logger.info("Testing Technical Overview Agent...")
    
    try:
        from agents.patterns.technical_overview import TechnicalOverviewProcessor, TechnicalOverviewCharts
        
        # Generate test data
        stock_data, indicators = generate_test_data(100)
        
        # Test processor
        processor = TechnicalOverviewProcessor()
        logger.info(f"Created {processor.name}: {processor.description}")
        
        # Run analysis
        result = await processor.analyze_async(stock_data, indicators, "Test context")
        
        # Verify results
        assert result['agent_name'] == 'technical_overview'
        assert 'trend_analysis' in result
        assert 'volume_analysis' in result
        assert 'momentum_analysis' in result
        assert 'support_resistance' in result
        assert 'risk_assessment' in result
        assert 'confidence_score' in result
        
        trend = result['trend_analysis']
        logger.info(f"✅ Technical Overview Processor: {trend['overall_trend']} trend "
                   f"({trend['trend_strength']}) - confidence: {result['confidence_score']}")
        logger.info(f"   Volume: {result['volume_analysis']['volume_confirmation']}, "
                   f"Momentum: {result['momentum_analysis']['momentum_alignment']}")
        
        # Test charts
        charts = TechnicalOverviewCharts()
        chart_data = await charts.create_chart(stock_data, indicators)
        
        assert isinstance(chart_data, bytes)
        assert len(chart_data) > 0
        
        logger.info(f"✅ Technical Overview Charts: Generated {len(chart_data)} bytes")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Technical Overview Agent failed: {str(e)}")
        traceback.print_exc()
        return False

async def test_patterns_orchestrator():
    """Test the Pattern Agents Orchestrator"""
    logger.info("Testing Pattern Agents Orchestrator...")
    
    try:
        from agents.patterns import PatternAgentsOrchestrator
        
        # Generate test data
        stock_data, indicators = generate_test_data(100)
        
        # Test orchestrator
        orchestrator = PatternAgentsOrchestrator()
        logger.info("Created Pattern Agents Orchestrator")
        
        # Run comprehensive analysis
        result = await orchestrator.analyze_patterns_comprehensive(
            symbol="TEST", 
            stock_data=stock_data, 
            indicators=indicators,
            context="Test orchestrator analysis"
        )
        
        # Verify results
        assert result.successful_agents >= 0
        assert result.total_processing_time > 0
        assert isinstance(result.individual_results, dict)
        assert isinstance(result.unified_analysis, dict)
        
        logger.info(f"✅ Patterns Orchestrator: {result.successful_agents}/{result.successful_agents + result.failed_agents} agents succeeded")
        logger.info(f"   Processing time: {result.total_processing_time:.2f}s")
        logger.info(f"   Overall confidence: {result.overall_confidence:.2f}")
        
        # Check individual agent results
        for agent_name, agent_result in result.individual_results.items():
            if agent_result.success:
                logger.info(f"   ✅ {agent_name}: {agent_result.processing_time:.2f}s (confidence: {agent_result.confidence_score:.2f})")
            else:
                logger.warning(f"   ❌ {agent_name}: {agent_result.error_message}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Pattern Agents Orchestrator failed: {str(e)}")
        traceback.print_exc()
        return False

async def test_pattern_package_imports():
    """Test that the pattern package imports work correctly"""
    logger.info("Testing Pattern Package Imports...")
    
    try:
        # Test main package import
        from agents.patterns import (
            ReversalPatternsProcessor,
            ReversalPatternsCharts,
            ContinuationPatternsProcessor, 
            ContinuationPatternsCharts,
            TechnicalOverviewProcessor,
            TechnicalOverviewCharts,
            PatternAgentsOrchestrator,
            patterns_orchestrator
        )
        
        logger.info("✅ All pattern package imports successful")
        
        # Test that singleton orchestrator is available
        assert patterns_orchestrator is not None
        assert isinstance(patterns_orchestrator, PatternAgentsOrchestrator)
        
        logger.info("✅ Singleton patterns orchestrator available")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Pattern package imports failed: {str(e)}")
        traceback.print_exc()
        return False

def save_test_chart(chart_data, filename):
    """Save chart data to file for visual inspection"""
    try:
        with open(f"test_outputs/{filename}", 'wb') as f:
            f.write(chart_data)
        logger.info(f"📊 Saved test chart: test_outputs/{filename}")
    except Exception as e:
        logger.warning(f"Could not save chart {filename}: {str(e)}")

async def main():
    """Run all pattern agent tests"""
    logger.info("🧪 Starting Pattern Agents Test Suite")
    logger.info("=" * 60)
    
    # Create output directory
    os.makedirs("test_outputs", exist_ok=True)
    
    test_results = []
    
    # Run all tests
    tests = [
        ("Package Imports", test_pattern_package_imports()),
        ("Reversal Patterns Agent", test_reversal_patterns_agent()),
        ("Continuation Patterns Agent", test_continuation_patterns_agent()),
        ("Technical Overview Agent", test_technical_overview_agent()),
        ("Patterns Orchestrator", test_patterns_orchestrator())
    ]
    
    for test_name, test_coro in tests:
        logger.info(f"\n🔍 Running: {test_name}")
        logger.info("-" * 40)
        
        try:
            result = await test_coro
            test_results.append((test_name, result))
            
            if result:
                logger.info(f"✅ {test_name} PASSED")
            else:
                logger.error(f"❌ {test_name} FAILED")
                
        except Exception as e:
            logger.error(f"❌ {test_name} CRASHED: {str(e)}")
            test_results.append((test_name, False))
    
    # Generate test charts for visual inspection
    logger.info(f"\n📊 Generating test charts...")
    try:
        from agents.patterns import (
            ReversalPatternsCharts, 
            ContinuationPatternsCharts,
            TechnicalOverviewCharts
        )
        
        stock_data, indicators = generate_test_data(60)  # 60 days of data
        
        # Save reversal patterns chart
        reversal_charts = ReversalPatternsCharts()
        reversal_chart_data = await reversal_charts.create_chart(stock_data, indicators)
        save_test_chart(reversal_chart_data, "reversal_patterns_test.png")
        
        # Save continuation patterns chart
        continuation_charts = ContinuationPatternsCharts()
        continuation_chart_data = await continuation_charts.create_chart(stock_data, indicators)
        save_test_chart(continuation_chart_data, "continuation_patterns_test.png")
        
        # Save technical overview chart
        technical_charts = TechnicalOverviewCharts()
        technical_chart_data = await technical_charts.create_chart(stock_data, indicators)
        save_test_chart(technical_chart_data, "technical_overview_test.png")
        
    except Exception as e:
        logger.warning(f"Chart generation failed: {str(e)}")
    
    # Summary
    logger.info(f"\n🏁 Test Suite Complete")
    logger.info("=" * 60)
    
    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"  {status} {test_name}")
    
    logger.info(f"\nResults: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        logger.info("🎉 All tests passed! Pattern agents are ready for integration.")
        return True
    else:
        logger.error("❌ Some tests failed. Please review the errors above.")
        return False

if __name__ == "__main__":
    # Run the test suite
    success = asyncio.run(main())
    exit(0 if success else 1)