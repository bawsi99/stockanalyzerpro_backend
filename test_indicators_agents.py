#!/usr/bin/env python3
"""
Test for Indicators Agents System

Tests the indicators agents system to ensure all components work correctly.
"""

import sys
import os
import asyncio
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

# Add the backend directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_test_data(periods=100):
    """Generate test stock data and indicators"""
    
    dates = pd.date_range(start=datetime.now() - timedelta(days=periods), periods=periods, freq='D')
    
    # Generate realistic price data
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, periods)
    
    prices = [100.0]
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    closes = np.array(prices)
    highs = closes * (1 + np.random.uniform(0, 0.02, periods))
    lows = closes * (1 - np.random.uniform(0, 0.02, periods))
    opens = closes * (1 + np.random.uniform(-0.01, 0.01, periods))
    
    # Ensure OHLC relationship
    for i in range(periods):
        highs[i] = max(highs[i], opens[i], closes[i])
        lows[i] = min(lows[i], opens[i], closes[i])
    
    # Generate volume
    volume = np.random.lognormal(np.log(1000000), 0.3, periods).astype(int)
    
    stock_data = pd.DataFrame({
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volume
    }, index=dates)
    
    # Generate indicators
    indicators = generate_indicators(stock_data)
    
    return stock_data, indicators

def generate_indicators(stock_data):
    """Generate technical indicators for testing"""
    
    close = stock_data['close']
    high = stock_data['high']
    low = stock_data['low']
    
    indicators = {}
    
    # Moving averages
    indicators['sma_20'] = close.rolling(window=20, min_periods=1).mean().values
    indicators['sma_50'] = close.rolling(window=50, min_periods=1).mean().values
    indicators['ema_12'] = close.ewm(span=12).mean().values
    indicators['ema_26'] = close.ewm(span=26).mean().values
    
    # RSI
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
    rs = gain / loss
    indicators['rsi'] = (100 - (100 / (1 + rs))).values
    
    # MACD
    ema_12 = close.ewm(span=12).mean()
    ema_26 = close.ewm(span=26).mean()
    macd = ema_12 - ema_26
    macd_signal = macd.ewm(span=9).mean()
    indicators['macd'] = macd.values
    indicators['macd_signal'] = macd_signal.values
    indicators['macd_histogram'] = (macd - macd_signal).values
    
    # Stochastic
    high_14 = high.rolling(window=14, min_periods=1).max()
    low_14 = low.rolling(window=14, min_periods=1).min()
    stoch_k = ((close - low_14) / (high_14 - low_14) * 100).fillna(50)
    indicators['stoch_k'] = stoch_k.values
    indicators['stoch_d'] = stoch_k.rolling(window=3, min_periods=1).mean().values
    
    # ADX (approximation)
    returns = close.pct_change().fillna(0)
    volatility = returns.rolling(window=14, min_periods=1).std()
    indicators['adx'] = (volatility * 100).clip(0, 100).fillna(25).values
    
    return indicators

async def test_individual_agents():
    """Test individual indicator agents"""
    logger.info("🧪 Testing Individual Indicator Agents")
    logger.info("=" * 50)
    
    from agents.indicators import (
        TrendIndicatorsProcessor,
        MomentumIndicatorsProcessor,
        VolatilityIndicatorsProcessor,
        SupportResistanceIndicatorsProcessor
    )
    
    # Generate test data
    stock_data, indicators = generate_test_data(100)
    
    agents = [
        ("Trend", TrendIndicatorsProcessor()),
        ("Momentum", MomentumIndicatorsProcessor()), 
        ("Volatility", VolatilityIndicatorsProcessor()),
        ("Support/Resistance", SupportResistanceIndicatorsProcessor())
    ]
    
    test_results = {}
    
    for agent_name, agent in agents:
        logger.info(f"\n📊 Testing {agent_name} Agent...")
        
        try:
            start_time = datetime.now()
            result = await agent.analyze_async(stock_data, indicators, f"Test context for {agent_name}")
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Validate result structure
            assert isinstance(result, dict), f"{agent_name} should return a dictionary"
            assert 'agent_name' in result, f"{agent_name} result should have agent_name"
            assert 'confidence_score' in result, f"{agent_name} result should have confidence_score"
            
            confidence = result.get('confidence_score', 0.0)
            
            test_results[agent_name] = {
                'success': True,
                'execution_time': execution_time,
                'confidence': confidence,
                'result_keys': len(result.keys())
            }
            
            logger.info(f"  ✅ {agent_name}: {execution_time:.3f}s (confidence: {confidence:.2f}, keys: {len(result.keys())})")
            
        except Exception as e:
            test_results[agent_name] = {
                'success': False,
                'error': str(e)
            }
            logger.error(f"  ❌ {agent_name}: Failed - {str(e)}")
    
    return test_results

async def test_indicators_orchestrator():
    """Test the indicators agents orchestrator"""
    logger.info("\n🎭 Testing Indicators Agents Orchestrator")
    logger.info("=" * 50)
    
    from agents.indicators import IndicatorAgentsOrchestrator
    
    orchestrator = IndicatorAgentsOrchestrator()
    
    # Generate test data
    stock_data, indicators = generate_test_data(100)
    
    logger.info("\n📊 Running comprehensive indicators analysis...")
    
    try:
        start_time = datetime.now()
        
        result = await orchestrator.analyze_indicators_comprehensive(
            symbol="TEST_SYMBOL",
            stock_data=stock_data,
            indicators=indicators,
            context="Test orchestrator analysis"
        )
        
        total_time = (datetime.now() - start_time).total_seconds()
        
        # Validate orchestrator result
        assert hasattr(result, 'individual_results'), "Result should have individual_results"
        assert hasattr(result, 'unified_analysis'), "Result should have unified_analysis" 
        assert hasattr(result, 'successful_agents'), "Result should have successful_agents"
        assert hasattr(result, 'overall_confidence'), "Result should have overall_confidence"
        
        logger.info(f"  ⏱️  Total time: {total_time:.3f}s")
        logger.info(f"  🏃 Processing time: {result.total_processing_time:.3f}s")
        logger.info(f"  ✅ Success rate: {result.successful_agents}/{result.successful_agents + result.failed_agents}")
        logger.info(f"  🎯 Overall confidence: {result.overall_confidence:.3f}")
        
        # Individual agent performance
        for agent_name, agent_result in result.individual_results.items():
            if agent_result.success:
                logger.info(f"    {agent_name}: {agent_result.processing_time:.3f}s (conf: {agent_result.confidence_score:.2f})")
            else:
                logger.info(f"    {agent_name}: FAILED - {agent_result.error_message}")
        
        # Unified analysis validation
        unified = result.unified_analysis
        logger.info(f"  📊 Unified analysis keys: {list(unified.keys())}")
        
        if 'signal_consensus' in unified:
            consensus = unified['signal_consensus']
            logger.info(f"  🎯 Signal consensus: {consensus.get('consensus', 'none')} (strength: {consensus.get('strength', 'none')})")
        
        return {
            'success': True,
            'execution_time': total_time,
            'successful_agents': result.successful_agents,
            'failed_agents': result.failed_agents,
            'overall_confidence': result.overall_confidence
        }
        
    except Exception as e:
        logger.error(f"  ❌ Orchestrator test failed: {str(e)}")
        return {'success': False, 'error': str(e)}

async def test_package_imports():
    """Test that all package imports work correctly"""
    logger.info("\n📦 Testing Package Imports")
    logger.info("=" * 50)
    
    try:
        # Test imports
        from agents.indicators import (
            TrendIndicatorsProcessor,
            TrendIndicatorsCharts,
            MomentumIndicatorsProcessor,
            VolatilityIndicatorsProcessor,
            SupportResistanceIndicatorsProcessor,
            IndicatorAgentsOrchestrator,
            indicators_orchestrator
        )
        
        logger.info("  ✅ All indicator agent imports successful")
        
        # Test orchestrator singleton
        assert indicators_orchestrator is not None, "Global orchestrator should be available"
        assert isinstance(indicators_orchestrator, IndicatorAgentsOrchestrator), "Should be correct type"
        
        logger.info("  ✅ Global orchestrator singleton working")
        
        return {'success': True}
        
    except Exception as e:
        logger.error(f"  ❌ Import test failed: {str(e)}")
        return {'success': False, 'error': str(e)}

async def main():
    """Run all indicator agent tests"""
    logger.info("🚀 Starting Indicators Agents Test Suite")
    logger.info("=" * 60)
    
    start_time = datetime.now()
    test_results = {}
    
    try:
        # Run all tests
        test_results['individual_agents'] = await test_individual_agents()
        test_results['orchestrator'] = await test_indicators_orchestrator() 
        test_results['imports'] = await test_package_imports()
        
        total_time = (datetime.now() - start_time).total_seconds()
        
        # Summary
        logger.info(f"\n🏁 Indicators Test Suite Complete")
        logger.info(f"Total execution time: {total_time:.2f}s")
        
        # Count successes
        individual_successes = sum(1 for r in test_results['individual_agents'].values() if r.get('success', False))
        orchestrator_success = test_results['orchestrator'].get('success', False)
        import_success = test_results['imports'].get('success', False)
        
        total_tests = len(test_results['individual_agents']) + 2  # +2 for orchestrator and imports
        total_successes = individual_successes + (1 if orchestrator_success else 0) + (1 if import_success else 0)
        
        logger.info(f"Tests passed: {total_successes}/{total_tests}")
        
        if total_successes == total_tests:
            logger.info("🎉 All indicator agents tests passed!")
        else:
            logger.warning(f"⚠️  {total_tests - total_successes} tests failed")
        
        return test_results
        
    except Exception as e:
        logger.error(f"❌ Test suite failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return test_results

if __name__ == "__main__":
    asyncio.run(main())