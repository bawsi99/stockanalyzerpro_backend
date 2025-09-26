#!/usr/bin/env python3
"""
Performance Test for Pattern Analysis Agents

Tests the performance of pattern agents with different data sizes and scenarios.
"""

import sys
import os
import asyncio
import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime, timedelta

# Add the backend directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_test_data_with_patterns(periods=100, add_patterns=True):
    """Generate test data with specific patterns for more realistic testing"""
    
    dates = pd.date_range(start=datetime.now() - timedelta(days=periods), periods=periods, freq='D')
    
    # Create base price movement
    np.random.seed(42)
    start_price = 100.0
    
    if add_patterns:
        # Add specific patterns to the data
        returns = []
        
        # Create uptrend with pullbacks
        for i in range(periods):
            if i < periods // 3:  # First third - uptrend
                trend_return = np.random.normal(0.008, 0.015)  # Strong uptrend
            elif i < 2 * periods // 3:  # Second third - consolidation with patterns
                if i % 10 < 3:  # Create triangle pattern
                    trend_return = np.random.normal(-0.002, 0.005)  # Slight pullback
                else:
                    trend_return = np.random.normal(0.001, 0.008)  # Sideways
            else:  # Final third - reversal setup
                trend_return = np.random.normal(-0.003, 0.012)  # Potential reversal
            
            returns.append(trend_return)
    else:
        # Random walk
        returns = np.random.normal(0.001, 0.02, periods)
    
    # Create price series
    prices = [start_price]
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
    
    # Generate volume with spikes during pattern formations
    base_volume = 1000000
    volume = np.random.lognormal(np.log(base_volume), 0.3, periods).astype(int)
    
    if add_patterns:
        # Add volume spikes during breakouts
        for i in range(1, periods):
            price_change = abs(closes[i] - closes[i-1]) / closes[i-1]
            if price_change > 0.03:  # Significant move
                volume[i] = int(volume[i] * (2 + price_change * 10))
    
    stock_data = pd.DataFrame({
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volume
    }, index=dates)
    
    # Generate comprehensive indicators
    indicators = generate_comprehensive_indicators(stock_data)
    
    return stock_data, indicators

def generate_comprehensive_indicators(stock_data):
    """Generate comprehensive technical indicators for testing"""
    
    close = stock_data['close']
    high = stock_data['high']
    low = stock_data['low']
    volume = stock_data['volume']
    
    indicators = {}
    
    # Moving Averages
    for period in [5, 10, 20, 50, 100, 200]:
        if len(close) >= period:
            indicators[f'sma_{period}'] = close.rolling(window=period, min_periods=1).mean().values
            indicators[f'ema_{period}'] = close.ewm(span=period).mean().values
    
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
    
    # Bollinger Bands
    bb_middle = close.rolling(window=20, min_periods=1).mean()
    bb_std = close.rolling(window=20, min_periods=1).std()
    indicators['bb_upper'] = (bb_middle + (bb_std * 2)).values
    indicators['bb_middle'] = bb_middle.values
    indicators['bb_lower'] = (bb_middle - (bb_std * 2)).values
    
    # Stochastic
    high_14 = high.rolling(window=14, min_periods=1).max()
    low_14 = low.rolling(window=14, min_periods=1).min()
    stoch_k = ((close - low_14) / (high_14 - low_14) * 100).fillna(50)
    indicators['stoch_k'] = stoch_k.values
    indicators['stoch_d'] = stoch_k.rolling(window=3, min_periods=1).mean().values
    
    # ADX approximation
    returns = close.pct_change().fillna(0)
    volatility = returns.rolling(window=14, min_periods=1).std()
    indicators['adx'] = (volatility * 100).clip(0, 100).fillna(25).values
    
    return indicators

async def performance_test_individual_agents():
    """Test performance of individual agents with different data sizes"""
    logger.info("🏃 Performance Testing Individual Agents")
    logger.info("=" * 50)
    
    from agents.patterns import (
        ReversalPatternsProcessor,
        ContinuationPatternsProcessor, 
        TechnicalOverviewProcessor
    )
    
    # Test different data sizes
    test_sizes = [50, 100, 250, 500, 1000]
    agents = [
        ("Reversal", ReversalPatternsProcessor()),
        ("Continuation", ContinuationPatternsProcessor()),
        ("Technical", TechnicalOverviewProcessor())
    ]
    
    results = {}
    
    for size in test_sizes:
        logger.info(f"\n📊 Testing with {size} data points...")
        
        # Generate test data
        stock_data, indicators = generate_test_data_with_patterns(size, add_patterns=True)
        
        size_results = {}
        
        for agent_name, agent in agents:
            # Time the agent execution
            start_time = time.time()
            
            try:
                result = await agent.analyze_async(stock_data, indicators, f"Performance test {size}")
                execution_time = time.time() - start_time
                
                confidence = result.get('confidence_score', 0)
                patterns_found = 0
                
                # Count patterns found
                if 'reversal_patterns' in result:
                    for pattern_type, pattern_list in result['reversal_patterns'].items():
                        patterns_found += len(pattern_list)
                elif 'continuation_patterns' in result:
                    for pattern_type, pattern_list in result['continuation_patterns'].items():
                        patterns_found += len(pattern_list)
                elif 'patterns' in result:
                    patterns_found = len(result['patterns'])
                
                size_results[agent_name] = {
                    'execution_time': execution_time,
                    'confidence': confidence,
                    'patterns_found': patterns_found,
                    'success': True
                }
                
                logger.info(f"  ✅ {agent_name}: {execution_time:.3f}s (confidence: {confidence:.2f}, patterns: {patterns_found})")
                
            except Exception as e:
                size_results[agent_name] = {
                    'execution_time': 999.0,
                    'confidence': 0,
                    'patterns_found': 0,
                    'success': False,
                    'error': str(e)
                }
                logger.error(f"  ❌ {agent_name}: Failed - {str(e)}")
        
        results[size] = size_results
    
    # Performance summary
    logger.info(f"\n📈 Performance Summary")
    logger.info("-" * 50)
    
    for agent_name, _ in agents:
        logger.info(f"\n{agent_name} Agent Performance:")
        for size in test_sizes:
            if size in results and agent_name in results[size]:
                result = results[size][agent_name]
                if result['success']:
                    logger.info(f"  {size:4d} points: {result['execution_time']:6.3f}s - {result['patterns_found']:3d} patterns")
                else:
                    logger.info(f"  {size:4d} points: FAILED")
    
    return results

async def performance_test_orchestrator():
    """Test orchestrator performance with concurrent execution"""
    logger.info("\n🎭 Performance Testing Orchestrator")
    logger.info("=" * 50)
    
    from agents.patterns import PatternAgentsOrchestrator
    
    orchestrator = PatternAgentsOrchestrator()
    test_sizes = [100, 250, 500]
    
    for size in test_sizes:
        logger.info(f"\n📊 Orchestrator test with {size} data points...")
        
        # Generate test data
        stock_data, indicators = generate_test_data_with_patterns(size, add_patterns=True)
        
        # Time orchestrator execution
        start_time = time.time()
        
        try:
            result = await orchestrator.analyze_patterns_comprehensive(
                symbol=f"PERF_TEST_{size}",
                stock_data=stock_data,
                indicators=indicators,
                context=f"Performance test with {size} data points"
            )
            
            total_time = time.time() - start_time
            
            logger.info(f"  ⏱️  Total time: {total_time:.3f}s")
            logger.info(f"  🏃 Processing time: {result.total_processing_time:.3f}s")
            logger.info(f"  ✅ Success rate: {result.successful_agents}/{result.successful_agents + result.failed_agents}")
            logger.info(f"  🎯 Overall confidence: {result.overall_confidence:.2f}")
            logger.info(f"  🧩 Total patterns: {result.unified_analysis['pattern_summary']['total_patterns_identified']}")
            
            # Individual agent performance
            for agent_name, agent_result in result.individual_results.items():
                if agent_result.success:
                    logger.info(f"    {agent_name}: {agent_result.processing_time:.3f}s (conf: {agent_result.confidence_score:.2f})")
                else:
                    logger.info(f"    {agent_name}: FAILED")
            
        except Exception as e:
            logger.error(f"  ❌ Orchestrator failed: {str(e)}")

async def stress_test_patterns():
    """Stress test with challenging pattern scenarios"""
    logger.info("\n💪 Stress Testing Pattern Detection")
    logger.info("=" * 50)
    
    from agents.patterns import PatternAgentsOrchestrator
    
    orchestrator = PatternAgentsOrchestrator()
    
    # Test scenarios
    scenarios = [
        ("High Volatility", 200, True, 0.05),
        ("Low Volatility", 200, True, 0.005),
        ("Random Walk", 200, False, 0.02),
        ("Strong Trends", 300, True, 0.03),
        ("Large Dataset", 2000, True, 0.02)
    ]
    
    for scenario_name, periods, add_patterns, volatility in scenarios:
        logger.info(f"\n🎯 Testing: {scenario_name}")
        
        # Generate scenario-specific data
        np.random.seed(int(time.time()) % 100)  # Different seed each time
        
        dates = pd.date_range(start=datetime.now() - timedelta(days=periods), periods=periods, freq='D')
        
        if add_patterns:
            returns = np.random.normal(0.001, volatility, periods)
            if "Strong Trends" in scenario_name:
                # Add strong trending periods
                trend_periods = periods // 4
                returns[:trend_periods] = np.random.normal(0.008, volatility, trend_periods)  # Uptrend
                returns[-trend_periods:] = np.random.normal(-0.008, volatility, trend_periods)  # Downtrend
        else:
            returns = np.random.normal(0, volatility, periods)
        
        # Create price data
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
        
        volume = np.random.lognormal(np.log(1000000), 0.4, periods).astype(int)
        
        stock_data = pd.DataFrame({
            'open': opens, 'high': highs, 'low': lows, 'close': closes, 'volume': volume
        }, index=dates)
        
        indicators = generate_comprehensive_indicators(stock_data)
        
        # Run stress test
        start_time = time.time()
        
        try:
            result = await orchestrator.analyze_patterns_comprehensive(
                symbol=f"STRESS_{scenario_name.replace(' ', '_')}",
                stock_data=stock_data,
                indicators=indicators,
                context=f"Stress test: {scenario_name}"
            )
            
            total_time = time.time() - start_time
            
            logger.info(f"  ⏱️  Completed in: {total_time:.3f}s")
            logger.info(f"  📊 Data points: {periods}")
            logger.info(f"  ✅ Agents succeeded: {result.successful_agents}/{result.successful_agents + result.failed_agents}")
            logger.info(f"  🎯 Overall confidence: {result.overall_confidence:.2f}")
            logger.info(f"  🧩 Patterns found: {result.unified_analysis['pattern_summary']['total_patterns_identified']}")
            logger.info(f"  📈 Pattern types: {result.unified_analysis['pattern_summary']['pattern_types']}")
            
            if result.successful_agents == 0:
                logger.warning(f"  ⚠️  All agents failed for {scenario_name}")
            
        except Exception as e:
            logger.error(f"  ❌ Stress test failed: {str(e)}")

async def main():
    """Run all performance tests"""
    logger.info("🚀 Starting Pattern Agents Performance Test Suite")
    logger.info("=" * 60)
    
    start_time = time.time()
    
    try:
        # Run performance tests
        await performance_test_individual_agents()
        await performance_test_orchestrator()
        await stress_test_patterns()
        
        total_time = time.time() - start_time
        
        logger.info(f"\n🏁 Performance Test Suite Complete")
        logger.info(f"Total execution time: {total_time:.2f}s")
        logger.info("🎉 Pattern agents performance validated!")
        
    except Exception as e:
        logger.error(f"❌ Performance test suite failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())