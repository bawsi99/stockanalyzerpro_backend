#!/usr/bin/env python3
"""
Comprehensive Integration Test for All Agent Systems

Tests the integration of all agent systems including indicators agents,
risk analysis agents, pattern agents, and volume agents to ensure they
work together properly.
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

def generate_comprehensive_test_data(periods=150):
    """Generate comprehensive test stock data and indicators for integration testing"""
    
    dates = pd.date_range(start=datetime.now() - timedelta(days=periods), periods=periods, freq='D')
    
    # Generate more realistic price data with trends and patterns
    np.random.seed(42)
    
    # Create multiple regime changes
    regime1_returns = np.random.normal(0.002, 0.015, periods//3)  # Bull market
    regime2_returns = np.random.normal(-0.001, 0.025, periods//3)  # Volatile/bear market
    regime3_returns = np.random.normal(0.001, 0.018, periods - 2*(periods//3))  # Recovery
    
    all_returns = np.concatenate([regime1_returns, regime2_returns, regime3_returns])
    
    # Generate prices with compound returns
    prices = [100.0]
    for ret in all_returns:
        prices.append(prices[-1] * (1 + ret))
    
    closes = np.array(prices[1:])  # Remove initial price
    
    # Generate OHLC data
    highs = closes * (1 + np.random.uniform(0, 0.025, periods))
    lows = closes * (1 - np.random.uniform(0, 0.025, periods))
    opens = np.roll(closes, 1)  # Previous close as open
    opens[0] = closes[0] * 0.995  # First open slightly below first close
    
    # Ensure OHLC relationship
    for i in range(periods):
        highs[i] = max(highs[i], opens[i], closes[i])
        lows[i] = min(lows[i], opens[i], closes[i])
    
    # Generate volume with realistic patterns
    base_volume = 1000000
    volume_noise = np.random.lognormal(0, 0.4, periods)
    trend_volume = np.where(abs(all_returns) > 0.03, 2.0, 1.0)  # Higher volume on big moves
    volume = (base_volume * volume_noise * trend_volume).astype(int)
    
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
    """Generate comprehensive technical indicators for integration testing"""
    
    close = stock_data['close']
    high = stock_data['high']
    low = stock_data['low']
    volume = stock_data['volume']
    
    indicators = {}
    
    # Moving averages
    indicators['sma_10'] = close.rolling(window=10, min_periods=1).mean().values
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
    
    # ADX approximation
    returns = close.pct_change().fillna(0)
    volatility = returns.rolling(window=14, min_periods=1).std()
    indicators['adx'] = (volatility * 100).clip(0, 100).fillna(25).values
    
    # Volume indicators
    indicators['volume_sma'] = volume.rolling(window=20, min_periods=1).mean().values
    indicators['volume_ratio'] = (volume / volume.rolling(window=20, min_periods=1).mean()).fillna(1.0).values
    
    # Bollinger Bands
    bb_middle = close.rolling(window=20, min_periods=1).mean()
    bb_std = close.rolling(window=20, min_periods=1).std()
    indicators['bb_upper'] = (bb_middle + 2 * bb_std).values
    indicators['bb_lower'] = (bb_middle - 2 * bb_std).values
    indicators['bb_middle'] = bb_middle.values
    
    return indicators

async def test_indicators_system_integration():
    """Test the indicators system with comprehensive data"""
    logger.info("🔧 Testing Indicators System Integration")
    logger.info("=" * 50)
    
    try:
        from agents.indicators import indicators_orchestrator
        
        # Generate comprehensive test data
        stock_data, indicators = generate_comprehensive_test_data(120)
        
        logger.info(f"📊 Generated test data: {len(stock_data)} periods")
        logger.info(f"📈 Price range: ${stock_data['close'].min():.2f} - ${stock_data['close'].max():.2f}")
        logger.info(f"📊 Generated {len(indicators)} technical indicators")
        
        # Run comprehensive indicators analysis
        start_time = datetime.now()
        result = await indicators_orchestrator.analyze_indicators_comprehensive(
            symbol="INTEGRATION_TEST",
            stock_data=stock_data,
            indicators=indicators,
            context="Integration testing with comprehensive data"
        )
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Validate results
        assert result.successful_agents > 0, "At least some indicator agents should succeed"
        assert result.unified_analysis is not None, "Should have unified analysis"
        
        logger.info(f"  ✅ Execution time: {execution_time:.3f}s")
        logger.info(f"  ✅ Successful agents: {result.successful_agents}/4")
        logger.info(f"  ✅ Overall confidence: {result.overall_confidence:.3f}")
        
        # Check unified analysis content
        unified = result.unified_analysis
        if 'signal_consensus' in unified:
            consensus = unified['signal_consensus']
            logger.info(f"  📊 Signal consensus: {consensus.get('consensus', 'none')} ({consensus.get('strength', 'none')})")
        
        return {
            'success': True,
            'execution_time': execution_time,
            'successful_agents': result.successful_agents,
            'confidence': result.overall_confidence,
            'unified_keys': len(unified)
        }
        
    except Exception as e:
        logger.error(f"  ❌ Indicators integration test failed: {str(e)}")
        return {'success': False, 'error': str(e)}

async def test_risk_analysis_integration():
    """Test the risk analysis system with comprehensive data"""
    logger.info("\n🛡️  Testing Risk Analysis System Integration")
    logger.info("=" * 50)
    
    try:
        from agents.risk_analysis import risk_orchestrator
        
        # Generate test data with some risk scenarios
        stock_data, indicators = generate_comprehensive_test_data(100)
        
        # Add some risk elements to the data
        # Simulate a volatility spike
        stock_data.loc[stock_data.index[-20:-10], 'volume'] *= 2.5
        returns = stock_data['close'].pct_change()
        stock_data.loc[stock_data.index[-15:-5], 'close'] *= np.random.uniform(0.92, 1.08, 10)
        
        logger.info(f"📊 Generated risk test data: {len(stock_data)} periods")
        logger.info(f"📈 Added volatility spikes for risk testing")
        
        # Run comprehensive risk analysis
        start_time = datetime.now()
        result = await risk_orchestrator.analyze_risk_comprehensive(
            symbol="RISK_TEST",
            stock_data=stock_data,
            indicators=indicators,
            context="Integration testing with risk scenarios"
        )
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Validate results
        assert result.successful_agents > 0, "At least some risk agents should succeed"
        assert result.unified_analysis is not None, "Should have unified risk analysis"
        
        logger.info(f"  ✅ Execution time: {execution_time:.3f}s")
        logger.info(f"  ✅ Successful agents: {result.successful_agents}/4")
        logger.info(f"  ✅ Overall risk score: {result.overall_risk_score:.3f}")
        logger.info(f"  ✅ Overall confidence: {result.overall_confidence:.3f}")
        
        # Check unified analysis content
        unified = result.unified_analysis
        if 'risk_summary' in unified:
            summary = unified['risk_summary']
            logger.info(f"  🎯 Risk level: {summary.get('overall_level', 'unknown')}")
            logger.info(f"  📊 Key risks: {len(summary.get('key_risk_factors', []))}")
        
        if 'trading_implications' in unified:
            implications = unified['trading_implications']
            logger.info(f"  📈 Position sizing: {implications.get('position_sizing_recommendation', 'unknown')}")
        
        return {
            'success': True,
            'execution_time': execution_time,
            'successful_agents': result.successful_agents,
            'risk_score': result.overall_risk_score,
            'confidence': result.overall_confidence
        }
        
    except Exception as e:
        logger.error(f"  ❌ Risk analysis integration test failed: {str(e)}")
        return {'success': False, 'error': str(e)}

async def test_combined_analysis():
    """Test running both indicator and risk analysis systems together"""
    logger.info("\n🔄 Testing Combined Analysis Systems")
    logger.info("=" * 50)
    
    try:
        from agents.indicators import indicators_orchestrator
        from agents.risk_analysis import risk_orchestrator
        
        # Generate comprehensive test data
        stock_data, indicators = generate_comprehensive_test_data(80)
        
        logger.info(f"📊 Running combined analysis on {len(stock_data)} periods of data")
        
        # Run both analyses concurrently
        start_time = datetime.now()
        
        indicators_task = indicators_orchestrator.analyze_indicators_comprehensive(
            symbol="COMBINED_TEST",
            stock_data=stock_data,
            indicators=indicators,
            context="Combined analysis - indicators"
        )
        
        risk_task = risk_orchestrator.analyze_risk_comprehensive(
            symbol="COMBINED_TEST",
            stock_data=stock_data,
            indicators=indicators,
            context="Combined analysis - risk"
        )
        
        # Wait for both to complete
        indicators_result, risk_result = await asyncio.gather(indicators_task, risk_task)
        
        total_time = (datetime.now() - start_time).total_seconds()
        
        # Validate both results
        assert indicators_result.successful_agents > 0, "Indicators analysis should succeed"
        assert risk_result.successful_agents > 0, "Risk analysis should succeed"
        
        logger.info(f"  ✅ Combined execution time: {total_time:.3f}s")
        logger.info(f"  ✅ Indicators agents: {indicators_result.successful_agents}/4")
        logger.info(f"  ✅ Risk agents: {risk_result.successful_agents}/4")
        
        # Create combined summary
        combined_confidence = (indicators_result.overall_confidence + risk_result.overall_confidence) / 2
        
        logger.info(f"  📊 Combined confidence: {combined_confidence:.3f}")
        logger.info(f"  🎯 Risk level: {risk_result.unified_analysis.get('risk_summary', {}).get('overall_level', 'unknown')}")
        
        # Check for signal alignment
        indicators_consensus = indicators_result.unified_analysis.get('signal_consensus', {})
        risk_level = risk_result.unified_analysis.get('risk_summary', {}).get('overall_level', 'moderate')
        
        signal_risk_alignment = "aligned" if (
            (indicators_consensus.get('consensus') == 'bullish' and risk_level == 'low') or
            (indicators_consensus.get('consensus') == 'bearish' and risk_level == 'high')
        ) else "divergent"
        
        logger.info(f"  🔄 Signal-risk alignment: {signal_risk_alignment}")
        
        return {
            'success': True,
            'total_time': total_time,
            'indicators_success': indicators_result.successful_agents,
            'risk_success': risk_result.successful_agents,
            'combined_confidence': combined_confidence,
            'alignment': signal_risk_alignment
        }
        
    except Exception as e:
        logger.error(f"  ❌ Combined analysis test failed: {str(e)}")
        return {'success': False, 'error': str(e)}

async def test_main_agents_import():
    """Test importing all systems from main agents package"""
    logger.info("\n📦 Testing Main Agents Package Integration")
    logger.info("=" * 50)
    
    try:
        # Test importing all orchestrators from main agents package
        from agents import (
            indicators_orchestrator,
            risk_orchestrator,
            patterns_orchestrator
        )
        
        logger.info("  ✅ Successfully imported all orchestrators")
        
        # Test that they're the correct types
        from agents.indicators import IndicatorAgentsOrchestrator
        from agents.risk_analysis import RiskAgentsOrchestrator
        from agents.patterns import PatternAgentsOrchestrator
        
        assert isinstance(indicators_orchestrator, IndicatorAgentsOrchestrator)
        assert isinstance(risk_orchestrator, RiskAgentsOrchestrator) 
        assert isinstance(patterns_orchestrator, PatternAgentsOrchestrator)
        
        logger.info("  ✅ All orchestrators have correct types")
        
        # Test that they're singleton instances
        from agents import indicators_orchestrator as indicators_2
        from agents import risk_orchestrator as risk_2
        
        assert indicators_orchestrator is indicators_2
        assert risk_orchestrator is risk_2
        
        logger.info("  ✅ Singleton pattern working correctly")
        
        return {'success': True}
        
    except Exception as e:
        logger.error(f"  ❌ Main agents import test failed: {str(e)}")
        return {'success': False, 'error': str(e)}

async def test_error_handling():
    """Test error handling and resilience"""
    logger.info("\n⚠️  Testing Error Handling and Resilience")
    logger.info("=" * 50)
    
    try:
        from agents.indicators import indicators_orchestrator
        from agents.risk_analysis import risk_orchestrator
        
        # Test with minimal/edge case data
        minimal_data = pd.DataFrame({
            'open': [100, 101, 99],
            'high': [101, 102, 100],
            'low': [99, 100, 98],
            'close': [100.5, 101.5, 99.5],
            'volume': [1000, 1100, 900]
        }, index=pd.date_range(start='2023-01-01', periods=3, freq='D'))
        
        minimal_indicators = {'rsi': [50, 51, 49]}
        
        logger.info("  📊 Testing with minimal data (3 periods)")
        
        # Test indicators with minimal data
        indicators_result = await indicators_orchestrator.analyze_indicators_comprehensive(
            symbol="MINIMAL_TEST",
            stock_data=minimal_data,
            indicators=minimal_indicators,
            context="Error handling test - minimal data"
        )
        
        # Test risk analysis with minimal data
        risk_result = await risk_orchestrator.analyze_risk_comprehensive(
            symbol="MINIMAL_TEST",
            stock_data=minimal_data,
            indicators=minimal_indicators,
            context="Error handling test - minimal data"
        )
        
        logger.info(f"  ✅ Indicators handled minimal data: {indicators_result.successful_agents}/4 agents")
        logger.info(f"  ✅ Risk analysis handled minimal data: {risk_result.successful_agents}/4 agents")
        
        # Test with empty indicators
        logger.info("  📊 Testing with empty indicators")
        
        empty_result = await indicators_orchestrator.analyze_indicators_comprehensive(
            symbol="EMPTY_TEST",
            stock_data=minimal_data,
            indicators={},
            context="Error handling test - empty indicators"
        )
        
        logger.info(f"  ✅ Handled empty indicators: {empty_result.successful_agents}/4 agents")
        
        return {
            'success': True,
            'minimal_indicators': indicators_result.successful_agents,
            'minimal_risk': risk_result.successful_agents,
            'empty_indicators': empty_result.successful_agents
        }
        
    except Exception as e:
        logger.error(f"  ❌ Error handling test failed: {str(e)}")
        return {'success': False, 'error': str(e)}

async def main():
    """Run comprehensive integration tests"""
    logger.info("🚀 Starting Comprehensive Agent Systems Integration Test")
    logger.info("=" * 70)
    
    start_time = datetime.now()
    test_results = {}
    
    try:
        # Run all integration tests
        test_results['indicators_integration'] = await test_indicators_system_integration()
        test_results['risk_integration'] = await test_risk_analysis_integration() 
        test_results['combined_analysis'] = await test_combined_analysis()
        test_results['main_import'] = await test_main_agents_import()
        test_results['error_handling'] = await test_error_handling()
        
        total_time = (datetime.now() - start_time).total_seconds()
        
        # Summary
        logger.info(f"\n🏁 Integration Test Suite Complete")
        logger.info(f"Total execution time: {total_time:.2f}s")
        
        # Count successes
        successes = sum(1 for result in test_results.values() if result.get('success', False))
        total_tests = len(test_results)
        
        logger.info(f"Integration tests passed: {successes}/{total_tests}")
        
        if successes == total_tests:
            logger.info("🎉 All integration tests passed!")
        else:
            logger.warning(f"⚠️  {total_tests - successes} integration tests failed")
        
        # Performance summary
        logger.info(f"\n📊 Performance Summary:")
        
        if test_results['indicators_integration'].get('success'):
            indicators_perf = test_results['indicators_integration']
            logger.info(f"  Indicators System: {indicators_perf['execution_time']:.3f}s ({indicators_perf['successful_agents']}/4 agents)")
        
        if test_results['risk_integration'].get('success'):
            risk_perf = test_results['risk_integration']
            logger.info(f"  Risk Analysis: {risk_perf['execution_time']:.3f}s ({risk_perf['successful_agents']}/4 agents)")
        
        if test_results['combined_analysis'].get('success'):
            combined_perf = test_results['combined_analysis']
            logger.info(f"  Combined Analysis: {combined_perf['total_time']:.3f}s")
            logger.info(f"  Signal-Risk Alignment: {combined_perf['alignment']}")
        
        # System health summary
        logger.info(f"\n🏥 System Health Summary:")
        logger.info(f"  Package Integration: {'✅' if test_results['main_import']['success'] else '❌'}")
        logger.info(f"  Error Resilience: {'✅' if test_results['error_handling']['success'] else '❌'}")
        logger.info(f"  Concurrent Execution: {'✅' if test_results['combined_analysis']['success'] else '❌'}")
        
        return test_results
        
    except Exception as e:
        logger.error(f"❌ Integration test suite failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return test_results

if __name__ == "__main__":
    asyncio.run(main())