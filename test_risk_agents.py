#!/usr/bin/env python3
"""
Test for Risk Analysis Agents System

Tests the risk analysis agents system to ensure all components work correctly.
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
    indicators['macd'] = macd.values
    
    return indicators

async def test_individual_risk_agents():
    """Test individual risk agents"""
    logger.info("🛡️  Testing Individual Risk Agents")
    logger.info("=" * 50)
    
    from agents.risk_analysis import (
        MarketRiskProcessor,
        VolatilityRiskProcessor,
        LiquidityRiskProcessor,
        TechnicalRiskProcessor
    )
    
    # Generate test data
    stock_data, indicators = generate_test_data(100)
    
    agents = [
        ("Market Risk", MarketRiskProcessor()),
        ("Volatility Risk", VolatilityRiskProcessor()), 
        ("Liquidity Risk", LiquidityRiskProcessor()),
        ("Technical Risk", TechnicalRiskProcessor())
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
            
            # Check for risk-specific fields
            has_risk_level = 'risk_level' in result or 'overall_risk_level' in result
            assert has_risk_level, f"{agent_name} should have risk_level or overall_risk_level"
            
            test_results[agent_name] = {
                'success': True,
                'execution_time': execution_time,
                'confidence': confidence,
                'result_keys': len(result.keys()),
                'risk_level': result.get('risk_level', result.get('overall_risk_level', 'unknown'))
            }
            
            logger.info(f"  ✅ {agent_name}: {execution_time:.3f}s (confidence: {confidence:.2f}, risk: {test_results[agent_name]['risk_level']})")
            
        except Exception as e:
            test_results[agent_name] = {
                'success': False,
                'error': str(e)
            }
            logger.error(f"  ❌ {agent_name}: Failed - {str(e)}")
    
    return test_results

async def test_risk_orchestrator():
    """Test the risk analysis orchestrator"""
    logger.info("\n🎭 Testing Risk Analysis Orchestrator")
    logger.info("=" * 50)
    
    from agents.risk_analysis import RiskAgentsOrchestrator
    
    orchestrator = RiskAgentsOrchestrator()
    
    # Generate test data
    stock_data, indicators = generate_test_data(100)
    
    logger.info("\n📊 Running comprehensive risk analysis...")
    
    try:
        start_time = datetime.now()
        
        result = await orchestrator.analyze_risk_comprehensive(
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
        assert hasattr(result, 'overall_risk_score'), "Result should have overall_risk_score"
        
        logger.info(f"  ⏱️  Total time: {total_time:.3f}s")
        logger.info(f"  🏃 Processing time: {result.total_processing_time:.3f}s")
        logger.info(f"  ✅ Success rate: {result.successful_agents}/{result.successful_agents + result.failed_agents}")
        logger.info(f"  🎯 Overall risk score: {result.overall_risk_score:.3f}")
        logger.info(f"  📊 Overall confidence: {result.overall_confidence:.3f}")
        
        # Individual agent performance
        for agent_name, agent_result in result.individual_results.items():
            if agent_result.success:
                risk_level = agent_result.result_data.get('risk_level', 
                           agent_result.result_data.get('overall_risk_level', 'unknown'))
                logger.info(f"    {agent_name}: {agent_result.processing_time:.3f}s (risk: {risk_level}, conf: {agent_result.confidence_score:.2f})")
            else:
                logger.info(f"    {agent_name}: FAILED - {agent_result.error_message}")
        
        # Unified analysis validation
        unified = result.unified_analysis
        logger.info(f"  📊 Unified analysis keys: {list(unified.keys())}")
        
        if 'risk_summary' in unified:
            summary = unified['risk_summary']
            logger.info(f"  🎯 Risk summary: {summary.get('overall_level', 'none')} (score: {summary.get('average_risk_score', 0):.2f})")
        
        if 'trading_implications' in unified:
            implications = unified['trading_implications']
            logger.info(f"  📈 Position sizing: {implications.get('position_sizing_recommendation', 'none')}")
            logger.info(f"  🛡️  Hedging strategy: {implications.get('hedging_strategy', 'none')}")
        
        return {
            'success': True,
            'execution_time': total_time,
            'successful_agents': result.successful_agents,
            'failed_agents': result.failed_agents,
            'overall_risk_score': result.overall_risk_score,
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
        from agents.risk_analysis import (
            MarketRiskProcessor,
            VolatilityRiskProcessor,
            LiquidityRiskProcessor,
            TechnicalRiskProcessor,
            RiskAgentsOrchestrator,
            RiskAgentResult,
            RiskAnalysisResult,
            risk_orchestrator
        )
        
        logger.info("  ✅ All risk agent imports successful")
        
        # Test orchestrator singleton
        assert risk_orchestrator is not None, "Global orchestrator should be available"
        assert isinstance(risk_orchestrator, RiskAgentsOrchestrator), "Should be correct type"
        
        logger.info("  ✅ Global orchestrator singleton working")
        
        # Test data classes by creating instances
        from dataclasses import fields
        risk_result_fields = [f.name for f in fields(RiskAgentResult)]
        analysis_result_fields = [f.name for f in fields(RiskAnalysisResult)]
        
        assert 'agent_name' in risk_result_fields, "RiskAgentResult should have agent_name field"
        assert 'individual_results' in analysis_result_fields, "RiskAnalysisResult should have individual_results field"
        
        logger.info("  ✅ Data classes structure validated")
        
        return {'success': True}
        
    except Exception as e:
        logger.error(f"  ❌ Import test failed: {str(e)}")
        return {'success': False, 'error': str(e)}

async def test_risk_scenarios():
    """Test risk scenario generation and analysis"""
    logger.info("\n🎲 Testing Risk Scenario Analysis")
    logger.info("=" * 50)
    
    try:
        from agents.risk_analysis import MarketRiskProcessor
        
        # Generate test data with higher volatility for interesting scenarios
        stock_data, indicators = generate_test_data(60)
        
        # Add some volatility spikes
        stock_data.loc[stock_data.index[-10:], 'volume'] *= 3
        returns = stock_data['close'].pct_change()
        stock_data.loc[stock_data.index[-5:], 'close'] *= 0.95  # Simulate decline
        
        agent = MarketRiskProcessor()
        result = await agent.analyze_async(stock_data, indicators, "Scenario testing")
        
        # Check scenario generation
        if 'risk_scenarios' in result:
            scenarios = result['risk_scenarios']
            logger.info(f"  📊 Generated {len(scenarios)} scenarios")
            
            for scenario in scenarios[:3]:  # Show top 3
                logger.info(f"    {scenario['name']}: {scenario['probability']:.1%} probability, {scenario['impact']:.1%} impact")
        
        if 'mitigation_strategies' in result:
            strategies = result['mitigation_strategies']
            logger.info(f"  🛡️  Generated {len(strategies)} mitigation strategies")
            
            for strategy in strategies[:2]:  # Show top 2
                logger.info(f"    - {strategy}")
        
        logger.info("  ✅ Risk scenario analysis working")
        return {'success': True}
        
    except Exception as e:
        logger.error(f"  ❌ Risk scenario test failed: {str(e)}")
        return {'success': False, 'error': str(e)}

async def main():
    """Run all risk analysis tests"""
    logger.info("🚀 Starting Risk Analysis Agents Test Suite")
    logger.info("=" * 60)
    
    start_time = datetime.now()
    test_results = {}
    
    try:
        # Run all tests
        test_results['individual_agents'] = await test_individual_risk_agents()
        test_results['orchestrator'] = await test_risk_orchestrator() 
        test_results['imports'] = await test_package_imports()
        test_results['scenarios'] = await test_risk_scenarios()
        
        total_time = (datetime.now() - start_time).total_seconds()
        
        # Summary
        logger.info(f"\n🏁 Risk Analysis Test Suite Complete")
        logger.info(f"Total execution time: {total_time:.2f}s")
        
        # Count successes
        individual_successes = sum(1 for r in test_results['individual_agents'].values() if r.get('success', False))
        orchestrator_success = test_results['orchestrator'].get('success', False)
        import_success = test_results['imports'].get('success', False)
        scenario_success = test_results['scenarios'].get('success', False)
        
        total_tests = len(test_results['individual_agents']) + 3  # +3 for orchestrator, imports, scenarios
        total_successes = individual_successes + sum([orchestrator_success, import_success, scenario_success])
        
        logger.info(f"Tests passed: {total_successes}/{total_tests}")
        
        if total_successes == total_tests:
            logger.info("🎉 All risk analysis tests passed!")
        else:
            logger.warning(f"⚠️  {total_tests - total_successes} tests failed")
        
        # Risk analysis summary
        if orchestrator_success:
            orch_result = test_results['orchestrator']
            logger.info(f"\n📊 Risk Analysis Performance Summary:")
            logger.info(f"  Overall risk score: {orch_result.get('overall_risk_score', 0):.3f}")
            logger.info(f"  Analysis confidence: {orch_result.get('overall_confidence', 0):.3f}")
            logger.info(f"  Agent success rate: {orch_result.get('successful_agents', 0)}/4")
        
        return test_results
        
    except Exception as e:
        logger.error(f"❌ Test suite failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return test_results

if __name__ == "__main__":
    asyncio.run(main())