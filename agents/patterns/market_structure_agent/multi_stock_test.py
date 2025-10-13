#!/usr/bin/env python3
"""
Market Structure Agent - Multi-Stock Testing Framework

This module provides comprehensive testing for the Market Structure Agent
across multiple stocks and scenarios, saving prompts and responses for analysis.
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
from typing import Dict, Any, List
import logging

# Load environment variables
try:
    import dotenv
    # Load .env file from the backend/config directory
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'config', '.env')
    dotenv.load_dotenv(dotenv_path=env_path)
    print(f"✅ Environment variables loaded from: {env_path}")
except ImportError:
    print("⚠️ python-dotenv not available, using system environment variables")

from .agent import MarketStructureAgent
from .processor import MarketStructureProcessor
from .llm_agent import MarketStructureLLMAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketStructureAgentTester:
    """
    Comprehensive tester for Market Structure Agent with multi-stock support
    and prompt/response tracking for debugging and improvement.
    """
    
    def __init__(self, test_results_dir: str = None):
        self.agent = MarketStructureAgent()
        
        # Setup output directory - make it relative to the script location if it's a relative path
        if test_results_dir is None:
            test_results_dir = "test_results"
            
        if not os.path.isabs(test_results_dir):
            script_dir = Path(__file__).parent
            self.test_results_dir = script_dir / test_results_dir
        else:
            self.test_results_dir = Path(test_results_dir)
        
        self.test_results_dir.mkdir(parents=True, exist_ok=True)
        
        # Test tracking
        self.test_session_id = f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.results = []
        
        logger.info(f"Market Structure Agent Tester initialized")
        logger.info(f"Test results directory: {self.test_results_dir}")
        logger.info(f"Test session ID: {self.test_session_id}")
    
    def generate_test_data(self, symbol: str, scenario: str = "uptrend") -> pd.DataFrame:
        """
        Generate synthetic test data for different market scenarios.
        
        Args:
            symbol: Stock symbol for testing
            scenario: Market scenario type
            
        Returns:
            DataFrame with OHLCV data
        """
        np.random.seed(hash(symbol) % 2147483647)  # Deterministic but varied per symbol
        
        periods = 100
        base_price = 100.0 + (hash(symbol) % 200)  # Vary base price by symbol
        
        dates = pd.date_range(start=datetime.now() - timedelta(days=periods), periods=periods, freq='D')
        
        if scenario == "uptrend":
            # Generate uptrending data with swing points
            trend = np.linspace(0, 20, periods)
            noise = np.random.normal(0, 2, periods)
            swing_pattern = 3 * np.sin(np.linspace(0, 6*np.pi, periods))
            
            prices = base_price + trend + noise + swing_pattern
            
        elif scenario == "downtrend":
            # Generate downtrending data
            trend = np.linspace(0, -15, periods)
            noise = np.random.normal(0, 1.5, periods)
            swing_pattern = 2 * np.sin(np.linspace(0, 5*np.pi, periods))
            
            prices = base_price + trend + noise + swing_pattern
            
        elif scenario == "sideways":
            # Generate sideways/ranging data
            noise = np.random.normal(0, 1, periods)
            swing_pattern = 4 * np.sin(np.linspace(0, 8*np.pi, periods))
            
            prices = base_price + noise + swing_pattern
            
        elif scenario == "volatile":
            # Generate highly volatile data
            trend = np.linspace(0, 10, periods) 
            volatility = np.random.normal(0, 4, periods)
            swing_pattern = 5 * np.sin(np.linspace(0, 10*np.pi, periods))
            
            prices = base_price + trend + volatility + swing_pattern
            
        else:  # mixed
            # Generate mixed scenario data
            trend1 = np.linspace(0, 15, periods//2)
            trend2 = np.linspace(15, 5, periods//2)
            trend = np.concatenate([trend1, trend2])
            
            noise = np.random.normal(0, 2, periods)
            swing_pattern = 3 * np.sin(np.linspace(0, 7*np.pi, periods))
            
            prices = base_price + trend + noise + swing_pattern
        
        # Ensure positive prices
        prices = np.maximum(prices, base_price * 0.5)
        
        # Generate OHLC from close prices
        highs = prices * (1 + np.abs(np.random.normal(0, 0.02, periods)))
        lows = prices * (1 - np.abs(np.random.normal(0, 0.02, periods)))
        opens = np.roll(prices, 1)
        opens[0] = prices[0]
        
        # Generate volume
        volumes = np.random.lognormal(10, 0.5, periods).astype(int)
        
        data = pd.DataFrame({
            'open': opens,
            'high': highs,
            'low': lows,
            'close': prices,
            'volume': volumes
        }, index=dates)
        
        return data
    
    async def test_single_stock(self, symbol: str, scenario: str = "uptrend") -> Dict[str, Any]:
        """
        Test the agent on a single stock with specific scenario.
        
        Args:
            symbol: Stock symbol to test
            scenario: Market scenario for test data
            
        Returns:
            Test results dictionary
        """
        logger.info(f"Testing {symbol} with {scenario} scenario")
        
        try:
            # Generate test data
            stock_data = self.generate_test_data(symbol, scenario)
            
            # Test complete analysis
            start_time = datetime.now()
            result = await self.agent.analyze_complete(
                stock_data=stock_data,
                symbol=symbol,
                context=f"Test scenario: {scenario}"
            )
            end_time = datetime.now()
            
            # Extract key metrics for evaluation
            test_result = {
                'symbol': symbol,
                'scenario': scenario,
                'timestamp': start_time.isoformat(),
                'test_duration': (end_time - start_time).total_seconds(),
                'success': result.get('success', False),
                'confidence_score': result.get('confidence_score', 0),
                'has_llm_analysis': result.get('has_llm_analysis', False),
                'processing_time': result.get('processing_time', 0),
                'error': result.get('error'),
                'insights': self.agent.get_key_insights(result),
                'technical_analysis_summary': self._extract_technical_summary(result),
                'result': result  # Full result for detailed analysis
            }
            
            # Save detailed results
            await self._save_test_result(test_result)
            
            logger.info(f"✅ Test completed for {symbol}: {test_result['success']}")
            return test_result
            
        except Exception as e:
            logger.error(f"❌ Test failed for {symbol}: {e}")
            error_result = {
                'symbol': symbol,
                'scenario': scenario,
                'timestamp': datetime.now().isoformat(),
                'success': False,
                'error': str(e),
                'confidence_score': 0,
                'has_llm_analysis': False
            }
            await self._save_test_result(error_result)
            return error_result
    
    def _extract_technical_summary(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract technical analysis summary for evaluation."""
        if not result.get('success') or 'technical_analysis' not in result:
            return {'error': 'No technical analysis available'}
        
        technical = result['technical_analysis']
        
        return {
            'swing_points': {
                'total': technical.get('swing_points', {}).get('total_swings', 0),
                'density': technical.get('swing_points', {}).get('swing_density', 0)
            },
            'trend': {
                'direction': technical.get('trend_analysis', {}).get('trend_direction', 'unknown'),
                'strength': technical.get('trend_analysis', {}).get('trend_strength', 'unknown'),
                'quality': technical.get('trend_analysis', {}).get('trend_quality', 'unknown')
            },
            'structure_quality': {
                'score': technical.get('structure_quality', {}).get('quality_score', 0),
                'rating': technical.get('structure_quality', {}).get('quality_rating', 'unknown')
            },
            'bos_choch': {
                'structural_bias': technical.get('bos_choch_analysis', {}).get('structural_bias', 'unknown'),
                'total_events': (technical.get('bos_choch_analysis', {}).get('total_bos_events', 0) + 
                               technical.get('bos_choch_analysis', {}).get('total_choch_events', 0))
            },
            'current_state': technical.get('current_state', {}).get('structure_state', 'unknown'),
            'confidence': technical.get('confidence_score', 0)
        }
    
    async def _save_test_result(self, test_result: Dict[str, Any]):
        """Save individual test result to file."""
        try:
            # Create session directory
            session_dir = self.test_results_dir / self.test_session_id
            session_dir.mkdir(exist_ok=True)
            
            # Save individual test result
            result_file = session_dir / f"{test_result['symbol']}_{test_result['scenario']}.json"
            
            # Prepare result for JSON serialization
            serializable_result = self._make_json_serializable(test_result)
            
            with open(result_file, 'w') as f:
                json.dump(serializable_result, f, indent=2, default=str)
            
            # Also save the prompt and response if available
            if test_result.get('has_llm_analysis') and 'result' in test_result:
                await self._save_prompt_response(test_result)
                
        except Exception as e:
            logger.error(f"Failed to save test result for {test_result.get('symbol', 'unknown')}: {e}")
    
    async def _save_prompt_response(self, test_result: Dict[str, Any]):
        """Save LLM prompt and response for debugging."""
        try:
            session_dir = self.test_results_dir / self.test_session_id / "prompts_responses"
            session_dir.mkdir(exist_ok=True)
            
            symbol = test_result['symbol']
            scenario = test_result['scenario']
            
            # Extract technical analysis for prompt building
            if 'result' in test_result and 'technical_analysis' in test_result['result']:
                technical_analysis = test_result['result']['technical_analysis']
                
                # Build the prompt using the LLM agent
                llm_agent = MarketStructureLLMAgent()
                prompt = llm_agent.build_analysis_prompt(technical_analysis, symbol)
                
                # Save prompt
                prompt_file = session_dir / f"{symbol}_{scenario}_prompt.txt"
                with open(prompt_file, 'w', encoding='utf-8') as f:
                    f.write(prompt)
                
                # Save response if available
                llm_response = test_result['result'].get('llm_analysis')
                if llm_response:
                    response_file = session_dir / f"{symbol}_{scenario}_response.txt"
                    with open(response_file, 'w', encoding='utf-8') as f:
                        f.write(llm_response)
                        
        except Exception as e:
            logger.error(f"Failed to save prompt/response for {test_result.get('symbol', 'unknown')}: {e}")
    
    def _make_json_serializable(self, obj):
        """Make object JSON serializable."""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (datetime, pd.Timestamp)):
            return obj.isoformat()
        elif isinstance(obj, bytes):
            return f"<bytes: {len(obj)} bytes>"
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif hasattr(obj, '__dict__'):
            return self._make_json_serializable(obj.__dict__)
        else:
            return obj
    
    async def run_comprehensive_test(self, 
                                   test_stocks: List[str] = None,
                                   scenarios: List[str] = None) -> Dict[str, Any]:
        """
        Run comprehensive tests across multiple stocks and scenarios.
        
        Args:
            test_stocks: List of stock symbols to test (default: standard test set)
            scenarios: List of scenarios to test (default: all scenarios)
            
        Returns:
            Comprehensive test results
        """
        if test_stocks is None:
            test_stocks = ["RELIANCE"]
        
        if scenarios is None:
            scenarios = ["uptrend", "downtrend", "sideways", "volatile", "mixed"]
        
        logger.info(f"Starting comprehensive test: {len(test_stocks)} stocks × {len(scenarios)} scenarios")
        logger.info(f"Total tests: {len(test_stocks) * len(scenarios)}")
        
        all_results = []
        start_time = datetime.now()
        
        # Test each combination
        for scenario in scenarios:
            logger.info(f"\n📊 Testing scenario: {scenario}")
            scenario_results = []
            
            for stock in test_stocks:
                try:
                    result = await self.test_single_stock(stock, scenario)
                    scenario_results.append(result)
                    all_results.append(result)
                    
                    # Brief pause between tests
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    logger.error(f"Failed test for {stock} in {scenario}: {e}")
                    continue
            
            logger.info(f"Completed scenario {scenario}: {len(scenario_results)} tests")
        
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()
        
        # Compile comprehensive results
        comprehensive_results = {
            'test_session_id': self.test_session_id,
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'total_duration': total_duration,
            'total_tests': len(all_results),
            'test_stocks': test_stocks,
            'scenarios': scenarios,
            'summary_stats': self._calculate_summary_stats(all_results),
            'all_results': all_results
        }
        
        # Save comprehensive results
        await self._save_comprehensive_results(comprehensive_results)
        
        # Print summary
        self._print_test_summary(comprehensive_results)
        
        return comprehensive_results
    
    def _calculate_summary_stats(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate summary statistics from test results."""
        if not results:
            return {}
        
        successful_tests = [r for r in results if r.get('success', False)]
        llm_tests = [r for r in results if r.get('has_llm_analysis', False)]
        
        confidence_scores = [r.get('confidence_score', 0) for r in successful_tests]
        processing_times = [r.get('processing_time', 0) for r in successful_tests]
        
        return {
            'total_tests': len(results),
            'successful_tests': len(successful_tests),
            'failed_tests': len(results) - len(successful_tests),
            'success_rate': len(successful_tests) / len(results) if results else 0,
            'llm_analysis_rate': len(llm_tests) / len(results) if results else 0,
            'average_confidence': np.mean(confidence_scores) if confidence_scores else 0,
            'confidence_std': np.std(confidence_scores) if confidence_scores else 0,
            'average_processing_time': np.mean(processing_times) if processing_times else 0,
            'processing_time_std': np.std(processing_times) if processing_times else 0,
            'by_scenario': self._stats_by_scenario(results),
            'by_stock': self._stats_by_stock(results)
        }
    
    def _stats_by_scenario(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate statistics by scenario."""
        by_scenario = {}
        for result in results:
            scenario = result.get('scenario', 'unknown')
            if scenario not in by_scenario:
                by_scenario[scenario] = []
            by_scenario[scenario].append(result)
        
        return {
            scenario: {
                'count': len(scenario_results),
                'success_rate': len([r for r in scenario_results if r.get('success', False)]) / len(scenario_results),
                'avg_confidence': np.mean([r.get('confidence_score', 0) for r in scenario_results if r.get('success', False)]) if scenario_results else 0
            }
            for scenario, scenario_results in by_scenario.items()
        }
    
    def _stats_by_stock(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate statistics by stock."""
        by_stock = {}
        for result in results:
            stock = result.get('symbol', 'unknown')
            if stock not in by_stock:
                by_stock[stock] = []
            by_stock[stock].append(result)
        
        return {
            stock: {
                'count': len(stock_results),
                'success_rate': len([r for r in stock_results if r.get('success', False)]) / len(stock_results),
                'avg_confidence': np.mean([r.get('confidence_score', 0) for r in stock_results if r.get('success', False)]) if stock_results else 0
            }
            for stock, stock_results in by_stock.items()
        }
    
    async def _save_comprehensive_results(self, results: Dict[str, Any]):
        """Save comprehensive test results."""
        try:
            session_dir = self.test_results_dir / self.test_session_id
            results_file = session_dir / "comprehensive_results.json"
            
            # Make JSON serializable
            serializable_results = self._make_json_serializable(results)
            
            with open(results_file, 'w') as f:
                json.dump(serializable_results, f, indent=2, default=str)
                
            logger.info(f"Comprehensive results saved to {results_file}")
            
        except Exception as e:
            logger.error(f"Failed to save comprehensive results: {e}")
    
    def _print_test_summary(self, results: Dict[str, Any]):
        """Print test summary to console."""
        stats = results.get('summary_stats', {})
        
        print("\n" + "="*60)
        print("🧪 MARKET STRUCTURE AGENT - TEST SUMMARY")
        print("="*60)
        print(f"Test Session: {results['test_session_id']}")
        print(f"Total Duration: {results['total_duration']:.1f}s")
        print(f"Total Tests: {stats.get('total_tests', 0)}")
        print(f"Success Rate: {stats.get('success_rate', 0):.2%}")
        print(f"LLM Analysis Rate: {stats.get('llm_analysis_rate', 0):.2%}")
        print(f"Average Confidence: {stats.get('average_confidence', 0):.2f}")
        print(f"Average Processing Time: {stats.get('average_processing_time', 0):.2f}s")
        
        print("\n📊 BY SCENARIO:")
        by_scenario = stats.get('by_scenario', {})
        for scenario, scenario_stats in by_scenario.items():
            print(f"  {scenario:>12}: {scenario_stats['success_rate']:.2%} success, "
                  f"{scenario_stats['avg_confidence']:.2f} avg confidence")
        
        print("\n📈 BY STOCK:")
        by_stock = stats.get('by_stock', {})
        for stock, stock_stats in by_stock.items():
            print(f"  {stock:>6}: {stock_stats['success_rate']:.2%} success, "
                  f"{stock_stats['avg_confidence']:.2f} avg confidence")
        
        print("="*60)


# Main test function
async def main():
    """Run the comprehensive market structure agent test."""
    print("🚀 Starting Market Structure Agent Comprehensive Test")
    
    # Create tester
    tester = MarketStructureAgentTester()
    
    # Run comprehensive test
    results = await tester.run_comprehensive_test()
    
    print(f"\n✅ Test completed! Results saved in: {tester.test_results_dir / tester.test_session_id}")
    return results


if __name__ == "__main__":
    asyncio.run(main())