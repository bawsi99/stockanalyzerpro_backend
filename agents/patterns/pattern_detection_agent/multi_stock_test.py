#!/usr/bin/env python3
"""
Pattern Detection Agent - Multi-Stock Testing Framework

This module provides comprehensive testing capabilities for the pattern detection agent including:
- Multi-stock batch testing
- Performance benchmarking
- Pattern detection accuracy validation
- Result analysis and reporting
"""

import asyncio
import pandas as pd
import numpy as np
import json
import os
import sys
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
import time
import traceback
from pathlib import Path

# Add the backend directory to sys.path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# Import the pattern detection agent
from agents.patterns.pattern_detection_agent.agent import PatternDetectionAgent

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PatternDetectionMultiStockTester:
    """
    Comprehensive testing framework for the Pattern Detection Agent.
    
    Provides capabilities for:
    - Multi-stock batch testing
    - Performance analysis and benchmarking
    - Pattern detection validation
    - Result aggregation and reporting
    """
    
    def __init__(self, output_dir: str = "pattern_detection_test_results"):
        self.name = "pattern_detection_multi_stock_tester"
        self.version = "1.0.0"
        
        # Initialize agent for testing
        self.agent = PatternDetectionAgent()
        
        # Setup output directory - make it relative to the script location if it's a relative path
        if not os.path.isabs(output_dir):
            script_dir = Path(__file__).parent
            self.output_dir = script_dir / output_dir
        else:
            self.output_dir = Path(output_dir)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Test configuration
        self.test_config = {
            'include_charts': True,
            'save_individual_results': True,
            'save_charts': True
        }
        
        # Results storage
        self.test_results = []
        self.performance_metrics = {}
        self.summary_statistics = {}
    
    async def run_multi_stock_test(
        self,
        test_stocks: List[str],
        test_periods: List[int] = [30, 60, 90],
        max_concurrent: int = 3,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """
        Run comprehensive multi-stock pattern detection testing.
        
        Args:
            test_stocks: List of stock symbols to test
            test_periods: List of time periods (in days) to test
            max_concurrent: Maximum concurrent tests to run
            save_results: Whether to save detailed results to files
            
        Returns:
            Dictionary containing comprehensive test results and analysis
        """
        test_start_time = datetime.now()
        
        try:
            logger.info(f"[PATTERN_DETECTION_TESTER] Starting multi-stock test with {len(test_stocks)} stocks")
            logger.info(f"[PATTERN_DETECTION_TESTER] Test periods: {test_periods}")
            logger.info(f"[PATTERN_DETECTION_TESTER] Max concurrent: {max_concurrent}")
            
            # Generate test cases
            test_cases = self._generate_test_cases(test_stocks, test_periods)
            logger.info(f"[PATTERN_DETECTION_TESTER] Generated {len(test_cases)} test cases")
            
            # Initialize results tracking
            self.test_results = []
            self.performance_metrics = {}
            
            # Run tests with concurrency control
            await self._execute_batch_tests(test_cases, max_concurrent)
            
            # Analyze results
            analysis_results = self._analyze_test_results()
            
            # Generate comprehensive report
            test_report = self._generate_test_report(test_start_time, analysis_results)
            
            # Save results if requested
            if save_results:
                await self._save_test_results(test_report)
            
            total_time = (datetime.now() - test_start_time).total_seconds()
            logger.info(f"[PATTERN_DETECTION_TESTER] Multi-stock test completed in {total_time:.2f}s")
            
            return test_report
            
        except Exception as e:
            logger.error(f"[PATTERN_DETECTION_TESTER] Multi-stock test failed: {e}")
            logger.error(f"[PATTERN_DETECTION_TESTER] Traceback: {traceback.format_exc()}")
            return self._build_error_report(str(e), test_start_time)
    
    def _generate_test_cases(self, test_stocks: List[str], test_periods: List[int]) -> List[Dict[str, Any]]:
        """Generate test cases for all stock/period combinations"""
        test_cases = []
        
        for symbol in test_stocks:
            for period in test_periods:
                test_case = {
                    'symbol': symbol,
                    'period_days': period,
                    'test_id': f"{symbol}_{period}d",
                    'start_date': datetime.now() - timedelta(days=period + 10),  # Add buffer for data
                    'end_date': datetime.now()
                }
                test_cases.append(test_case)
        
        return test_cases
    
    async def _execute_batch_tests(self, test_cases: List[Dict[str, Any]], max_concurrent: int):
        """Execute test cases with concurrency control"""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def run_single_test(test_case: Dict[str, Any]):
            async with semaphore:
                return await self._run_single_pattern_test(test_case)
        
        # Run all tests concurrently with limit
        tasks = [run_single_test(test_case) for test_case in test_cases]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"[PATTERN_DETECTION_TESTER] Test case {i} failed: {result}")
                self.test_results.append(self._build_failed_test_result(test_cases[i], str(result)))
            else:
                self.test_results.append(result)
    
    async def _run_single_pattern_test(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Run pattern detection test for a single stock/period combination"""
        test_start = time.time()
        
        try:
            symbol = test_case['symbol']
            period_days = test_case['period_days']
            test_id = test_case['test_id']
            
            logger.info(f"[PATTERN_DETECTION_TESTER] Testing {test_id}")
            
            # Generate synthetic stock data (since we might not have real data access)
            stock_data = self._generate_synthetic_stock_data(symbol, period_days)
            
            if stock_data is None or len(stock_data) < 20:
                return self._build_failed_test_result(test_case, "Insufficient stock data")
            
            # Prepare save path for this test
            test_output_dir = self.output_dir / test_id
            test_output_dir.mkdir(exist_ok=True)
            save_path = str(test_output_dir / f"{test_id}_charts.html")
            
            # Run pattern detection analysis
            analysis_start = time.time()
            analysis_results = await self.agent.analyze_patterns(
                stock_data=stock_data,
                symbol=symbol,
                include_charts=self.test_config['include_charts'],
                save_path=save_path if self.test_config['save_charts'] else None
            )
            analysis_time = time.time() - analysis_start
            
            # Build test result
            test_result = {
                'test_id': test_id,
                'symbol': symbol,
                'period_days': period_days,
                'test_timestamp': datetime.now().isoformat(),
                'test_duration': time.time() - test_start,
                'analysis_duration': analysis_time,
                'data_points': len(stock_data),
                
                # Analysis results
                'analysis_success': analysis_results.get('success', False),
                'patterns_detected': analysis_results.get('total_patterns_detected', 0),
                'detected_patterns': analysis_results.get('detected_patterns', []),
                'overall_confidence': analysis_results.get('overall_confidence', 0.0),
                'technical_confidence': analysis_results.get('technical_confidence', 0.0),
                
                # Pattern summary
                'pattern_summary': analysis_results.get('pattern_summary', {}),
                'formation_stage': analysis_results.get('formation_stage', {}),
                'key_levels': analysis_results.get('key_levels', {}),
                
                # Component status
                'charts_generated': analysis_results.get('charts_generated', 0),
                'components_executed': analysis_results.get('analysis_summary', {}).get('components_list', []),
                
                # Executive summary
                'executive_summary': analysis_results.get('executive_summary', {}),
                
                # Full results for detailed analysis
                'full_analysis_results': analysis_results
            }
            
            # Save individual test result if configured
            if self.test_config['save_individual_results']:
                await self._save_individual_test_result(test_result, test_output_dir)
            
            logger.info(f"[PATTERN_DETECTION_TESTER] Completed {test_id} - "
                       f"Success: {test_result['analysis_success']}, "
                       f"Patterns: {test_result['patterns_detected']}, "
                       f"Time: {test_result['test_duration']:.2f}s")
            
            return test_result
            
        except Exception as e:
            logger.error(f"[PATTERN_DETECTION_TESTER] Single test failed for {test_case.get('test_id', 'unknown')}: {e}")
            return self._build_failed_test_result(test_case, str(e))
    
    def _generate_synthetic_stock_data(self, symbol: str, period_days: int) -> pd.DataFrame:
        """Generate realistic synthetic stock data for testing"""
        try:
            # Generate dates
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=period_days)
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            
            # Generate synthetic price data with realistic patterns
            np.random.seed(hash(symbol) % 2**32)  # Consistent seed based on symbol
            
            base_price = np.random.uniform(50, 500)  # Random base price
            returns = np.random.normal(0.0005, 0.02, len(dates))  # Daily returns with slight upward bias
            
            # Add some trend and pattern elements
            trend = np.linspace(0, np.random.uniform(-0.1, 0.1), len(dates))
            cycle = 0.05 * np.sin(np.linspace(0, 4*np.pi, len(dates)))  # Add cyclical pattern
            
            # Generate price series
            log_prices = np.cumsum(returns + trend + cycle)
            prices = base_price * np.exp(log_prices)
            
            # Generate OHLC data
            data = []
            for i, (date, close_price) in enumerate(zip(dates, prices)):
                # Generate realistic OHLC based on close price
                daily_volatility = abs(np.random.normal(0, 0.015))
                
                open_price = close_price * (1 + np.random.normal(0, 0.005))
                high_price = max(open_price, close_price) * (1 + daily_volatility * np.random.uniform(0, 1))
                low_price = min(open_price, close_price) * (1 - daily_volatility * np.random.uniform(0, 1))
                
                # Generate volume (with some correlation to price movement)
                price_change = abs(close_price - open_price) / open_price if i == 0 else abs(close_price - prices[i-1]) / prices[i-1]
                base_volume = np.random.uniform(100000, 1000000)
                volume = base_volume * (1 + price_change * 5)  # Higher volume on larger moves
                
                data.append({
                    'date': date,
                    'open': round(open_price, 2),
                    'high': round(high_price, 2),
                    'low': round(low_price, 2),
                    'close': round(close_price, 2),
                    'volume': int(volume)
                })
            
            df = pd.DataFrame(data)
            df.set_index('date', inplace=True)
            
            logger.debug(f"[PATTERN_DETECTION_TESTER] Generated {len(df)} days of synthetic data for {symbol}")
            
            return df
            
        except Exception as e:
            logger.error(f"[PATTERN_DETECTION_TESTER] Synthetic data generation failed for {symbol}: {e}")
            return None
    
    async def _save_individual_test_result(self, test_result: Dict[str, Any], output_dir: Path):
        """Save individual test result to file"""
        try:
            result_file = output_dir / "test_result.json"
            
            # Create a clean version without the full analysis results for JSON serialization
            clean_result = {k: v for k, v in test_result.items() if k != 'full_analysis_results'}
            
            with open(result_file, 'w') as f:
                json.dump(clean_result, f, indent=2, default=str)
            
            # Save detailed analysis separately
            detailed_file = output_dir / "detailed_analysis.json"
            with open(detailed_file, 'w') as f:
                json.dump(test_result['full_analysis_results'], f, indent=2, default=str)
            
            logger.debug(f"[PATTERN_DETECTION_TESTER] Saved individual result to {output_dir}")
            
        except Exception as e:
            logger.error(f"[PATTERN_DETECTION_TESTER] Failed to save individual result: {e}")
    
    def _analyze_test_results(self) -> Dict[str, Any]:
        """Analyze aggregated test results"""
        try:
            if not self.test_results:
                return {'error': 'No test results to analyze'}
            
            # Basic statistics
            total_tests = len(self.test_results)
            successful_tests = len([r for r in self.test_results if r.get('analysis_success', False)])
            failed_tests = total_tests - successful_tests
            
            success_rate = (successful_tests / total_tests) * 100 if total_tests > 0 else 0
            
            # Performance metrics
            test_durations = [r.get('test_duration', 0) for r in self.test_results if r.get('test_duration')]
            analysis_durations = [r.get('analysis_duration', 0) for r in self.test_results if r.get('analysis_duration')]
            
            # Pattern detection statistics
            patterns_detected = [r.get('patterns_detected', 0) for r in self.test_results]
            confidence_scores = [r.get('overall_confidence', 0) for r in self.test_results if r.get('analysis_success')]
            
            # Component success rates
            charts_generated_count = len([r for r in self.test_results if r.get('charts_generated', 0) > 0])
            
            # Pattern type analysis
            pattern_types = {}
            for result in self.test_results:
                for pattern in result.get('detected_patterns', []):
                    pattern_name = pattern.get('pattern_name', 'unknown')
                    pattern_types[pattern_name] = pattern_types.get(pattern_name, 0) + 1
            
            analysis = {
                'test_summary': {
                    'total_tests': total_tests,
                    'successful_tests': successful_tests,
                    'failed_tests': failed_tests,
                    'success_rate_percent': round(success_rate, 2)
                },
                'performance_metrics': {
                    'avg_test_duration': np.mean(test_durations) if test_durations else 0,
                    'max_test_duration': np.max(test_durations) if test_durations else 0,
                    'min_test_duration': np.min(test_durations) if test_durations else 0,
                    'avg_analysis_duration': np.mean(analysis_durations) if analysis_durations else 0,
                    'total_test_time': np.sum(test_durations) if test_durations else 0
                },
                'pattern_detection_stats': {
                    'total_patterns_found': np.sum(patterns_detected),
                    'avg_patterns_per_test': np.mean(patterns_detected) if patterns_detected else 0,
                    'max_patterns_per_test': np.max(patterns_detected) if patterns_detected else 0,
                    'tests_with_patterns': len([p for p in patterns_detected if p > 0]),
                    'pattern_detection_rate': (len([p for p in patterns_detected if p > 0]) / total_tests * 100) if total_tests > 0 else 0
                },
                'confidence_analysis': {
                    'avg_confidence': np.mean(confidence_scores) if confidence_scores else 0,
                    'max_confidence': np.max(confidence_scores) if confidence_scores else 0,
                    'min_confidence': np.min(confidence_scores) if confidence_scores else 0,
                    'high_confidence_tests': len([c for c in confidence_scores if c >= 0.8]),
                    'medium_confidence_tests': len([c for c in confidence_scores if 0.5 <= c < 0.8]),
                    'low_confidence_tests': len([c for c in confidence_scores if c < 0.5])
                },
                'component_success_rates': {
                    'charts_generation_rate': (charts_generated_count / total_tests * 100) if total_tests > 0 else 0
                },
                'pattern_types_detected': pattern_types,
                'most_common_patterns': sorted(pattern_types.items(), key=lambda x: x[1], reverse=True)[:10]
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"[PATTERN_DETECTION_TESTER] Result analysis failed: {e}")
            return {'error': str(e)}
    
    def _generate_test_report(self, test_start_time: datetime, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        
        total_duration = (datetime.now() - test_start_time).total_seconds()
        
        return {
            'test_report_metadata': {
                'test_framework': self.name,
                'framework_version': self.version,
                'test_start_time': test_start_time.isoformat(),
                'test_end_time': datetime.now().isoformat(),
                'total_test_duration': total_duration,
                'agent_tested': self.agent.name,
                'agent_version': self.agent.version
            },
            'test_configuration': self.test_config,
            'analysis_results': analysis_results,
            'individual_test_results': self.test_results,
            'test_success': analysis_results.get('test_summary', {}).get('success_rate_percent', 0) > 70,
            'recommendations': self._generate_recommendations(analysis_results),
            'summary': self._generate_test_summary(analysis_results, total_duration)
        }
    
    def _generate_recommendations(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        try:
            success_rate = analysis_results.get('test_summary', {}).get('success_rate_percent', 0)
            avg_confidence = analysis_results.get('confidence_analysis', {}).get('avg_confidence', 0)
            pattern_detection_rate = analysis_results.get('pattern_detection_stats', {}).get('pattern_detection_rate', 0)
            
            if success_rate < 80:
                recommendations.append(f"Success rate ({success_rate:.1f}%) below target (80%). Investigate failure causes.")
            
            if avg_confidence < 0.6:
                recommendations.append(f"Average confidence ({avg_confidence:.1%}) is low. Consider improving pattern detection algorithms.")
            
            if pattern_detection_rate < 60:
                recommendations.append(f"Pattern detection rate ({pattern_detection_rate:.1f}%) is low. Review pattern detection sensitivity.")
            
            if not recommendations:
                recommendations.append("All metrics are performing well. Consider expanding test coverage.")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"[PATTERN_DETECTION_TESTER] Recommendation generation failed: {e}")
            return ["Error generating recommendations"]
    
    def _generate_test_summary(self, analysis_results: Dict[str, Any], total_duration: float) -> str:
        """Generate human-readable test summary"""
        
        try:
            summary = analysis_results.get('test_summary', {})
            total_tests = summary.get('total_tests', 0)
            success_rate = summary.get('success_rate_percent', 0)
            
            patterns_stats = analysis_results.get('pattern_detection_stats', {})
            total_patterns = patterns_stats.get('total_patterns_found', 0)
            avg_patterns = patterns_stats.get('avg_patterns_per_test', 0)
            
            confidence_stats = analysis_results.get('confidence_analysis', {})
            avg_confidence = confidence_stats.get('avg_confidence', 0)
            
            return f"""
Pattern Detection Agent Test Summary:
- Executed {total_tests} tests in {total_duration:.2f} seconds
- Success rate: {success_rate:.1f}%
- Total patterns detected: {total_patterns}
- Average patterns per test: {avg_patterns:.1f}
- Average confidence score: {avg_confidence:.1%}
- Test framework performance: {'PASS' if success_rate > 70 else 'FAIL'}
"""
            
        except Exception as e:
            logger.error(f"[PATTERN_DETECTION_TESTER] Summary generation failed: {e}")
            return f"Summary generation failed: {str(e)}"
    
    async def _save_test_results(self, test_report: Dict[str, Any]):
        """Save comprehensive test results to files"""
        try:
            # Save main test report
            report_file = self.output_dir / f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_file, 'w') as f:
                json.dump(test_report, f, indent=2, default=str)
            
            # Save summary report
            summary_file = self.output_dir / "test_summary.txt"
            with open(summary_file, 'w') as f:
                f.write(test_report.get('summary', ''))
                f.write('\n\n')
                f.write('\n'.join(test_report.get('recommendations', [])))
            
            logger.info(f"[PATTERN_DETECTION_TESTER] Test results saved to {self.output_dir}")
            
        except Exception as e:
            logger.error(f"[PATTERN_DETECTION_TESTER] Failed to save test results: {e}")
    
    def _build_failed_test_result(self, test_case: Dict[str, Any], error_message: str) -> Dict[str, Any]:
        """Build result structure for failed tests"""
        return {
            'test_id': test_case.get('test_id', 'unknown'),
            'symbol': test_case.get('symbol', 'unknown'),
            'period_days': test_case.get('period_days', 0),
            'test_timestamp': datetime.now().isoformat(),
            'test_duration': 0.0,
            'analysis_success': False,
            'failure_reason': error_message,
            'patterns_detected': 0,
            'overall_confidence': 0.0,
            'technical_confidence': 0.0
        }
    
    def _build_error_report(self, error_message: str, test_start_time: datetime) -> Dict[str, Any]:
        """Build error report for failed test run"""
        return {
            'test_report_metadata': {
                'test_framework': self.name,
                'framework_version': self.version,
                'test_start_time': test_start_time.isoformat(),
                'test_end_time': datetime.now().isoformat(),
                'total_test_duration': (datetime.now() - test_start_time).total_seconds(),
                'test_success': False,
                'error': error_message
            },
            'analysis_results': {'error': error_message},
            'individual_test_results': [],
            'recommendations': ['Fix the critical error before running tests again'],
            'summary': f"Test run failed due to error: {error_message}"
        }

# Main execution function for standalone testing
async def main():
    """Main function for running pattern detection tests"""
    
    # Test configuration
    test_stocks = ["RELIANCE"]
    test_periods = [30, 60, 90]
    max_concurrent = 2
    
    # Initialize tester
    tester = PatternDetectionMultiStockTester("test_results")
    
    # Run tests
    print("Starting Pattern Detection Agent Multi-Stock Test...")
    test_results = await tester.run_multi_stock_test(
        test_stocks=test_stocks,
        test_periods=test_periods,
        max_concurrent=max_concurrent,
        save_results=True
    )
    
    # Display results
    print("\n" + "="*60)
    print("TEST RESULTS SUMMARY")
    print("="*60)
    print(test_results.get('summary', 'No summary available'))
    
    print("\nRECOMMENDations:")
    for rec in test_results.get('recommendations', []):
        print(f"- {rec}")
    
    print(f"\nDetailed results saved to: {tester.output_dir}")
    
    return test_results

# Allow running this script directly
if __name__ == "__main__":
    # Run the test
    results = asyncio.run(main())