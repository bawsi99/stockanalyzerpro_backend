#!/usr/bin/env python3
"""
Cross-Validation Agent - Multi-Stock Testing Framework

This module provides comprehensive testing capabilities for the cross-validation agent including:
- Multi-stock validation testing with REAL MARKET DATA
- Validation method performance analysis on real patterns
- Cross-validation accuracy assessment using real market conditions
- Comprehensive result analysis and reporting
- Fallback to synthetic data if real data sources are unavailable
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

# Load environment variables
try:
    import dotenv
    # Load .env file from the backend/config directory
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'config', '.env')
    dotenv.load_dotenv(dotenv_path=env_path)
    print(f"✅ Environment variables loaded from: {env_path}")
except ImportError:
    print("⚠️ python-dotenv not available, using system environment variables")

# Add the backend directory to sys.path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# Import the cross-validation agent
from agents.patterns.cross_validation_agent.agent import CrossValidationAgent
from agents.patterns.cross_validation_agent.llm_agent import CrossValidationLLMAgent

# Import data clients for real data
try:
    # Add the backend directory to sys.path for proper imports
    backend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..')
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)
    
    from core.orchestrator import StockAnalysisOrchestrator
    from zerodha.client import ZerodhaDataClient
    print("✅ Real data client imports successful")
except ImportError as e:
    print(f"⚠️ Real data client import failed: {e}")
    print("Will fall back to synthetic data if real data is not available")
    StockAnalysisOrchestrator = None
    ZerodhaDataClient = None

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CrossValidationMultiStockTester:
    """
    Comprehensive testing framework for the Cross-Validation Agent.
    
    Provides capabilities for:
    - Multi-stock validation testing
    - Validation method performance analysis
    - Pattern validation accuracy assessment
    - Result aggregation and comprehensive reporting
    """
    
    def __init__(self, output_dir: str = "cross_validation_test_results"):
        self.name = "cross_validation_multi_stock_tester"
        self.version = "1.0.0"
        
        # Initialize agent for testing
        self.agent = CrossValidationAgent()
        
        # Initialize real data clients
        self.orchestrator = None
        self.zerodha_client = None
        self.use_real_data = False
        
        # Try to initialize real data clients
        if StockAnalysisOrchestrator and ZerodhaDataClient:
            try:
                self.orchestrator = StockAnalysisOrchestrator()
                self.zerodha_client = ZerodhaDataClient()
                self.use_real_data = True
                print("✅ Real data clients initialized successfully")
            except Exception as e:
                print(f"⚠️ Could not initialize real data clients: {e}")
                print("Will use synthetic data for testing")
        else:
            print("⚠️ Real data client classes not available - using synthetic data")
        
        # Setup output directory - make it relative to the script location if it's a relative path
        if not os.path.isabs(output_dir):
            script_dir = Path(__file__).parent
            self.output_dir = script_dir / output_dir
        else:
            self.output_dir = Path(output_dir)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Test configuration
        self.test_config = {
            'include_charts': False,
            'include_llm_analysis': True,
            'save_individual_results': False,
            'save_charts': False
        }
        
        # Results storage
        self.test_results = []
        self.performance_metrics = {}
        self.summary_statistics = {}
    
    async def run_multi_stock_validation_test(
        self,
        test_stocks: List[str],
        test_periods: List[int] = [30, 60, 90],
        max_concurrent: int = 3,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """
        Run comprehensive multi-stock cross-validation testing.
        
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
            logger.info(f"[CROSS_VALIDATION_TESTER] Starting multi-stock validation test with {len(test_stocks)} stocks")
            logger.info(f"[CROSS_VALIDATION_TESTER] Test periods: {test_periods}")
            logger.info(f"[CROSS_VALIDATION_TESTER] Max concurrent: {max_concurrent}")
            
            # Generate test cases
            test_cases = self._generate_test_cases(test_stocks, test_periods)
            logger.info(f"[CROSS_VALIDATION_TESTER] Generated {len(test_cases)} test cases")
            
            # Initialize results tracking
            self.test_results = []
            self.performance_metrics = {}
            
            # Run tests with concurrency control
            await self._execute_batch_validation_tests(test_cases, max_concurrent)
            
            # Analyze results
            analysis_results = self._analyze_validation_test_results()
            
            # Generate comprehensive report
            test_report = self._generate_validation_test_report(test_start_time, analysis_results)
            
            # Save results if requested
            if save_results:
                await self._save_validation_test_results(test_report)
            
            total_time = (datetime.now() - test_start_time).total_seconds()
            logger.info(f"[CROSS_VALIDATION_TESTER] Multi-stock validation test completed in {total_time:.2f}s")
            
            return test_report
            
        except Exception as e:
            logger.error(f"[CROSS_VALIDATION_TESTER] Multi-stock validation test failed: {e}")
            logger.error(f"[CROSS_VALIDATION_TESTER] Traceback: {traceback.format_exc()}")
            return self._build_error_report(str(e), test_start_time)
    
    def _generate_test_cases(self, test_stocks: List[str], test_periods: List[int]) -> List[Dict[str, Any]]:
        """Generate test cases for all stock/period combinations"""
        test_cases = []
        
        for symbol in test_stocks:
            for period in test_periods:
                test_case = {
                    'symbol': symbol,
                    'period_days': period,
                    'test_id': f"{symbol}_{period}d_validation",
                    'start_date': datetime.now() - timedelta(days=period + 10),  # Add buffer for data
                    'end_date': datetime.now()
                }
                test_cases.append(test_case)
        
        return test_cases
    
    async def _execute_batch_validation_tests(self, test_cases: List[Dict[str, Any]], max_concurrent: int):
        """Execute validation test cases with concurrency control"""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def run_single_validation_test(test_case: Dict[str, Any]):
            async with semaphore:
                return await self._run_single_validation_test(test_case)
        
        # Run all tests concurrently with limit
        tasks = [run_single_validation_test(test_case) for test_case in test_cases]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"[CROSS_VALIDATION_TESTER] Test case {i} failed: {result}")
                self.test_results.append(self._build_failed_test_result(test_cases[i], str(result)))
            else:
                self.test_results.append(result)
    
    async def _run_single_validation_test(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Run cross-validation test for a single stock/period combination"""
        test_start = time.time()
        
        try:
            symbol = test_case['symbol']
            period_days = test_case['period_days']
            test_id = test_case['test_id']
            
            logger.info(f"[CROSS_VALIDATION_TESTER] Testing {test_id}")
            
            # Get real stock data or fall back to synthetic
            data_source = "unknown"
            if self.use_real_data:
                stock_data = await self._get_real_stock_data(symbol, period_days)
                if stock_data is not None and len(stock_data) >= 20:
                    data_source = "real_market_data"
                    logger.info(f"[CROSS_VALIDATION_TESTER] Using REAL market data for {symbol} ({len(stock_data)} days)")
                else:
                    logger.warning(f"[CROSS_VALIDATION_TESTER] Real data unavailable/insufficient for {symbol}, falling back to synthetic")
                    stock_data = self._generate_synthetic_stock_data(symbol, period_days)
                    data_source = "synthetic_fallback"
            else:
                stock_data = self._generate_synthetic_stock_data(symbol, period_days)
                data_source = "synthetic_only"
                logger.info(f"[CROSS_VALIDATION_TESTER] Using synthetic data for {symbol}")
            
            if stock_data is None or len(stock_data) < 20:
                return self._build_failed_test_result(test_case, "Insufficient stock data")
            
            # Prepare save path for this test
            test_output_dir = self.output_dir / test_id
            test_output_dir.mkdir(exist_ok=True)
            save_path = str(test_output_dir / f"{test_id}_charts.html")
            
            # Run complete analysis (pattern detection + validation)
            analysis_start = time.time()
            analysis_results = await self.agent.analyze_and_validate_patterns(
                stock_data=stock_data,
                symbol=symbol,
                include_charts=self.test_config['include_charts'],
                include_llm_analysis=self.test_config['include_llm_analysis'],
                save_path=save_path if self.test_config['save_charts'] else None
            )
            analysis_time = time.time() - analysis_start
            
            # Extract validation results from the complete analysis
            validation_results = analysis_results.get('cross_validation', {})
            pattern_detection_results = analysis_results.get('pattern_detection', {})
            detected_patterns = pattern_detection_results.get('detected_patterns', [])
            
            # Build test result
            test_result = {
                'test_id': test_id,
                'symbol': symbol,
                'period_days': period_days,
                'test_timestamp': datetime.now().isoformat(),
                'test_duration': time.time() - test_start,
                'analysis_duration': analysis_time,
                'data_points': len(stock_data),
                'patterns_detected_count': len(detected_patterns),
                'data_source': data_source,  # Track whether real or synthetic data was used
                
                # Validation results
                'validation_success': validation_results.get('success', False),
                'patterns_validated': validation_results.get('patterns_validated', 0),
                'validation_methods_used': validation_results.get('validation_methods_used', 0),
                'overall_validation_score': validation_results.get('overall_validation_score', 0),
                'overall_validation_confidence': validation_results.get('overall_validation_confidence', 0),
                
                # Validation scores breakdown
                'validation_scores': validation_results.get('validation_scores', {}),
                'final_confidence_assessment': validation_results.get('final_confidence_assessment', {}),
                
                # Method-specific results
                'statistical_validation': validation_results.get('statistical_validation', {}),
                'volume_confirmation': validation_results.get('volume_confirmation', {}),
                'time_series_validation': validation_results.get('time_series_validation', {}),
                'historical_validation': validation_results.get('historical_validation', {}),
                'consistency_analysis': validation_results.get('consistency_analysis', {}),
                'alternative_validation': validation_results.get('alternative_validation', {}),
                
                # Component status
                'charts_generated': validation_results.get('charts_generated', 0),
                'llm_analysis_success': validation_results.get('llm_analysis', {}).get('success', False),
                'components_executed': validation_results.get('analysis_summary', {}).get('components_list', []),
                
                # AI insights (if available)
                'ai_insights_available': bool(validation_results.get('validation_insights')),
                'ai_analysis_quality': validation_results.get('ai_analysis_quality', 'unknown'),
                
                # Executive summary
                'validation_executive_summary': validation_results.get('validation_executive_summary', {}),
                'validation_recommendations': validation_results.get('validation_recommendations', []),
                
                # Full results for detailed analysis
                'full_analysis_results': analysis_results
            }
            
            # Save individual test result if configured
            if self.test_config['save_individual_results']:
                await self._save_individual_validation_result(test_result, test_output_dir)
            
            # Save prompt and response if LLM analysis was performed
            full_analysis = test_result.get('full_analysis_results', {})
            llm_analysis = full_analysis.get('llm_analysis', {})
            if llm_analysis.get('success', False):
                await self._save_prompt_response(test_result, test_output_dir)
            
            logger.info(f"[CROSS_VALIDATION_TESTER] Completed {test_id} - "
                       f"Success: {test_result['validation_success']}, "
                       f"Patterns Validated: {test_result['patterns_validated']}, "
                       f"Methods: {test_result['validation_methods_used']}, "
                       f"Data Source: {test_result['data_source']}, "
                       f"Time: {test_result['test_duration']:.2f}s")
            
            return test_result
            
        except Exception as e:
            logger.error(f"[CROSS_VALIDATION_TESTER] Single validation test failed for {test_case.get('test_id', 'unknown')}: {e}")
            return self._build_failed_test_result(test_case, str(e))
    
    async def _get_real_stock_data(self, symbol: str, period_days: int) -> pd.DataFrame:
        """
        Get real stock data using the orchestrator or Zerodha client.
        Falls back to synthetic data if real data fetch fails.
        """
        try:
            logger.info(f"[CROSS_VALIDATION_TESTER] Fetching real data for {symbol} ({period_days} days)")
            
            # Try using orchestrator first (preferred method)
            if self.orchestrator:
                try:
                    stock_data = await self.orchestrator.retrieve_stock_data(
                        symbol=symbol,
                        exchange="NSE",
                        interval="day",
                        period=period_days
                    )
                    
                    if stock_data is not None and len(stock_data) > 0:
                        logger.info(f"[CROSS_VALIDATION_TESTER] Retrieved {len(stock_data)} days of real data via orchestrator for {symbol}")
                        return stock_data
                    else:
                        logger.warning(f"[CROSS_VALIDATION_TESTER] Orchestrator returned empty data for {symbol}")
                        
                except Exception as e:
                    logger.warning(f"[CROSS_VALIDATION_TESTER] Orchestrator failed for {symbol}: {e}")
            
            # Fallback to direct Zerodha client
            if self.zerodha_client:
                try:
                    # Skip explicit authentication - trust that environment variables are set correctly
                    # The client will use ZERODHA_ACCESS_TOKEN from .env if available
                    logger.info(f"[CROSS_VALIDATION_TESTER] Using Zerodha client with environment credentials for {symbol}")
                    
                    # Check if async method exists
                    if hasattr(self.zerodha_client, 'get_historical_data_async'):
                        stock_data = await self.zerodha_client.get_historical_data_async(
                            symbol=symbol,
                            exchange="NSE",
                            interval="day",
                            period=period_days
                        )
                    else:
                        # Use sync method in executor
                        import asyncio
                        loop = asyncio.get_event_loop()
                        stock_data = await loop.run_in_executor(
                            None,
                            self.zerodha_client.get_historical_data,
                            symbol,
                            "NSE",
                            "day",
                            None,
                            None,
                            period_days
                        )
                    
                    if stock_data is not None and len(stock_data) > 0:
                        # Ensure proper data structure
                        required_columns = ['open', 'high', 'low', 'close', 'volume']
                        
                        # Handle date column properly
                        if 'date' not in stock_data.columns:
                            if stock_data.index.name == 'date' or hasattr(stock_data.index, 'date'):
                                stock_data = stock_data.reset_index()
                            else:
                                stock_data['date'] = stock_data.index
                                stock_data = stock_data.reset_index(drop=True)
                        
                        # Check for missing columns
                        missing_columns = [col for col in required_columns if col not in stock_data.columns]
                        if missing_columns:
                            logger.error(f"[CROSS_VALIDATION_TESTER] Missing required columns for {symbol}: {missing_columns}")
                            return None
                        
                        # Sort by date and set as index for pattern analysis
                        stock_data = stock_data.sort_values('date').reset_index(drop=True)
                        stock_data = stock_data.set_index('date')
                        
                        logger.info(f"[CROSS_VALIDATION_TESTER] Retrieved {len(stock_data)} days of real data via Zerodha for {symbol}")
                        logger.info(f"[CROSS_VALIDATION_TESTER] Date range: {stock_data.index.min()} to {stock_data.index.max()}")
                        logger.info(f"[CROSS_VALIDATION_TESTER] Volume range: {stock_data['volume'].min():,} to {stock_data['volume'].max():,}")
                        
                        return stock_data
                    else:
                        logger.warning(f"[CROSS_VALIDATION_TESTER] Zerodha returned empty data for {symbol}")
                        
                except Exception as e:
                    logger.warning(f"[CROSS_VALIDATION_TESTER] Zerodha client failed for {symbol}: {e}")
            
            # If all real data methods failed, return None (will trigger fallback to synthetic)
            logger.warning(f"[CROSS_VALIDATION_TESTER] All real data sources failed for {symbol}, falling back to synthetic data")
            return None
            
        except Exception as e:
            logger.error(f"[CROSS_VALIDATION_TESTER] Real data fetch failed for {symbol}: {e}")
            return None
    
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
            
            logger.debug(f"[CROSS_VALIDATION_TESTER] Generated {len(df)} days of synthetic data for {symbol}")
            
            return df
            
        except Exception as e:
            logger.error(f"[CROSS_VALIDATION_TESTER] Synthetic data generation failed for {symbol}: {e}")
            return None
    
    def _generate_synthetic_patterns(self, symbol: str, period_days: int) -> List[Dict[str, Any]]:
        """Generate synthetic patterns for testing validation"""
        try:
            np.random.seed((hash(symbol) + period_days) % 2**32)
            
            # Pattern types to generate
            pattern_types = [
                'ascending_triangle', 'descending_triangle', 'symmetrical_triangle',
                'bullish_flag', 'bearish_flag', 'rectangle',
                'head_and_shoulders', 'double_top', 'double_bottom'
            ]
            
            # Generate 1-4 synthetic patterns
            num_patterns = np.random.randint(1, 5)
            patterns = []
            
            for i in range(num_patterns):
                pattern_name = np.random.choice(pattern_types)
                
                # Determine pattern type
                if pattern_name in ['ascending_triangle', 'bullish_flag']:
                    pattern_type = 'continuation'
                elif pattern_name in ['head_and_shoulders', 'double_top', 'double_bottom']:
                    pattern_type = 'reversal'
                else:
                    pattern_type = np.random.choice(['continuation', 'reversal'])
                
                # Generate pattern characteristics
                completion = np.random.uniform(60, 95)
                reliability = np.random.choice(['high', 'medium', 'low'], p=[0.3, 0.5, 0.2])
                quality = np.random.choice(['strong', 'medium', 'weak'], p=[0.4, 0.4, 0.2])
                
                # Generate synthetic pattern data
                pattern_data = self._generate_pattern_specific_data(pattern_name)
                
                pattern = {
                    'pattern_name': pattern_name,
                    'pattern_type': pattern_type,
                    'completion_percentage': completion,
                    'reliability': reliability,
                    'pattern_quality': quality,
                    'start_date': (datetime.now() - timedelta(days=np.random.randint(5, period_days))).isoformat(),
                    'pattern_data': pattern_data
                }
                
                patterns.append(pattern)
            
            logger.debug(f"[CROSS_VALIDATION_TESTER] Generated {len(patterns)} synthetic patterns for {symbol}")
            return patterns
            
        except Exception as e:
            logger.error(f"[CROSS_VALIDATION_TESTER] Synthetic pattern generation failed for {symbol}: {e}")
            return []
    
    def _generate_pattern_specific_data(self, pattern_name: str) -> Dict[str, Any]:
        """Generate pattern-specific data based on pattern type"""
        base_price = np.random.uniform(100, 300)
        
        if 'triangle' in pattern_name.lower():
            return {
                'high_trend': [base_price * (1 + i * 0.001) for i in range(10)],
                'low_trend': [base_price * (0.95 + i * 0.002) for i in range(10)],
                'apex_price': base_price * 1.05
            }
        elif 'flag' in pattern_name.lower() or 'pennant' in pattern_name.lower():
            return {
                'flagpole_move': np.random.uniform(0.05, 0.15),
                'consolidation_range': base_price * 0.03,
                'volume_confirmation': np.random.choice(['present', 'absent'])
            }
        elif 'channel' in pattern_name.lower() or 'rectangle' in pattern_name.lower():
            return {
                'resistance_level': base_price * 1.05,
                'support_level': base_price * 0.95,
                'channel_width': base_price * 0.1,
                'resistance_tests': np.random.randint(2, 5),
                'support_tests': np.random.randint(2, 5)
            }
        elif 'head' in pattern_name.lower():
            return {
                'head_price': base_price * 1.1,
                'left_shoulder_price': base_price * 1.05,
                'right_shoulder_price': base_price * 1.06,
                'neckline_level': base_price * 0.98,
                'shoulder_symmetry': np.random.uniform(0.8, 0.98)
            }
        elif 'double' in pattern_name.lower():
            if 'top' in pattern_name.lower():
                return {
                    'first_peak': base_price * 1.08,
                    'second_peak': base_price * 1.07,
                    'valley_low': base_price * 0.95,
                    'peak_similarity': np.random.uniform(0.95, 0.99)
                }
            else:
                return {
                    'first_bottom': base_price * 0.92,
                    'second_bottom': base_price * 0.93,
                    'peak_high': base_price * 1.05,
                    'bottom_similarity': np.random.uniform(0.95, 0.99)
                }
        else:
            return {'base_price': base_price}
    
    def _generate_synthetic_pattern_summary(self, detected_patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate synthetic pattern summary"""
        if not detected_patterns:
            return {
                'total_patterns': 0,
                'dominant_pattern': 'none',
                'pattern_confluence': 'none',
                'overall_bias': 'neutral'
            }
        
        # Count pattern types
        reversal_patterns = len([p for p in detected_patterns if p['pattern_type'] == 'reversal'])
        continuation_patterns = len([p for p in detected_patterns if p['pattern_type'] == 'continuation'])
        
        # Determine dominant pattern
        dominant_pattern = max(detected_patterns, key=lambda x: x['completion_percentage'])['pattern_name']
        
        # Assess confluence
        if len(detected_patterns) >= 3:
            confluence = 'high'
        elif len(detected_patterns) == 2:
            confluence = 'medium'
        else:
            confluence = 'low'
        
        # Determine bias
        bullish_patterns = [p for p in detected_patterns if 'bullish' in p['pattern_name'] or 'ascending' in p['pattern_name']]
        bearish_patterns = [p for p in detected_patterns if 'bearish' in p['pattern_name'] or 'descending' in p['pattern_name']]
        
        if len(bullish_patterns) > len(bearish_patterns):
            bias = 'bullish'
        elif len(bearish_patterns) > len(bullish_patterns):
            bias = 'bearish'
        else:
            bias = 'neutral'
        
        return {
            'total_patterns': len(detected_patterns),
            'dominant_pattern': dominant_pattern,
            'pattern_confluence': confluence,
            'overall_bias': bias,
            'reversal_patterns': reversal_patterns,
            'continuation_patterns': continuation_patterns
        }
    
    async def _save_individual_validation_result(self, test_result: Dict[str, Any], output_dir: Path):
        """Save individual validation test result to file"""
        try:
            result_file = output_dir / "validation_test_result.json"
            
            # Create a clean version without the full analysis results for JSON serialization
            clean_result = {k: v for k, v in test_result.items() if k != 'full_analysis_results'}
            
            with open(result_file, 'w') as f:
                json.dump(clean_result, f, indent=2, default=str)
            
            # Save detailed analysis separately
            detailed_file = output_dir / "detailed_validation.json"
            with open(detailed_file, 'w') as f:
                json.dump(test_result['full_analysis_results'], f, indent=2, default=str)
            
            logger.debug(f"[CROSS_VALIDATION_TESTER] Saved individual validation result to {output_dir}")
            
        except Exception as e:
            logger.error(f"[CROSS_VALIDATION_TESTER] Failed to save individual validation result: {e}")
    
    async def _save_prompt_response(self, test_result: Dict[str, Any], test_output_dir: Path):
        """Save LLM prompt and response for debugging and analysis."""
        try:
            # Create prompts_responses directory
            prompts_dir = test_output_dir / "prompts_responses"
            prompts_dir.mkdir(exist_ok=True)
            
            symbol = test_result['symbol']
            test_id = test_result['test_id']
            
            # Extract analysis results
            full_analysis_results = test_result.get('full_analysis_results', {})
            
            if full_analysis_results:
                # Get the cross-validation results and detected patterns
                cross_validation_results = full_analysis_results.get('cross_validation', {})
                pattern_detection_results = full_analysis_results.get('pattern_detection', {})
                detected_patterns = pattern_detection_results.get('detected_patterns', [])
                llm_analysis = full_analysis_results.get('llm_analysis', {})
                
                # Extract market context from analysis results to match actual execution
                market_context_for_prompt = None
                try:
                    # First try to get it from cross-validation results (where it's stored during execution)
                    market_context_for_prompt = cross_validation_results.get('market_context')
                    # If not there, try the full analysis results
                    if not market_context_for_prompt:
                        market_context_for_prompt = full_analysis_results.get('market_context')
                    
                    # Debug: Log what we found
                    if market_context_for_prompt:
                        ctx_source = market_context_for_prompt.get('source', 'unknown') if isinstance(market_context_for_prompt, dict) else 'non-dict'
                        logger.info(f"[CROSS_VALIDATION_TESTER] Extracted market context for prompt: source={ctx_source}, type={type(market_context_for_prompt).__name__}")
                    else:
                        logger.warning(f"[CROSS_VALIDATION_TESTER] No market context found in results for {test_id}")
                        # Debug: Check what keys are available
                        cv_keys = list(cross_validation_results.keys()) if cross_validation_results else []
                        full_keys = list(full_analysis_results.keys()) if full_analysis_results else []
                        logger.debug(f"[CROSS_VALIDATION_TESTER] Available keys - CV: {cv_keys}, Full: {full_keys}")
                        
                except Exception as e:
                    logger.warning(f"[CROSS_VALIDATION_TESTER] Failed to extract market context for prompt: {e}")
                
                # Build the prompt using enhanced LLM agent with correct market context
                from agents.patterns.cross_validation_agent.llm_agent import CrossValidationLLMAgent
                llm_agent = CrossValidationLLMAgent()
                prompt = llm_agent._build_validation_analysis_prompt(
                    cross_validation_results, detected_patterns, symbol, market_context_for_prompt
                )
                
                # Save prompt
                prompt_file = prompts_dir / f"{test_id}_prompt.txt"
                with open(prompt_file, 'w', encoding='utf-8') as f:
                    f.write(prompt)
                
                # Save LLM response if available
                if llm_analysis.get('success', False):
                    response_content = llm_analysis.get('analysis', '')
                    if not response_content:
                        response_content = llm_analysis.get('full_analysis', '')
                    
                    if response_content:
                        response_file = prompts_dir / f"{test_id}_response.txt"
                        with open(response_file, 'w', encoding='utf-8') as f:
                            f.write(str(response_content))
                    
                    # Save structured LLM analysis
                    structured_file = prompts_dir / f"{test_id}_llm_analysis.json"
                    with open(structured_file, 'w', encoding='utf-8') as f:
                        json.dump(llm_analysis, f, indent=2, default=str)
                
                # Save validation insights if available from the integrated results
                validation_insights = full_analysis_results.get('validation_insights')
                if validation_insights:
                    insights_file = prompts_dir / f"{test_id}_insights.json"
                    with open(insights_file, 'w', encoding='utf-8') as f:
                        json.dump(validation_insights, f, indent=2, default=str)
                
                # Save token usage information if available
                token_usage = llm_analysis.get('token_usage')
                if token_usage:
                    usage_file = prompts_dir / f"{test_id}_token_usage.json"
                    with open(usage_file, 'w', encoding='utf-8') as f:
                        json.dump(token_usage, f, indent=2, default=str)
                        
                logger.debug(f"[CROSS_VALIDATION_TESTER] Saved prompts and responses to {prompts_dir}")
                        
        except Exception as e:
            logger.error(f"[CROSS_VALIDATION_TESTER] Failed to save prompt/response for {test_result.get('test_id', 'unknown')}: {e}")
    
    def _analyze_validation_test_results(self) -> Dict[str, Any]:
        """Analyze aggregated validation test results"""
        try:
            if not self.test_results:
                return {'error': 'No test results to analyze'}
            
            # Basic statistics
            total_tests = len(self.test_results)
            successful_tests = len([r for r in self.test_results if r.get('validation_success', False)])
            failed_tests = total_tests - successful_tests
            
            success_rate = (successful_tests / total_tests) * 100 if total_tests > 0 else 0
            
            # Performance metrics
            test_durations = [r.get('test_duration', 0) for r in self.test_results if r.get('test_duration')]
            validation_durations = [r.get('validation_duration', 0) for r in self.test_results if r.get('validation_duration')]
            
            # Validation statistics
            patterns_validated = [r.get('patterns_validated', 0) for r in self.test_results]
            validation_scores = [r.get('overall_validation_score', 0) for r in self.test_results if r.get('validation_success')]
            validation_confidences = [r.get('overall_validation_confidence', 0) for r in self.test_results if r.get('validation_success')]
            
            # Method usage statistics
            methods_used = [r.get('validation_methods_used', 0) for r in self.test_results]
            
            # Component success rates
            llm_success_count = len([r for r in self.test_results if r.get('llm_analysis_success', False)])
            charts_generated_count = len([r for r in self.test_results if r.get('charts_generated', 0) > 0])
            
            # Data source statistics
            real_data_count = len([r for r in self.test_results if r.get('data_source') == 'real_market_data'])
            synthetic_fallback_count = len([r for r in self.test_results if r.get('data_source') == 'synthetic_fallback'])
            synthetic_only_count = len([r for r in self.test_results if r.get('data_source') == 'synthetic_only'])
            
            # Validation method performance
            method_performance = {}
            for result in self.test_results:
                validation_scores_detail = result.get('validation_scores', {}).get('method_scores', {})
                for method, score in validation_scores_detail.items():
                    if method not in method_performance:
                        method_performance[method] = []
                    method_performance[method].append(score)
            
            # Calculate average performance for each method
            avg_method_performance = {}
            for method, scores in method_performance.items():
                avg_method_performance[method] = np.mean(scores) if scores else 0
            
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
                    'avg_validation_duration': np.mean(validation_durations) if validation_durations else 0,
                    'total_test_time': np.sum(test_durations) if test_durations else 0
                },
                'validation_statistics': {
                    'total_patterns_validated': np.sum(patterns_validated),
                    'avg_patterns_per_test': np.mean(patterns_validated) if patterns_validated else 0,
                    'avg_validation_score': np.mean(validation_scores) if validation_scores else 0,
                    'avg_validation_confidence': np.mean(validation_confidences) if validation_confidences else 0,
                    'avg_methods_used': np.mean(methods_used) if methods_used else 0
                },
                'confidence_analysis': {
                    'high_confidence_tests': len([c for c in validation_confidences if c >= 0.8]),
                    'medium_confidence_tests': len([c for c in validation_confidences if 0.6 <= c < 0.8]),
                    'low_confidence_tests': len([c for c in validation_confidences if c < 0.6])
                },
                'component_success_rates': {
                    'llm_analysis_success_rate': (llm_success_count / total_tests * 100) if total_tests > 0 else 0,
                    'charts_generation_rate': (charts_generated_count / total_tests * 100) if total_tests > 0 else 0
                },
                'data_source_statistics': {
                    'real_market_data': real_data_count,
                    'synthetic_fallback': synthetic_fallback_count,
                    'synthetic_only': synthetic_only_count,
                    'real_data_success_rate': (real_data_count / total_tests * 100) if total_tests > 0 else 0
                },
                'validation_method_performance': avg_method_performance,
                'best_performing_methods': sorted(avg_method_performance.items(), key=lambda x: x[1], reverse=True)[:5]
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"[CROSS_VALIDATION_TESTER] Result analysis failed: {e}")
            return {'error': str(e)}
    
    def _generate_validation_test_report(self, test_start_time: datetime, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive validation test report"""
        
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
            'recommendations': self._generate_validation_recommendations(analysis_results),
            'summary': self._generate_validation_test_summary(analysis_results, total_duration)
        }
    
    def _generate_validation_recommendations(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on validation test results"""
        recommendations = []
        
        try:
            success_rate = analysis_results.get('test_summary', {}).get('success_rate_percent', 0)
            avg_confidence = analysis_results.get('validation_statistics', {}).get('avg_validation_confidence', 0)
            avg_score = analysis_results.get('validation_statistics', {}).get('avg_validation_score', 0)
            
            if success_rate < 80:
                recommendations.append(f"Success rate ({success_rate:.1f}%) below target (80%). Investigate validation failure causes.")
            
            if avg_confidence < 0.6:
                recommendations.append(f"Average validation confidence ({avg_confidence:.1%}) is low. Review validation method reliability.")
            
            if avg_score < 0.6:
                recommendations.append(f"Average validation score ({avg_score:.2f}) indicates room for improvement in validation methods.")
            
            # Method-specific recommendations
            method_performance = analysis_results.get('validation_method_performance', {})
            if method_performance:
                best_methods = analysis_results.get('best_performing_methods', [])
                if best_methods:
                    best_method_name, best_score = best_methods[0]
                    recommendations.append(f"Best performing validation method: {best_method_name} ({best_score:.2f})")
                
                weak_methods = [(name, score) for name, score in method_performance.items() if score < 0.5]
                if weak_methods:
                    weak_method_names = [name for name, _ in weak_methods]
                    recommendations.append(f"Validation methods needing improvement: {', '.join(weak_method_names)}")
            
            llm_success_rate = analysis_results.get('component_success_rates', {}).get('llm_analysis_success_rate', 0)
            if llm_success_rate < 90:
                recommendations.append(f"LLM analysis success rate ({llm_success_rate:.1f}%) could be improved.")
            
            if not recommendations:
                recommendations.append("All validation metrics are performing well. Consider expanding test coverage or more challenging scenarios.")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"[CROSS_VALIDATION_TESTER] Recommendation generation failed: {e}")
            return ["Error generating recommendations"]
    
    def _generate_validation_test_summary(self, analysis_results: Dict[str, Any], total_duration: float) -> str:
        """Generate human-readable validation test summary"""
        
        try:
            summary = analysis_results.get('test_summary', {})
            total_tests = summary.get('total_tests', 0)
            success_rate = summary.get('success_rate_percent', 0)
            
            validation_stats = analysis_results.get('validation_statistics', {})
            total_patterns = validation_stats.get('total_patterns_validated', 0)
            avg_confidence = validation_stats.get('avg_validation_confidence', 0)
            avg_score = validation_stats.get('avg_validation_score', 0)
            
            return f"""
Cross-Validation Agent Test Summary:
- Executed {total_tests} validation tests in {total_duration:.2f} seconds
- Success rate: {success_rate:.1f}%
- Total patterns validated: {total_patterns}
- Average validation confidence: {avg_confidence:.1%}
- Average validation score: {avg_score:.2f}
- Test framework performance: {'PASS' if success_rate > 70 else 'FAIL'}
"""
            
        except Exception as e:
            logger.error(f"[CROSS_VALIDATION_TESTER] Summary generation failed: {e}")
            return f"Summary generation failed: {str(e)}"
    
    async def _save_validation_test_results(self, test_report: Dict[str, Any]):
        """Save comprehensive validation test results to files"""
        try:
            # Save main test report
            report_file = self.output_dir / f"validation_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_file, 'w') as f:
                json.dump(test_report, f, indent=2, default=str)
            
            # Save summary report
            summary_file = self.output_dir / "validation_test_summary.txt"
            with open(summary_file, 'w') as f:
                f.write(test_report.get('summary', ''))
                f.write('\n\n')
                f.write('\n'.join(test_report.get('recommendations', [])))
            
            logger.info(f"[CROSS_VALIDATION_TESTER] Test results saved to {self.output_dir}")
            
        except Exception as e:
            logger.error(f"[CROSS_VALIDATION_TESTER] Failed to save test results: {e}")
    
    def _build_failed_test_result(self, test_case: Dict[str, Any], error_message: str) -> Dict[str, Any]:
        """Build result structure for failed tests"""
        return {
            'test_id': test_case.get('test_id', 'unknown'),
            'symbol': test_case.get('symbol', 'unknown'),
            'period_days': test_case.get('period_days', 0),
            'test_timestamp': datetime.now().isoformat(),
            'test_duration': 0.0,
            'validation_success': False,
            'failure_reason': error_message,
            'patterns_validated': 0,
            'validation_methods_used': 0,
            'overall_validation_score': 0.0,
            'overall_validation_confidence': 0.0
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
            'recommendations': ['Fix the critical error before running validation tests again'],
            'summary': f"Validation test run failed due to error: {error_message}"
        }

# Main execution function for standalone testing
async def main():
    """Main function for running cross-validation tests"""
    
    # Test configuration - using real market stocks for comprehensive testing
    test_stocks = ["RELIANCE"]
    test_periods = [365]  # Different timeframes for pattern analysis
    max_concurrent = 3           # Increase concurrency for faster testing
    
    # Initialize tester
    tester = CrossValidationMultiStockTester("test_results")
    
    # Show data source configuration
    print(f"Data Source Configuration:")
    print(f"  Real Data Available: {tester.use_real_data}")
    print(f"  Orchestrator: {'Available' if tester.orchestrator else 'Not Available'}")
    print(f"  Zerodha Client: {'Available' if tester.zerodha_client else 'Not Available'}")
    print(f"  Will use: {'Real market data (with synthetic fallback)' if tester.use_real_data else 'Synthetic data only'}")
    
    # Run tests
    print("Starting Cross-Validation Agent Multi-Stock Test...")
    test_results = await tester.run_multi_stock_validation_test(
        test_stocks=test_stocks,
        test_periods=test_periods,
        max_concurrent=max_concurrent,
        save_results=True
    )
    
    # Display results
    print("\n" + "="*60)
    print("VALIDATION TEST RESULTS SUMMARY")
    print("="*60)
    print(test_results.get('summary', 'No summary available'))
    
    print("\nRECOMMENDATIONS:")
    for rec in test_results.get('recommendations', []):
        print(f"- {rec}")
    
    print(f"\nDetailed results saved to: {tester.output_dir}")
    
    return test_results

# Allow running this script directly
if __name__ == "__main__":
    # Run the test
    results = asyncio.run(main())