#!/usr/bin/env python3
"""
Multi-Stock Volume Confirmation Agent Testing Framework

Tests the volume_confirmation_analysis prompt across multiple stocks from different sectors
to validate consistency and quality of volume confirmation analysis.

Usage: python multi_stock_test.py
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add necessary paths
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
# Navigate to the main project directory and add backend
project_root = os.path.join(current_dir, '..', '..', '..', '..')
backend_dir = os.path.join(project_root, 'backend')
sys.path.append(src_dir)
sys.path.append(project_root)  # Add project root to path

# Import Volume Confirmation Agent components
try:
    from backend.agents.volume.volume_confirmation.processor import VolumeConfirmationProcessor
    from backend.agents.volume.volume_confirmation.charts import VolumeConfirmationChartGenerator
    from backend.agents.volume.volume_confirmation.context import VolumeConfirmationContextFormatter
    print("✅ Volume Confirmation Agent components loaded")
except ImportError as e:
    print(f"❌ Import Error for Volume Confirmation components: {e}")
    sys.exit(1)

# Import backend components
try:
    from backend.gemini.gemini_client import GeminiClient
    from backend.gemini.prompt_manager import PromptManager
    from backend.zerodha.client import ZerodhaDataClient
    HAS_BACKEND = True
    print("✅ Backend components loaded")
except ImportError as e:
    print(f"❌ Import Error for backend components: {e}")
    print("Make sure you're running this from the correct directory")
    HAS_BACKEND = False
    sys.exit(1)

class VolumeStockTestConfig:
    """Configuration for individual stock tests"""
    def __init__(self, symbol: str, name: str, sector: str, volume_profile: str):
        self.symbol = symbol
        self.name = name
        self.sector = sector
        self.volume_profile = volume_profile

class VolumeConfirmationCalculator:
    """Calculate volume confirmation metrics from real stock data"""
    
    @staticmethod
    def calculate_volume_averages(data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate volume averages and ratios"""
        volume = data['volume']
        
        # Volume moving averages
        volume_10d = volume.rolling(window=10).mean()
        volume_20d = volume.rolling(window=20).mean()
        volume_50d = volume.rolling(window=50).mean()
        
        # Current volume metrics
        current_volume = volume.iloc[-1] if not volume.empty else 0
        
        def get_latest_valid(series):
            valid_values = series.dropna()
            return float(valid_values.iloc[-1]) if not valid_values.empty else None
        
        return {
            'current_volume': int(current_volume),
            'volume_10d_avg': int(get_latest_valid(volume_10d)) if get_latest_valid(volume_10d) else 0,
            'volume_20d_avg': int(get_latest_valid(volume_20d)) if get_latest_valid(volume_20d) else 0,
            'volume_50d_avg': int(get_latest_valid(volume_50d)) if get_latest_valid(volume_50d) else 0,
            'volume_vs_10d': current_volume / get_latest_valid(volume_10d) if get_latest_valid(volume_10d) else 1.0,
            'volume_vs_20d': current_volume / get_latest_valid(volume_20d) if get_latest_valid(volume_20d) else 1.0,
            'volume_vs_50d': current_volume / get_latest_valid(volume_50d) if get_latest_valid(volume_50d) else 1.0
        }
    
    @staticmethod
    def calculate_price_volume_correlation(data: pd.DataFrame, window: int = 30) -> Dict[str, Any]:
        """Calculate price-volume correlation"""
        if len(data) < 20:
            return {'error': 'Insufficient data'}
        
        # Calculate price changes and volume changes
        price_changes = data['close'].pct_change().dropna()
        volume_changes = data['volume'].pct_change().dropna()
        
        # Align the series
        common_index = price_changes.index.intersection(volume_changes.index)
        price_changes = price_changes.loc[common_index]
        volume_changes = volume_changes.loc[common_index]
        
        if len(price_changes) < 10:
            return {'error': 'Insufficient aligned data'}
        
        # Calculate correlation
        correlation = price_changes.corr(volume_changes)
        correlation = 0 if pd.isna(correlation) else correlation
        
        return {
            'correlation_coefficient': round(correlation, 3),
            'correlation_strength': 'strong' if abs(correlation) > 0.5 else 'medium' if abs(correlation) > 0.3 else 'weak',
            'correlation_direction': 'positive' if correlation > 0.1 else 'negative' if correlation < -0.1 else 'neutral'
        }

class MultiStockVolumeConfirmationTester:
    """Test volume confirmation prompt across multiple stocks"""
    
    def __init__(self):
        # Initialize Volume Confirmation components
        self.processor = VolumeConfirmationProcessor()
        self.chart_generator = VolumeConfirmationChartGenerator()
        self.context_formatter = VolumeConfirmationContextFormatter()
        self.calculator = VolumeConfirmationCalculator()
        
        # Initialize Zerodha client
        if HAS_BACKEND:
            try:
                self.zerodha_client = ZerodhaDataClient()
                print("✅ Zerodha client initialized")
            except Exception as e:
                print(f"❌ Cannot initialize Zerodha client: {e}")
                sys.exit(1)
        
        # Initialize other components
        if HAS_BACKEND:
            self.prompt_manager = PromptManager()
            
            # Initialize Gemini client if API key is available
            self.gemini_client = None
            try:
                api_key = os.environ.get("GEMINI_API_KEY")
                if api_key:
                    self.gemini_client = GeminiClient(api_key=api_key)
                    print("✅ Gemini API client initialized")
                else:
                    print("⚠️  GEMINI_API_KEY not found - will show prompts only")
            except Exception as e:
                print(f"⚠️  Could not initialize Gemini client: {e}")
        
        # Define test stocks from different sectors with different volume profiles
        self.test_stocks = [
            VolumeStockTestConfig("RELIANCE", "Reliance Industries", "Energy/Petrochemicals", "high_liquidity"),
            VolumeStockTestConfig("TCS", "Tata Consultancy Services", "IT Services", "consistent_volume"),
            VolumeStockTestConfig("HDFCBANK", "HDFC Bank", "Banking", "high_liquidity"),
            VolumeStockTestConfig("ICICIBANK", "ICICI Bank", "Banking", "moderate_volatility"),
            VolumeStockTestConfig("ITC", "ITC Limited", "FMCG/Tobacco", "defensive_volume")
            # Note: Add more stocks here later for comprehensive testing:
            # VolumeStockTestConfig("INFY", "Infosys", "IT Services", "consistent_volume"),
            # VolumeStockTestConfig("BHARTIARTL", "Bharti Airtel", "Telecommunications", "cyclical_volume"),
            # VolumeStockTestConfig("HINDUNILVR", "Hindustan Unilever", "FMCG", "defensive_volume"),
            # VolumeStockTestConfig("MARUTI", "Maruti Suzuki", "Automotive", "cyclical_volume"),
            # VolumeStockTestConfig("BAJFINANCE", "Bajaj Finance", "NBFC", "growth_volume")
        ]
        
        self.results = []
    
    async def run_multi_stock_tests(self):
        """Run tests across all configured stocks"""
        print(f"🚀 Starting Multi-Stock Volume Confirmation Testing")
        print(f"Testing {len(self.test_stocks)} stocks with 90 days of data")
        print("=" * 80)
        
        # Authenticate with Zerodha first
        print("🔗 Authenticating with Zerodha...")
        if not self.zerodha_client.authenticate():
            print("❌ Zerodha authentication failed")
            return False
        
        print("✅ Zerodha authentication successful")
        
        # Create results directory
        results_dir = "volume_confirmation_test_results"
        os.makedirs(results_dir, exist_ok=True)
        
        # Create async tasks for all stocks to run them concurrently
        async def test_single_stock(stock_config, stock_index):
            """Test a single stock asynchronously"""
            print(f"\n📊 Testing Stock {stock_index}/{len(self.test_stocks)}: {stock_config.symbol}")
            print(f"   Company: {stock_config.name}")
            print(f"   Sector: {stock_config.sector}")
            print(f"   Volume Profile: {stock_config.volume_profile}")
            print("-" * 60)
            
            try:
                # Get stock data
                print(f"📈 Fetching 90 days of data for {stock_config.symbol}...")
                
                # Use async version of get_historical_data if available
                if hasattr(self.zerodha_client, 'get_historical_data_async'):
                    stock_data = await self.zerodha_client.get_historical_data_async(
                        symbol=stock_config.symbol,
                        exchange="NSE",
                        interval="day",
                        period=90
                    )
                else:
                    # Fallback to sync version in executor
                    import asyncio
                    loop = asyncio.get_event_loop()
                    stock_data = await loop.run_in_executor(
                        None,
                        self.zerodha_client.get_historical_data,
                        stock_config.symbol,
                        "NSE",
                        "day",
                        None,
                        None,
                        90
                    )
                
                if stock_data is None or stock_data.empty:
                    print(f"❌ No data available for {stock_config.symbol}")
                    return {
                        'stock_config': stock_config,
                        'success': False,
                        'error': 'No data available',
                        'execution_time': 0,
                        'quality_score': 0,
                        'data_quality': 'no_data'
                    }
                
                # Ensure we have the right columns
                required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
                
                # If date is the index, reset it
                if 'date' not in stock_data.columns and stock_data.index.name == 'date':
                    stock_data = stock_data.reset_index()
                elif 'date' not in stock_data.columns:
                    stock_data['date'] = stock_data.index
                    stock_data = stock_data.reset_index(drop=True)
                
                # Check for missing columns
                missing_columns = [col for col in required_columns if col not in stock_data.columns]
                if missing_columns:
                    print(f"❌ Missing required columns for {stock_config.symbol}: {missing_columns}")
                    return {
                        'stock_config': stock_config,
                        'success': False,
                        'error': f'Missing columns: {missing_columns}',
                        'execution_time': 0,
                        'quality_score': 0,
                        'data_quality': 'missing_columns'
                    }
                
                # Sort by date to ensure proper order
                stock_data = stock_data.sort_values('date').reset_index(drop=True)
                
                print(f"✅ Retrieved {len(stock_data)} days of data")
                print(f"   Date range: {stock_data['date'].min()} to {stock_data['date'].max()}")
                print(f"   Price range: ₹{stock_data['close'].min():.2f} to ₹{stock_data['close'].max():.2f}")
                print(f"   Volume range: {stock_data['volume'].min():,} to {stock_data['volume'].max():,}")
                
                # Set date as index for processing
                stock_data_indexed = stock_data.set_index('date')
                
                # Process volume confirmation analysis
                print("🔬 Processing volume confirmation analysis...")
                analysis_data = self.processor.process_volume_confirmation_data(stock_data_indexed)
                
                # Generate chart with timestamp to avoid conflicts
                print("📊 Generating volume confirmation chart...")
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]  # Include microseconds for uniqueness
                chart_filename = f"volume_chart_{stock_config.symbol}_{timestamp}.png"
                chart_path = os.path.join(results_dir, chart_filename)
                
                chart_bytes = self.chart_generator.generate_volume_confirmation_chart(
                    stock_data_indexed, analysis_data, stock_config.symbol, 
                    save_path=chart_path  # Save chart to results directory
                )
                
                # Test the prompt
                result = await self._test_stock_prompt(stock_config, analysis_data, results_dir)
                
                # Add chart information to result
                if chart_bytes:
                    result['chart_saved'] = True
                    result['chart_filename'] = chart_filename
                    result['chart_path'] = chart_path
                    result['chart_size_bytes'] = len(chart_bytes)
                else:
                    result['chart_saved'] = False
                
                print(f"✅ Test completed for {stock_config.symbol}")
                print(f"   Success: {result['success']}")
                print(f"   Quality Score: {result['quality_score']:.1f}/100")
                print(f"   Response Time: {result['execution_time']:.1f}s")
                if chart_bytes:
                    print(f"   Chart saved: {chart_filename}")
                
                return result
                
            except Exception as e:
                print(f"❌ Error testing {stock_config.symbol}: {e}")
                import traceback
                traceback.print_exc()
                
                # Return error result
                return {
                    'stock_config': stock_config,
                    'success': False,
                    'error': str(e),
                    'execution_time': 0,
                    'quality_score': 0,
                    'data_quality': 'failed'
                }
        
        # Run all stock tests concurrently with a semaphore to limit concurrency
        semaphore = asyncio.Semaphore(3)  # Limit to 3 concurrent tests
        
        async def test_with_semaphore(stock_config, index):
            async with semaphore:
                return await test_single_stock(stock_config, index)
        
        # Create tasks for all stocks
        tasks = [
            test_with_semaphore(stock_config, i + 1)
            for i, stock_config in enumerate(self.test_stocks)
        ]
        
        # Wait for all tasks to complete
        print(f"\n🔄 Running {len(tasks)} stock tests concurrently (max 3 at a time)...")
        results = await asyncio.gather(*tasks)
        self.results.extend(results)
        
        # Generate comprehensive report
        self._generate_multi_stock_report(results_dir)
        
        print(f"\n✅ Multi-stock volume confirmation testing completed!")
        print(f"📁 Results saved to: {results_dir}/")
        
        # Count charts saved
        charts_generated = sum(1 for r in results if r.get('chart_saved', False))
        if charts_generated > 0:
            print(f"📊 {charts_generated} volume confirmation charts saved to results directory")
        
        return True
    
    async def _test_stock_prompt(self, stock_config: VolumeStockTestConfig, analysis_data: Dict[str, Any], results_dir: str) -> Dict[str, Any]:
        """Test the volume confirmation prompt for a single stock"""
        start_time = time.time()
        
        # Helper function for safe dictionary access
        def safe_get(data, *keys):
            """Safely get nested dictionary values"""
            try:
                result = data
                for key in keys:
                    result = result[key]
                return result
            except (KeyError, TypeError, AttributeError):
                return None
        
        try:
            # Format context using the Volume Confirmation context formatter
            formatted_context = self.context_formatter.format_context(
                analysis_data, stock_config.symbol, stock_config.name, stock_config.sector
            )
            
            # Format the final prompt
            prompt = self.prompt_manager.format_prompt(
                "volume_confirmation_analysis",
                context=formatted_context
            )
            prompt += self.prompt_manager.SOLVING_LINE
            
            # Save prompt details with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            prompt_file = os.path.join(results_dir, f"volume_prompt_{stock_config.symbol}_{timestamp}.txt")
            with open(prompt_file, 'w') as f:
                f.write("VOLUME CONFIRMATION PROMPT ANALYSIS FOR LLM\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"Stock Symbol: {stock_config.symbol}\n")
                f.write(f"Company: {stock_config.name}\n")
                f.write(f"Sector: {stock_config.sector}\n")
                f.write(f"Volume Profile: {stock_config.volume_profile}\n")
                f.write(f"Generated: {datetime.now().isoformat()}\n")
                f.write(f"Prompt Length: {len(prompt)} characters\n")
                f.write(f"Context Length: {len(formatted_context)} characters\n\n")
                f.write("KEY VOLUME CONFIRMATION METRICS SUMMARY:\n")
                f.write("-" * 40 + "\n")
                
                # Extract key metrics for prompt analysis
                overall = safe_get(analysis_data, 'overall_assessment') or {}
                correlation = safe_get(analysis_data, 'price_volume_correlation') or {}
                volume_avg = safe_get(analysis_data, 'volume_averages') or {}
                trend_support = safe_get(analysis_data, 'trend_support') or {}
                
                key_metrics = {
                    "confirmation_status": overall.get('confirmation_status', 'unknown'),
                    "confirmation_strength": overall.get('confirmation_strength', 'unknown'),
                    "confidence_score": overall.get('confidence_score', 0),
                    "correlation_coefficient": correlation.get('correlation_coefficient', 0),
                    "correlation_strength": correlation.get('correlation_strength', 'unknown'),
                    "current_volume": volume_avg.get('current_volume', 0),
                    "volume_vs_20d": volume_avg.get('volume_vs_20d', 1.0),
                    "uptrend_volume_support": trend_support.get('uptrend_volume_support', 'unknown'),
                    "downtrend_volume_support": trend_support.get('downtrend_volume_support', 'unknown'),
                    "consolidation_pattern": trend_support.get('consolidation_volume_pattern', 'unknown'),
                    "data_quality": analysis_data.get('data_quality', 'unknown')
                }
                f.write(json.dumps(key_metrics, indent=2, default=str))
                f.write("\n\n")
                f.write("FINAL PROMPT SENT TO LLM:\n")
                f.write("-" * 40 + "\n")
                f.write(prompt)
            
            # Make API call if available
            llm_response = ""
            if self.gemini_client:
                try:
                    print(f"🚀 Making API call for {stock_config.symbol}...")
                    response, code_results, execution_results = await self.gemini_client.core.call_llm_with_code_execution(prompt)
                    llm_response = response
                    
                    # Save response with timestamp
                    response_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    response_file = os.path.join(results_dir, f"volume_response_{stock_config.symbol}_{response_timestamp}.txt")
                    with open(response_file, 'w') as f:
                        f.write("VOLUME CONFIRMATION LLM ANALYSIS RESULTS\n")
                        f.write("=" * 80 + "\n\n")
                        f.write(f"Stock Symbol: {stock_config.symbol}\n")
                        f.write(f"Company: {stock_config.name}\n")
                        f.write(f"Sector: {stock_config.sector}\n")
                        f.write(f"Volume Profile: {stock_config.volume_profile}\n")
                        f.write(f"Response Time: {datetime.now().isoformat()}\n")
                        f.write(f"Response Length: {len(response) if response else 0} characters\n")
                        if code_results:
                            f.write(f"Mathematical Calculations: {len(code_results)} code snippets executed\n")
                        if execution_results:
                            f.write(f"Calculation Results: {len(execution_results)} computational outputs\n")
                        f.write("\n")
                        
                        # Full response
                        f.write("COMPLETE LLM RESPONSE:\n")
                        f.write("-" * 40 + "\n")
                        f.write(response or "No response received")
                        f.write("\n")
                    
                except Exception as e:
                    print(f"❌ API call failed for {stock_config.symbol}: {e}")
                    llm_response = f"API_ERROR: {str(e)}"
            
            execution_time = time.time() - start_time
            
            # Evaluate quality
            quality_metrics = self._evaluate_volume_analysis(stock_config, analysis_data, llm_response)
            
            return {
                'stock_config': stock_config,
                'success': True,
                'execution_time': execution_time,
                'quality_score': quality_metrics['overall_score'],
                'data_quality': quality_metrics['data_quality'],
                'response_length': len(llm_response) if llm_response else 0,
                'volume_analysis': {
                    'confirmation_status': safe_get(analysis_data, 'overall_assessment', 'confirmation_status'),
                    'confirmation_strength': safe_get(analysis_data, 'overall_assessment', 'confirmation_strength'),
                    'confidence_score': safe_get(analysis_data, 'overall_assessment', 'confidence_score'),
                    'correlation_coefficient': safe_get(analysis_data, 'price_volume_correlation', 'correlation_coefficient'),
                    'correlation_strength': safe_get(analysis_data, 'price_volume_correlation', 'correlation_strength'),
                    'volume_vs_20d': safe_get(analysis_data, 'volume_averages', 'volume_vs_20d')
                },
                'quality_metrics': quality_metrics,
                'has_llm_response': bool(llm_response and not llm_response.startswith("API_ERROR"))
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            return {
                'stock_config': stock_config,
                'success': False,
                'error': str(e),
                'execution_time': execution_time,
                'quality_score': 0,
                'data_quality': 'error'
            }
    
    def _evaluate_volume_analysis(self, stock_config: VolumeStockTestConfig, analysis_data: Dict[str, Any], llm_response: str) -> Dict[str, Any]:
        """Evaluate the quality of volume confirmation analysis for a stock"""
        metrics = {
            'data_quality': 'unknown',
            'analysis_completeness': 0,
            'response_quality': 0,
            'volume_profile_appropriateness': 0,
            'overall_score': 0
        }
        
        # Evaluate data quality and analysis completeness
        required_components = ['overall_assessment', 'price_volume_correlation', 'volume_averages', 'trend_support']
        available_components = 0
        for component in required_components:
            if component in analysis_data and 'error' not in analysis_data[component]:
                available_components += 1
        
        metrics['analysis_completeness'] = (available_components / len(required_components)) * 100
        
        # Check correlation analysis quality
        correlation_data = analysis_data.get('price_volume_correlation', {})
        if 'error' not in correlation_data and correlation_data.get('correlation_coefficient') is not None:
            metrics['data_quality'] = 'excellent'
            metrics['analysis_completeness'] += 20
        else:
            metrics['data_quality'] = 'limited'
        
        # Evaluate response quality if available
        if llm_response and not llm_response.startswith("API_ERROR"):
            if len(llm_response) > 800:  # Substantial response
                metrics['response_quality'] = 80
            elif len(llm_response) > 400:  # Moderate response
                metrics['response_quality'] = 60
            else:
                metrics['response_quality'] = 40
            
            # Check for JSON format and required fields
            if 'volume_confirmation_status' in llm_response and 'confidence_score' in llm_response:
                metrics['response_quality'] += 20
        
        # Volume profile appropriateness (based on expected volume behavior)
        metrics['volume_profile_appropriateness'] = 70  # Default good score
        
        # Bonus for specific volume profile expectations
        overall_assessment = analysis_data.get('overall_assessment', {})
        if stock_config.volume_profile == 'high_liquidity':
            # Expect strong correlations for high liquidity stocks
            corr_strength = analysis_data.get('price_volume_correlation', {}).get('correlation_strength', 'weak')
            if corr_strength in ['strong', 'medium']:
                metrics['volume_profile_appropriateness'] += 15
        elif stock_config.volume_profile == 'consistent_volume':
            # Expect good confirmation scores for consistent volume stocks
            confidence = overall_assessment.get('confidence_score', 0)
            if confidence > 60:
                metrics['volume_profile_appropriateness'] += 15
        
        # Calculate overall score
        metrics['overall_score'] = min(100, (
            metrics['analysis_completeness'] * 0.4 +
            metrics['response_quality'] * 0.4 +
            metrics['volume_profile_appropriateness'] * 0.2
        ))
        
        return metrics
    
    def _generate_multi_stock_report(self, results_dir: str):
        """Generate comprehensive multi-stock volume confirmation analysis report"""
        # Prepare summary data
        summary_data = []
        successful_tests = [r for r in self.results if r['success']]
        
        for result in self.results:
            summary_data.append({
                'Symbol': result['stock_config'].symbol,
                'Company': result['stock_config'].name,
                'Sector': result['stock_config'].sector,
                'Volume Profile': result['stock_config'].volume_profile,
                'Success': result['success'],
                'Quality Score': result['quality_score'],
                'Execution Time (s)': result['execution_time'],
                'Data Quality': result.get('data_quality', 'unknown'),
                'Has LLM Response': result.get('has_llm_response', False),
                'Chart Saved': result.get('chart_saved', False),
                'Chart Filename': result.get('chart_filename', 'N/A'),
                'Chart Size (KB)': round(result.get('chart_size_bytes', 0) / 1024, 1) if result.get('chart_size_bytes') else 'N/A',
                'Confirmation Status': result.get('volume_analysis', {}).get('confirmation_status', 'N/A'),
                'Confirmation Strength': result.get('volume_analysis', {}).get('confirmation_strength', 'N/A'),
                'Confidence Score': result.get('volume_analysis', {}).get('confidence_score', 'N/A'),
                'Correlation Coefficient': result.get('volume_analysis', {}).get('correlation_coefficient', 'N/A'),
                'Volume vs 20D Avg': result.get('volume_analysis', {}).get('volume_vs_20d', 'N/A')
            })
        
        # Save to Excel
        summary_df = pd.DataFrame(summary_data)
        excel_path = os.path.join(results_dir, "volume_confirmation_summary.xlsx")
        summary_df.to_excel(excel_path, index=False)
        
        # Generate detailed text report
        report_path = os.path.join(results_dir, "volume_confirmation_comprehensive_report.txt")
        with open(report_path, 'w') as f:
            f.write("VOLUME CONFIRMATION MULTI-STOCK TESTING REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Stocks Tested: {len(self.results)}\n")
            f.write(f"Successful Tests: {len(successful_tests)}\n")
            f.write(f"Success Rate: {len(successful_tests)/len(self.results)*100:.1f}%\n\n")
            
            # Overall statistics
            if successful_tests:
                avg_quality = sum(r['quality_score'] for r in successful_tests) / len(successful_tests)
                avg_execution = sum(r['execution_time'] for r in successful_tests) / len(successful_tests)
                
                # Chart statistics
                charts_saved = sum(1 for r in successful_tests if r.get('chart_saved', False))
                total_chart_size = sum(r.get('chart_size_bytes', 0) for r in successful_tests if r.get('chart_saved', False))
                avg_chart_size_kb = round(total_chart_size / charts_saved / 1024, 1) if charts_saved > 0 else 0
                
                f.write("OVERALL STATISTICS\n")
                f.write("-" * 40 + "\n")
                f.write(f"Average Quality Score: {avg_quality:.1f}/100\n")
                f.write(f"Average Execution Time: {avg_execution:.1f}s\n")
                f.write(f"Charts Generated: {charts_saved}/{len(successful_tests)}\n")
                f.write(f"Chart Success Rate: {charts_saved/len(successful_tests)*100:.1f}%\n")
                if charts_saved > 0:
                    f.write(f"Average Chart Size: {avg_chart_size_kb} KB\n")
                    f.write(f"Total Chart Storage: {round(total_chart_size/1024/1024, 2)} MB\n")
                f.write("\n")
            
            # Volume profile analysis
            f.write("VOLUME PROFILE ANALYSIS\n")
            f.write("-" * 40 + "\n")
            profiles = {}
            for result in successful_tests:
                profile = result['stock_config'].volume_profile
                if profile not in profiles:
                    profiles[profile] = []
                profiles[profile].append(result)
            
            for profile, profile_results in profiles.items():
                profile_avg_quality = sum(r['quality_score'] for r in profile_results) / len(profile_results)
                f.write(f"\n{profile.replace('_', ' ').title()}:\n")
                f.write(f"  Stocks Tested: {len(profile_results)}\n")
                f.write(f"  Average Quality: {profile_avg_quality:.1f}/100\n")
                f.write(f"  Companies: {', '.join(r['stock_config'].symbol for r in profile_results)}\n")
            
            # Sector-wise analysis
            f.write("\n\nSECTOR-WISE ANALYSIS\n")
            f.write("-" * 40 + "\n")
            sectors = {}
            for result in successful_tests:
                sector = result['stock_config'].sector
                if sector not in sectors:
                    sectors[sector] = []
                sectors[sector].append(result)
            
            for sector, sector_results in sectors.items():
                sector_avg_quality = sum(r['quality_score'] for r in sector_results) / len(sector_results)
                f.write(f"\n{sector}:\n")
                f.write(f"  Stocks Tested: {len(sector_results)}\n")
                f.write(f"  Average Quality: {sector_avg_quality:.1f}/100\n")
                f.write(f"  Companies: {', '.join(r['stock_config'].symbol for r in sector_results)}\n")
            
            # Individual stock details
            f.write("\n\nINDIVIDUAL STOCK ANALYSIS\n")
            f.write("-" * 40 + "\n")
            for result in self.results:
                f.write(f"\n{result['stock_config'].symbol} ({result['stock_config'].name}):\n")
                f.write(f"  Sector: {result['stock_config'].sector}\n")
                f.write(f"  Volume Profile: {result['stock_config'].volume_profile}\n")
                f.write(f"  Success: {'✅' if result['success'] else '❌'}\n")
                if result['success']:
                    f.write(f"  Quality Score: {result['quality_score']:.1f}/100\n")
                    f.write(f"  Execution Time: {result['execution_time']:.1f}s\n")
                    f.write(f"  Data Quality: {result.get('data_quality', 'unknown')}\n")
                    f.write(f"  Chart Saved: {'✅' if result.get('chart_saved', False) else '❌'}\n")
                    if result.get('chart_saved', False):
                        f.write(f"  Chart File: {result.get('chart_filename', 'N/A')}\n")
                        chart_size_kb = round(result.get('chart_size_bytes', 0) / 1024, 1) if result.get('chart_size_bytes') else 0
                        f.write(f"  Chart Size: {chart_size_kb} KB\n")
                    if 'volume_analysis' in result:
                        va = result['volume_analysis']
                        f.write(f"  Confirmation Status: {va.get('confirmation_status', 'N/A')}\n")
                        f.write(f"  Confirmation Strength: {va.get('confirmation_strength', 'N/A')}\n")
                        f.write(f"  Confidence Score: {va.get('confidence_score', 'N/A')}%\n")
                        f.write(f"  Correlation: {va.get('correlation_coefficient', 'N/A')}\n")
                        if va.get('volume_vs_20d'):
                            f.write(f"  Volume vs 20D Avg: {va['volume_vs_20d']:.2f}x\n")
                else:
                    f.write(f"  Error: {result.get('error', 'Unknown error')}\n")
        
        print(f"📊 Volume confirmation report saved to: {report_path}")
        print(f"📈 Summary data saved to: {excel_path}")

async def main():
    """Main function"""
    print("🔍 Multi-Stock Volume Confirmation Testing Framework")
    print("Testing volume_confirmation_analysis across multiple stocks from different sectors")
    
    tester = MultiStockVolumeConfirmationTester()
    success = await tester.run_multi_stock_tests()
    
    if success:
        print("\n🎉 Multi-stock volume confirmation testing completed successfully!")
    else:
        print("\n❌ Multi-stock volume confirmation testing failed")

if __name__ == "__main__":
    asyncio.run(main())