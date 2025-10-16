#!/usr/bin/env python3
"""
Multi-Stock Volume Momentum Testing Framework

Tests the volume_trend_momentum prompt across multiple stocks from different sectors
to validate consistency and quality of volume momentum analysis.

Usage: python multi_stock_test.py
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import openpyxl
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
import requests

# Add backend and relevant paths to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '../../../../')
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'backend'))

try:
    from backend.llm import get_llm_client
    from backend.zerodha.client import ZerodhaDataClient
    # Import volume momentum processor and chart generator from local package
    from backend.agents.volume.volume_momentum.processor import VolumeTrendMomentumProcessor
    from backend.agents.volume.volume_momentum.charts import VolumeTrendMomentumChartGenerator
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Make sure you're running this from the correct directory")
    print(f"Current directory: {os.getcwd()}")
    print(f"Project root: {project_root}")
    sys.exit(1)

class StockTestConfig:
    """Configuration for individual stock tests"""
    def __init__(self, symbol: str, name: str, sector: str, expected_behavior: str):
        self.symbol = symbol
        self.name = name
        self.sector = sector
        self.expected_behavior = expected_behavior

class VolumeMomentumMultiStockTester:
    """Test volume momentum prompt across multiple stocks"""
    
    def __init__(self):
        # Initialize Zerodha client
        try:
            self.zerodha_client = ZerodhaDataClient()
            print("✅ Zerodha client initialized")
        except Exception as e:
            print(f"❌ Cannot initialize Zerodha client: {e}")
            sys.exit(1)
        
        # Initialize other components
        self.volume_momentum_processor = VolumeTrendMomentumProcessor()
        self.chart_generator = VolumeTrendMomentumChartGenerator()
        
        # Initialize LLM client using the new backend/llm system
        self.llm_client = None
        try:
            # Use volume_agent configuration for volume momentum analysis
            self.llm_client = get_llm_client("volume_agent")
            print("✅ LLM client initialized using volume_agent configuration")
        except Exception as e:
            print(f"⚠️  Could not initialize LLM client: {e}")
            print("Will show prompts only without LLM responses")
        
        # Define test stocks from different sectors
        self.test_stocks = [
            StockTestConfig("RELIANCE", "Reliance Industries", "Energy/Petrochemicals", "high_volume_stability"),
            StockTestConfig("TCS", "Tata Consultancy Services", "IT Services", "consistent_volume_growth"),
            StockTestConfig("HDFCBANK", "HDFC Bank", "Banking", "volume_momentum_shifts"),
            StockTestConfig("ICICIBANK", "ICICI Bank", "Banking", "institutional_volume_patterns"),
            StockTestConfig("ITC", "ITC Limited", "FMCG/Tobacco", "defensive_volume_behavior"),
            StockTestConfig("INFY", "Infosys", "IT Services", "tech_sector_momentum"),
            StockTestConfig("BHARTIARTL", "Bharti Airtel", "Telecommunications", "cyclical_volume_patterns"),
            # Add more stocks for comprehensive testing
            # StockTestConfig("HINDUNILVR", "Hindustan Unilever", "FMCG", "stable_consumer_volume"),
            # StockTestConfig("MARUTI", "Maruti Suzuki", "Automotive", "cyclical_auto_volume"),
            # StockTestConfig("BAJFINANCE", "Bajaj Finance", "NBFC", "growth_volume_momentum")
        ]
        
        self.results = []
    
    async def run_multi_stock_tests(self):
        """Run tests across all configured stocks"""
        print(f"🚀 Starting Multi-Stock Volume Momentum Testing")
        print(f"Testing {len(self.test_stocks)} stocks with 90 days of data")
        print("==" * 40)
        
        # Authenticate with Zerodha first
        print("🔗 Authenticating with Zerodha...")
        if not self.zerodha_client.authenticate():
            print("❌ Zerodha authentication failed")
            return False
        
        print("✅ Zerodha authentication successful")
        
        # Create results directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_dir = f"volume_momentum_test_results_{timestamp}"
        os.makedirs(results_dir, exist_ok=True)
        
        # Create async tasks for all stocks to run them concurrently
        async def test_single_stock(stock_config, stock_index):
            """Test a single stock asynchronously"""
            print(f"\n📊 Testing Stock {stock_index}/{len(self.test_stocks)}: {stock_config.symbol}")
            print(f"   Company: {stock_config.name}")
            print(f"   Sector: {stock_config.sector}")
            print(f"   Expected Behavior: {stock_config.expected_behavior}")
            print("-" * 60)
            
            try:
                # Get stock data (90 days for volume momentum analysis)
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
                
                # Process volume momentum data
                print("📊 Processing volume momentum data...")
                
                # Set date as index for processing
                stock_data_indexed = stock_data.set_index('date')
                
                # Use the volume momentum processor
                volume_momentum_analysis = self.volume_momentum_processor.process_volume_trend_momentum_data(
                    stock_data_indexed
                )
                
                if 'error' in volume_momentum_analysis:
                    print(f"❌ Volume momentum processing failed: {volume_momentum_analysis['error']}")
                    return {
                        'stock_config': stock_config,
                        'success': False,
                        'error': volume_momentum_analysis['error'],
                        'execution_time': 0,
                        'quality_score': 0,
                        'data_quality': 'processing_failed'
                    }
                
                # Test the prompt
                result = await self._test_volume_momentum_prompt(
                    stock_config, 
                    volume_momentum_analysis, 
                    stock_data_indexed,
                    results_dir
                )
                
                print(f"✅ Test completed for {stock_config.symbol}")
                print(f"   Success: {result['success']}")
                print(f"   Quality Score: {result['quality_score']:.1f}/100")
                print(f"   Response Time: {result['execution_time']:.1f}s")
                
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
        semaphore = asyncio.Semaphore(2)  # Limit to 2 concurrent tests for volume analysis
        
        async def test_with_semaphore(stock_config, index):
            async with semaphore:
                return await test_single_stock(stock_config, index)
        
        # Create tasks for all stocks
        tasks = [
            test_with_semaphore(stock_config, i + 1)
            for i, stock_config in enumerate(self.test_stocks)
        ]
        
        # Wait for all tasks to complete
        print(f"\n🔄 Running {len(tasks)} volume momentum tests concurrently (max 2 at a time)...")
        results = await asyncio.gather(*tasks)
        self.results.extend(results)
        
        # Generate comprehensive report
        self._generate_multi_stock_report(results_dir)
        
        print(f"\n✅ Multi-stock volume momentum testing completed!")
        print(f"📁 Results saved to: {results_dir}/")
        
        return True
    
    async def _test_volume_momentum_prompt(self, stock_config: StockTestConfig, 
                                         volume_momentum_analysis: Dict[str, Any], 
                                         stock_data: pd.DataFrame,
                                         results_dir: str) -> Dict[str, Any]:
        """Test the volume momentum prompt for a single stock"""
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
            # Create context for the volume momentum prompt
            context = self._create_volume_momentum_context(volume_momentum_analysis, stock_config)
            
            # Create the volume momentum analysis prompt
            prompt = f"""Analyze the volume momentum patterns for {stock_config.symbol}:

{context}

Please provide a comprehensive volume momentum analysis including:
1. Volume trend direction and strength
2. Momentum phase assessment
3. Volume momentum patterns
4. Future trend implications
5. Confidence score (0-100)

Format your response as a JSON object with the structure:
{{
    "volume_trend_direction": "bullish/bearish/neutral",
    "trend_strength": "strong/moderate/weak",
    "momentum_analysis": {{
        "current_phase": "accumulation/distribution/consolidation",
        "momentum_strength": "high/medium/low"
    }},
    "volume_momentum_phases": ["phase1", "phase2", "phase3"],
    "future_implications": {{
        "trend_continuation_probability": 0.75,
        "key_levels": ["level1", "level2"]
    }},
    "confidence_score": 85
}}

Provide detailed explanations for each component."""
            
            # Save prompt details
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            prompt_file = os.path.join(results_dir, f"volume_momentum_prompt_{stock_config.symbol}_{timestamp}.txt")
            with open(prompt_file, 'w') as f:
                f.write("VOLUME MOMENTUM PROMPT ANALYSIS FOR LLM\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"Stock Symbol: {stock_config.symbol}\n")
                f.write(f"Company: {stock_config.name}\n")
                f.write(f"Sector: {stock_config.sector}\n")
                f.write(f"Expected Behavior: {stock_config.expected_behavior}\n")
                f.write(f"Generated: {datetime.now().isoformat()}\n")
                f.write(f"Prompt Length: {len(prompt)} characters\n")
                f.write(f"Context Length: {len(context)} characters\n\n")
                
                f.write("KEY VOLUME MOMENTUM INDICATORS SUMMARY:\n")
                f.write("-" * 40 + "\n")
                
                # Extract key metrics for summary
                key_metrics = {
                    "volume_trend_direction": safe_get(volume_momentum_analysis, 'volume_trend_direction'),
                    "trend_strength": safe_get(volume_momentum_analysis, 'trend_strength'),
                    "momentum_phase": safe_get(volume_momentum_analysis, 'momentum_phase'),
                    "primary_trend_strength": safe_get(volume_momentum_analysis, 'volume_trend_analysis', 'primary_trend_strength'),
                    "overall_momentum_direction": safe_get(volume_momentum_analysis, 'momentum_analysis', 'overall_momentum_direction'),
                    "momentum_strength": safe_get(volume_momentum_analysis, 'momentum_analysis', 'momentum_strength'),
                    "current_phase": safe_get(volume_momentum_analysis, 'cycle_analysis', 'current_phase'),
                    "trend_continuation_probability": safe_get(volume_momentum_analysis, 'future_implications', 'trend_continuation_probability'),
                    "sustainability_score": safe_get(volume_momentum_analysis, 'sustainability_assessment', 'sustainability_score'),
                    "quality_score": safe_get(volume_momentum_analysis, 'quality_assessment', 'overall_score'),
                }
                f.write(json.dumps(key_metrics, indent=2, default=str))
                f.write("\n\n")
                
                f.write("FINAL PROMPT SENT TO LLM:\n")
                f.write("-" * 40 + "\n")
                f.write(prompt)
            
            # Make API call if available
            llm_response = ""
            parsed_response = {}
            if self.llm_client:
                try:
                    print(f"🚀 Making LLM API call for {stock_config.symbol}...")
                    # Use the new LLM client with code execution enabled for calculations
                    llm_response = await self.llm_client.generate(
                        prompt=prompt,
                        enable_code_execution=True,
                        timeout=90  # 90 seconds for volume analysis
                    )
                    
                    # Try to parse JSON response
                    try:
                        # Extract JSON from response if it's wrapped in text
                        json_start = llm_response.find('{')
                        json_end = llm_response.rfind('}') + 1
                        if json_start != -1 and json_end > json_start:
                            json_str = llm_response[json_start:json_end]
                            parsed_response = json.loads(json_str)
                    except json.JSONDecodeError:
                        print(f"⚠️  Could not parse JSON response for {stock_config.symbol}")
                    
                    # Save response
                    response_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    response_file = os.path.join(results_dir, f"volume_momentum_response_{stock_config.symbol}_{response_timestamp}.txt")
                    with open(response_file, 'w') as f:
                        f.write("VOLUME MOMENTUM ANALYSIS RESULTS\n")
                        f.write("=" * 80 + "\n\n")
                        f.write(f"Stock Symbol: {stock_config.symbol}\n")
                        f.write(f"Company: {stock_config.name}\n")
                        f.write(f"Sector: {stock_config.sector}\n")
                        f.write(f"Response Time: {datetime.now().isoformat()}\n")
                        f.write(f"Response Length: {len(llm_response) if llm_response else 0} characters\n")
                        f.write(f"JSON Parsed Successfully: {bool(parsed_response)}\n")
                        f.write(f"LLM Provider: {self.llm_client.get_provider_info() if self.llm_client else 'None'}\n")
                        f.write("\n")
                        
                        # Parsed JSON response (if available)
                        if parsed_response:
                            f.write("PARSED JSON RESPONSE:\n")
                            f.write("-" * 40 + "\n")
                            f.write(json.dumps(parsed_response, indent=2))
                            f.write("\n\n")
                        
                        # Full response
                        f.write("COMPLETE LLM RESPONSE:\n")
                        f.write("-" * 40 + "\n")
                        f.write(llm_response or "No response received")
                        f.write("\n")
                    
                except Exception as e:
                    print(f"❌ API call failed for {stock_config.symbol}: {e}")
                    llm_response = f"API_ERROR: {str(e)}"
            
            # Generate volume momentum chart
            chart_bytes = None
            chart_path = None
            chart_success = False
            try:
                print(f"🎨 Generating volume momentum chart for {stock_config.symbol}...")
                chart_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                chart_filename = f"volume_momentum_chart_{stock_config.symbol}_{chart_timestamp}.png"
                chart_path = os.path.join(results_dir, chart_filename)
                
                # Generate the chart
                chart_bytes = self.chart_generator.generate_volume_momentum_chart(
                    stock_data, 
                    volume_momentum_analysis, 
                    f"{stock_config.symbol} ({stock_config.name})",
                    save_path=chart_path
                )
                
                if chart_bytes:
                    chart_success = True
                    print(f"✅ Chart generated: {len(chart_bytes)} bytes saved to {chart_filename}")
                else:
                    print(f"⚠️  Chart generation returned no data for {stock_config.symbol}")
                    
            except Exception as e:
                print(f"❌ Chart generation failed for {stock_config.symbol}: {e}")
                chart_path = None
            
            execution_time = time.time() - start_time
            
            # Evaluate quality
            quality_metrics = self._evaluate_volume_momentum_analysis(
                stock_config, 
                volume_momentum_analysis, 
                llm_response, 
                parsed_response
            )
            
            return {
                'stock_config': stock_config,
                'success': True,
                'execution_time': execution_time,
                'quality_score': quality_metrics['overall_score'],
                'data_quality': quality_metrics['data_quality'],
                'response_length': len(llm_response) if llm_response else 0,
                'json_parsed_successfully': bool(parsed_response),
                'chart_generated': chart_success,
                'chart_path': chart_path,
                'chart_bytes': len(chart_bytes) if chart_bytes else 0,
                'volume_momentum_metrics': {
                    'trend_direction': safe_get(volume_momentum_analysis, 'volume_trend_direction'),
                    'trend_strength': safe_get(volume_momentum_analysis, 'trend_strength'),
                    'momentum_phase': safe_get(volume_momentum_analysis, 'momentum_phase'),
                    'momentum_strength': safe_get(volume_momentum_analysis, 'momentum_analysis', 'momentum_strength'),
                    'sustainability_score': safe_get(volume_momentum_analysis, 'sustainability_assessment', 'sustainability_score'),
                    'quality_assessment_score': safe_get(volume_momentum_analysis, 'quality_assessment', 'overall_score')
                },
                'parsed_llm_response': parsed_response,
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
    
    def _create_volume_momentum_context(self, volume_momentum_analysis: Dict[str, Any], 
                                      stock_config: StockTestConfig) -> str:
        """Create streamlined context string for volume momentum prompt - essential data only"""
        context_parts = []
        
        # Stock information
        context_parts.append(f"Stock: {stock_config.symbol} ({stock_config.name})")
        context_parts.append(f"Sector: {stock_config.sector}")
        context_parts.append(f"Analysis Period: 90 days")
        
        # Essential momentum data only
        volume_trends = volume_momentum_analysis.get('volume_trend_analysis', {})
        momentum_analysis = volume_momentum_analysis.get('momentum_analysis', {})
        
        if volume_trends and 'error' not in volume_trends:
            context_parts.append("\nMOMENTUM METRICS:")
            context_parts.append(f"Primary Trend: {volume_trends.get('primary_trend_direction', 'unknown')} ({volume_trends.get('primary_trend_strength', 'unknown')})")
            context_parts.append(f"Trend Agreement: {volume_trends.get('trend_agreement_score', 0):.2f}")
        
        if momentum_analysis and 'error' not in momentum_analysis:
            # Only use most relevant ROC indicator (10-day)
            roc_10d = momentum_analysis.get('rate_of_change_indicators', {}).get('roc_10d', {})
            if roc_10d:
                context_parts.append(f"ROC 10-Day: {roc_10d.get('current_value', 0):+.1f}% (Avg: {roc_10d.get('average_value', 0):+.1f}%)")
            
            context_parts.append(f"Momentum Phase: {momentum_analysis.get('overall_momentum_direction', 'unknown')}")
        
        # Essential sustainability data only
        sustainability = volume_momentum_analysis.get('sustainability_assessment', {})
        if sustainability and 'error' not in sustainability:
            context_parts.append(f"\nSUSTAINABILITY:")
            context_parts.append(f"Sustainability Score: {sustainability.get('sustainability_score', 0)}/100")
            context_parts.append(f"Continuation Probability: {volume_momentum_analysis.get('future_implications', {}).get('trend_continuation_probability', 0):.1%}")
        
        return "\n".join(context_parts)
    
    def _evaluate_volume_momentum_analysis(self, stock_config: StockTestConfig, 
                                         volume_momentum_analysis: Dict[str, Any], 
                                         llm_response: str, 
                                         parsed_response: Dict) -> Dict[str, Any]:
        """Evaluate the quality of volume momentum analysis for a stock"""
        metrics = {
            'data_quality': 'unknown',
            'processing_completeness': 0,
            'response_quality': 0,
            'json_format_compliance': 0,
            'volume_analysis_depth': 0,
            'overall_score': 0
        }
        
        # Evaluate data processing completeness
        required_components = ['volume_trend_analysis', 'momentum_analysis', 'cycle_analysis', 
                             'future_implications', 'sustainability_assessment']
        available_components = 0
        for component in required_components:
            if component in volume_momentum_analysis and 'error' not in volume_momentum_analysis[component]:
                available_components += 1
        
        metrics['processing_completeness'] = (available_components / len(required_components)) * 100
        
        # Data quality assessment
        quality_assessment = volume_momentum_analysis.get('quality_assessment', {})
        if quality_assessment and 'error' not in quality_assessment:
            quality_score = quality_assessment.get('overall_score', 0)
            if quality_score >= 80:
                metrics['data_quality'] = 'excellent'
                metrics['processing_completeness'] += 20
            elif quality_score >= 60:
                metrics['data_quality'] = 'good'
                metrics['processing_completeness'] += 10
            elif quality_score >= 40:
                metrics['data_quality'] = 'fair'
            else:
                metrics['data_quality'] = 'poor'
        
        # Evaluate LLM response quality
        if llm_response and not llm_response.startswith("API_ERROR"):
            if len(llm_response) > 1000:  # Substantial response
                metrics['response_quality'] = 80
            elif len(llm_response) > 500:  # Moderate response
                metrics['response_quality'] = 60
            else:
                metrics['response_quality'] = 40
            
            # Check for expected volume momentum fields
            expected_fields = ['volume_trend_direction', 'momentum_analysis', 'trend_quality_indicators', 
                             'future_implications', 'volume_momentum_phases']
            field_count = sum(1 for field in expected_fields if field in llm_response.lower())
            if field_count >= 4:
                metrics['response_quality'] += 20
        
        # JSON format compliance
        if parsed_response:
            metrics['json_format_compliance'] = 80
            
            # Check for required JSON fields from prompt
            required_json_fields = ['volume_trend_direction', 'trend_strength', 'momentum_analysis', 
                                  'volume_momentum_phases', 'confidence_score']
            json_field_count = sum(1 for field in required_json_fields if field in parsed_response)
            if json_field_count >= 4:
                metrics['json_format_compliance'] = 100
        
        # Volume analysis depth (based on available analysis components)
        volume_trends = volume_momentum_analysis.get('volume_trend_analysis', {})
        momentum_analysis = volume_momentum_analysis.get('momentum_analysis', {})
        cycle_analysis = volume_momentum_analysis.get('cycle_analysis', {})
        
        depth_score = 0
        if volume_trends and 'error' not in volume_trends:
            depth_score += 25
            if volume_trends.get('trend_agreement_score', 0) > 0.5:
                depth_score += 10
        
        if momentum_analysis and 'error' not in momentum_analysis:
            depth_score += 25
            roc_indicators = momentum_analysis.get('rate_of_change_indicators', {})
            if len(roc_indicators) >= 2:
                depth_score += 10
        
        if cycle_analysis and 'error' not in cycle_analysis:
            depth_score += 25
            if cycle_analysis.get('cycle_count', 0) > 0:
                depth_score += 15
        
        metrics['volume_analysis_depth'] = min(depth_score, 100)
        
        # Calculate overall score
        metrics['overall_score'] = (
            metrics['processing_completeness'] * 0.25 +
            metrics['response_quality'] * 0.25 +
            metrics['json_format_compliance'] * 0.2 +
            metrics['volume_analysis_depth'] * 0.3
        )
        
        return metrics
    
    def _generate_multi_stock_report(self, results_dir: str):
        """Generate comprehensive multi-stock volume momentum analysis report"""
        # Prepare summary data
        summary_data = []
        successful_tests = [r for r in self.results if r['success']]
        
        for result in self.results:
            summary_data.append({
                'Symbol': result['stock_config'].symbol,
                'Company': result['stock_config'].name,
                'Sector': result['stock_config'].sector,
                'Expected Behavior': result['stock_config'].expected_behavior,
                'Success': result['success'],
                'Quality Score': result['quality_score'],
                'Execution Time (s)': result['execution_time'],
                'Data Quality': result.get('data_quality', 'unknown'),
                'JSON Parsed': result.get('json_parsed_successfully', False),
                'Has LLM Response': result.get('has_llm_response', False),
                'Chart Generated': result.get('chart_generated', False),
                'Chart Path': result.get('chart_path', 'N/A'),
                'Trend Direction': result.get('volume_momentum_metrics', {}).get('trend_direction', 'N/A'),
                'Trend Strength': result.get('volume_momentum_metrics', {}).get('trend_strength', 'N/A'),
                'Momentum Phase': result.get('volume_momentum_metrics', {}).get('momentum_phase', 'N/A'),
                'Sustainability Score': result.get('volume_momentum_metrics', {}).get('sustainability_score', 'N/A')
            })
        
        # Save to Excel
        summary_df = pd.DataFrame(summary_data)
        excel_path = os.path.join(results_dir, "volume_momentum_multi_stock_summary.xlsx")
        summary_df.to_excel(excel_path, index=False)
        
        # Generate detailed text report
        report_path = os.path.join(results_dir, "volume_momentum_comprehensive_report.txt")
        with open(report_path, 'w') as f:
            f.write("MULTI-STOCK VOLUME MOMENTUM TESTING REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Stocks Tested: {len(self.results)}\n")
            f.write(f"Successful Tests: {len(successful_tests)}\n")
            f.write(f"Success Rate: {len(successful_tests)/len(self.results)*100:.1f}%\n\n")
            
            # Overall statistics
            if successful_tests:
                avg_quality = sum(r['quality_score'] for r in successful_tests) / len(successful_tests)
                avg_execution = sum(r['execution_time'] for r in successful_tests) / len(successful_tests)
                json_success_rate = sum(1 for r in successful_tests if r.get('json_parsed_successfully', False)) / len(successful_tests) * 100
                chart_success_rate = sum(1 for r in successful_tests if r.get('chart_generated', False)) / len(successful_tests) * 100
                
                f.write("OVERALL STATISTICS\n")
                f.write("-" * 40 + "\n")
                f.write(f"Average Quality Score: {avg_quality:.1f}/100\n")
                f.write(f"Average Execution Time: {avg_execution:.1f}s\n")
                f.write(f"JSON Parsing Success Rate: {json_success_rate:.1f}%\n")
                f.write(f"Chart Generation Success Rate: {chart_success_rate:.1f}%\n\n")
            
            # Sector-wise analysis
            f.write("SECTOR-WISE ANALYSIS\n")
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
            
            # Volume momentum insights
            f.write("\n\nVOLUME MOMENTUM INSIGHTS\n")
            f.write("-" * 40 + "\n")
            
            # Trend directions
            trend_directions = {}
            for result in successful_tests:
                trend_dir = result.get('volume_momentum_metrics', {}).get('trend_direction', 'unknown')
                trend_directions[trend_dir] = trend_directions.get(trend_dir, 0) + 1
            
            f.write("Trend Direction Distribution:\n")
            for direction, count in trend_directions.items():
                percentage = (count / len(successful_tests)) * 100
                f.write(f"  {direction}: {count} stocks ({percentage:.1f}%)\n")
            
            # Individual stock details
            f.write("\n\nINDIVIDUAL STOCK ANALYSIS\n")
            f.write("-" * 40 + "\n")
            for result in self.results:
                f.write(f"\n{result['stock_config'].symbol} ({result['stock_config'].name}):\n")
                f.write(f"  Sector: {result['stock_config'].sector}\n")
                f.write(f"  Expected Behavior: {result['stock_config'].expected_behavior}\n")
                f.write(f"  Success: {'✅' if result['success'] else '❌'}\n")
                if result['success']:
                    f.write(f"  Quality Score: {result['quality_score']:.1f}/100\n")
                    f.write(f"  Execution Time: {result['execution_time']:.1f}s\n")
                    f.write(f"  Data Quality: {result.get('data_quality', 'unknown')}\n")
                    f.write(f"  JSON Parsed: {'✅' if result.get('json_parsed_successfully', False) else '❌'}\n")
                    f.write(f"  Chart Generated: {'✅' if result.get('chart_generated', False) else '❌'}\n")
                    if result.get('chart_path'):
                        chart_filename = os.path.basename(result['chart_path'])
                        f.write(f"  Chart File: {chart_filename}\n")
                    
                    if 'volume_momentum_metrics' in result:
                        vm = result['volume_momentum_metrics']
                        f.write(f"  Trend Direction: {vm.get('trend_direction', 'N/A')}\n")
                        f.write(f"  Trend Strength: {vm.get('trend_strength', 'N/A')}\n")
                        f.write(f"  Momentum Phase: {vm.get('momentum_phase', 'N/A')}\n")
                        if vm.get('sustainability_score') is not None:
                            f.write(f"  Sustainability Score: {vm['sustainability_score']}/100\n")
                else:
                    f.write(f"  Error: {result.get('error', 'Unknown error')}\n")
        
        print(f"📊 Volume momentum report saved to: {report_path}")
        print(f"📈 Summary data saved to: {excel_path}")

async def main():
    """Main function"""
    print("🔍 Multi-Stock Volume Momentum Testing Framework")
    print("Testing volume_trend_momentum prompt across multiple stocks from different sectors")
    
    tester = VolumeMomentumMultiStockTester()
    success = await tester.run_multi_stock_tests()
    
    if success:
        print("\n🎉 Multi-stock volume momentum testing completed successfully!")
    else:
        print("\n❌ Multi-stock volume momentum testing failed")

if __name__ == "__main__":
    asyncio.run(main())