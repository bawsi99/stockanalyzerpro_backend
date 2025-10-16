#!/usr/bin/env python3
"""
Volume Anomaly Multi-Stock Prompt Testing Framework

Tests the volume_anomaly_detection prompt across multiple stocks from different sectors
to validate consistency and quality of volume anomaly detection analysis.

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

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'backend'))

try:
    from backend.llm import get_llm_client
    from backend.zerodha.client import ZerodhaDataClient
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Make sure you're running this from the correct directory")
    sys.exit(1)

# Import volume anomaly agent components
try:
    from backend.agents.volume.volume_anomaly.processor import VolumeAnomalyProcessor
    from backend.agents.volume.volume_anomaly.charts import VolumeAnomalyChartGenerator
except ImportError as e:
    print(f"❌ Volume Anomaly Agent Import Error: {e}")
    print("Make sure the volume anomaly agent components are properly installed")
    sys.exit(1)

class StockTestConfig:
    """Configuration for individual stock tests"""
    def __init__(self, symbol: str, name: str, sector: str, expected_volume_behavior: str):
        self.symbol = symbol
        self.name = name
        self.sector = sector
        self.expected_volume_behavior = expected_volume_behavior

class VolumeAnomalyMultiStockTester:
    """Test volume anomaly detection prompt across multiple stocks"""
    
    def __init__(self):
        # Initialize Zerodha client
        try:
            self.zerodha_client = ZerodhaDataClient()
            print("✅ Zerodha client initialized")
        except Exception as e:
            print(f"❌ Cannot initialize Zerodha client: {e}")
            sys.exit(1)
        
        # Initialize volume anomaly components
        self.volume_processor = VolumeAnomalyProcessor()
        self.chart_generator = VolumeAnomalyChartGenerator()
        
        # Initialize LLM client for volume analysis
        self.llm_client = None
        try:
            self.llm_client = get_llm_client("volume_agent")  # Uses pre-configured volume agent
            print("✅ LLM client initialized for volume analysis")
        except Exception as e:
            print(f"⚠️  Could not initialize LLM client: {e}")
            print("⚠️  Will show prompts only")
        
        # Define test stocks from different sectors with volume behavior expectations
        self.test_stocks = [
            StockTestConfig("RELIANCE", "Reliance Industries", "Energy/Petrochemicals", "moderate_spikes"),
            StockTestConfig("TCS", "Tata Consultancy Services", "IT Services", "low_volatility"),
            StockTestConfig("HDFCBANK", "HDFC Bank", "Banking", "moderate_consistent"),
            StockTestConfig("ICICIBANK", "ICICI Bank", "Banking", "moderate_spikes"),
            StockTestConfig("ITC", "ITC Limited", "FMCG/Tobacco", "low_volatility"),
            StockTestConfig("INFY", "Infosys", "IT Services", "low_volatility"),
            StockTestConfig("BHARTIARTL", "Bharti Airtel", "Telecommunications", "high_volatility"),
            StockTestConfig("HINDUNILVR", "Hindustan Unilever", "FMCG", "low_volatility"),
            StockTestConfig("MARUTI", "Maruti Suzuki", "Automotive", "seasonal_spikes"),
            StockTestConfig("BAJFINANCE", "Bajaj Finance", "NBFC", "high_volatility")
        ]
        
        self.results = []
    
    async def run_multi_stock_tests(self):
        """Run volume anomaly tests across all configured stocks"""
        print(f"🚀 Starting Volume Anomaly Multi-Stock Testing")
        print(f"Testing {len(self.test_stocks)} stocks with 365 days of data")
        print("=" * 80)
        
        # Authenticate with Zerodha first
        print("🔗 Authenticating with Zerodha...")
        if not self.zerodha_client.authenticate():
            print("❌ Zerodha authentication failed")
            return False
        
        print("✅ Zerodha authentication successful")
        
        # Create results directory
        results_dir = "volume_anomaly_test_results"
        os.makedirs(results_dir, exist_ok=True)
        
        # Create async tasks for all stocks to run them concurrently
        async def test_single_stock(stock_config, stock_index):
            """Test a single stock asynchronously"""
            print(f"\n📊 Testing Stock {stock_index}/{len(self.test_stocks)}: {stock_config.symbol}")
            print(f"   Company: {stock_config.name}")
            print(f"   Sector: {stock_config.sector}")
            print(f"   Expected Volume Behavior: {stock_config.expected_volume_behavior}")
            print("-" * 60)
            
            try:
                # Get stock data
                print(f"📈 Fetching 365 days of data for {stock_config.symbol}...")
                
                # Use async version of get_historical_data if available
                if hasattr(self.zerodha_client, 'get_historical_data_async'):
                    stock_data = await self.zerodha_client.get_historical_data_async(
                        symbol=stock_config.symbol,
                        exchange="NSE",
                        interval="day",
                        period=365
                    )
                else:
                    # Fallback to sync version in executor
                    loop = asyncio.get_event_loop()
                    stock_data = await loop.run_in_executor(
                        None,
                        self.zerodha_client.get_historical_data,
                        stock_config.symbol,
                        "NSE",
                        "day",
                        None,
                        None,
                        365
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
                stock_data_indexed = stock_data.set_index('date')
                
                print(f"✅ Retrieved {len(stock_data)} days of data")
                print(f"   Date range: {stock_data['date'].min()} to {stock_data['date'].max()}")
                print(f"   Price range: ₹{stock_data['close'].min():.2f} to ₹{stock_data['close'].max():.2f}")
                print(f"   Volume range: {stock_data['volume'].min():,} to {stock_data['volume'].max():,}")
                
                # Process volume anomaly data
                print("🔍 Processing volume anomaly data...")
                volume_analysis_data = self.volume_processor.process_volume_anomaly_data(stock_data_indexed)
                
                if 'error' in volume_analysis_data:
                    print(f"❌ Volume analysis failed: {volume_analysis_data['error']}")
                    return {
                        'stock_config': stock_config,
                        'success': False,
                        'error': volume_analysis_data['error'],
                        'execution_time': 0,
                        'quality_score': 0,
                        'data_quality': 'analysis_failed'
                    }
                
                # Generate volume anomaly chart
                print("📊 Generating volume anomaly chart...")
                chart_path = os.path.join(results_dir, f"chart_{stock_config.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
                chart_bytes = self.chart_generator.generate_volume_anomaly_chart(
                    stock_data_indexed, 
                    volume_analysis_data, 
                    stock_config.symbol, 
                    chart_path
                )
                
                # Test the volume anomaly prompt
                result = await self._test_volume_anomaly_prompt(
                    stock_config, 
                    volume_analysis_data, 
                    chart_bytes, 
                    results_dir
                )
                
                print(f"✅ Test completed for {stock_config.symbol}")
                print(f"   Success: {result['success']}")
                print(f"   Quality Score: {result['quality_score']:.1f}/100")
                print(f"   Response Time: {result['execution_time']:.1f}s")
                print(f"   Anomalies Detected: {result.get('anomalies_detected', 0)}")
                
                return result
                
            except Exception as e:
                print(f"❌ Error testing {stock_config.symbol}: {e}")
                import traceback
                traceback.print_exc()
                
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
        
        print(f"\n✅ Volume anomaly multi-stock testing completed!")
        print(f"📁 Results saved to: {results_dir}/")
        
        return True
    
    async def _test_volume_anomaly_prompt(self, 
                                        stock_config: StockTestConfig, 
                                        volume_analysis_data: Dict[str, Any],
                                        chart_bytes: bytes,
                                        results_dir: str) -> Dict[str, Any]:
        """Test the volume anomaly detection prompt for a single stock"""
        start_time = time.time()
        
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
            # Create context for volume anomaly detection
            context_data = {
                'stock_symbol': stock_config.symbol,
                'company_name': stock_config.name,
                'sector': stock_config.sector,
                'analysis_type': 'volume_anomaly_detection',
                'time_period': '1 year, daily',
                'volume_statistics': volume_analysis_data.get('volume_statistics', {}),
                'significant_anomalies': volume_analysis_data.get('significant_anomalies', []),
                'anomaly_patterns': volume_analysis_data.get('anomaly_patterns', {}),
                'current_volume_status': volume_analysis_data.get('current_volume_status', {}),
                'data_quality': volume_analysis_data.get('data_quality', 'unknown')
            }
            
            # Format context string
            context = f"""Stock: {stock_config.symbol} ({stock_config.name})
Sector: {stock_config.sector}
Analysis Period: {volume_analysis_data.get('data_range', 'Unknown')}
Data Quality: {volume_analysis_data.get('data_quality', 'Unknown')}

Volume Statistics:
- Mean Volume: {safe_get(volume_analysis_data, 'volume_statistics', 'volume_mean') or 0:,.0f}
- Current Volume: {safe_get(volume_analysis_data, 'volume_statistics', 'current_volume') or 0:,.0f}
- Volume Volatility (CV): {safe_get(volume_analysis_data, 'volume_statistics', 'volume_cv') or 0:.2f}
- Current Z-Score: {safe_get(volume_analysis_data, 'volume_statistics', 'current_z_score') or 0:.2f}

Detected Anomalies: {len(volume_analysis_data.get('significant_anomalies', []))}
Current Volume Status: {safe_get(volume_analysis_data, 'current_volume_status', 'current_status') or 'unknown'}
Volume Percentile: {safe_get(volume_analysis_data, 'current_volume_status', 'volume_percentile') or 0}th

Anomaly Patterns:
- Frequency: {safe_get(volume_analysis_data, 'anomaly_patterns', 'anomaly_frequency') or 'unknown'}
- Pattern: {safe_get(volume_analysis_data, 'anomaly_patterns', 'anomaly_pattern') or 'unknown'}
- Dominant Causes: {', '.join(safe_get(volume_analysis_data, 'anomaly_patterns', 'dominant_causes') or [])}

Recent Significant Anomalies:"""
            
            # Add recent anomalies to context
            anomalies = volume_analysis_data.get('significant_anomalies', [])
            for i, anomaly in enumerate(anomalies[:5]):  # Top 5 anomalies
                if 'error' not in anomaly:
                    context += f"""
- {anomaly.get('date', 'unknown')}: {anomaly.get('volume_ratio', 0):.1f}x volume ({anomaly.get('significance', 'unknown')} significance)
  Context: {anomaly.get('price_context', 'unknown')}, Likely Cause: {anomaly.get('likely_cause', 'unknown')}"""
            
            # Create the volume anomaly detection prompt
            prompt = f"""Please analyze the volume anomaly data for this stock:

{context}

Provide a comprehensive volume anomaly analysis including:
1. Assessment of detected anomalies and their significance
2. Current volume status and trends
3. Anomaly patterns and frequency analysis
4. Risk assessment based on volume behavior
5. Trading implications and recommendations

Format your response as a structured analysis with clear sections for each component."""
            
            # Save prompt details
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            prompt_file = os.path.join(results_dir, f"prompt_volume_anomaly_{stock_config.symbol}_{timestamp}.txt")
            with open(prompt_file, 'w') as f:
                f.write("VOLUME ANOMALY DETECTION PROMPT ANALYSIS FOR LLM\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"Stock Symbol: {stock_config.symbol}\n")
                f.write(f"Company: {stock_config.name}\n")
                f.write(f"Sector: {stock_config.sector}\n")
                f.write(f"Expected Volume Behavior: {stock_config.expected_volume_behavior}\n")
                f.write(f"Generated: {datetime.now().isoformat()}\n")
                f.write(f"Prompt Length: {len(prompt)} characters\n")
                f.write(f"Context Length: {len(context)} characters\n\n")
                
                f.write("KEY VOLUME ANOMALY METRICS:\n")
                f.write("-" * 40 + "\n")
                key_metrics = {
                    "total_anomalies": len(anomalies),
                    "high_significance_count": len([a for a in anomalies if a.get('significance') == 'high']),
                    "medium_significance_count": len([a for a in anomalies if a.get('significance') == 'medium']),
                    "low_significance_count": len([a for a in anomalies if a.get('significance') == 'low']),
                    "current_volume_status": safe_get(volume_analysis_data, 'current_volume_status', 'current_status'),
                    "volume_percentile": safe_get(volume_analysis_data, 'current_volume_status', 'volume_percentile'),
                    "anomaly_frequency": safe_get(volume_analysis_data, 'anomaly_patterns', 'anomaly_frequency'),
                    "dominant_causes": safe_get(volume_analysis_data, 'anomaly_patterns', 'dominant_causes'),
                    "volume_cv": safe_get(volume_analysis_data, 'volume_statistics', 'volume_cv'),
                    "current_z_score": safe_get(volume_analysis_data, 'volume_statistics', 'current_z_score')
                }
                f.write(json.dumps(key_metrics, indent=2, default=str))
                f.write("\n\n")
                f.write("FINAL PROMPT SENT TO LLM:\n")
                f.write("-" * 40 + "\n")
                f.write(prompt)
            
            # Make API call if available
            llm_response = ""
            if self.llm_client:
                try:
                    print(f"🚀 Making API call for volume anomaly analysis of {stock_config.symbol}...")
                    
                    # Use the new LLM backend with code execution enabled
                    llm_response = await self.llm_client.generate(
                        prompt=prompt,
                        enable_code_execution=True,  # Enable calculations for better analysis
                        timeout=90,  # 90 second timeout for volume analysis
                        max_retries=3
                    )
                    
                    # Save response
                    response_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    response_file = os.path.join(results_dir, f"response_volume_anomaly_{stock_config.symbol}_{response_timestamp}.txt")
                    with open(response_file, 'w') as f:
                        f.write("VOLUME ANOMALY DETECTION LLM RESPONSE\n")
                        f.write("=" * 80 + "\n\n")
                        f.write(f"Stock Symbol: {stock_config.symbol}\n")
                        f.write(f"Company: {stock_config.name}\n")
                        f.write(f"Sector: {stock_config.sector}\n")
                        f.write(f"Response Time: {datetime.now().isoformat()}\n")
                        f.write(f"Response Length: {len(llm_response) if llm_response else 0} characters\n")
                        f.write(f"Provider: {self.llm_client.get_provider_info()}\n")
                        f.write("\n")
                        
                        f.write("COMPLETE LLM RESPONSE:\n")
                        f.write("-" * 40 + "\n")
                        f.write(llm_response or "No response received")
                        f.write("\n")
                    
                except Exception as e:
                    print(f"❌ API call failed for {stock_config.symbol}: {e}")
                    llm_response = f"API_ERROR: {str(e)}"
            
            execution_time = time.time() - start_time
            
            # Evaluate quality
            quality_metrics = self._evaluate_volume_anomaly_analysis(
                stock_config, 
                volume_analysis_data, 
                llm_response
            )
            
            return {
                'stock_config': stock_config,
                'success': True,
                'execution_time': execution_time,
                'quality_score': quality_metrics['overall_score'],
                'data_quality': quality_metrics['data_quality'],
                'response_length': len(llm_response) if llm_response else 0,
                'anomalies_detected': len(anomalies),
                'high_significance_anomalies': len([a for a in anomalies if a.get('significance') == 'high']),
                'volume_metrics': {
                    'current_status': safe_get(volume_analysis_data, 'current_volume_status', 'current_status'),
                    'volume_percentile': safe_get(volume_analysis_data, 'current_volume_status', 'volume_percentile'),
                    'z_score': safe_get(volume_analysis_data, 'volume_statistics', 'current_z_score'),
                    'anomaly_frequency': safe_get(volume_analysis_data, 'anomaly_patterns', 'anomaly_frequency'),
                    'volume_cv': safe_get(volume_analysis_data, 'volume_statistics', 'volume_cv')
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
    
    def _evaluate_volume_anomaly_analysis(self, 
                                        stock_config: StockTestConfig, 
                                        volume_analysis_data: Dict[str, Any], 
                                        llm_response: str) -> Dict[str, Any]:
        """Evaluate the quality of volume anomaly analysis for a stock"""
        metrics = {
            'data_quality': 'unknown',
            'anomaly_detection_quality': 0,
            'response_quality': 0,
            'sector_appropriateness': 0,
            'overall_score': 0
        }
        
        # Evaluate data quality
        if 'error' not in volume_analysis_data:
            metrics['data_quality'] = volume_analysis_data.get('data_quality', 'good')
            
        # Evaluate anomaly detection quality
        anomalies = volume_analysis_data.get('significant_anomalies', [])
        quality_assessment = volume_analysis_data.get('quality_assessment', {})
        
        if anomalies and not any('error' in a for a in anomalies):
            # Base score for having valid anomalies
            metrics['anomaly_detection_quality'] = 40
            
            # Bonus for high-quality anomalies
            high_sig_count = len([a for a in anomalies if a.get('significance') == 'high'])
            medium_sig_count = len([a for a in anomalies if a.get('significance') == 'medium'])
            
            if high_sig_count > 0:
                metrics['anomaly_detection_quality'] += 30
            elif medium_sig_count > 0:
                metrics['anomaly_detection_quality'] += 20
            else:
                metrics['anomaly_detection_quality'] += 10
                
            # Quality assessment bonus
            overall_quality_score = quality_assessment.get('overall_score', 0)
            if overall_quality_score > 80:
                metrics['anomaly_detection_quality'] += 20
            elif overall_quality_score > 60:
                metrics['anomaly_detection_quality'] += 10
                
        # Cap at 100
        metrics['anomaly_detection_quality'] = min(100, metrics['anomaly_detection_quality'])
        
        # Evaluate response quality if available
        if llm_response and not llm_response.startswith("API_ERROR"):
            if len(llm_response) > 1000:  # Substantial response
                metrics['response_quality'] = 80
            elif len(llm_response) > 500:  # Moderate response
                metrics['response_quality'] = 60
            else:
                metrics['response_quality'] = 40
            
            # Check for expected JSON structure for volume anomaly detection
            expected_fields = [
                'significant_anomalies', 'anomaly_frequency', 'anomaly_pattern',
                'current_volume_status', 'volume_percentile', 'recent_volume_trend',
                'confidence_score'
            ]
            
            fields_found = sum(1 for field in expected_fields if field in llm_response)
            if fields_found >= len(expected_fields) * 0.8:  # 80% of expected fields
                metrics['response_quality'] += 20
        
        # Sector appropriateness based on expected volume behavior
        expected_behavior = stock_config.expected_volume_behavior
        actual_frequency = volume_analysis_data.get('anomaly_patterns', {}).get('anomaly_frequency', 'unknown')
        
        # Map expected behaviors to actual frequency expectations
        behavior_mapping = {
            'low_volatility': 'low',
            'moderate_consistent': 'medium',
            'moderate_spikes': 'medium',
            'seasonal_spikes': 'medium',
            'high_volatility': 'high'
        }
        
        expected_frequency = behavior_mapping.get(expected_behavior, 'medium')
        if actual_frequency == expected_frequency:
            metrics['sector_appropriateness'] = 90
        elif abs(['low', 'medium', 'high'].index(actual_frequency) - ['low', 'medium', 'high'].index(expected_frequency)) == 1:
            metrics['sector_appropriateness'] = 70  # Close match
        else:
            metrics['sector_appropriateness'] = 50  # Different but not necessarily wrong
        
        # Calculate overall score
        metrics['overall_score'] = min(100, (
            metrics['anomaly_detection_quality'] * 0.5 +
            metrics['response_quality'] * 0.3 +
            metrics['sector_appropriateness'] * 0.2
        ))
        
        return metrics
    
    def _generate_multi_stock_report(self, results_dir: str):
        """Generate comprehensive multi-stock volume anomaly analysis report"""
        # Prepare summary data
        summary_data = []
        successful_tests = [r for r in self.results if r['success']]
        
        for result in self.results:
            summary_data.append({
                'Symbol': result['stock_config'].symbol,
                'Company': result['stock_config'].name,
                'Sector': result['stock_config'].sector,
                'Expected Volume Behavior': result['stock_config'].expected_volume_behavior,
                'Success': result['success'],
                'Quality Score': result['quality_score'],
                'Execution Time (s)': result['execution_time'],
                'Data Quality': result.get('data_quality', 'unknown'),
                'Has LLM Response': result.get('has_llm_response', False),
                'Anomalies Detected': result.get('anomalies_detected', 0),
                'High Significance Anomalies': result.get('high_significance_anomalies', 0),
                'Volume Status': result.get('volume_metrics', {}).get('current_status', 'N/A'),
                'Volume Percentile': result.get('volume_metrics', {}).get('volume_percentile', 'N/A'),
                'Volume Z-Score': result.get('volume_metrics', {}).get('z_score', 'N/A'),
                'Anomaly Frequency': result.get('volume_metrics', {}).get('anomaly_frequency', 'N/A'),
                'Volume CV': result.get('volume_metrics', {}).get('volume_cv', 'N/A')
            })
        
        # Save to Excel
        summary_df = pd.DataFrame(summary_data)
        excel_path = os.path.join(results_dir, "volume_anomaly_multi_stock_summary.xlsx")
        summary_df.to_excel(excel_path, index=False)
        
        # Generate detailed text report
        report_path = os.path.join(results_dir, "volume_anomaly_comprehensive_report.txt")
        with open(report_path, 'w') as f:
            f.write("VOLUME ANOMALY MULTI-STOCK TESTING REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Stocks Tested: {len(self.results)}\n")
            f.write(f"Successful Tests: {len(successful_tests)}\n")
            f.write(f"Success Rate: {len(successful_tests)/len(self.results)*100:.1f}%\n\n")
            
            # Overall statistics
            if successful_tests:
                avg_quality = sum(r['quality_score'] for r in successful_tests) / len(successful_tests)
                avg_execution = sum(r['execution_time'] for r in successful_tests) / len(successful_tests)
                total_anomalies = sum(r.get('anomalies_detected', 0) for r in successful_tests)
                avg_anomalies = total_anomalies / len(successful_tests)
                
                f.write("OVERALL STATISTICS\n")
                f.write("-" * 40 + "\n")
                f.write(f"Average Quality Score: {avg_quality:.1f}/100\n")
                f.write(f"Average Execution Time: {avg_execution:.1f}s\n")
                f.write(f"Total Anomalies Detected: {total_anomalies}\n")
                f.write(f"Average Anomalies per Stock: {avg_anomalies:.1f}\n\n")
            
            # Sector-wise analysis
            f.write("SECTOR-WISE VOLUME ANOMALY ANALYSIS\n")
            f.write("-" * 40 + "\n")
            sectors = {}
            for result in successful_tests:
                sector = result['stock_config'].sector
                if sector not in sectors:
                    sectors[sector] = []
                sectors[sector].append(result)
            
            for sector, sector_results in sectors.items():
                sector_avg_quality = sum(r['quality_score'] for r in sector_results) / len(sector_results)
                sector_avg_anomalies = sum(r.get('anomalies_detected', 0) for r in sector_results) / len(sector_results)
                f.write(f"\n{sector}:\n")
                f.write(f"  Stocks Tested: {len(sector_results)}\n")
                f.write(f"  Average Quality: {sector_avg_quality:.1f}/100\n")
                f.write(f"  Average Anomalies: {sector_avg_anomalies:.1f}\n")
                f.write(f"  Companies: {', '.join(r['stock_config'].symbol for r in sector_results)}\n")
            
            # Individual stock details
            f.write("\n\nINDIVIDUAL STOCK VOLUME ANALYSIS\n")
            f.write("-" * 40 + "\n")
            for result in self.results:
                f.write(f"\n{result['stock_config'].symbol} ({result['stock_config'].name}):\n")
                f.write(f"  Sector: {result['stock_config'].sector}\n")
                f.write(f"  Expected Behavior: {result['stock_config'].expected_volume_behavior}\n")
                f.write(f"  Success: {'✅' if result['success'] else '❌'}\n")
                if result['success']:
                    f.write(f"  Quality Score: {result['quality_score']:.1f}/100\n")
                    f.write(f"  Execution Time: {result['execution_time']:.1f}s\n")
                    f.write(f"  Data Quality: {result.get('data_quality', 'unknown')}\n")
                    f.write(f"  Anomalies Detected: {result.get('anomalies_detected', 0)}\n")
                    f.write(f"  High Significance: {result.get('high_significance_anomalies', 0)}\n")
                    if 'volume_metrics' in result:
                        vm = result['volume_metrics']
                        f.write(f"  Volume Status: {vm.get('current_status', 'N/A')}\n")
                        f.write(f"  Volume Percentile: {vm.get('volume_percentile', 'N/A')}th\n")
                        if vm.get('z_score') is not None:
                            f.write(f"  Z-Score: {vm['z_score']:.2f}\n")
                        f.write(f"  Anomaly Frequency: {vm.get('anomaly_frequency', 'N/A')}\n")
                else:
                    f.write(f"  Error: {result.get('error', 'Unknown error')}\n")
        
        print(f"📊 Volume anomaly report saved to: {report_path}")
        print(f"📈 Summary data saved to: {excel_path}")

async def main():
    """Main function"""
    print("🔍 Volume Anomaly Multi-Stock Testing Framework")
    print("Testing volume_anomaly_detection prompt across multiple stocks from different sectors")
    
    tester = VolumeAnomalyMultiStockTester()
    success = await tester.run_multi_stock_tests()
    
    if success:
        print("\n🎉 Volume anomaly multi-stock testing completed successfully!")
    else:
        print("\n❌ Volume anomaly multi-stock testing failed")

if __name__ == "__main__":
    asyncio.run(main())