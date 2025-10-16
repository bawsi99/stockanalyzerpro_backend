#!/usr/bin/env python3
"""
Multi-Stock Institutional Activity Testing Framework

Tests the institutional_activity_analysis prompt across multiple stocks from different sectors
to validate consistency and quality of volume-based institutional analysis.

NOTE: Updated to use the new backend/llm system instead of backend/gemini.
Uses 'volume_agent' configuration from llm_assignments.yaml for LLM requests.

Usage: python multi_stock_institutional_test.py

Requirements:
- GEMINI_API_KEY environment variable set
- Redis running for caching
- Zerodha credentials configured
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

# Add paths relative to the script location
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, '..', '..', '..', '..')
backend_path = os.path.join(project_root, 'backend')
agent_path = os.path.join(script_dir, 'src')

# Add paths to Python path
sys.path.insert(0, project_root)
sys.path.insert(0, backend_path)
sys.path.insert(0, agent_path)

try:
    from backend.llm import get_llm_client
    from backend.zerodha.client import ZerodhaDataClient
    # Import institutional activity components (updated after file move)
    from backend.agents.volume.institutional_activity.processor import InstitutionalActivityProcessor
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print(f"Script dir: {script_dir}")
    print(f"Project root: {project_root}")
    print(f"Backend path: {backend_path}")
    print(f"Agent path: {agent_path}")
    print("Python path:")
    for path in sys.path[:10]:  # Show first 10 paths
        print(f"  {path}")
    print("Make sure you're running this from the correct directory and dependencies are installed")
    sys.exit(1)

class StockTestConfig:
    """Configuration for individual stock tests"""
    def __init__(self, symbol: str, name: str, sector: str, expected_behavior: str):
        self.symbol = symbol
        self.name = name
        self.sector = sector
        self.expected_behavior = expected_behavior

class InstitutionalVolumeAnalyzer:
    """Analyze volume patterns for institutional activity detection"""
    
    @staticmethod
    def calculate_volume_metrics(df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate volume-specific metrics for institutional analysis"""
        volume = df['volume']
        close = df['close']
        high = df['high']
        low = df['low']
        
        # Volume moving averages
        volume_sma_20 = volume.rolling(window=20).mean()
        volume_sma_50 = volume.rolling(window=50).mean()
        
        # Volume ratio calculations
        volume_ratio = volume / volume_sma_20
        
        # Large volume detection (3x normal volume)
        large_volume_threshold = volume_sma_20 * 3
        large_volume_days = volume > large_volume_threshold
        
        # Price-volume relationships
        price_change = close.pct_change()
        volume_price_correlation = volume.rolling(20).corr(close)
        
        # On Balance Volume (OBV) calculation
        obv = (volume * np.sign(price_change)).fillna(0).cumsum()
        
        # Volume Rate of Change
        volume_roc = volume.pct_change(periods=5)
        
        # Accumulation/Distribution Line
        money_flow_multiplier = ((close - low) - (high - close)) / (high - low)
        money_flow_multiplier = money_flow_multiplier.fillna(0)
        money_flow_volume = money_flow_multiplier * volume
        ad_line = money_flow_volume.cumsum()
        
        def get_latest_valid(series):
            valid_values = series.dropna()
            return float(valid_values.iloc[-1]) if not valid_values.empty else None
        
        def get_recent_values(series, count=20):
            valid_values = series.dropna()
            return valid_values.tail(count).tolist() if not valid_values.empty else []
        
        # Build comprehensive volume metrics
        volume_metrics = {
            # Basic volume data
            'volume_data': {
                'current_volume': get_latest_valid(volume),
                'volume_sma_20': get_latest_valid(volume_sma_20),
                'volume_sma_50': get_latest_valid(volume_sma_50),
                'volume_ratio': get_latest_valid(volume_ratio),
                'average_daily_volume': float(volume.mean()),
                'volume_volatility': float(volume.std()),
                'recent_volumes': get_recent_values(volume, 10)
            },
            
            # Large volume analysis
            'large_volume_analysis': {
                'large_volume_days': int(large_volume_days.sum()),
                'large_volume_percentage': float(large_volume_days.sum() / len(df) * 100),
                'max_volume_ratio': float(volume_ratio.max()) if not volume_ratio.empty else 0,
                'recent_large_volumes': len([v for v in get_recent_values(volume_ratio, 10) if v > 3])
            },
            
            # Institutional patterns
            'institutional_patterns': {
                'obv_current': get_latest_valid(obv),
                'obv_trend': 'bullish' if len(get_recent_values(obv, 5)) >= 2 and get_recent_values(obv, 5)[-1] > get_recent_values(obv, 5)[0] else 'bearish',
                'ad_line_current': get_latest_valid(ad_line),
                'ad_line_trend': 'accumulation' if len(get_recent_values(ad_line, 5)) >= 2 and get_recent_values(ad_line, 5)[-1] > get_recent_values(ad_line, 5)[0] else 'distribution',
                'volume_roc': get_latest_valid(volume_roc),
                'price_volume_correlation': get_latest_valid(volume_price_correlation)
            },
            
            # Smart money indicators
            'smart_money_indicators': {
                'volume_on_up_days': float(volume[price_change > 0].mean()) if (price_change > 0).any() else 0,
                'volume_on_down_days': float(volume[price_change < 0].mean()) if (price_change < 0).any() else 0,
                'buying_pressure_ratio': float(volume[price_change > 0].sum() / volume.sum()) if volume.sum() > 0 else 0,
                'selling_pressure_ratio': float(volume[price_change < 0].sum() / volume.sum()) if volume.sum() > 0 else 0,
                'quiet_accumulation_score': float((large_volume_days & (abs(price_change) < 0.02)).sum()),
                'distribution_score': float((large_volume_days & (price_change < -0.02)).sum())
            },
            
            # Historical context
            'historical_context': {
                'volume_history': get_recent_values(volume, 50),
                'price_history': get_recent_values(close, 50),
                'volume_ratio_history': get_recent_values(volume_ratio, 30),
                'obv_history': get_recent_values(obv, 30),
                'ad_line_history': get_recent_values(ad_line, 30)
            }
        }
        
        return volume_metrics

class MultiStockInstitutionalTester:
    """Test institutional activity prompt across multiple stocks"""
    
    def __init__(self):
        # Initialize Zerodha client
        try:
            self.zerodha_client = ZerodhaDataClient()
            print("✅ Zerodha client initialized")
        except Exception as e:
            print(f"❌ Cannot initialize Zerodha client: {e}")
            sys.exit(1)
        
        # Initialize other components
        self.volume_analyzer = InstitutionalVolumeAnalyzer()
        self.institutional_processor = InstitutionalActivityProcessor()
        
        # Initialize chart generator
        try:
            from backend.agents.volume.institutional_activity.charts import InstitutionalActivityChartGenerator
            self.chart_generator = InstitutionalActivityChartGenerator()
            print("✅ Institutional Activity Chart Generator initialized")
        except Exception as e:
            print(f"⚠️  Chart generator initialization failed: {e}")
            self.chart_generator = None
        
        # Initialize LLM client using new backend/llm system
        self.llm_client = None
        try:
            # Use volume_agent configuration for institutional analysis
            self.llm_client = get_llm_client("volume_agent")
            print("✅ LLM client initialized (volume_agent)")
        except Exception as e:
            print(f"⚠️  Could not initialize LLM client: {e}")
            print("⚠️  Will show prompts only")
        
        # Define test stocks with focus on volume patterns
        self.test_stocks = [
            StockTestConfig("RELIANCE", "Reliance Industries", "Energy/Petrochemicals", "institutional_favorite"),
            StockTestConfig("TCS", "Tata Consultancy Services", "IT Services", "steady_institutional"),
            StockTestConfig("HDFCBANK", "HDFC Bank", "Banking", "high_institutional_activity"),
            StockTestConfig("ICICIBANK", "ICICI Bank", "Banking", "active_institutional_trading"),
            StockTestConfig("ITC", "ITC Limited", "FMCG/Tobacco", "defensive_institutional"),
            # Additional stocks for comprehensive volume analysis
            StockTestConfig("INFY", "Infosys", "IT Services", "institutional_accumulation"),
            StockTestConfig("BHARTIARTL", "Bharti Airtel", "Telecommunications", "cyclical_institutional"),
        ]
        
        self.results = []
    
    async def run_multi_stock_tests(self):
        """Run institutional activity tests across all configured stocks"""
        print(f"🚀 Starting Multi-Stock Institutional Activity Testing")
        print(f"Testing {len(self.test_stocks)} stocks for institutional volume patterns")
        print("==" * 40)
        
        # Authenticate with Zerodha first
        print("🔗 Authenticating with Zerodha...")
        if not self.zerodha_client.authenticate():
            print("❌ Zerodha authentication failed")
            return False
        
        print("✅ Zerodha authentication successful")
        
        # Create results directory
        results_dir = "institutional_activity_test_results"
        os.makedirs(results_dir, exist_ok=True)
        
        # Create async tasks for all stocks
        async def test_single_stock(stock_config, stock_index):
            """Test institutional activity for a single stock"""
            print(f"\n🏛️ Testing Stock {stock_index}/{len(self.test_stocks)}: {stock_config.symbol}")
            print(f"   Company: {stock_config.name}")
            print(f"   Sector: {stock_config.sector}")
            print(f"   Expected Pattern: {stock_config.expected_behavior}")
            print("-" * 60)
            
            try:
                # Get stock data - more data for better volume analysis
                print(f"📊 Fetching 365 days of data for {stock_config.symbol}...")
                
                if hasattr(self.zerodha_client, 'get_historical_data_async'):
                    stock_data = await self.zerodha_client.get_historical_data_async(
                        symbol=stock_config.symbol,
                        exchange="NSE",
                        interval="day",
                        period=365
                    )
                else:
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
                        365
                    )
                
                if stock_data is None or stock_data.empty:
                    print(f"❌ No data available for {stock_config.symbol}")
                    return self._create_error_result(stock_config, 'No data available')
                
                # Ensure proper data structure
                required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
                if 'date' not in stock_data.columns and stock_data.index.name == 'date':
                    stock_data = stock_data.reset_index()
                elif 'date' not in stock_data.columns:
                    stock_data['date'] = stock_data.index
                    stock_data = stock_data.reset_index(drop=True)
                
                missing_columns = [col for col in required_columns if col not in stock_data.columns]
                if missing_columns:
                    print(f"❌ Missing required columns for {stock_config.symbol}: {missing_columns}")
                    return self._create_error_result(stock_config, f'Missing columns: {missing_columns}')
                
                # Sort by date
                stock_data = stock_data.sort_values('date').reset_index(drop=True)
                
                print(f"✅ Retrieved {len(stock_data)} days of data")
                print(f"   Date range: {stock_data['date'].min()} to {stock_data['date'].max()}")
                print(f"   Volume range: {stock_data['volume'].min():,} to {stock_data['volume'].max():,}")
                print(f"   Avg daily volume: {stock_data['volume'].mean():,.0f}")
                
                # Calculate volume metrics for institutional analysis
                print("📈 Calculating institutional volume metrics...")
                volume_metrics = self.volume_analyzer.calculate_volume_metrics(stock_data)
                
                # Use institutional activity processor for advanced analysis
                print("🔍 Processing institutional activity patterns...")
                stock_data_indexed = stock_data.set_index('date')
                institutional_analysis = self.institutional_processor.process_institutional_activity_data(stock_data_indexed)
                
                # Combine metrics
                combined_analysis = {
                    **volume_metrics,
                    'institutional_analysis': institutional_analysis
                }
                
                # Test the institutional activity prompt
                result = await self._test_institutional_prompt(stock_config, combined_analysis, stock_data, results_dir)
                
                print(f"✅ Institutional analysis completed for {stock_config.symbol}")
                print(f"   Success: {result['success']}")
                print(f"   Quality Score: {result['quality_score']:.1f}/100")
                print(f"   Response Time: {result['execution_time']:.1f}s")
                print(f"   Chart Generated: {'Yes' if result.get('chart_generated') else 'No'}")
                print(f"   Analysis Mode: {result.get('analysis_mode', 'Unknown')}")
                
                return result
                
            except Exception as e:
                print(f"❌ Error testing {stock_config.symbol}: {e}")
                import traceback
                traceback.print_exc()
                return self._create_error_result(stock_config, str(e))
        
        # Run all stock tests with concurrency control
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
        print(f"\n🔄 Running {len(tasks)} institutional activity tests concurrently (max 3 at a time)...")
        results = await asyncio.gather(*tasks)
        self.results.extend(results)
        
        # Generate comprehensive report
        self._generate_institutional_report(results_dir)
        
        print(f"\n✅ Multi-stock institutional testing completed!")
        print(f"📁 Results saved to: {results_dir}/")
        
        return True
    
    async def _test_institutional_prompt(self, stock_config: StockTestConfig, analysis_data: Dict[str, Any], stock_data: pd.DataFrame, results_dir: str) -> Dict[str, Any]:
        """Test the institutional activity analysis prompt for a single stock"""
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
            # Prepare context for institutional activity analysis
            context = self._prepare_institutional_context(stock_config, analysis_data)
            
            # Create institutional activity analysis prompt directly
            prompt = f"""
            INSTITUTIONAL ACTIVITY ANALYSIS REQUEST
            
            {context}
            
            Please provide a comprehensive analysis of institutional activity patterns for this stock.
            
            Focus on:
            1. Institutional Activity Level (low/medium/high)
            2. Primary Activity Type (accumulation/distribution/mixed)
            3. Smart Money Timing patterns
            4. Volume-based institutional signals
            5. Key support/resistance levels based on institutional activity
            6. Confidence score and reasoning
            
            Provide your analysis in a structured format with specific insights and actionable intelligence.
            """
            
            # Save prompt details
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            prompt_file = os.path.join(results_dir, f"institutional_prompt_{stock_config.symbol}_{timestamp}.txt")
            with open(prompt_file, 'w') as f:
                f.write("INSTITUTIONAL ACTIVITY ANALYSIS PROMPT\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"Stock Symbol: {stock_config.symbol}\n")
                f.write(f"Company: {stock_config.name}\n")
                f.write(f"Sector: {stock_config.sector}\n")
                f.write(f"Expected Pattern: {stock_config.expected_behavior}\n")
                f.write(f"Generated: {datetime.now().isoformat()}\n")
                f.write(f"Prompt Length: {len(prompt)} characters\n")
                f.write(f"Context Length: {len(context)} characters\n\n")
                
                f.write("KEY VOLUME METRICS SUMMARY:\n")
                f.write("-" * 40 + "\n")
                
                volume_summary = {
                    "current_volume": safe_get(analysis_data, 'volume_data', 'current_volume'),
                    "volume_ratio": safe_get(analysis_data, 'volume_data', 'volume_ratio'),
                    "large_volume_days": safe_get(analysis_data, 'large_volume_analysis', 'large_volume_days'),
                    "obv_trend": safe_get(analysis_data, 'institutional_patterns', 'obv_trend'),
                    "ad_line_trend": safe_get(analysis_data, 'institutional_patterns', 'ad_line_trend'),
                    "buying_pressure": safe_get(analysis_data, 'smart_money_indicators', 'buying_pressure_ratio'),
                    "selling_pressure": safe_get(analysis_data, 'smart_money_indicators', 'selling_pressure_ratio'),
                    "quiet_accumulation_score": safe_get(analysis_data, 'smart_money_indicators', 'quiet_accumulation_score'),
                    "institutional_activity_level": safe_get(analysis_data, 'institutional_analysis', 'institutional_activity_level')
                }
                f.write(json.dumps(volume_summary, indent=2, default=str))
                f.write("\n\n")
                f.write("FINAL PROMPT SENT TO LLM:\n")
                f.write("-" * 40 + "\n")
                f.write(prompt)
            
            # Generate chart if chart generator is available
            chart_bytes = None
            chart_path = None
            if self.chart_generator:
                try:
                    print(f"📊 Generating institutional activity chart for {stock_config.symbol}...")
                    chart_path = os.path.join(results_dir, f"institutional_chart_{stock_config.symbol}_{timestamp}.png")
                    
                    # Prepare stock data with date as index for chart generation
                    stock_data_for_chart = stock_data.copy()
                    if 'date' in stock_data_for_chart.columns:
                        stock_data_for_chart = stock_data_for_chart.set_index('date')
                    
                    chart_bytes = self.chart_generator.generate_institutional_activity_chart(
                        stock_data_for_chart, analysis_data, stock_config.symbol, save_path=chart_path
                    )
                    if chart_bytes:
                        print(f"✅ Chart generated and saved: {chart_path}")
                    else:
                        print(f"❌ Chart generation failed for {stock_config.symbol}")
                except Exception as e:
                    print(f"❌ Chart generation error for {stock_config.symbol}: {e}")
                    chart_bytes = None
                    chart_path = None
            
            # Make API call if available
            llm_response = ""
            if self.llm_client:
                try:
                    print(f"🚀 Making institutional analysis API call for {stock_config.symbol}...")
                    
                    # If we have a chart, send it with the prompt
                    if chart_bytes and chart_path:
                        from PIL import Image
                        import io
                        
                        # Convert bytes to PIL Image
                        chart_image = Image.open(io.BytesIO(chart_bytes))
                        
                        # Enhanced prompt with chart instruction
                        enhanced_prompt = prompt + "\n\nIMAGE ANALYSIS INSTRUCTION:\nThe attached chart shows comprehensive institutional activity analysis including:\n1. Volume Profile with Point of Control (POC)\n2. Large Block Detection with institutional thresholds\n3. Accumulation/Distribution Analysis\n4. Smart Money Timing Analysis\n5. Analysis Summary and Key Metrics\n\nPlease analyze both the numerical data provided above AND the visual patterns in the chart to provide more comprehensive insights."
                        
                        response = await self.llm_client.generate_with_images(
                            prompt=enhanced_prompt,
                            images=[chart_image]
                        )
                        llm_response = response
                        print(f"✅ LLM call with chart completed for {stock_config.symbol}")
                    else:
                        # Fallback to text-only analysis with code execution
                        response = await self.llm_client.generate(
                            prompt=prompt,
                            enable_code_execution=True
                        )
                        llm_response = response
                        print(f"✅ LLM call (text-only) completed for {stock_config.symbol}")
                    
                    # Save response
                    response_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    response_file = os.path.join(results_dir, f"institutional_response_{stock_config.symbol}_{response_timestamp}.txt")
                    with open(response_file, 'w') as f:
                        f.write("INSTITUTIONAL ACTIVITY ANALYSIS RESULTS\n")
                        f.write("=" * 80 + "\n\n")
                        f.write(f"Stock Symbol: {stock_config.symbol}\n")
                        f.write(f"Company: {stock_config.name}\n")
                        f.write(f"Sector: {stock_config.sector}\n")
                        f.write(f"Response Time: {datetime.now().isoformat()}\n")
                        f.write(f"Response Length: {len(llm_response) if llm_response else 0} characters\n")
                        f.write(f"Chart Generated: {'Yes' if chart_bytes else 'No'}\n")
                        if chart_path:
                            f.write(f"Chart Path: {chart_path}\n")
                        f.write(f"Analysis Mode: {'Visual + Text' if chart_bytes else 'Text Only'}\n\n")
                        
                        f.write("COMPLETE LLM RESPONSE:\n")
                        f.write("-" * 40 + "\n")
                        f.write(llm_response or "No response received")
                        f.write("\n")
                    
                except Exception as e:
                    print(f"❌ API call failed for {stock_config.symbol}: {e}")
                    llm_response = f"API_ERROR: {str(e)}"
            
            execution_time = time.time() - start_time
            
            # Evaluate quality of institutional analysis
            quality_metrics = self._evaluate_institutional_analysis(stock_config, analysis_data, llm_response)
            
            return {
                'stock_config': stock_config,
                'success': True,
                'execution_time': execution_time,
                'quality_score': quality_metrics['overall_score'],
                'data_quality': quality_metrics['data_quality'],
                'response_length': len(llm_response) if llm_response else 0,
                'chart_generated': bool(chart_bytes),
                'chart_path': chart_path,
                'analysis_mode': 'Visual + Text' if chart_bytes else 'Text Only',
                'institutional_metrics': {
                    'activity_level': safe_get(analysis_data, 'institutional_analysis', 'institutional_activity_level'),
                    'primary_activity': safe_get(analysis_data, 'institutional_analysis', 'primary_activity'),
                    'volume_ratio': safe_get(analysis_data, 'volume_data', 'volume_ratio'),
                    'large_volume_days': safe_get(analysis_data, 'large_volume_analysis', 'large_volume_days'),
                    'obv_trend': safe_get(analysis_data, 'institutional_patterns', 'obv_trend'),
                    'buying_pressure': safe_get(analysis_data, 'smart_money_indicators', 'buying_pressure_ratio'),
                    'quiet_accumulation': safe_get(analysis_data, 'smart_money_indicators', 'quiet_accumulation_score')
                },
                'quality_metrics': quality_metrics,
                'has_llm_response': bool(llm_response and not llm_response.startswith("API_ERROR"))
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            return self._create_error_result(stock_config, str(e), execution_time)
    
    def _prepare_institutional_context(self, stock_config: StockTestConfig, analysis_data: Dict[str, Any]) -> str:
        """Prepare context string for institutional activity analysis"""
        def safe_get(data, *keys):
            try:
                result = data
                for key in keys:
                    result = result[key]
                return result
            except (KeyError, TypeError, AttributeError):
                return "N/A"
        
        # Build streamlined context focused on institutional activity
        # Get institutional analysis data first
        inst_analysis = safe_get(analysis_data, 'institutional_analysis')
        
        context = f"""Stock: {stock_config.symbol} ({stock_config.name})
Sector: {stock_config.sector}
Analysis Period: 1 Year Daily Data

INSTITUTIONAL ACTIVITY PATTERNS:
OBV Trend: {safe_get(analysis_data, 'institutional_patterns', 'obv_trend')}
A/D Line Trend: {safe_get(analysis_data, 'institutional_patterns', 'ad_line_trend')}
Price-Volume Correlation: {safe_get(analysis_data, 'institutional_patterns', 'price_volume_correlation'):.3f}

SMART MONEY INDICATORS:
Buying Pressure Ratio: {safe_get(analysis_data, 'smart_money_indicators', 'buying_pressure_ratio'):.1%}
Selling Pressure Ratio: {safe_get(analysis_data, 'smart_money_indicators', 'selling_pressure_ratio'):.1%}
Quiet Accumulation Score: {safe_get(analysis_data, 'smart_money_indicators', 'quiet_accumulation_score')}
Distribution Score: {safe_get(analysis_data, 'smart_money_indicators', 'distribution_score')}

LARGE BLOCK ANALYSIS:
Institutional Blocks Detected: {safe_get(inst_analysis, 'large_block_analysis', 'institutional_block_count') if inst_analysis else 'N/A'}
Total Large Blocks: {safe_get(inst_analysis, 'large_block_analysis', 'total_large_blocks') if inst_analysis else 'N/A'}
Activity Level: {safe_get(inst_analysis, 'institutional_activity_level') if inst_analysis else 'N/A'}
Primary Activity: {safe_get(inst_analysis, 'primary_activity') if inst_analysis else 'N/A'}"""

        # Add institutional processor results if available
        inst_analysis = safe_get(analysis_data, 'institutional_analysis')
        if inst_analysis and inst_analysis != "N/A" and 'error' not in inst_analysis:
            context += f"""

ADVANCED INSTITUTIONAL ANALYSIS:
Activity Level: {safe_get(inst_analysis, 'institutional_activity_level')}
Primary Activity: {safe_get(inst_analysis, 'primary_activity')}
Activity Confidence: {safe_get(inst_analysis, 'activity_confidence')}"""

        return context
    
    def _evaluate_institutional_analysis(self, stock_config: StockTestConfig, analysis_data: Dict[str, Any], llm_response: str) -> Dict[str, Any]:
        """Evaluate the quality of institutional activity analysis"""
        metrics = {
            'data_quality': 'unknown',
            'volume_completeness': 0,
            'response_quality': 0,
            'institutional_insights': 0,
            'overall_score': 0
        }
        
        def safe_get(data, *keys):
            try:
                result = data
                for key in keys:
                    result = result[key]
                return result
            except (KeyError, TypeError, AttributeError):
                return None
        
        # Evaluate volume data completeness
        volume_metrics = ['volume_data', 'large_volume_analysis', 'institutional_patterns', 'smart_money_indicators']
        available_metrics = 0
        for metric in volume_metrics:
            if metric in analysis_data and analysis_data[metric]:
                available_metrics += 1
        
        metrics['volume_completeness'] = (available_metrics / len(volume_metrics)) * 100
        
        # Assess data quality based on volume analysis depth
        current_volume = safe_get(analysis_data, 'volume_data', 'current_volume')
        volume_ratio = safe_get(analysis_data, 'volume_data', 'volume_ratio')
        large_volume_days = safe_get(analysis_data, 'large_volume_analysis', 'large_volume_days')
        
        if all(x is not None for x in [current_volume, volume_ratio, large_volume_days]):
            metrics['data_quality'] = 'excellent'
            metrics['volume_completeness'] += 20
        else:
            metrics['data_quality'] = 'good'
        
        # Evaluate response quality for institutional insights
        if llm_response and not llm_response.startswith("API_ERROR"):
            response_lower = llm_response.lower()
            
            # Check for institutional analysis keywords
            institutional_keywords = [
                'institutional', 'smart money', 'accumulation', 'distribution',
                'volume profile', 'large blocks', 'institutional activity',
                'buying pressure', 'selling pressure', 'institutional sentiment'
            ]
            
            keyword_score = sum(1 for keyword in institutional_keywords if keyword in response_lower)
            metrics['response_quality'] = min(100, keyword_score * 10)
            
            # Check for JSON structure specific to institutional analysis
            if all(key in response_lower for key in ['institutional_activity_level', 'primary_activity', 'smart_money_timing']):
                metrics['response_quality'] += 30
            
            # Evaluate institutional insights depth
            insight_indicators = [
                'quiet accumulation', 'volume on dips', 'institutional zones',
                'activity clusters', 'predictive indicators', 'confidence score'
            ]
            
            insight_score = sum(1 for indicator in insight_indicators if indicator in response_lower)
            metrics['institutional_insights'] = min(100, insight_score * 15)
        
        # Calculate overall score with institutional focus
        metrics['overall_score'] = min(100, (
            metrics['volume_completeness'] * 0.3 +
            metrics['response_quality'] * 0.4 +
            metrics['institutional_insights'] * 0.3
        ))
        
        return metrics
    
    def _create_error_result(self, stock_config: StockTestConfig, error_message: str, execution_time: float = 0) -> Dict[str, Any]:
        """Create error result dictionary"""
        return {
            'stock_config': stock_config,
            'success': False,
            'error': error_message,
            'execution_time': execution_time,
            'quality_score': 0,
            'data_quality': 'failed'
        }
    
    def _generate_institutional_report(self, results_dir: str):
        """Generate comprehensive multi-stock institutional analysis report"""
        # Prepare summary data
        summary_data = []
        successful_tests = [r for r in self.results if r['success']]
        
        for result in self.results:
            summary_data.append({
                'Symbol': result['stock_config'].symbol,
                'Company': result['stock_config'].name,
                'Sector': result['stock_config'].sector,
                'Expected Pattern': result['stock_config'].expected_behavior,
                'Success': result['success'],
                'Quality Score': result['quality_score'],
                'Execution Time (s)': result['execution_time'],
                'Data Quality': result.get('data_quality', 'unknown'),
                'Has LLM Response': result.get('has_llm_response', False),
                'Chart Generated': result.get('chart_generated', False),
                'Analysis Mode': result.get('analysis_mode', 'Unknown'),
                'Activity Level': result.get('institutional_metrics', {}).get('activity_level', 'N/A'),
                'Primary Activity': result.get('institutional_metrics', {}).get('primary_activity', 'N/A'),
                'Volume Ratio': result.get('institutional_metrics', {}).get('volume_ratio', 'N/A'),
                'Large Volume Days': result.get('institutional_metrics', {}).get('large_volume_days', 'N/A'),
                'OBV Trend': result.get('institutional_metrics', {}).get('obv_trend', 'N/A'),
                'Buying Pressure': result.get('institutional_metrics', {}).get('buying_pressure', 'N/A')
            })
        
        # Save to Excel
        summary_df = pd.DataFrame(summary_data)
        excel_path = os.path.join(results_dir, "institutional_activity_summary.xlsx")
        summary_df.to_excel(excel_path, index=False)
        
        # Generate detailed text report
        report_path = os.path.join(results_dir, "institutional_comprehensive_report.txt")
        with open(report_path, 'w') as f:
            f.write("MULTI-STOCK INSTITUTIONAL ACTIVITY TESTING REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Stocks Tested: {len(self.results)}\n")
            f.write(f"Successful Tests: {len(successful_tests)}\n")
            f.write(f"Success Rate: {len(successful_tests)/len(self.results)*100:.1f}%\n")
            
            # Chart generation statistics
            charts_generated = len([r for r in successful_tests if r.get('chart_generated', False)])
            f.write(f"Charts Generated: {charts_generated}/{len(successful_tests)}\n")
            f.write(f"Chart Success Rate: {charts_generated/len(successful_tests)*100:.1f}% (of successful tests)\n")
            visual_analysis_count = len([r for r in successful_tests if r.get('analysis_mode') == 'Visual + Text'])
            f.write(f"Visual Analysis Mode: {visual_analysis_count}/{len(successful_tests)}\n\n")
            
            # Overall statistics
            if successful_tests:
                avg_quality = sum(r['quality_score'] for r in successful_tests) / len(successful_tests)
                avg_execution = sum(r['execution_time'] for r in successful_tests) / len(successful_tests)
                
                f.write("OVERALL STATISTICS\n")
                f.write("-" * 40 + "\n")
                f.write(f"Average Quality Score: {avg_quality:.1f}/100\n")
                f.write(f"Average Execution Time: {avg_execution:.1f}s\n\n")
            
            # Sector-wise institutional activity analysis
            f.write("SECTOR-WISE INSTITUTIONAL ANALYSIS\n")
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
                
                # Sector-specific institutional insights
                high_activity_stocks = [r for r in sector_results if r.get('institutional_metrics', {}).get('activity_level') == 'high']
                if high_activity_stocks:
                    f.write(f"  High Activity Stocks: {', '.join(r['stock_config'].symbol for r in high_activity_stocks)}\n")
            
            # Individual stock details with institutional focus
            f.write("\n\nINDIVIDUAL INSTITUTIONAL ANALYSIS\n")
            f.write("-" * 40 + "\n")
            for result in self.results:
                f.write(f"\n{result['stock_config'].symbol} ({result['stock_config'].name}):\n")
                f.write(f"  Sector: {result['stock_config'].sector}\n")
                f.write(f"  Expected Pattern: {result['stock_config'].expected_behavior}\n")
                f.write(f"  Success: {'✅' if result['success'] else '❌'}\n")
                if result['success']:
                    f.write(f"  Quality Score: {result['quality_score']:.1f}/100\n")
                    f.write(f"  Execution Time: {result['execution_time']:.1f}s\n")
                    f.write(f"  Data Quality: {result.get('data_quality', 'unknown')}\n")
                    f.write(f"  Chart Generated: {'Yes' if result.get('chart_generated') else 'No'}\n")
                    f.write(f"  Analysis Mode: {result.get('analysis_mode', 'Unknown')}\n")
                    if result.get('chart_path'):
                        f.write(f"  Chart Path: {result['chart_path']}\n")
                    if 'institutional_metrics' in result:
                        im = result['institutional_metrics']
                        f.write(f"  Activity Level: {im.get('activity_level', 'N/A')}\n")
                        f.write(f"  Primary Activity: {im.get('primary_activity', 'N/A')}\n")
                        f.write(f"  Volume Ratio: {im.get('volume_ratio', 'N/A'):.2f}x\n")
                        f.write(f"  Large Volume Days: {im.get('large_volume_days', 'N/A')}\n")
                        f.write(f"  OBV Trend: {im.get('obv_trend', 'N/A')}\n")
                        if im.get('buying_pressure'):
                            f.write(f"  Buying Pressure: {im['buying_pressure']:.1%}\n")
                else:
                    f.write(f"  Error: {result.get('error', 'Unknown error')}\n")
        
        print(f"📊 Institutional activity report saved to: {report_path}")
        print(f"📈 Summary data saved to: {excel_path}")

async def main():
    """Main function"""
    print("🏛️ Multi-Stock Institutional Activity Testing Framework")
    print("Testing institutional_activity_analysis across multiple stocks from different sectors")
    
    tester = MultiStockInstitutionalTester()
    success = await tester.run_multi_stock_tests()
    
    if success:
        print("\n🎉 Multi-stock institutional activity testing completed successfully!")
    else:
        print("\n❌ Multi-stock institutional activity testing failed")

if __name__ == "__main__":
    asyncio.run(main())