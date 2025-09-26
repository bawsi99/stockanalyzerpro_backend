#!/usr/bin/env python3
"""
Multi-Stock Support/Resistance Prompt Testing Framework

Tests the volume_support_resistance prompt across multiple stocks from different sectors
to validate consistency and quality of support/resistance analysis.

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
from PIL import Image

# Add backend to path
backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../backend'))
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
sys.path.insert(0, root_path)  # Add root path for 'backend' imports
sys.path.insert(0, backend_path)  # Add backend path for direct imports

try:
    from gemini.gemini_client import GeminiClient
    from gemini.prompt_manager import PromptManager
    from gemini.context_engineer import ContextEngineer, AnalysisType
    from zerodha.client import ZerodhaDataClient
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Make sure you're running this from the correct directory")
    print(f"Root path: {root_path}")
    print(f"Backend path: {backend_path}")
    sys.exit(1)

# Import Support/Resistance agent and charts from package
try:
    from backend.agents.volume.support_resistance.agent import SupportResistanceAgent
    from backend.agents.volume.support_resistance.charts import SupportResistanceCharts
except ImportError as e:
    print(f"❌ Import Error for Support/Resistance modules: {e}")
    print("Make sure you're running this from the project root so 'backend' is importable.")
    print(f"Root path: {root_path}")
    print(f"Backend path: {backend_path}")
    sys.exit(1)

class StockTestConfig:
    """Configuration for individual stock tests"""
    def __init__(self, symbol: str, name: str, sector: str, expected_behavior: str):
        self.symbol = symbol
        self.name = name
        self.sector = sector
        self.expected_behavior = expected_behavior

class SupportResistanceMultiStockTester:
    """Test volume_support_resistance prompt across multiple stocks"""
    
    def __init__(self):
        # Initialize Zerodha client
        try:
            self.zerodha_client = ZerodhaDataClient()
            print("✅ Zerodha client initialized")
        except Exception as e:
            print(f"❌ Cannot initialize Zerodha client: {e}")
            sys.exit(1)
        
        # Initialize support resistance agent
        try:
            self.sr_agent = SupportResistanceAgent()
            print("✅ Support/Resistance agent initialized")
        except Exception as e:
            print(f"❌ Cannot initialize Support/Resistance agent: {e}")
            sys.exit(1)
        
        # Initialize other components
        self.prompt_manager = PromptManager()
        self.context_engineer = ContextEngineer()
        self.chart_maker = SupportResistanceCharts()
        
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
        
        # Define test stocks focusing on those with strong support/resistance patterns
        self.test_stocks = [
            StockTestConfig("RELIANCE", "Reliance Industries", "Energy/Petrochemicals", "strong_levels_large_cap"),
            StockTestConfig("TCS", "Tata Consultancy Services", "IT Services", "range_bound_strong_levels"),
            StockTestConfig("HDFCBANK", "HDFC Bank", "Banking", "institutional_levels"),
            StockTestConfig("ICICIBANK", "ICICI Bank", "Banking", "volatile_with_levels"),
            StockTestConfig("ITC", "ITC Limited", "FMCG/Tobacco", "stable_range_bound"),
            # Additional stocks known for clear support/resistance
            StockTestConfig("HINDUNILVR", "Hindustan Unilever", "FMCG", "defensive_clear_levels"),
            StockTestConfig("MARUTI", "Maruti Suzuki", "Automotive", "cyclical_with_levels")
        ]
        
        self.results = []
    
    async def run_multi_stock_tests(self):
        """Run tests across all configured stocks"""
        print(f"🚀 Starting Multi-Stock Support/Resistance Prompt Testing")
        print(f"Testing {len(self.test_stocks)} stocks with 365 days of data")
        print("=" * 80)
        
        # Authenticate with Zerodha first
        print("🔗 Authenticating with Zerodha...")
        if not self.zerodha_client.authenticate():
            print("❌ Zerodha authentication failed")
            return False
        
        print("✅ Zerodha authentication successful")
        
        # Create results directory
        results_dir = "support_resistance_test_results"
        os.makedirs(results_dir, exist_ok=True)
        
        # Create async tasks for all stocks to run them concurrently
        async def test_single_stock(stock_config, stock_index):
            """Test a single stock asynchronously"""
            print(f"\n📊 Testing Stock {stock_index}/{len(self.test_stocks)}: {stock_config.symbol}")
            print(f"   Company: {stock_config.name}")
            print(f"   Sector: {stock_config.sector}")
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
                
                # Ensure proper data structure (same as working institutional test)
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
                print(f"   Price range: ₹{stock_data['close'].min():.2f} to ₹{stock_data['close'].max():.2f}")
                
                # Run support/resistance analysis using the agent
                print("📊 Running Support/Resistance analysis...")
                
                # Set date as index for the agent (same approach as institutional test)
                stock_data_indexed = stock_data.set_index('date')
                
                # Create a fresh agent instance for each test to avoid state sharing issues
                sr_agent = SupportResistanceAgent()
                sr_analysis = sr_agent.analyze(stock_data_indexed, stock_config.symbol)
                
                if 'error' in sr_analysis:
                    print(f"❌ Support/Resistance analysis failed: {sr_analysis['error']}")
                    return self._create_error_result(stock_config, f"SR Analysis failed: {sr_analysis['error']}")
                
                # Generate charts for the analysis
                print("📈 Generating analysis charts...")
                chart_paths = await self._generate_analysis_charts(stock_config, stock_data_indexed, sr_analysis, results_dir)
                
                # Test the prompt with charts
                result = await self._test_stock_prompt(stock_config, sr_analysis, results_dir, chart_paths)
                
                print(f"✅ Test completed for {stock_config.symbol}")
                print(f"   Success: {result['success']}")
                print(f"   Quality Score: {result['quality_score']:.1f}/100")
                print(f"   Response Time: {result['execution_time']:.1f}s")
                
                return result
                
            except Exception as e:
                print(f"❌ Error testing {stock_config.symbol}: {e}")
                import traceback
                traceback.print_exc()
                return self._create_error_result(stock_config, str(e))
        
        # Run all stock tests with a semaphore to limit concurrency (reduced to prevent race conditions)
        semaphore = asyncio.Semaphore(1)  # Limit to 1 concurrent test to prevent race conditions
        
        async def test_with_semaphore(stock_config, index):
            async with semaphore:
                return await test_single_stock(stock_config, index)
        
        # Create tasks for all stocks
        tasks = [
            test_with_semaphore(stock_config, i + 1)
            for i, stock_config in enumerate(self.test_stocks)
        ]
        
        # Wait for all tasks to complete
        print(f"\n🔄 Running {len(tasks)} stock tests sequentially (1 at a time to prevent race conditions)...")
        results = await asyncio.gather(*tasks)
        self.results.extend(results)
        
        # Generate comprehensive report
        self._generate_multi_stock_report(results_dir)
        
        print(f"\n✅ Multi-stock testing completed!")
        print(f"📁 Results saved to: {results_dir}/")
        
        return True
    
    async def _generate_analysis_charts(self, stock_config: StockTestConfig, stock_data: pd.DataFrame, 
                                       sr_analysis: Dict[str, Any], results_dir: str) -> Dict[str, str]:
        """Generate analysis charts and return their file paths"""
        try:
            chart_paths = {}
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Create comprehensive chart (includes price, levels, volume profile, and summary)
            comprehensive_path = os.path.join(results_dir, f"{stock_config.symbol}_comprehensive_{timestamp}.png")
            print(f"   Creating comprehensive chart: {comprehensive_path}")
            fig1 = self.chart_maker.create_comprehensive_chart(
                stock_data, sr_analysis, symbol=stock_config.symbol, save_path=comprehensive_path
            )
            chart_paths['comprehensive'] = comprehensive_path
            
            # Create strength analysis chart (unique analytical breakdown)
            strength_path = os.path.join(results_dir, f"{stock_config.symbol}_strength_{timestamp}.png")
            print(f"   Creating strength chart: {strength_path}")
            fig2 = self.chart_maker.create_levels_strength_chart(
                sr_analysis, save_path=strength_path
            )
            chart_paths['strength'] = strength_path
            
            # Close figures to free memory
            import matplotlib.pyplot as plt
            plt.close(fig1)
            plt.close(fig2)
            
            print(f"✅ Generated {len(chart_paths)} charts for {stock_config.symbol}")
            return chart_paths
            
        except Exception as e:
            print(f"❌ Chart generation failed for {stock_config.symbol}: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    async def _test_stock_prompt(self, stock_config: StockTestConfig, sr_analysis: Dict[str, Any], 
                               results_dir: str, chart_paths: Dict[str, str] = None) -> Dict[str, Any]:
        """Test the volume_support_resistance prompt for a single stock"""
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
            # Format the context for the volume_support_resistance prompt
            # Extract key data from the support resistance analysis
            support_levels = sr_analysis.get('support_levels', [])
            resistance_levels = sr_analysis.get('resistance_levels', [])
            current_position = sr_analysis.get('current_position', {})
            trading_implications = sr_analysis.get('trading_implications', {})
            
            # Create context in the format expected by the volume_support_resistance prompt
            context_data = {
                'symbol': stock_config.symbol,
                'company_name': stock_config.name,
                'sector': stock_config.sector,
                'current_price': safe_get(current_position, 'current_price'),
                'support_levels': support_levels,
                'resistance_levels': resistance_levels,
                'volume_analysis': sr_analysis.get('detailed_analysis', {}),
                'trading_implications': trading_implications,
                'analysis_quality': sr_analysis.get('quality_assessment', {})
            }
            
            # Create formatted context string
            context = self._format_sr_context(context_data)
            
            # Add chart information to context if available
            if chart_paths:
                context += "\n\nCHART ANALYSIS:\n"
                context += "The following optimized charts have been generated for visual analysis:\n"
                for chart_type, path in chart_paths.items():
                    if chart_type == 'comprehensive':
                        context += f"- {chart_type.title()} Chart: {os.path.basename(path)} (Price + S/R Levels + Volume Profile + Analysis Summary)\n"
                    elif chart_type == 'strength':
                        context += f"- {chart_type.title()} Chart: {os.path.basename(path)} (Level Strength Scores + Component Breakdown)\n"
                    else:
                        context += f"- {chart_type.title()} Chart: {os.path.basename(path)}\n"
                context += "Please analyze these charts along with the numerical data provided.\n"
            
            # Format the final prompt
            prompt = self.prompt_manager.format_prompt(
                "volume_support_resistance",
                context=context
            )
            prompt += self.prompt_manager.SOLVING_LINE
            
            # Add chart analysis instruction to prompt
            if chart_paths:
                prompt += "\n\nIMPORTANT: Analyze the provided charts to validate and enhance your support/resistance analysis:\n"
                prompt += "- Comprehensive Chart: Use for visual validation of price action, S/R levels, volume profile, and overall market structure\n"
                prompt += "- Strength Chart: Use to understand the relative reliability and scoring methodology of each identified level\n"
                prompt += "Compare visual patterns with numerical data and identify any discrepancies between calculated levels and chart formations."
            
            # Save prompt details with timestamp format
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            prompt_file = os.path.join(results_dir, f"prompt_sr_analysis_{stock_config.symbol}_{timestamp}.txt")
            with open(prompt_file, 'w') as f:
                f.write("SUPPORT/RESISTANCE PROMPT ANALYSIS FOR LLM\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"Stock Symbol: {stock_config.symbol}\n")
                f.write(f"Company: {stock_config.name}\n")
                f.write(f"Sector: {stock_config.sector}\n")
                f.write(f"Expected Behavior: {stock_config.expected_behavior}\n")
                f.write(f"Generated: {datetime.now().isoformat()}\n")
                f.write(f"Prompt Length: {len(prompt)} characters\n")
                f.write(f"Context Length: {len(context)} characters\n")
                
                # Add chart information
                if chart_paths:
                    f.write(f"Charts Generated: {len(chart_paths)}\n")
                    for chart_type, path in chart_paths.items():
                        f.write(f"  - {chart_type.title()}: {os.path.basename(path)}\n")
                f.write("\n")
                f.write("KEY SUPPORT/RESISTANCE SUMMARY:\n")
                f.write("-" * 40 + "\n")
                
                # Extract key metrics for debugging
                key_metrics = {
                    "current_price": safe_get(current_position, 'current_price'),
                    "support_levels_count": len(support_levels),
                    "resistance_levels_count": len(resistance_levels),
                    "nearest_support": safe_get(current_position, 'nearest_support'),
                    "nearest_resistance": safe_get(current_position, 'nearest_resistance'),
                    "range_position": safe_get(current_position, 'range_position_classification'),
                    "analysis_quality_score": safe_get(sr_analysis, 'analysis_summary', 'analysis_quality_score'),
                    "total_validated_levels": safe_get(sr_analysis, 'analysis_summary', 'total_validated_levels'),
                    "strongest_support": safe_get(sr_analysis, 'analysis_summary', 'strongest_support_price'),
                    "strongest_resistance": safe_get(sr_analysis, 'analysis_summary', 'strongest_resistance_price')
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
                    
                    # If charts are available, send them with the prompt
                    if chart_paths:
                        print(f"📊 Including {len(chart_paths)} charts in API call...")
                        
                        # Prepare images for the API call
                        image_objects = []
                        for chart_type, path in chart_paths.items():
                            if os.path.exists(path):
                                try:
                                    from PIL import Image
                                    img = Image.open(path)
                                    image_objects.append(img)
                                    print(f"   Including {chart_type} chart: {os.path.basename(path)}")
                                except Exception as img_error:
                                    print(f"   ⚠️ Failed to load {chart_type} chart: {img_error}")
                        
                        if image_objects:
                            # Call LLM with images and code execution
                            response = await self.gemini_client.core.call_llm_with_images(
                                prompt, image_objects, enable_code_execution=True
                            )
                            # For consistency with code execution format, we'll format as tuple
                            code_results = []
                            execution_results = []
                        else:
                            print("⚠️ No valid chart files found, proceeding without images")
                            response, code_results, execution_results = await self.gemini_client.core.call_llm_with_code_execution(prompt)
                    else:
                        response, code_results, execution_results = await self.gemini_client.core.call_llm_with_code_execution(prompt)
                    
                    llm_response = response
                    
                    # Save response with timestamp format
                    response_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    response_file = os.path.join(results_dir, f"response_sr_analysis_{stock_config.symbol}_{response_timestamp}.txt")
                    with open(response_file, 'w') as f:
                        f.write("LLM SUPPORT/RESISTANCE ANALYSIS RESULTS\n")
                        f.write("=" * 80 + "\n\n")
                        f.write(f"Stock Symbol: {stock_config.symbol}\n")
                        f.write(f"Company: {stock_config.name}\n")
                        f.write(f"Sector: {stock_config.sector}\n")
                        f.write(f"Response Time: {datetime.now().isoformat()}\n")
                        f.write(f"Response Length: {len(response) if response else 0} characters\n")
                        if code_results:
                            f.write(f"Mathematical Calculations: {len(code_results)} code snippets executed\n")
                        if execution_results:
                            f.write(f"Calculation Results: {len(execution_results)} computational outputs\n")
                        
                        # Add chart information to response file
                        if chart_paths:
                            f.write(f"Charts Analyzed: {len(chart_paths)}\n")
                            for chart_type, path in chart_paths.items():
                                f.write(f"  - {chart_type.title()}: {os.path.basename(path)}\n")
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
            quality_metrics = self._evaluate_sr_analysis(stock_config, sr_analysis, llm_response)
            
            return {
                'stock_config': stock_config,
                'success': True,
                'execution_time': execution_time,
                'quality_score': quality_metrics['overall_score'],
                'data_quality': quality_metrics['data_quality'],
                'response_length': len(llm_response) if llm_response else 0,
                'support_resistance_metrics': {
                    'support_levels_found': len(support_levels),
                    'resistance_levels_found': len(resistance_levels),
                    'current_price': safe_get(current_position, 'current_price'),
                    'range_position': safe_get(current_position, 'range_position_classification'),
                    'analysis_quality_score': safe_get(sr_analysis, 'analysis_summary', 'analysis_quality_score'),
                    'total_validated_levels': safe_get(sr_analysis, 'analysis_summary', 'total_validated_levels')
                },
                'quality_metrics': quality_metrics,
                'has_llm_response': bool(llm_response and not llm_response.startswith("API_ERROR")),
                'charts_generated': len(chart_paths) if chart_paths else 0,
                'chart_paths': chart_paths if chart_paths else {}
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            return self._create_error_result(stock_config, str(e), execution_time)
    
    def _format_sr_context(self, context_data: Dict[str, Any]) -> str:
        """Format essential context data for trading-focused support/resistance validation"""
        context_lines = []
        
        # Basic stock information
        context_lines.append(f"Stock: {context_data['symbol']} - {context_data['company_name']}")
        context_lines.append(f"Sector: {context_data['sector']}")
        current_price = context_data['current_price'] or 0
        context_lines.append(f"Current Price: ₹{current_price:.2f}")
        context_lines.append("")
        
        # Top 3 Support levels only (most relevant for trading)
        support_levels = context_data.get('support_levels', [])
        if support_levels:
            context_lines.append("TOP SUPPORT LEVELS (Pre-calculated & Validated):")
            for i, level in enumerate(support_levels[:3], 1):  # Only top 3
                price = level.get('price_level', 0) or 0
                reliability = level.get('reliability', 'unknown') or 'unknown'
                success_rate = level.get('success_rate', 0) or 0
                distance_pct = abs(current_price - price) / current_price * 100 if current_price > 0 else 0
                context_lines.append(f"  {i}. ₹{price:.2f} - {reliability} reliability ({success_rate:.1%} success) - {distance_pct:.1f}% away")
            context_lines.append("")
        
        # Top 3 Resistance levels only
        resistance_levels = context_data.get('resistance_levels', [])
        if resistance_levels:
            context_lines.append("TOP RESISTANCE LEVELS (Pre-calculated & Validated):")
            for i, level in enumerate(resistance_levels[:3], 1):  # Only top 3
                price = level.get('price_level', 0) or 0
                reliability = level.get('reliability', 'unknown') or 'unknown'
                success_rate = level.get('success_rate', 0) or 0
                distance_pct = abs(price - current_price) / current_price * 100 if current_price > 0 else 0
                context_lines.append(f"  {i}. ₹{price:.2f} - {reliability} reliability ({success_rate:.1%} success) - {distance_pct:.1f}% away")
            context_lines.append("")
        
        # Current Position Context (essential for trading decisions)
        context_lines.append("CURRENT POSITION CONTEXT:")
        
        # Find nearest levels
        all_levels = []
        for level in support_levels[:3]:
            price = level.get('price_level', 0)
            if price < current_price:
                distance = current_price - price
                all_levels.append((price, 'support', distance, level.get('reliability', 'unknown')))
        
        for level in resistance_levels[:3]:
            price = level.get('price_level', 0)
            if price > current_price:
                distance = price - current_price
                all_levels.append((price, 'resistance', distance, level.get('reliability', 'unknown')))
        
        # Sort by distance to find nearest levels
        all_levels.sort(key=lambda x: x[2])
        
        if all_levels:
            nearest_level = all_levels[0]
            context_lines.append(f"  Nearest Key Level: ₹{nearest_level[0]:.2f} ({nearest_level[1]}) - {nearest_level[3]} reliability")
            context_lines.append(f"  Distance to Nearest Level: {(nearest_level[2]/current_price*100):.1f}%")
        
        # Risk/Reward context (crucial for trading)
        trading_implications = context_data.get('trading_implications', {})
        if trading_implications:
            ratio = trading_implications.get('risk_reward_ratio')
            if ratio is not None and ratio > 0:
                context_lines.append(f"  Current Risk/Reward Ratio: {ratio:.2f}:1")
            
            strategy = trading_implications.get('trading_strategy', '')
            if strategy:
                strategy_readable = strategy.replace('_', ' ').title()
                context_lines.append(f"  Suggested Strategy: {strategy_readable}")
        
        context_lines.append("")
        
        # Analysis confidence (helps with validation)
        analysis_quality = context_data.get('analysis_quality', {})
        quality_score = analysis_quality.get('overall_score', 0)
        if quality_score > 0:
            context_lines.append(f"Pre-Analysis Confidence: {quality_score}/100")
        
        return "\n".join(context_lines)
    
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
    
    def _evaluate_sr_analysis(self, stock_config: StockTestConfig, sr_analysis: Dict[str, Any], llm_response: str) -> Dict[str, Any]:
        """Evaluate the quality of support/resistance analysis"""
        metrics = {
            'data_quality': 'unknown',
            'level_completeness': 0,
            'response_quality': 0,
            'sector_appropriateness': 0,
            'overall_score': 0
        }
        
        # Handle error cases first
        if 'error' in sr_analysis:
            metrics['data_quality'] = 'failed'
            metrics['overall_score'] = 0
            return metrics
        
        # Evaluate level completeness
        support_count = len(sr_analysis.get('support_levels', []))
        resistance_count = len(sr_analysis.get('resistance_levels', []))
        total_levels = support_count + resistance_count
        
        # Score based on number of levels found
        if total_levels >= 5:
            metrics['level_completeness'] = 90
        elif total_levels >= 3:
            metrics['level_completeness'] = 75
        elif total_levels >= 1:
            metrics['level_completeness'] = 50
        else:
            metrics['level_completeness'] = 20
        
        # Check analysis quality from the agent
        analysis_summary = sr_analysis.get('analysis_summary', {})
        quality_score = analysis_summary.get('analysis_quality_score', 0)
        
        if quality_score >= 80:
            metrics['data_quality'] = 'excellent'
            metrics['level_completeness'] += 10
        elif quality_score >= 60:
            metrics['data_quality'] = 'good'
            metrics['level_completeness'] += 5
        else:
            metrics['data_quality'] = 'moderate'
        
        # Evaluate response quality if available
        if llm_response and not llm_response.startswith("API_ERROR"):
            if len(llm_response) > 1500:  # Substantial response
                metrics['response_quality'] = 85
            elif len(llm_response) > 800:  # Moderate response
                metrics['response_quality'] = 70
            else:
                metrics['response_quality'] = 50
            
            # Check for JSON format (expected for volume_support_resistance prompt)
            if 'volume_based_support_levels' in llm_response and 'volume_based_resistance_levels' in llm_response:
                metrics['response_quality'] += 15
            elif '{' in llm_response and '}' in llm_response:
                metrics['response_quality'] += 10
        
        # Sector appropriateness (support/resistance analysis is universally applicable)
        metrics['sector_appropriateness'] = 80  # Default good score
        
        # Bonus for expected behavior match
        expected_behavior = stock_config.expected_behavior
        if 'range_bound' in expected_behavior and total_levels >= 4:
            metrics['sector_appropriateness'] += 10
        elif 'strong_levels' in expected_behavior and quality_score >= 70:
            metrics['sector_appropriateness'] += 10
        elif 'institutional' in expected_behavior and total_levels >= 3:
            metrics['sector_appropriateness'] += 5
        
        # Calculate overall score
        metrics['overall_score'] = min(100, (
            metrics['level_completeness'] * 0.4 +
            metrics['response_quality'] * 0.35 +
            metrics['sector_appropriateness'] * 0.25
        ))
        
        return metrics
    
    def _generate_multi_stock_report(self, results_dir: str):
        """Generate comprehensive multi-stock support/resistance analysis report"""
        # Prepare summary data
        summary_data = []
        successful_tests = [r for r in self.results if r['success']]
        
        for result in self.results:
            summary_data.append({
                'Symbol': result['stock_config'].symbol,
                'Company': result['stock_config'].name,
                'Sector': result['stock_config'].sector,
                'Success': result['success'],
                'Quality Score': result['quality_score'],
                'Execution Time (s)': result['execution_time'],
                'Data Quality': result.get('data_quality', 'unknown'),
                'Has LLM Response': result.get('has_llm_response', False),
                'Charts Generated': result.get('charts_generated', 0),
                'Support Levels Found': result.get('support_resistance_metrics', {}).get('support_levels_found', 'N/A'),
                'Resistance Levels Found': result.get('support_resistance_metrics', {}).get('resistance_levels_found', 'N/A'),
                'Total Validated Levels': result.get('support_resistance_metrics', {}).get('total_validated_levels', 'N/A'),
                'Range Position': result.get('support_resistance_metrics', {}).get('range_position', 'N/A'),
                'Analysis Quality Score': result.get('support_resistance_metrics', {}).get('analysis_quality_score', 'N/A')
            })
        
        # Save to Excel
        summary_df = pd.DataFrame(summary_data)
        excel_path = os.path.join(results_dir, "support_resistance_multi_stock_summary.xlsx")
        summary_df.to_excel(excel_path, index=False)
        
        # Generate detailed text report
        report_path = os.path.join(results_dir, "support_resistance_comprehensive_report.txt")
        with open(report_path, 'w') as f:
            f.write("MULTI-STOCK SUPPORT/RESISTANCE PROMPT TESTING REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Stocks Tested: {len(self.results)}\n")
            f.write(f"Successful Tests: {len(successful_tests)}\n")
            f.write(f"Success Rate: {len(successful_tests)/len(self.results)*100:.1f}%\n\n")
            
            # Overall statistics
            if successful_tests:
                avg_quality = sum(r['quality_score'] for r in successful_tests) / len(successful_tests)
                avg_execution = sum(r['execution_time'] for r in successful_tests) / len(successful_tests)
                
                f.write("OVERALL STATISTICS\n")
                f.write("-" * 40 + "\n")
                f.write(f"Average Quality Score: {avg_quality:.1f}/100\n")
                f.write(f"Average Execution Time: {avg_execution:.1f}s\n\n")
                
                # Support/Resistance specific statistics
                total_support_levels = sum(r.get('support_resistance_metrics', {}).get('support_levels_found', 0) for r in successful_tests if isinstance(r.get('support_resistance_metrics', {}).get('support_levels_found'), int))
                total_resistance_levels = sum(r.get('support_resistance_metrics', {}).get('resistance_levels_found', 0) for r in successful_tests if isinstance(r.get('support_resistance_metrics', {}).get('resistance_levels_found'), int))
                
                f.write("SUPPORT/RESISTANCE STATISTICS\n")
                f.write("-" * 40 + "\n")
                f.write(f"Total Support Levels Found: {total_support_levels}\n")
                f.write(f"Total Resistance Levels Found: {total_resistance_levels}\n")
                f.write(f"Average Support Levels per Stock: {total_support_levels/len(successful_tests):.1f}\n")
                f.write(f"Average Resistance Levels per Stock: {total_resistance_levels/len(successful_tests):.1f}\n\n")
            
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
                    f.write(f"  Charts Generated: {result.get('charts_generated', 0)}\n")
                    if result.get('chart_paths'):
                        f.write(f"  Chart Files:\n")
                        for chart_type, path in result['chart_paths'].items():
                            f.write(f"    - {chart_type.title()}: {os.path.basename(path)}\n")
                    if 'support_resistance_metrics' in result:
                        srm = result['support_resistance_metrics']
                        f.write(f"  Support Levels: {srm.get('support_levels_found', 'N/A')}\n")
                        f.write(f"  Resistance Levels: {srm.get('resistance_levels_found', 'N/A')}\n")
                        f.write(f"  Total Validated Levels: {srm.get('total_validated_levels', 'N/A')}\n")
                        f.write(f"  Range Position: {srm.get('range_position', 'N/A')}\n")
                        if srm.get('analysis_quality_score'):
                            f.write(f"  Analysis Quality: {srm['analysis_quality_score']:.1f}/100\n")
                else:
                    f.write(f"  Error: {result.get('error', 'Unknown error')}\n")
        
        print(f"📊 Multi-stock support/resistance report saved to: {report_path}")
        print(f"📈 Summary data saved to: {excel_path}")

async def main():
    """Main function"""
    print("🔍 Multi-Stock Support/Resistance Prompt Testing Framework")
    print("Testing volume_support_resistance across multiple stocks from different sectors")
    
    tester = SupportResistanceMultiStockTester()
    success = await tester.run_multi_stock_tests()
    
    if success:
        print("\n🎉 Multi-stock support/resistance testing completed successfully!")
    else:
        print("\n❌ Multi-stock support/resistance testing failed")

if __name__ == "__main__":
    # Run the async multi-stock test
    asyncio.run(main())
