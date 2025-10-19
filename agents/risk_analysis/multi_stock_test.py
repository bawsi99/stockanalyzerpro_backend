#!/usr/bin/env python3
"""
Multi-Stock Risk Analysis Testing Framework

Tests the risk analysis system (orchestrator + LLM agent) across multiple stocks from different sectors
to validate consistency and quality of comprehensive risk assessment.

Usage: python multi_stock_test.py
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import asyncio
import time
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple

# Add paths relative to the script location
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, '..', '..', '..')
backend_path = os.path.join(project_root, 'backend')

# Add paths to Python path
sys.path.insert(0, project_root)
sys.path.insert(0, backend_path)

try:
    from backend.zerodha.client import ZerodhaDataClient
from backend.analysis.technical_indicators import TechnicalIndicators
    from backend.agents.risk_analysis.quantitative_risk.processor import QuantitativeRiskProcessor
    from backend.agents.risk_analysis.risk_llm_agent import risk_llm_agent
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print(f"Script dir: {script_dir}")
    print(f"Project root: {project_root}")
    print(f"Backend path: {backend_path}")
    print("Python path:")
    for path in sys.path[:10]:  # Show first 10 paths
        print(f"  {path}")
    print("Make sure you're running this from the correct directory and dependencies are installed")
    sys.exit(1)

class StockTestConfig:
    """Configuration for individual stock tests"""
    def __init__(self, symbol: str, name: str, sector: str, expected_risk_profile: str):
        self.symbol = symbol
        self.name = name
        self.sector = sector
        self.expected_risk_profile = expected_risk_profile

class MultiStockRiskTester:
    """Test risk analysis system across multiple stocks"""
    
    def __init__(self):
        # Initialize Zerodha client
        try:
            self.zerodha_client = ZerodhaDataClient()
            print("✅ Zerodha client initialized")
        except Exception as e:
            print(f"❌ Cannot initialize Zerodha client: {e}")
            sys.exit(1)
        
        # Initialize technical indicators calculator
        self.technical_indicators = TechnicalIndicators()
        
        # Define test stocks with different risk profiles
        self.test_stocks = [
            StockTestConfig("RELIANCE", "Reliance Industries", "Energy/Petrochemicals", "moderate_systematic_risk"),
            StockTestConfig("TCS", "Tata Consultancy Services", "IT Services", "low_beta_defensive"),
            StockTestConfig("HDFCBANK", "HDFC Bank", "Banking", "moderate_regulatory_risk"),
            StockTestConfig("ICICIBANK", "ICICI Bank", "Banking", "moderate_credit_risk"),
            StockTestConfig("ITC", "ITC Limited", "FMCG/Tobacco", "regulatory_defensive"),
            StockTestConfig("INFY", "Infosys", "IT Services", "low_volatility_stable"),
            StockTestConfig("BHARTIARTL", "Bharti Airtel", "Telecommunications", "high_competitive_risk"),
        ]
        
        self.results = []
        self.no_llm = False  # Default: run LLM analysis
    
    async def run_multi_stock_tests(self):
        """Run risk analysis tests across all configured stocks"""
        print(f"🚀 Starting Multi-Stock Risk Analysis Testing")
        print(f"Testing {len(self.test_stocks)} stocks for comprehensive risk assessment")
        print("==" * 40)
        
        # Authenticate with Zerodha first
        print("🔗 Authenticating with Zerodha...")
        if not self.zerodha_client.authenticate():
            print("❌ Zerodha authentication failed")
            return False
        
        print("✅ Zerodha authentication successful")
        
        # Create results directory within the risk_analysis directory
        results_dir = os.path.join(os.path.dirname(__file__), "test_results")
        os.makedirs(results_dir, exist_ok=True)
        
        # Create async tasks for all stocks
        async def test_single_stock(stock_config, stock_index):
            """Test risk analysis for a single stock"""
            print(f"\n⚠️ Testing Stock {stock_index}/{len(self.test_stocks)}: {stock_config.symbol}")
            print(f"   Company: {stock_config.name}")
            print(f"   Sector: {stock_config.sector}")
            print(f"   Expected Risk Profile: {stock_config.expected_risk_profile}")
            print("-" * 60)
            
            try:
                # Get stock data
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
                if 'date' in stock_data.columns and not isinstance(stock_data.index, pd.DatetimeIndex):
                    stock_data['date'] = pd.to_datetime(stock_data['date'])
                    stock_data = stock_data.set_index('date')
                elif not isinstance(stock_data.index, pd.DatetimeIndex):
                    stock_data.index = pd.to_datetime(stock_data.index)
                
                # Sort by date
                stock_data = stock_data.sort_index()
                
                print(f"✅ Retrieved {len(stock_data)} days of data")
                print(f"   Date range: {stock_data.index.min()} to {stock_data.index.max()}")
                print(f"   Price range: {stock_data['close'].min():.2f} to {stock_data['close'].max():.2f}")
                print(f"   Average volume: {stock_data['volume'].mean():,.0f}")
                
                # Calculate technical indicators
                print("📈 Calculating technical indicators...")
                indicators = TechnicalIndicators.calculate_all_indicators_optimized(
                    stock_data, stock_config.symbol
                )
                
                print(f"✅ Calculated indicators: {list(indicators.keys())}")
                
                # Test the risk analysis system
                result = await self._test_risk_analysis(stock_config, stock_data, indicators, results_dir)
                
                print(f"✅ Risk analysis completed for {stock_config.symbol}")
                print(f"   Success: {result['success']}")
                print(f"   Overall Risk Score: {result.get('overall_risk_score', 'N/A'):.3f}" if result.get('overall_risk_score') else "   Overall Risk Score: N/A")
                print(f"   Risk Level: {result.get('risk_level', 'N/A')}")
                print(f"   Successful Agents: {result.get('successful_agents', 0)}")
                print(f"   LLM Analysis: {'Yes' if result.get('llm_success') else 'No'}")
                print(f"   Execution Time: {result['execution_time']:.1f}s")
                
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
        print(f"\n🔄 Running {len(tasks)} risk analysis tests concurrently (max 3 at a time)...")
        results = await asyncio.gather(*tasks)
        self.results.extend(results)
        
        print(f"\n✅ Multi-stock risk analysis testing completed!")
        print(f"📁 Individual analysis results saved to: {results_dir}/")
        
        return True
    
    async def _test_risk_analysis(self, stock_config: StockTestConfig, stock_data: pd.DataFrame, indicators: Dict, results_dir: str) -> Dict[str, Any]:
        """Test the comprehensive risk analysis system for a single stock"""
        start_time = time.time()
        
        try:
            # Step 1: Test Quantitative Risk Processor
            print(f"🔍 Running Quantitative Risk Analysis for {stock_config.symbol}...")
            processor_start = time.time()
            
            processor = QuantitativeRiskProcessor()
            risk_analysis_result = await processor.analyze_async(
                stock_data=stock_data,
                indicators=indicators,
                context=f"Test analysis for {stock_config.symbol} ({stock_config.sector})"
            )
            
            processor_time = time.time() - processor_start
            print(f"✅ Quantitative Risk Analysis completed in {processor_time:.2f}s")
            
            # Extract key metrics from the result
            advanced_metrics = risk_analysis_result.get('advanced_risk_metrics', {})
            overall_risk_assessment = risk_analysis_result.get('overall_risk_assessment', {})
            
            print(f"   Risk score: {advanced_metrics.get('risk_score', 0)}")
            print(f"   Risk level: {advanced_metrics.get('risk_level', 'unknown')}")
            print(f"   Sharpe ratio: {advanced_metrics.get('sharpe_ratio', 0):.3f}")
            print(f"   Max drawdown: {advanced_metrics.get('max_drawdown', 0):.3f}")
        
            # Step 2: Test Risk LLM Agent (Human-readable analysis) - Skip if no_llm flag is set
            if self.no_llm:
                print(f"🚫 Skipping LLM analysis for {stock_config.symbol} (--no-llm flag)")
                print(f"📄 Generating prompt for inspection...")
                
                # Still generate the enhanced prompt for inspection even when skipping LLM
                llm_start = time.time()
                from backend.agents.risk_analysis.risk_llm_agent import RiskLLMAgent
                temp_llm_agent = RiskLLMAgent()
                
                # Build enhanced prompt using the comprehensive quantitative data
                prompt = self._build_enhanced_risk_prompt(
                    symbol=stock_config.symbol,
                    risk_analysis_result=risk_analysis_result,
                    context=f"Test analysis for {stock_config.symbol} ({stock_config.sector})",
                    company_name=stock_config.name,
                    sector=stock_config.sector
                )
                
                # Save prompt but don't call LLM
                llm_success = False
                risk_llm_analysis = {
                    "success": False, 
                    "skipped": True, 
                    "prompt_generated": True,
                    "prompt_length": len(prompt),
                    "risk_bullets": "[LLM SKIPPED] Prompt generated for inspection only",
                    "generated_prompt": prompt  # Store the actual prompt for saving
                }
                llm_time = time.time() - llm_start
                
                print(f"✅ Prompt generated (no LLM call) in {llm_time:.2f}s")
                print(f"   Prompt length: {len(prompt)} characters")
                
            else:
                print(f"🤖 Running Risk LLM Agent for {stock_config.symbol}...")
                llm_start = time.time()
                
                # Use the quantitative risk analysis result directly - enhanced format
                # Add metadata that the enhanced prompt builder expects
                enhanced_result = dict(risk_analysis_result)
                enhanced_result['symbol'] = stock_config.symbol
                enhanced_result['company'] = stock_config.name
                enhanced_result['sector'] = stock_config.sector
                enhanced_result['timestamp'] = datetime.now().isoformat()
                
                llm_success, risk_llm_analysis = await risk_llm_agent.analyze_risk_with_llm(
                    symbol=stock_config.symbol,
                    risk_analysis_result=enhanced_result,
                    context=f"Enhanced multi-timeframe risk analysis for {stock_config.symbol} ({stock_config.sector})"
                )
                
                llm_time = time.time() - llm_start
                print(f"{'✅' if llm_success else '❌'} Risk LLM Agent completed in {llm_time:.2f}s")
                
                if llm_success:
                    bullets_length = len(risk_llm_analysis.get('risk_bullets', ''))
                    print(f"   Risk bullets generated: {bullets_length} characters")
                    print(f"   LLM confidence: {risk_llm_analysis.get('confidence', 0):.3f}")
                else:
                    print(f"   LLM Error: {risk_llm_analysis.get('error', 'Unknown error')}")
            
            # Save detailed analysis to files
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self._save_analysis_details(stock_config, risk_analysis_result, risk_llm_analysis, results_dir, timestamp)
            
            execution_time = time.time() - start_time
            
            # Extract key insights for evaluation from quantitative processor result
            overall_risk_score = advanced_metrics.get('risk_score', 50) / 100.0  # Convert to 0-1 range
            risk_level = advanced_metrics.get('risk_level', 'medium')
            
            return {
                'stock_config': stock_config,
                'success': True,
                'execution_time': execution_time,
                'processor_time': processor_time,
                'llm_time': llm_time,
                
                # Quantitative Risk Processor Results
                'overall_risk_score': overall_risk_score,
                'overall_confidence': 0.85,  # Mock confidence since processor doesn't provide this
                'successful_agents': 1,  # Single processor success
                'failed_agents': 0,
                'risk_level': risk_level,
                'consensus_strength': 'quantitative',
                
                # Risk Breakdown
                'risk_breakdown': advanced_metrics.get('risk_components', {}),
                
                # Risk Metrics
                'sharpe_ratio': advanced_metrics.get('sharpe_ratio', 0),
                'max_drawdown': advanced_metrics.get('max_drawdown', 0),
                'volatility': advanced_metrics.get('annualized_volatility', 0),
                'var_95': advanced_metrics.get('var_95', 0),
                
                # LLM Analysis Results
                'llm_success': llm_success,
                'llm_analysis': risk_llm_analysis if llm_success else {},
                'risk_bullets': risk_llm_analysis.get('risk_bullets', '') if llm_success else '',
                'bullets_length': len(risk_llm_analysis.get('risk_bullets', '')) if llm_success else 0,
                
                # Quality metrics
                'quality_score': self._calculate_quality_score_quantitative(advanced_metrics, risk_llm_analysis if llm_success else {}),
                'quantitative_results': {
                    'processor': {
                        'success': 'error' not in risk_analysis_result,
                        'risk_level': risk_level,
                        'confidence': 0.85
                    }
                }
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            return self._create_error_result(stock_config, str(e), execution_time)
    
    def _save_analysis_details(self, stock_config: StockTestConfig, risk_analysis_result: Dict, risk_llm_analysis: Dict, results_dir: str, timestamp: str):
        """Save detailed analysis results to files"""
        
        def safe_get(data, *keys):
            """Safely get nested dictionary values"""
            try:
                result = data
                for key in keys:
                    result = result[key]
                return result
            except (KeyError, TypeError, AttributeError):
                return None
        
        # Save quantitative processor results
        processor_file = os.path.join(results_dir, f"quantitative_processor_{stock_config.symbol}_{timestamp}.json")
        advanced_metrics = risk_analysis_result.get('advanced_risk_metrics', {})
        
        processor_data = {
            'symbol': stock_config.symbol,
            'company': stock_config.name,
            'sector': stock_config.sector,
            'timestamp': datetime.now().isoformat(),
            'agent_name': risk_analysis_result.get('agent_name', 'quantitative_risk'),
            'analysis_timestamp': risk_analysis_result.get('analysis_timestamp'),
            'context': risk_analysis_result.get('context', ''),
            'advanced_risk_metrics': advanced_metrics,
            'stress_testing': risk_analysis_result.get('stress_testing', {}),
            'scenario_analysis': risk_analysis_result.get('scenario_analysis', {}),
            'overall_risk_assessment': risk_analysis_result.get('overall_risk_assessment', {}),
            'error': risk_analysis_result.get('error', None)
        }
        
        with open(processor_file, 'w') as f:
            json.dump(processor_data, f, indent=2, default=str)
        
        # Save detailed prompt that was sent to LLM (following institutional activity pattern)
        prompt_file = os.path.join(results_dir, f"risk_prompt_{stock_config.symbol}_{timestamp}.txt")
        
        # Use the generated prompt if available (from --no-llm mode), otherwise create basic info
        if isinstance(risk_llm_analysis, dict) and 'generated_prompt' in risk_llm_analysis:
            prompt = risk_llm_analysis['generated_prompt']
        else:
            # Fallback for when LLM was actually called or other cases
            prompt = f"Quantitative Risk Analysis for {stock_config.symbol}\nContext: Test analysis for {stock_config.symbol} ({stock_config.sector})"
        
        with open(prompt_file, 'w') as f:
            if 'generated_prompt' in risk_llm_analysis:
                # We have the actual optimized prompt - save it directly
                f.write("# OPTIMIZED RISK ANALYSIS PROMPT\n")
                f.write(f"# Generated: {datetime.now().isoformat()}\n")
                f.write(f"# Stock: {stock_config.symbol} - {stock_config.name}\n")
                f.write(f"# Sector: {stock_config.sector}\n\n")
                f.write(prompt)
            else:
                # Fallback format for when LLM was called
                f.write("QUANTITATIVE RISK ANALYSIS PROMPT\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"Stock Symbol: {stock_config.symbol}\n")
                f.write(f"Company: {stock_config.name}\n")
                f.write(f"Sector: {stock_config.sector}\n")
                f.write(f"Expected Risk Profile: {stock_config.expected_risk_profile}\n")
                f.write(f"Generated: {datetime.now().isoformat()}\n")
                f.write(f"Analysis Type: Quantitative Risk Processing\n\n")
                
                f.write("KEY RISK METRICS SUMMARY:\n")
                f.write("-" * 40 + "\n")
                
                risk_summary = {
                    "risk_score": advanced_metrics.get('risk_score', 0),
                    "risk_level": advanced_metrics.get('risk_level', 'unknown'),
                    "sharpe_ratio": advanced_metrics.get('sharpe_ratio', 0),
                    "max_drawdown": advanced_metrics.get('max_drawdown', 0),
                    "annualized_volatility": advanced_metrics.get('annualized_volatility', 0),
                    "var_95": advanced_metrics.get('var_95', 0),
                    "var_99": advanced_metrics.get('var_99', 0),
                    "risk_components": advanced_metrics.get('risk_components', {}),
                    "mitigation_strategies": advanced_metrics.get('mitigation_strategies', [])
                }
                f.write(json.dumps(risk_summary, indent=2, default=str))
                f.write("\n\n")
                f.write("ANALYSIS CONTEXT:\n")
                f.write("-" * 40 + "\n")
                f.write(prompt)
        
        # Save LLM response if successful
        if isinstance(risk_llm_analysis, dict) and risk_llm_analysis.get('success'):
            response_file = os.path.join(results_dir, f"risk_response_{stock_config.symbol}_{timestamp}.txt")
            with open(response_file, 'w') as f:
                f.write("RISK ANALYSIS LLM RESPONSE\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"Stock Symbol: {stock_config.symbol}\n")
                f.write(f"Company: {stock_config.name}\n")
                f.write(f"Sector: {stock_config.sector}\n")
                f.write(f"Response Time: {datetime.now().isoformat()}\n")
                f.write(f"Response Length: {risk_llm_analysis.get('response_length', 0)} characters\n")
                f.write(f"Processing Time: {risk_llm_analysis.get('processing_time', 0):.2f}s\n")
                f.write(f"LLM Processing Time: {risk_llm_analysis.get('llm_processing_time', 0):.2f}s\n")
                f.write(f"Confidence: {risk_llm_analysis.get('confidence', 0):.3f}\n")
                f.write(f"Code Executions: {risk_llm_analysis.get('code_executions', 0)}\n")
                f.write(f"Calculation Results: {risk_llm_analysis.get('calculation_results', 0)}\n\n")
                
                f.write("COMPLETE LLM RESPONSE:\n")
                f.write("-" * 40 + "\n")
                f.write(risk_llm_analysis.get('risk_bullets', 'No risk bullets generated') or 'No response received')
                f.write("\n\n")
                
                f.write("RISK ANALYSIS CONTEXT (ORCHESTRATOR DATA):\n")
                f.write("-" * 40 + "\n")
                f.write(f"Risk Summary: {json.dumps(risk_llm_analysis.get('risk_summary', {}), indent=2, default=str)}\n\n")
                f.write(f"Risk Breakdown: {json.dumps(risk_llm_analysis.get('risk_breakdown', {}), indent=2, default=str)}\n")
        
        
        print(f"📄 Saved detailed analysis for {stock_config.symbol}:")
        print(f"   📊 Quantitative data: {processor_file}")
        print(f"   📝 Analysis details: {prompt_file}")
        if isinstance(risk_llm_analysis, dict) and risk_llm_analysis.get('success'):
            print(f"   🤖 LLM response: {response_file}")
    
    def _calculate_quality_score_quantitative(self, advanced_metrics: Dict, llm_analysis: Dict) -> float:
        """Calculate overall quality score for quantitative risk analysis"""
        
        # Base score from successful analysis (no errors)
        base_score = 50 if 'error' not in advanced_metrics and advanced_metrics else 0
        
        # Data completeness score
        required_metrics = ['risk_score', 'risk_level', 'sharpe_ratio', 'max_drawdown', 'annualized_volatility']
        completeness = sum(1 for metric in required_metrics if metric in advanced_metrics and advanced_metrics[metric] is not None)
        completeness_score = (completeness / len(required_metrics)) * 20  # Up to 20 points
        
        # Risk components score
        risk_components = advanced_metrics.get('risk_components', {})
        components_score = min(len(risk_components) * 2, 10)  # Up to 10 points
        
        # LLM analysis quality
        llm_score = 0
        if llm_analysis.get('success') and llm_analysis.get('risk_bullets'):
            bullets_length = len(llm_analysis.get('risk_bullets', ''))
            # Good bullets should be substantial but not too verbose
            if 200 <= bullets_length <= 2000:
                llm_score = 20
            elif bullets_length > 100:
                llm_score = 10
        
        total_score = base_score + completeness_score + components_score + llm_score
        return min(100.0, total_score)
    
    def _calculate_quality_score(self, risk_analysis_result, llm_analysis: Dict) -> float:
        """Calculate overall quality score for risk analysis"""
        
        # Base score from orchestrator success rate
        success_rate = risk_analysis_result.successful_agents / (risk_analysis_result.successful_agents + risk_analysis_result.failed_agents)
        base_score = success_rate * 50  # Up to 50 points
        
        # Confidence score
        confidence_score = risk_analysis_result.overall_confidence * 20  # Up to 20 points
        
        # LLM analysis quality
        llm_score = 0
        if llm_analysis.get('success') and llm_analysis.get('risk_bullets'):
            bullets_length = len(llm_analysis.get('risk_bullets', ''))
            # Good bullets should be substantial but not too verbose
            if 200 <= bullets_length <= 2000:
                llm_score = 20
            elif bullets_length > 100:
                llm_score = 10
        
        # Data completeness score
        unified_analysis = risk_analysis_result.unified_analysis
        completeness_indicators = [
            'risk_summary' in unified_analysis,
            'risk_breakdown' in unified_analysis,
            'trading_implications' in unified_analysis,
            'mitigation_strategies' in unified_analysis
        ]
        completeness_score = sum(completeness_indicators) * 2.5  # Up to 10 points
        
        total_score = base_score + confidence_score + llm_score + completeness_score
        return min(100.0, total_score)
    
    def _build_enhanced_risk_prompt(self, symbol: str, risk_analysis_result: Dict, context: str, 
                                   company_name: str, sector: str) -> str:
        """
        Build enhanced quantitative risk analysis prompt with multi-timeframe analysis.
        
        Leverages the comprehensive data from QuantitativeRiskProcessor to provide
        timeframe-specific risk analysis with scenario probabilities and stress testing.
        """
        try:
            # Extract comprehensive data from quantitative risk analysis
            advanced_metrics = risk_analysis_result.get('advanced_risk_metrics', {})
            stress_testing = risk_analysis_result.get('stress_testing', {})
            scenario_analysis = risk_analysis_result.get('scenario_analysis', {})
            overall_assessment = risk_analysis_result.get('overall_risk_assessment', {})
            
            # Extract metadata
            timestamp = risk_analysis_result.get('timestamp', datetime.now().isoformat())
            
            # Advanced risk metrics
            risk_score = advanced_metrics.get('risk_score', 0)
            risk_level = advanced_metrics.get('risk_level', 'Medium')
            sharpe_ratio = advanced_metrics.get('sharpe_ratio', 0)
            sortino_ratio = advanced_metrics.get('sortino_ratio', 0)
            calmar_ratio = advanced_metrics.get('calmar_ratio', 0)
            max_drawdown = advanced_metrics.get('max_drawdown', 0)
            current_drawdown = advanced_metrics.get('current_drawdown', 0)
            drawdown_duration = advanced_metrics.get('drawdown_duration', 0)
            var_95 = advanced_metrics.get('var_95', 0)
            var_99 = advanced_metrics.get('var_99', 0)
            expected_shortfall_95 = advanced_metrics.get('expected_shortfall_95', 0)
            annualized_volatility = advanced_metrics.get('annualized_volatility', 0)
            skewness = advanced_metrics.get('skewness', 0)
            kurtosis = advanced_metrics.get('kurtosis', 0)
            tail_frequency = advanced_metrics.get('tail_frequency', 0)
            
            # Risk components
            risk_components = advanced_metrics.get('risk_components', {})
            volatility_risk = risk_components.get('volatility_risk', 'Unknown')
            drawdown_risk = risk_components.get('drawdown_risk', 'Unknown')
            tail_risk = risk_components.get('tail_risk', 'Unknown')
            liquidity_risk = risk_components.get('liquidity_risk', 'Unknown')
            sector_risk = risk_components.get('sector_risk', 'Unknown')
            
            # Stress testing data
            stress_scenarios = stress_testing.get('stress_scenarios', {})
            historical_stress = stress_scenarios.get('historical_stress', {})
            monte_carlo_stress = stress_scenarios.get('monte_carlo_stress', {})
            sector_stress = stress_scenarios.get('sector_stress', {})
            crash_scenarios = stress_scenarios.get('crash_scenarios', {})
            
            # Historical stress metrics
            worst_20_day_period = historical_stress.get('worst_20_day_period', 0)
            second_worst_period = historical_stress.get('second_worst_period', 0)
            stress_frequency = historical_stress.get('stress_frequency', 0)
            
            # Monte Carlo stress metrics
            worst_case = monte_carlo_stress.get('worst_case', 0)
            fifth_percentile = monte_carlo_stress.get('fifth_percentile', 0)
            expected_loss = monte_carlo_stress.get('expected_loss', 0)
            probability_of_loss = monte_carlo_stress.get('probability_of_loss', 0)
            
            # Sector stress metrics
            sector_rotation_stress = sector_stress.get('sector_rotation_stress', 0)
            regulatory_stress = sector_stress.get('regulatory_stress', 0)
            economic_recession = crash_scenarios.get('economic_recession', 0)
            
            # Market crash scenarios
            black_swan_event = crash_scenarios.get('black_swan_event', 0)
            systemic_crisis = crash_scenarios.get('systemic_crisis', 0)
            geopolitical_crisis = crash_scenarios.get('geopolitical_crisis', 0)
            
            # Scenario analysis data
            expected_outcomes = scenario_analysis.get('expected_outcomes', {})
            probability_scores = scenario_analysis.get('probability_scores', {})
            
            # Bull scenario
            bull_scenario = expected_outcomes.get('bull_scenario', {})
            bull_probability = probability_scores.get('bull', 0)  # Already in 0-1 range
            bull_timeframe = bull_scenario.get('timeframe', '6-12 months')
            bull_price_target = bull_scenario.get('price_target', 0)
            bull_return_expectation = bull_scenario.get('return_expectation', 0)  # Already in decimal format
            bull_key_drivers = ', '.join(bull_scenario.get('key_drivers', ['Market recovery']))
            bull_confidence = bull_scenario.get('confidence_level', 0.5)  # Keep as decimal
            
            # Bear scenario
            bear_scenario = expected_outcomes.get('bear_scenario', {})
            bear_probability = probability_scores.get('bear', 0)  # Already in 0-1 range
            bear_timeframe = bear_scenario.get('timeframe', '3-6 months')
            bear_price_target = bear_scenario.get('price_target', 0)
            bear_return_expectation = bear_scenario.get('return_expectation', 0)  # Already in decimal format
            bear_key_drivers = ', '.join(bear_scenario.get('key_drivers', ['Market correction']))
            bear_confidence = bear_scenario.get('confidence_level', 0.5)  # Keep as decimal
            
            # Sideways scenario
            sideways_scenario = expected_outcomes.get('sideways_scenario', {})
            sideways_probability = probability_scores.get('sideways', 0)  # Already in 0-1 range
            sideways_timeframe = sideways_scenario.get('timeframe', '3-9 months')
            sideways_price_target = sideways_scenario.get('price_target', 0)
            sideways_return_expectation = sideways_scenario.get('return_expectation', 0)  # Already in decimal format
            sideways_key_drivers = ', '.join(sideways_scenario.get('key_drivers', ['Range-bound trading']))
            
            # Volatility scenario
            volatility_scenario = expected_outcomes.get('volatility_scenario', {})
            volatility_probability = probability_scores.get('volatility', 0)  # Already in 0-1 range
            volatility_timeframe = volatility_scenario.get('timeframe', '1-3 months')
            volatility_return_expectation = volatility_scenario.get('return_expectation', 0)  # Already in decimal format
            volatility_key_drivers = ', '.join(volatility_scenario.get('key_drivers', ['Market uncertainty']))
            
            # Build the enhanced prompt
            prompt = f"""# ENHANCED QUANTITATIVE RISK ANALYSIS PROMPT
# Generated: {timestamp}
# Stock: {symbol} - {company_name}
# Sector: {sector}
# Context: {context}

# MULTI-TIMEFRAME RISK ASSESSMENT for {symbol}

## QUANTITATIVE RISK METRICS
**Overall Risk**: {risk_level} (Score: {risk_score}/100)
**Performance Risk**: Sharpe Ratio {sharpe_ratio:.2f} | Sortino Ratio {sortino_ratio:.2f} | Calmar Ratio {calmar_ratio:.2f}
**Tail Risk**: VaR(95%) {var_95:.2%} | VaR(99%) {var_99:.2%} | ES(95%) {expected_shortfall_95:.2%}
**Volatility Profile**: {annualized_volatility:.1%} annualized | Skewness {skewness:.2f} | Kurtosis {kurtosis:.2f}
**Drawdown Analysis**: Max DD {max_drawdown:.1%} | Current DD {current_drawdown:.1%} | Duration {drawdown_duration} days

## ADVANCED RISK COMPONENT BREAKDOWN
- **Volatility Risk**: {volatility_risk} - Annualized volatility of {annualized_volatility:.1%}
- **Drawdown Risk**: {drawdown_risk} - Current drawdown streak of {drawdown_duration} days
- **Tail Risk**: {tail_risk} - {tail_frequency:.1%} tail events frequency  
- **Liquidity Risk**: {liquidity_risk} - Volume and spread analysis
- **Sector Risk**: {sector_risk} - Technical and fundamental sector factors

## COMPREHENSIVE STRESS TESTING SCENARIOS
### Historical Stress Events
- **Worst 20-Day Period**: {worst_20_day_period:.1%}
- **Second Worst Period**: {second_worst_period:.1%}
- **Stress Event Frequency**: {stress_frequency:.1%}

### Monte Carlo Stress Analysis
- **Worst Case (1%ile)**: {worst_case:.1%}
- **Fifth Percentile**: {fifth_percentile:.1%}
- **Expected Loss Scenario**: {expected_loss:.1%}
- **Probability of Loss**: {probability_of_loss:.0%}

### Sector-Specific Stress Tests
- **Sector Rotation Impact**: {sector_rotation_stress:.1%}
- **Regulatory Stress**: {regulatory_stress:.1%}
- **Economic Recession**: {economic_recession:.1%}

### Market Crash Scenarios
- **Black Swan Event**: {black_swan_event:.1%}
- **Systemic Crisis**: {systemic_crisis:.1%}
- **Geopolitical Crisis**: {geopolitical_crisis:.1%}

## SCENARIO ANALYSIS WITH PROBABILITIES
### Bull Market Scenario (Probability: {bull_probability:.1%})
- **Timeframe**: {bull_timeframe}
- **Price Target**: ${bull_price_target:.2f} ({bull_return_expectation:+.1%} return)
- **Key Drivers**: {bull_key_drivers}
- **Confidence**: {bull_confidence:.0%}

### Bear Market Scenario (Probability: {bear_probability:.1%})
- **Timeframe**: {bear_timeframe}  
- **Price Target**: ${bear_price_target:.2f} ({bear_return_expectation:+.1%} return)
- **Key Drivers**: {bear_key_drivers}
- **Confidence**: {bear_confidence:.0%}

### Sideways Market Scenario (Probability: {sideways_probability:.1%})
- **Timeframe**: {sideways_timeframe}
- **Price Target**: ${sideways_price_target:.2f} ({sideways_return_expectation:+.1%} return)
- **Key Drivers**: {sideways_key_drivers}

### Volatility Spike Scenario (Probability: {volatility_probability:.1%})
- **Timeframe**: {volatility_timeframe}
- **Expected Impact**: {volatility_return_expectation:+.1%}
- **Key Triggers**: {volatility_key_drivers}

---

## INSTRUCTIONS: Multi-Timeframe Risk Assessment for Trading Decisions

Analyze the comprehensive quantitative risk data above and provide **exactly 15 actionable risk bullets** structured as follows:

### **SHORT-TERM RISK ASSESSMENT (1-3 months)** [5 bullets]
Focus on immediate trading risks, volatility spikes, liquidity concerns, and near-term scenario probabilities.

• **[Primary Short-Term Risk]**: [Level] - [Specific 1-3 month trading implication] ([Probability/Impact])
• **[Liquidity & Execution Risk]**: [Level] - [Position entry/exit timing guidance] ([Market Conditions])
• **[Volatility Regime Risk]**: [Analysis] - [Expected volatility patterns] ([Short-term Scenarios])
• **[Technical Risk Factors]**: [Assessment] - [Key technical levels and breakdowns] ([Signal Reliability])
• **[Short-Term Position Sizing]**: [Recommendation] - [Specific sizing for 1-3 month horizon] ([Risk Budget])

### **MEDIUM-TERM RISK ASSESSMENT (3-12 months)** [5 bullets]  
Focus on trend sustainability, scenario probabilities, sector rotation risks, and drawdown management.

• **[Primary Medium-Term Risk]**: [Level] - [3-12 month strategic implications] ([Scenario Probability])
• **[Trend & Momentum Risk]**: [Analysis] - [Trend sustainability and reversal risks] ([Technical Confluence])
• **[Scenario Probability Risk]**: [Assessment] - [Most likely scenarios and transitions] ([Confidence Levels])
• **[Sector & Correlation Risk]**: [Level] - [Sector rotation and correlation breakdown risks] ([Market Regime])
• **[Medium-Term Hedging Strategy]**: [Recommendation] - [Optimal hedging for 3-12 month exposure] ([Cost-Benefit])

### **LONG-TERM RISK ASSESSMENT (1+ years)** [5 bullets]
Focus on structural risks, regime changes, maximum drawdown potential, and strategic positioning.

• **[Structural Risk Factors]**: [Level] - [Long-term fundamental and structural risks] ([Regime Analysis])
• **[Maximum Drawdown Risk]**: [Assessment] - [Worst-case drawdown scenarios and recovery] ([Historical Context])
• **[Regime Change Risk]**: [Analysis] - [Market regime sustainability and transition risks] ([Macro Factors])
• **[Strategic Position Risk]**: [Recommendation] - [Long-term position strategy and adjustments] ([Risk-Return Profile])
• **[Long-Term Monitoring Framework]**: [Key Metrics] - [Critical indicators for regime/trend changes] ([Early Warning System])

### Focus Areas:
- **Quantitative Priority**: Utilize all VaR, stress test, and scenario analysis metrics
- **Timeframe Specificity**: Tailor recommendations to each time horizon's unique characteristics
- **Scenario Integration**: Incorporate probability-weighted scenario analysis into recommendations
- **Risk Compounding**: Analyze how short-term risks can compound into medium/long-term issues
- **Dynamic Positioning**: Provide guidance that adapts to changing market conditions
- **Actionable Metrics**: Include specific thresholds, probabilities, and monitoring triggers

### Avoid:
- Generic timeframe-agnostic advice
- Ignoring the rich scenario and stress test data
- Vague probability assessments
- Static risk recommendations
- Fundamental analysis outside technical risk scope

Provide concise, quantitative risk bullets that directly inform multi-timeframe trading and position management decisions. Each bullet should reference specific metrics from the comprehensive risk analysis above."""
            
            return prompt
            
        except Exception as e:
            print(f"❌ Error building enhanced risk prompt: {e}")
            # Fallback to basic prompt
            return f"""# FALLBACK RISK ANALYSIS PROMPT
# Generated: {datetime.now().isoformat()}
# Stock: {symbol} - {company_name}
# Sector: {sector}
# Context: {context}
# Error: {str(e)}

Provide basic risk analysis for {symbol} based on available data."""
    
    def _create_error_result(self, stock_config: StockTestConfig, error_message: str, execution_time: float = 0) -> Dict[str, Any]:
        """Create error result dictionary"""
        return {
            'stock_config': stock_config,
            'success': False,
            'error': error_message,
            'execution_time': execution_time,
            'overall_risk_score': 0.0,
            'quality_score': 0.0,
            'risk_level': 'unknown'
        }
    

async def main():
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Multi-Stock Risk Analysis Testing Framework')
    parser.add_argument('--stock', '-s', type=str, help='Test only one specific stock (e.g., RELIANCE, TCS, HDFCBANK)')
    parser.add_argument('--list-stocks', '-l', action='store_true', help='List available stocks and exit')
    parser.add_argument('--no-llm', action='store_true', help='Generate prompts only, skip LLM analysis (faster testing)')
    args = parser.parse_args()
    
    tester = MultiStockRiskTester()
    
    # List stocks option
    if args.list_stocks:
        print("📋 Available stocks for testing:")
        for i, stock in enumerate(tester.test_stocks, 1):
            print(f"  {i}. {stock.symbol} - {stock.name} ({stock.sector})")
        print("\n💡 Usage examples:")
        print("  python multi_stock_test.py --stock RELIANCE")
        print("  python multi_stock_test.py --stock TCS --no-llm  # Generate prompts only")
        print("  python multi_stock_test.py --no-llm              # All stocks, prompts only")
        return
    
    # Single stock testing
    if args.stock:
        stock_upper = args.stock.upper()
        matching_stocks = [s for s in tester.test_stocks if s.symbol == stock_upper]
        
        if not matching_stocks:
            print(f"❌ Stock '{args.stock}' not found in test list.")
            print("📋 Available stocks:")
            for stock in tester.test_stocks:
                print(f"  - {stock.symbol}")
            return
        
        print("⚠️ Single-Stock Risk Analysis Testing Framework")
        print(f"Testing risk analysis for {stock_upper} only")
        if args.no_llm:
            print("🚫 LLM calls disabled - generating prompts only")
        
        # Override test_stocks to only include the selected stock
        tester.test_stocks = matching_stocks
        tester.no_llm = args.no_llm  # Pass flag to tester
        success = await tester.run_multi_stock_tests()
        
        if success:
            print(f"\n🎉 Risk analysis testing for {stock_upper} completed successfully!")
        else:
            print(f"\n❌ Risk analysis testing for {stock_upper} failed")
    else:
        # Multi-stock testing (default behavior)
        print("⚠️ Multi-Stock Risk Analysis Testing Framework")
        if args.no_llm:
            print("Testing quantitative risk analysis (prompts only, no LLM) across multiple stocks")
            print("🚫 LLM calls disabled - generating prompts only")
        else:
            print("Testing comprehensive risk analysis (orchestrator + LLM) across multiple stocks")
        
        tester.no_llm = args.no_llm  # Pass flag to tester
        success = await tester.run_multi_stock_tests()
        
        if success:
            print("\n🎉 Multi-stock risk analysis testing completed successfully!")
        else:
            print("\n❌ Multi-stock risk analysis testing failed")

if __name__ == "__main__":
    asyncio.run(main())
