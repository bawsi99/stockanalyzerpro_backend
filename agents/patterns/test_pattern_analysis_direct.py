#!/usr/bin/env python3
"""
Pattern Analysis Agent Direct Test Suite

This script provides comprehensive testing for the Pattern LLM Agent including:
- Full pattern analysis with real stock data
- LLM prompt inspection and debugging
- Response analysis and validation
- No-LLM mode for prompt-only testing
- File output for prompt and response analysis

Run from backend directory:
    cd backend
    python agents/patterns/test_pattern_analysis_direct.py --symbol RELIANCE --exchange NSE
"""

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, Any, Optional
import pandas as pd

# Add current directory to path for proper imports
sys.path.insert(0, os.getcwd())

# Imports with proper paths from backend directory
from agents.patterns.pattern_llm_agent import PatternLLMAgent
from agents.patterns.pattern_context_builder import PatternContextBuilder
from core.orchestrator import StockAnalysisOrchestrator
from ml.indicators.technical_indicators import TechnicalIndicators


class PatternAnalysisTestSuite:
    """Comprehensive test suite for Pattern Analysis Agent"""
    
    def __init__(self, output_dir: str = None):
        self.output_dir = output_dir or os.path.join(os.getcwd(), "agents", "patterns", "test_outputs")
        self.ensure_output_dir()
        
    def ensure_output_dir(self):
        """Create output directory if it doesn't exist"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"📁 Created output directory: {self.output_dir}")
    
    async def run_full_test(
        self,
        symbol: str,
        exchange: str = "NSE",
        period: int = 365,
        interval: str = "day",
        context: str = "",
        no_llm: bool = False,
        save_files: bool = False
    ) -> Dict[str, Any]:
        """Run comprehensive pattern analysis test"""
        
        print("🚀 Starting Pattern Analysis Test Suite")
        print("=" * 70)
        print(f"Symbol: {symbol}")
        print(f"Exchange: {exchange}")
        print(f"Period: {period} days")
        print(f"Interval: {interval}")
        print(f"Context: {context or 'Default analysis context'}")
        print(f"No-LLM Mode: {no_llm}")
        print(f"Save Files: {save_files}")
        print("=" * 70)
        
        start_time = time.time()
        test_results = {
            "test_info": {
                "symbol": symbol,
                "exchange": exchange,
                "period": period,
                "interval": interval,
                "context": context,
                "no_llm_mode": no_llm,
                "timestamp": datetime.now().isoformat()
            },
            "steps": {}
        }
        
        try:
            # Step 1: Data Retrieval
            print("\n📊 Step 1: Retrieving Stock Data")
            print("-" * 40)
            
            orchestrator = StockAnalysisOrchestrator()
            if not orchestrator.authenticate():
                raise Exception("Failed to authenticate with data provider")
            
            stock_data = await orchestrator.retrieve_stock_data(
                symbol=symbol,
                exchange=exchange,
                interval=interval,
                period=period
            )
            
            print(f"✅ Retrieved {len(stock_data)} days of stock data")
            print(f"   Date range: {stock_data.index[0]} to {stock_data.index[-1]}")
            print(f"   Current price: ${stock_data['close'].iloc[-1]:.2f}")
            
            test_results["steps"]["data_retrieval"] = {
                "success": True,
                "data_points": len(stock_data),
                "date_range": f"{stock_data.index[0]} to {stock_data.index[-1]}",
                "current_price": float(stock_data['close'].iloc[-1])
            }
            
            # Step 2: Technical Indicators
            print("\n🔧 Step 2: Calculating Technical Indicators")
            print("-" * 40)
            
            indicators = TechnicalIndicators.calculate_all_indicators_optimized(stock_data, symbol)
            print(f"✅ Calculated {len(indicators)} technical indicators")
            
            # Show key indicators (handle both Series and dict types)
            key_indicators = ["sma_20", "rsi", "macd", "bollinger_upper", "bollinger_lower"]
            for indicator in key_indicators:
                if indicator in indicators:
                    try:
                        # Handle pandas Series
                        if hasattr(indicators[indicator], 'iloc') and len(indicators[indicator]) > 0:
                            latest_value = indicators[indicator].iloc[-1]
                            if not pd.isna(latest_value):
                                print(f"   {indicator.upper()}: {latest_value:.2f}")
                        # Handle dict type
                        elif isinstance(indicators[indicator], dict):
                            # For dict indicators, show a summary or key value
                            if 'value' in indicators[indicator]:
                                print(f"   {indicator.upper()}: {indicators[indicator]['value']:.2f}")
                            elif 'current' in indicators[indicator]:
                                print(f"   {indicator.upper()}: {indicators[indicator]['current']:.2f}")
                            else:
                                print(f"   {indicator.upper()}: {type(indicators[indicator]).__name__} data")
                        # Handle scalar values
                        elif isinstance(indicators[indicator], (int, float)):
                            print(f"   {indicator.upper()}: {indicators[indicator]:.2f}")
                        else:
                            print(f"   {indicator.upper()}: {type(indicators[indicator]).__name__} type")
                    except Exception as e:
                        print(f"   {indicator.upper()}: Error displaying ({str(e)})")
            
            # Build key_indicators dict with proper error handling
            key_indicators_dict = {}
            for k in key_indicators:
                if k in indicators:
                    try:
                        if hasattr(indicators[k], 'iloc') and len(indicators[k]) > 0:
                            val = indicators[k].iloc[-1]
                            key_indicators_dict[k] = float(val) if not pd.isna(val) else None
                        elif isinstance(indicators[k], dict):
                            if 'value' in indicators[k]:
                                key_indicators_dict[k] = float(indicators[k]['value'])
                            elif 'current' in indicators[k]:
                                key_indicators_dict[k] = float(indicators[k]['current'])
                            else:
                                key_indicators_dict[k] = str(indicators[k])
                        elif isinstance(indicators[k], (int, float)):
                            key_indicators_dict[k] = float(indicators[k])
                        else:
                            key_indicators_dict[k] = str(type(indicators[k]).__name__)
                    except Exception as e:
                        key_indicators_dict[k] = f"Error: {str(e)}"
                else:
                    key_indicators_dict[k] = None
            
            test_results["steps"]["indicators"] = {
                "success": True,
                "indicator_count": len(indicators),
                "key_indicators": key_indicators_dict
            }
            
            # Step 3: Pattern Context Building
            print("\n🧠 Step 3: Building Pattern Context")
            print("-" * 40)
            
            # Create Pattern LLM Agent
            pattern_agent = PatternLLMAgent(gemini_client=None)
            
            # Get pattern analysis from orchestrator (without LLM)
            pattern_analysis_result = await pattern_agent._execute_pattern_analysis(
                symbol, stock_data, indicators, context
            )
            
            if not pattern_analysis_result.get('success', False):
                raise Exception(f"Pattern analysis failed: {pattern_analysis_result.get('error', 'Unknown error')}")
            
            technical_analysis = pattern_analysis_result.get('analysis_data', {})
            print(f"✅ Pattern technical analysis completed")
            
            # Handle both dict and AggregatedPatternAnalysis object
            try:
                if hasattr(technical_analysis, 'overall_confidence'):
                    overall_confidence = technical_analysis.overall_confidence or 0
                elif isinstance(technical_analysis, dict):
                    overall_confidence = technical_analysis.get('overall_confidence', 0)
                else:
                    overall_confidence = 0
                print(f"   Overall confidence: {overall_confidence:.2%}")
            except Exception as e:
                print(f"   Overall confidence: Could not determine ({str(e)})")
                overall_confidence = 0
            
            # Build LLM context
            current_price = stock_data['close'].iloc[-1]
            llm_context = pattern_agent.context_builder.build_comprehensive_pattern_context(
                technical_analysis, symbol, current_price
            )
            
            print(f"✅ Built LLM context ({len(llm_context)} characters)")
            
            # Convert technical_analysis to serializable format
            try:
                if hasattr(technical_analysis, '__dict__'):
                    # Convert object to dict for JSON serialization
                    technical_analysis_dict = {
                        'overall_confidence': getattr(technical_analysis, 'overall_confidence', 0),
                        'type': type(technical_analysis).__name__,
                        'summary': str(technical_analysis) if hasattr(technical_analysis, '__str__') else 'AggregatedPatternAnalysis object'
                    }
                else:
                    technical_analysis_dict = technical_analysis if isinstance(technical_analysis, dict) else {}
            except Exception as e:
                technical_analysis_dict = {'error': f'Could not serialize: {str(e)}'}
            
            test_results["steps"]["context_building"] = {
                "success": True,
                "technical_analysis": technical_analysis_dict,
                "context_length": len(llm_context),
                "overall_confidence": overall_confidence
            }
            
            # Step 4: LLM Prompt Construction
            print("\n🤖 Step 4: Constructing LLM Prompt")
            print("-" * 40)
            
            llm_prompt = pattern_agent._build_llm_prompt(llm_context, symbol)
            print(f"✅ Built LLM prompt ({len(llm_prompt)} characters)")
            
            # Show prompt preview
            prompt_lines = llm_prompt.split('\n')
            print(f"   Prompt has {len(prompt_lines)} lines")
            print(f"   First 3 lines:")
            for i, line in enumerate(prompt_lines[:3], 1):
                print(f"     {i}. {line[:80]}{'...' if len(line) > 80 else ''}")
            
            test_results["steps"]["prompt_construction"] = {
                "success": True,
                "prompt_length": len(llm_prompt),
                "prompt_lines": len(prompt_lines),
                "prompt_preview": prompt_lines[:5] if len(prompt_lines) >= 5 else prompt_lines
            }
            
            # Step 5: LLM Analysis (or skip if no-llm flag)
            llm_response = None
            if no_llm:
                print("\n⏭️  Step 5: Skipping LLM Analysis (No-LLM Mode)")
                print("-" * 40)
                print("✅ LLM prompt ready but not sent (no-llm flag enabled)")
                
                test_results["steps"]["llm_analysis"] = {
                    "success": True,
                    "skipped": True,
                    "reason": "no_llm flag enabled"
                }
            else:
                print("\n🤖 Step 5: LLM Analysis")
                print("-" * 40)
                
                try:
                    llm_result = await pattern_agent._synthesize_with_llm(llm_context, symbol)
                    
                    if llm_result.get('success', False):
                        llm_response = llm_result.get('raw_response', '')
                        print(f"✅ LLM analysis completed")
                        print(f"   Response length: {len(llm_response)} characters")
                        
                        # Show response preview
                        response_lines = llm_response.split('\n')
                        print(f"   Response has {len(response_lines)} lines")
                        print(f"   First 3 lines:")
                        for i, line in enumerate(response_lines[:3], 1):
                            print(f"     {i}. {line[:80]}{'...' if len(line) > 80 else ''}")
                        
                        test_results["steps"]["llm_analysis"] = {
                            "success": True,
                            "response_length": len(llm_response),
                            "response_lines": len(response_lines),
                            "response_preview": response_lines[:5] if len(response_lines) >= 5 else response_lines
                        }
                    else:
                        error_msg = llm_result.get('error', 'Unknown LLM error')
                        print(f"❌ LLM analysis failed: {error_msg}")
                        test_results["steps"]["llm_analysis"] = {
                            "success": False,
                            "error": error_msg
                        }
                except Exception as e:
                    print(f"❌ LLM analysis error: {e}")
                    test_results["steps"]["llm_analysis"] = {
                        "success": False,
                        "error": str(e)
                    }
            
            # Step 6: Save Files (if requested)
            if save_files:
                await self._save_test_files(
                    symbol=symbol,
                    exchange=exchange,
                    llm_prompt=llm_prompt,
                    llm_response=llm_response,
                    llm_context=llm_context,
                    technical_analysis=technical_analysis,
                    test_results=test_results
                )
            
            # Final Results
            total_time = time.time() - start_time
            test_results["total_processing_time"] = total_time
            test_results["overall_success"] = all(step.get("success", False) for step in test_results["steps"].values())
            
            print("\n🎯 Final Results")
            print("=" * 70)
            print(f"Overall Success: {'✅ PASS' if test_results['overall_success'] else '❌ FAIL'}")
            print(f"Total Processing Time: {total_time:.2f} seconds")
            
            # Step summary
            for step_name, step_result in test_results["steps"].items():
                status = "✅ PASS" if step_result.get("success", False) else "❌ FAIL"
                skipped = " (SKIPPED)" if step_result.get("skipped", False) else ""
                print(f"  {step_name.replace('_', ' ').title()}: {status}{skipped}")
            
            return test_results
            
        except Exception as e:
            total_time = time.time() - start_time
            print(f"\n❌ Test Failed: {e}")
            import traceback
            traceback.print_exc()
            test_results["error"] = str(e)
            test_results["total_processing_time"] = total_time
            test_results["overall_success"] = False
            return test_results
    
    async def _save_test_files(
        self,
        symbol: str,
        exchange: str,
        llm_prompt: str,
        llm_response: Optional[str],
        llm_context: str,
        technical_analysis: Dict[str, Any],
        test_results: Dict[str, Any]
    ):
        """Save test files for analysis"""
        print("\n💾 Step 6: Saving Test Files")
        print("-" * 40)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_prefix = f"{symbol}_{exchange}_{timestamp}"
        
        files_saved = []
        
        # Save LLM prompt
        prompt_file = os.path.join(self.output_dir, f"{file_prefix}_prompt.txt")
        with open(prompt_file, 'w', encoding='utf-8') as f:
            f.write(f"# Pattern Analysis LLM Prompt for {symbol}\n")
            f.write(f"# Generated: {datetime.now().isoformat()}\n")
            f.write(f"# Exchange: {exchange}\n")
            f.write("# " + "="*60 + "\n\n")
            f.write(llm_prompt)
        files_saved.append(prompt_file)
        print(f"✅ Saved LLM prompt: {os.path.basename(prompt_file)}")
        
        # Save LLM response (if available)
        if llm_response:
            response_file = os.path.join(self.output_dir, f"{file_prefix}_response.txt")
            with open(response_file, 'w', encoding='utf-8') as f:
                f.write(f"# Pattern Analysis LLM Response for {symbol}\n")
                f.write(f"# Generated: {datetime.now().isoformat()}\n")
                f.write(f"# Exchange: {exchange}\n")
                f.write("# " + "="*60 + "\n\n")
                f.write(llm_response)
            files_saved.append(response_file)
            print(f"✅ Saved LLM response: {os.path.basename(response_file)}")
        
        # Save pattern context
        context_file = os.path.join(self.output_dir, f"{file_prefix}_context.txt")
        with open(context_file, 'w', encoding='utf-8') as f:
            f.write(f"# Pattern Analysis Context for {symbol}\n")
            f.write(f"# Generated: {datetime.now().isoformat()}\n")
            f.write(f"# Exchange: {exchange}\n")
            f.write("# " + "="*60 + "\n\n")
            f.write(llm_context)
        files_saved.append(context_file)
        print(f"✅ Saved pattern context: {os.path.basename(context_file)}")
        
        # Save technical analysis data
        technical_file = os.path.join(self.output_dir, f"{file_prefix}_technical.json")
        with open(technical_file, 'w', encoding='utf-8') as f:
            json.dump(technical_analysis, f, indent=2, default=str)
        files_saved.append(technical_file)
        print(f"✅ Saved technical analysis: {os.path.basename(technical_file)}")
        
        # Save complete test results
        results_file = os.path.join(self.output_dir, f"{file_prefix}_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(test_results, f, indent=2, default=str)
        files_saved.append(results_file)
        print(f"✅ Saved test results: {os.path.basename(results_file)}")
        
        print(f"📁 All files saved to: {self.output_dir}")
        test_results["saved_files"] = files_saved


def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description="Test Pattern Analysis Agent with comprehensive prompt and response analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic test with RELIANCE
  cd backend
  python agents/patterns/test_pattern_analysis_direct.py --symbol RELIANCE
  
  # Test with custom parameters and save files
  python agents/patterns/test_pattern_analysis_direct.py --symbol TATAMOTORS --period 180 --save-files
  
  # No-LLM mode (prompt only, no API call)
  python agents/patterns/test_pattern_analysis_direct.py --symbol AAPL --exchange NASDAQ --no-llm
  
  # Full test with custom context
  python agents/patterns/test_pattern_analysis_direct.py --symbol INFY --context "Swing trading analysis for tech stock"
        """
    )
    
    parser.add_argument("--symbol", "-s", required=True, help="Stock symbol to analyze (e.g., RELIANCE, TATAMOTORS)")
    parser.add_argument("--exchange", "-e", default="NSE", help="Stock exchange (default: NSE)")
    parser.add_argument("--period", "-p", type=int, default=365, help="Analysis period in days (default: 365)")
    parser.add_argument("--interval", "-i", default="day", help="Data interval (default: day)")
    parser.add_argument("--context", "-c", default="", help="Additional context for pattern analysis")
    parser.add_argument("--no-llm", action="store_true", help="Build prompt only, don't call LLM")
    parser.add_argument("--save-files", action="store_true", help="Save prompt, response, and analysis files")
    parser.add_argument("--output-dir", "-o", help="Custom output directory for saved files")
    
    args = parser.parse_args()
    
    # Create test suite
    test_suite = PatternAnalysisTestSuite(output_dir=args.output_dir)
    
    # Run the test
    try:
        results = asyncio.run(test_suite.run_full_test(
            symbol=args.symbol.upper(),
            exchange=args.exchange.upper(),
            period=args.period,
            interval=args.interval,
            context=args.context,
            no_llm=args.no_llm,
            save_files=args.save_files
        ))
        
        # Exit with appropriate code
        exit_code = 0 if results.get("overall_success", False) else 1
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Test suite error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()