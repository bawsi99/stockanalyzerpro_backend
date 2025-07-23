#!/usr/bin/env python3
"""
Test script for enhanced analysis with code execution.
This script tests the new enhanced analysis capabilities that use
Google Gemini's code execution feature for mathematical validation.
"""

import asyncio
import json
import time
from agent_capabilities import StockAnalysisOrchestrator
from gemini.gemini_client import GeminiClient

async def test_enhanced_analysis():
    """
    Test the enhanced analysis with code execution.
    """
    print("🚀 Starting Enhanced Analysis Test")
    print("=" * 50)
    
    try:
        # Create orchestrator
        orchestrator = StockAnalysisOrchestrator()
        
        # Test authentication
        print("🔐 Testing authentication...")
        auth_success = orchestrator.authenticate()
        if not auth_success:
            print("❌ Authentication failed. Please check your Zerodha credentials.")
            return
        print("✅ Authentication successful")
        
        # Test stock symbol
        test_symbol = "RELIANCE"  # Reliance Industries
        print(f"\n📊 Testing enhanced analysis for {test_symbol}")
        
        # Perform enhanced analysis
        start_time = time.time()
        result = await orchestrator.enhanced_analyze_stock(
            symbol=test_symbol,
            exchange="NSE",
            period=365,
            interval="day",
            output_dir="./test_output"
        )
        end_time = time.time()
        
        print(f"⏱️  Analysis completed in {end_time - start_time:.2f} seconds")
        
        # Validate the result structure
        print("\n🔍 Validating result structure...")
        
        required_fields = [
            "symbol", "exchange", "analysis_timestamp", "analysis_type",
            "mathematical_validation", "calculation_method", "accuracy_improvement",
            "current_price", "ai_analysis", "technical_indicators",
            "risk_level", "recommendation", "enhanced_metadata"
        ]
        
        for field in required_fields:
            if field in result:
                print(f"✅ {field}: {result[field]}")
            else:
                print(f"❌ Missing field: {field}")
        
        # Check for mathematical validation results
        print("\n🧮 Checking mathematical validation...")
        if "mathematical_validation_results" in result:
            math_val = result["mathematical_validation_results"]
            print("✅ Mathematical validation results found:")
            for key, value in math_val.items():
                print(f"   - {key}: {value}")
        else:
            print("⚠️  No mathematical validation results found")
        
        # Check for code execution metadata
        print("\n💻 Checking code execution metadata...")
        if "code_execution_metadata" in result:
            code_meta = result["code_execution_metadata"]
            print("✅ Code execution metadata found:")
            for key, value in code_meta.items():
                print(f"   - {key}: {value}")
        else:
            print("⚠️  No code execution metadata found")
        
        # Check AI analysis structure
        print("\n🤖 Checking AI analysis structure...")
        if "ai_analysis" in result:
            ai_analysis = result["ai_analysis"]
            
            # Check for mathematical validation in AI analysis
            if "mathematical_validation" in ai_analysis:
                print("✅ Mathematical validation in AI analysis found")
                math_val = ai_analysis["mathematical_validation"]
                
                # Check specific calculation results
                calculations = [
                    "price_volume_correlation",
                    "rsi_analysis", 
                    "trend_strength",
                    "volatility_metrics",
                    "moving_average_analysis",
                    "macd_analysis"
                ]
                
                for calc in calculations:
                    if calc in math_val:
                        print(f"   ✅ {calc}: {math_val[calc]}")
                    else:
                        print(f"   ⚠️  Missing calculation: {calc}")
            else:
                print("⚠️  No mathematical validation in AI analysis")
            
            # Check for trading strategy
            if "trading_strategy" in ai_analysis:
                print("✅ Trading strategy found")
                strategy = ai_analysis["trading_strategy"]
                
                timeframes = ["short_term", "medium_term", "long_term"]
                for tf in timeframes:
                    if tf in strategy:
                        tf_data = strategy[tf]
                        print(f"   - {tf}: {tf_data.get('bias', 'N/A')} (confidence: {tf_data.get('confidence', 'N/A')})")
            else:
                print("⚠️  No trading strategy found")
        else:
            print("❌ No AI analysis found")
        
        # Save detailed result to file
        output_file = f"./test_output/enhanced_analysis_{test_symbol}_{int(time.time())}.json"
        print(f"\n💾 Saving detailed result to {output_file}")
        
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        print("✅ Test completed successfully!")
        print(f"📁 Detailed results saved to: {output_file}")
        
        # Summary
        print("\n" + "=" * 50)
        print("📋 ENHANCED ANALYSIS TEST SUMMARY")
        print("=" * 50)
        print(f"Symbol: {result.get('symbol', 'N/A')}")
        print(f"Analysis Type: {result.get('analysis_type', 'N/A')}")
        print(f"Mathematical Validation: {result.get('mathematical_validation', 'N/A')}")
        print(f"Calculation Method: {result.get('calculation_method', 'N/A')}")
        print(f"Accuracy Improvement: {result.get('accuracy_improvement', 'N/A')}")
        print(f"Risk Level: {result.get('risk_level', 'N/A')}")
        print(f"Recommendation: {result.get('recommendation', 'N/A')}")
        print(f"Current Price: {result.get('current_price', 'N/A')}")
        print(f"Analysis Time: {end_time - start_time:.2f} seconds")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

async def test_gemini_code_execution():
    """
    Test the Gemini code execution functionality directly.
    """
    print("\n🧪 Testing Gemini Code Execution Directly")
    print("=" * 50)
    
    try:
        gemini_client = GeminiClient()
        
        # Test simple calculation
        test_prompt = """
        Calculate the following mathematical operations and provide the results:
        
        1. Calculate the correlation coefficient between these price and volume data:
           Prices: [100, 102, 98, 105, 103, 107, 101, 104, 106, 108]
           Volumes: [1000, 1200, 800, 1500, 1100, 1800, 900, 1300, 1600, 2000]
        
        2. Calculate RSI for these values and count oversold/overbought periods:
           RSI values: [25, 28, 30, 32, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85]
        
        3. Calculate trend strength using linear regression for these prices:
           Prices: [100, 102, 98, 105, 103, 107, 101, 104, 106, 108]
        
        Use Python code for all calculations and show the results.
        """
        
        print("📝 Sending test prompt to Gemini with code execution...")
        text_response, code_results, execution_results = gemini_client.core.call_llm_with_code_execution(test_prompt)
        
        print(f"✅ Code execution completed")
        print(f"📊 Text response length: {len(text_response)} characters")
        print(f"💻 Code snippets generated: {len(code_results)}")
        print(f"🔢 Execution outputs: {len(execution_results)}")
        
        print("\n📋 Code Execution Results:")
        for i, code in enumerate(code_results):
            print(f"\nCode Snippet {i+1}:")
            print(code)
        
        print("\n📊 Execution Outputs:")
        for i, output in enumerate(execution_results):
            print(f"\nOutput {i+1}:")
            print(output)
        
        print("\n📝 Text Response Preview:")
        print(text_response[:500] + "..." if len(text_response) > 500 else text_response)
        
    except Exception as e:
        print(f"❌ Gemini code execution test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🧪 Enhanced Analysis with Code Execution Test Suite")
    print("=" * 60)
    
    # Run tests
    asyncio.run(test_gemini_code_execution())
    asyncio.run(test_enhanced_analysis())
    
    print("\n🎉 All tests completed!") 