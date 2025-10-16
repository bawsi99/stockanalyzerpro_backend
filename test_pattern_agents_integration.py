#!/usr/bin/env python3
"""
Integration test for Pattern Agents with recent changes.

This script verifies that:
1. Pattern agents orchestrator works correctly
2. Market structure and cross-validation agents integrate properly  
3. The /agents/patterns/analyze-all endpoint works
4. Final decision agent integration works
"""

import asyncio
import sys
import os
import json
import httpx
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the backend directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_pattern_agents_orchestrator():
    """Test the pattern agents orchestrator directly"""
    print("🧪 Testing Pattern Agents Orchestrator")
    print("=" * 60)
    
    try:
        from agents.patterns.pattern_agents import PatternAgentIntegrationManager
        
        # Create test data
        dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
        np.random.seed(42)  # For reproducible results
        
        # Generate realistic stock data
        base_price = 100
        returns = np.random.normal(0.001, 0.02, len(dates))  # 0.1% daily return, 2% volatility
        prices = base_price * np.exp(np.cumsum(returns))
        
        stock_data = pd.DataFrame({
            'Date': dates,
            'Open': prices * (1 + np.random.normal(0, 0.005, len(dates))),
            'High': prices * (1 + np.abs(np.random.normal(0.01, 0.005, len(dates)))),
            'Low': prices * (1 - np.abs(np.random.normal(0.01, 0.005, len(dates)))),
            'Close': prices,
            'Volume': np.random.randint(100000, 1000000, len(dates))
        })
        stock_data.set_index('Date', inplace=True)
        
        print(f"📊 Created test stock data: {len(stock_data)} rows")
        print(f"   Price range: {stock_data['Close'].min():.2f} - {stock_data['Close'].max():.2f}")
        
        # Test orchestrator
        manager = PatternAgentIntegrationManager()
        print("✅ PatternAgentIntegrationManager initialized")
        
        start_time = datetime.now()
        result = await manager.get_comprehensive_pattern_analysis(
            stock_data=stock_data,
            symbol="TEST",
            context="Integration test",
            include_charts=False,  # Disable charts for faster testing
            include_llm_analysis=False  # Disable LLM for faster testing
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"⏱️  Analysis completed in {duration:.2f} seconds")
        print(f"✅ Success: {result.get('success', False)}")
        print(f"📈 Overall Confidence: {result.get('overall_confidence', 0):.2%}")
        
        # Check individual agent results
        ms_analysis = result.get('market_structure_analysis', {})
        cv_analysis = result.get('cross_validation_analysis', {})
        
        print(f"🏗️  Market Structure Agent: {ms_analysis.get('success', False)}")
        if ms_analysis.get('success'):
            print(f"   Confidence: {ms_analysis.get('confidence_score', 0):.2%}")
            print(f"   Processing Time: {ms_analysis.get('processing_time', 0):.2f}s")
        
        print(f"🔄 Cross-Validation Agent: {cv_analysis.get('success', False)}")
        if cv_analysis.get('success'):
            print(f"   Confidence: {cv_analysis.get('confidence_score', 0):.2%}")
            print(f"   Processing Time: {cv_analysis.get('processing_time', 0):.2f}s")
        
        # Check aggregated results
        agents_summary = result.get('agents_summary', {})
        print(f"📊 Agents Summary:")
        print(f"   Total Agents: {agents_summary.get('total_agents', 0)}")
        print(f"   Successful: {agents_summary.get('successful_agents', 0)}")
        print(f"   Success Rate: {agents_summary.get('success_rate', 0):.1%}")
        
        # Check consensus signals
        consensus = result.get('consensus_signals', {})
        if consensus:
            print(f"🤝 Consensus Signals Available: {bool(consensus)}")
            if consensus.get('detected_patterns'):
                patterns = consensus.get('detected_patterns', [])
                print(f"   Detected Patterns: {len(patterns)} patterns")
        
        return result.get('success', False) and agents_summary.get('success_rate', 0) > 0
        
    except Exception as e:
        print(f"❌ Orchestrator Test Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_pattern_agents_endpoint():
    """Test the /agents/patterns/analyze-all endpoint"""
    print("\n🧪 Testing Pattern Agents Endpoint")
    print("=" * 60)
    
    base_url = "http://localhost:8002"
    endpoint = "/agents/patterns/analyze-all"
    test_symbol = "RELIANCE"
    
    test_payload = {
        "symbol": test_symbol,
        "exchange": "NSE",
        "interval": "day",
        "period": 180,  # Shorter period for faster testing
        "context": "Integration test for pattern agents",
        "return_prompt": False
    }
    
    try:
        print(f"📡 Making request to: {base_url}{endpoint}")
        print(f"📦 Payload: {json.dumps(test_payload, indent=2)}")
        
        async with httpx.AsyncClient(timeout=180.0) as client:
            start_time = datetime.now()
            
            response = await client.post(
                f"{base_url}{endpoint}",
                json=test_payload
            )
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            print(f"⏱️  Request completed in {duration:.2f} seconds")
            print(f"📊 Response Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                
                print("✅ SUCCESS - Pattern Agents Analysis Completed")
                print("=" * 60)
                print(f"🎯 Symbol: {result.get('symbol', 'N/A')}")
                print(f"⚡ Success: {result.get('success', False)}")
                print(f"🕒 Processing Time: {result.get('processing_time', 0):.2f}s")
                print(f"📈 Overall Confidence: {result.get('overall_confidence', 0):.2%}")
                print(f"🤖 Agent: {result.get('agent', 'N/A')}")
                
                # Check individual agent results
                ms_analysis = result.get('market_structure_analysis', {})
                cv_analysis = result.get('cross_validation_analysis', {})
                
                print(f"🏗️  Market Structure: Success={ms_analysis.get('success', False)}")
                print(f"🔄 Cross-Validation: Success={cv_analysis.get('success', False)}")
                
                # Check aggregated results
                consensus_signals = result.get('consensus_signals', {})
                pattern_conflicts = result.get('pattern_conflicts', [])
                unified_analysis = result.get('unified_analysis', {})
                
                print(f"🤝 Consensus Signals: {bool(consensus_signals)}")
                print(f"⚠️  Pattern Conflicts: {len(pattern_conflicts)}")
                print(f"🔗 Unified Analysis: {bool(unified_analysis)}")
                
                # Check final decision integration
                pattern_insights = result.get('pattern_insights_for_decision')
                if pattern_insights:
                    insights_length = len(pattern_insights) if isinstance(pattern_insights, str) else 0
                    print(f"🎯 Pattern Insights for Decision: {insights_length} characters")
                
                # Check agents summary
                agents_summary = result.get('agents_summary', {})
                print(f"📊 Agents Summary:")
                print(f"   Success Rate: {agents_summary.get('success_rate', 0):.1%}")
                print(f"   Successful: {agents_summary.get('successful_agents', 0)}")
                print(f"   Failed: {agents_summary.get('failed_agents', 0)}")
                
                return result.get('success', False)
                
            else:
                print(f"❌ FAILURE - HTTP {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"Error: {error_data.get('error', 'Unknown error')}")
                except:
                    print(f"Response Text: {response.text}")
                return False
                
    except httpx.ConnectError:
        print("❌ CONNECTION ERROR")
        print("Make sure the analysis service is running on http://localhost:8002")
        print("You can start it with: python start_analysis_service.py")
        return False
        
    except httpx.TimeoutException:
        print("❌ TIMEOUT ERROR")
        print("The request took too long to complete (>180s)")
        return False
        
    except Exception as e:
        print(f"❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_final_decision_integration():
    """Test the final decision agent integration function"""
    print("\n🧪 Testing Final Decision Integration")
    print("=" * 60)
    
    try:
        from services.analysis_service import _extract_pattern_insights_for_decision
        
        # Create mock pattern results
        mock_pattern_results = {
            'success': True,
            'overall_confidence': 0.85,
            'consensus_signals': {
                'signal_direction': 'bullish',
                'signal_strength': 'strong',
                'detected_patterns': ['ascending_triangle', 'bullish_flag', 'support_breakout']
            },
            'market_structure_analysis': {
                'success': True,
                'confidence_score': 0.88,
                'technical_analysis': {
                    'bos_events': [{'direction': 'bullish', 'price': 150.25}]
                }
            },
            'cross_validation_analysis': {
                'success': True,
                'pattern_detection': {
                    'detected_patterns': ['triangle', 'flag']
                }
            },
            'pattern_conflicts': [],
            'unified_analysis': {
                'recommendation': 'Strong bullish setup with multiple confirmations',
                'risk_assessment': 'Moderate'
            }
        }
        
        # Test the extraction function
        insights = _extract_pattern_insights_for_decision(mock_pattern_results)
        
        print("✅ Final decision integration function works")
        print(f"📋 Generated insights ({len(insights)} characters):")
        if insights:
            lines = insights.split('\n')
            for i, line in enumerate(lines[:5], 1):  # Show first 5 lines
                print(f"   {i}. {line}")
            if len(lines) > 5:
                print(f"   ... and {len(lines) - 5} more lines")
        
        return bool(insights)
        
    except Exception as e:
        print(f"❌ Final Decision Integration Test Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_integration_tests():
    """Run all integration tests"""
    print("🚀 Pattern Agents Integration Test Suite")
    print("=" * 60)
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Run async tests
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        # Test 1: Direct orchestrator test
        test1_passed = loop.run_until_complete(test_pattern_agents_orchestrator())
        
        # Test 2: Endpoint test
        test2_passed = loop.run_until_complete(test_pattern_agents_endpoint())
        
        # Test 3: Final decision integration
        test3_passed = test_final_decision_integration()
        
        # Summary
        print("\n" + "=" * 60)
        print("📋 INTEGRATION TEST SUMMARY")
        print("=" * 60)
        print(f"✅ Orchestrator Test: {'PASSED' if test1_passed else 'FAILED'}")
        print(f"✅ Endpoint Test: {'PASSED' if test2_passed else 'FAILED'}")
        print(f"✅ Final Decision Integration: {'PASSED' if test3_passed else 'FAILED'}")
        
        overall_success = test1_passed and test2_passed and test3_passed
        print(f"\n🎯 Overall Result: {'ALL TESTS PASSED' if overall_success else 'SOME TESTS FAILED'}")
        
        if overall_success:
            print("\n🎉 Pattern Agents Integration is working correctly!")
            print("   ✅ Orchestrator properly coordinates both agents")
            print("   ✅ Endpoint integrates with analysis service")
            print("   ✅ Final decision agent receives proper insights")
            print("   ✅ All recent changes are properly integrated")
        else:
            print("\n⚠️  Some integration tests failed.")
            print("   📝 Check the error messages above for details")
            
        return overall_success
        
    finally:
        loop.close()

if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)