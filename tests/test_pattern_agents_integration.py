#!/usr/bin/env python3
"""
Test script for pattern agents integration with analysis service.

This script tests:
1. Individual pattern agent endpoints
2. Combined pattern analysis endpoint
3. Data flow and response format validation
4. Cache usage and performance
5. Integration with the final decision agent

Run this script to verify the pattern agents integration is working correctly.
"""

import asyncio
import httpx
import json
import time
from datetime import datetime
from typing import Dict, Any


class PatternAgentsIntegrationTester:
    """Test runner for pattern agents integration."""
    
    def __init__(self, base_url: str = "http://localhost:8002"):
        self.base_url = base_url
        self.test_symbol = "RELIANCE"
        self.test_exchange = "NSE"
        self.test_period = 90
        self.test_interval = "day"
        
    async def test_health_check(self) -> bool:
        """Test if the analysis service is running."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/health", timeout=10.0)
                if response.status_code == 200:
                    health_data = response.json()
                    print(f"✅ Analysis service is healthy")
                    print(f"   Service: {health_data.get('service')}")
                    print(f"   Status: {health_data.get('status')}")
                    return True
                else:
                    print(f"❌ Analysis service health check failed: {response.status_code}")
                    return False
        except Exception as e:
            print(f"❌ Failed to connect to analysis service: {e}")
            return False
    
    async def test_market_structure_agent(self) -> Dict[str, Any]:
        """Test individual market structure agent endpoint."""
        print(f"\n🧪 Testing Market Structure Agent for {self.test_symbol}")
        start_time = time.monotonic()
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/agents/patterns/market-structure",
                    json={
                        "symbol": self.test_symbol,
                        "exchange": self.test_exchange,
                        "period": self.test_period,
                        "interval": self.test_interval,
                        "context": f"Testing market structure analysis for {self.test_symbol}"
                    },
                    timeout=120.0
                )
                
                elapsed = time.monotonic() - start_time
                
                if response.status_code == 200:
                    result = response.json()
                    success = result.get('success', False)
                    
                    if success:
                        print(f"✅ Market Structure Agent completed in {elapsed:.2f}s")
                        print(f"   Confidence: {result.get('confidence_score', 0):.1%}")
                        print(f"   Agent: {result.get('agent')}")
                        print(f"   Has Chart: {result.get('has_chart', False)}")
                        
                        # Check technical analysis results
                        tech_analysis = result.get('technical_analysis', {})
                        if tech_analysis:
                            print(f"   Technical Analysis Keys: {list(tech_analysis.keys())}")
                        
                        return result
                    else:
                        print(f"❌ Market Structure Agent failed: {result.get('error')}")
                        return result
                else:
                    print(f"❌ Market Structure Agent HTTP error: {response.status_code}")
                    print(f"   Response: {response.text}")
                    return {"success": False, "error": f"HTTP {response.status_code}"}
                    
        except Exception as e:
            elapsed = time.monotonic() - start_time
            print(f"❌ Market Structure Agent exception after {elapsed:.2f}s: {e}")
            return {"success": False, "error": str(e)}
    
    async def test_cross_validation_agent(self) -> Dict[str, Any]:
        """Test individual cross-validation agent endpoint."""
        print(f"\n🧪 Testing Cross-Validation Agent for {self.test_symbol}")
        start_time = time.monotonic()
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/agents/patterns/cross-validation",
                    json={
                        "symbol": self.test_symbol,
                        "exchange": self.test_exchange,
                        "period": self.test_period,
                        "interval": self.test_interval,
                        "context": f"Testing cross-validation analysis for {self.test_symbol}"
                    },
                    timeout=180.0
                )
                
                elapsed = time.monotonic() - start_time
                
                if response.status_code == 200:
                    result = response.json()
                    success = result.get('success', False)
                    
                    if success:
                        print(f"✅ Cross-Validation Agent completed in {elapsed:.2f}s")
                        print(f"   Agent: {result.get('agent')}")
                        
                        # Check pattern detection results
                        pattern_detection = result.get('pattern_detection', {})
                        if pattern_detection:
                            patterns = pattern_detection.get('detected_patterns', [])
                            print(f"   Patterns Detected: {len(patterns)}")
                            
                        # Check cross-validation results
                        cross_validation = result.get('cross_validation', {})
                        if cross_validation:
                            print(f"   Cross-Validation Keys: {list(cross_validation.keys())}")
                        
                        # Check components executed
                        components = result.get('components_executed', [])
                        print(f"   Components Executed: {len(components)} - {components}")
                        
                        return result
                    else:
                        print(f"❌ Cross-Validation Agent failed: {result.get('error')}")
                        return result
                else:
                    print(f"❌ Cross-Validation Agent HTTP error: {response.status_code}")
                    print(f"   Response: {response.text}")
                    return {"success": False, "error": f"HTTP {response.status_code}"}
                    
        except Exception as e:
            elapsed = time.monotonic() - start_time
            print(f"❌ Cross-Validation Agent exception after {elapsed:.2f}s: {e}")
            return {"success": False, "error": str(e)}
    
    async def test_comprehensive_pattern_analysis(self) -> Dict[str, Any]:
        """Test the comprehensive pattern analysis endpoint (analyze-all)."""
        print(f"\n🧪 Testing Comprehensive Pattern Analysis for {self.test_symbol}")
        start_time = time.monotonic()
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/agents/patterns/analyze-all",
                    json={
                        "symbol": self.test_symbol,
                        "exchange": self.test_exchange,
                        "period": self.test_period,
                        "interval": self.test_interval,
                        "context": f"Testing comprehensive pattern analysis for {self.test_symbol}",
                        "return_prompt": False
                    },
                    timeout=300.0  # 5 minutes for comprehensive analysis
                )
                
                elapsed = time.monotonic() - start_time
                
                if response.status_code == 200:
                    result = response.json()
                    success = result.get('success', False)
                    
                    if success:
                        print(f"✅ Comprehensive Pattern Analysis completed in {elapsed:.2f}s")
                        print(f"   Overall Confidence: {result.get('overall_confidence', 0):.1%}")
                        print(f"   Agent: {result.get('agent')}")
                        
                        # Check individual agent results
                        ms_analysis = result.get('market_structure_analysis', {})
                        cv_analysis = result.get('cross_validation_analysis', {})
                        
                        print(f"   Market Structure Success: {ms_analysis.get('success', False)}")
                        print(f"   Cross-Validation Success: {cv_analysis.get('success', False)}")
                        
                        # Check consensus signals
                        consensus_signals = result.get('consensus_signals', {})
                        if consensus_signals:
                            print(f"   Consensus Signal Direction: {consensus_signals.get('signal_direction', 'unknown')}")
                            print(f"   Consensus Signal Strength: {consensus_signals.get('signal_strength', 'unknown')}")
                            detected_patterns = consensus_signals.get('detected_patterns', [])
                            print(f"   Consensus Detected Patterns: {len(detected_patterns)}")
                        
                        # Check pattern conflicts
                        conflicts = result.get('pattern_conflicts', [])
                        print(f"   Pattern Conflicts: {len(conflicts)}")
                        
                        # Check agents summary
                        agents_summary = result.get('agents_summary', {})
                        if agents_summary:
                            success_rate = agents_summary.get('success_rate', 0)
                            print(f"   Agent Success Rate: {success_rate:.1%}")
                        
                        # CRITICAL: Check pattern_insights_for_decision (final decision integration)
                        pattern_insights = result.get('pattern_insights_for_decision', '')
                        if pattern_insights:
                            print(f"   Pattern Insights for Decision: {len(pattern_insights)} chars")
                            print(f"   Preview: {pattern_insights[:200]}...")
                        else:
                            print(f"   ⚠️  No pattern insights for decision agent")
                        
                        return result
                    else:
                        print(f"❌ Comprehensive Pattern Analysis failed: {result.get('error')}")
                        return result
                else:
                    print(f"❌ Comprehensive Pattern Analysis HTTP error: {response.status_code}")
                    print(f"   Response: {response.text}")
                    return {"success": False, "error": f"HTTP {response.status_code}"}
                    
        except Exception as e:
            elapsed = time.monotonic() - start_time
            print(f"❌ Comprehensive Pattern Analysis exception after {elapsed:.2f}s: {e}")
            return {"success": False, "error": str(e)}
    
    async def test_enhanced_analyze_integration(self) -> Dict[str, Any]:
        """Test pattern agents integration with enhanced analyze endpoint."""
        print(f"\n🧪 Testing Enhanced Analyze with Pattern Integration for {self.test_symbol}")
        start_time = time.monotonic()
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/analyze/enhanced",
                    json={
                        "stock": self.test_symbol,
                        "exchange": self.test_exchange,
                        "period": self.test_period,
                        "interval": self.test_interval,
                        "output": "json",
                        "enable_code_execution": True,
                        "user_id": "test_user_pattern_integration"
                    },
                    timeout=600.0  # 10 minutes for full analysis
                )
                
                elapsed = time.monotonic() - start_time
                
                if response.status_code == 200:
                    result = response.json()
                    success = result.get('success', False)
                    
                    if success:
                        print(f"✅ Enhanced Analyze with Pattern Integration completed in {elapsed:.2f}s")
                        
                        # Check if AI analysis includes pattern insights
                        ai_analysis = result.get('ai_analysis', {})
                        if ai_analysis:
                            # Look for pattern-related content in the final decision
                            analysis_text = str(ai_analysis)
                            pattern_keywords = ['pattern', 'structure', 'BOS', 'CHOCH', 'triangle', 'flag', 'channel']
                            pattern_mentions = sum(1 for keyword in pattern_keywords if keyword.lower() in analysis_text.lower())
                            print(f"   Pattern-related mentions in AI analysis: {pattern_mentions}")
                            
                            if pattern_mentions > 0:
                                print(f"   ✅ Pattern insights successfully integrated into final decision")
                            else:
                                print(f"   ⚠️  No obvious pattern insights in final decision")
                        
                        # Check processing times
                        processing_time = result.get('processing_time', 0)
                        print(f"   Total Processing Time: {processing_time:.2f}s")
                        
                        # Check token usage if available
                        token_usage = result.get('token_usage_summary')
                        if token_usage:
                            total_tokens = token_usage.get('total_tokens', 0)
                            print(f"   Total Tokens Used: {total_tokens}")
                            
                            # Check for pattern agent token usage
                            agent_tokens = token_usage.get('by_agent', {})
                            pattern_tokens = agent_tokens.get('pattern_analysis', 0)
                            if pattern_tokens > 0:
                                print(f"   Pattern Analysis Tokens: {pattern_tokens}")
                        
                        return result
                    else:
                        print(f"❌ Enhanced Analyze failed: {result.get('error')}")
                        return result
                else:
                    print(f"❌ Enhanced Analyze HTTP error: {response.status_code}")
                    print(f"   Response: {response.text}")
                    return {"success": False, "error": f"HTTP {response.status_code}"}
                    
        except Exception as e:
            elapsed = time.monotonic() - start_time
            print(f"❌ Enhanced Analyze exception after {elapsed:.2f}s: {e}")
            return {"success": False, "error": str(e)}
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all pattern agents integration tests."""
        print(f"🚀 Starting Pattern Agents Integration Tests")
        print(f"Target: {self.base_url}")
        print(f"Test Symbol: {self.test_symbol}")
        print(f"Test Parameters: {self.test_period} {self.test_interval} periods")
        print("=" * 80)
        
        overall_start = time.monotonic()
        results = {}
        
        # Test 1: Health check
        health_ok = await self.test_health_check()
        results['health_check'] = health_ok
        
        if not health_ok:
            print("❌ Cannot proceed with tests - service is not healthy")
            return results
        
        # Test 2: Individual agents
        ms_result = await self.test_market_structure_agent()
        results['market_structure'] = ms_result
        
        cv_result = await self.test_cross_validation_agent()
        results['cross_validation'] = cv_result
        
        # Test 3: Comprehensive analysis
        comp_result = await self.test_comprehensive_pattern_analysis()
        results['comprehensive_analysis'] = comp_result
        
        # Test 4: Full integration
        enhanced_result = await self.test_enhanced_analyze_integration()
        results['enhanced_integration'] = enhanced_result
        
        # Summary
        overall_elapsed = time.monotonic() - overall_start
        print(f"\n" + "=" * 80)
        print(f"🏁 Pattern Agents Integration Tests Completed in {overall_elapsed:.2f}s")
        print(f"=" * 80)
        
        # Test results summary
        test_names = ['health_check', 'market_structure', 'cross_validation', 'comprehensive_analysis', 'enhanced_integration']
        passed_tests = 0
        total_tests = len(test_names)
        
        for test_name in test_names:
            result = results.get(test_name)
            if test_name == 'health_check':
                status = "✅ PASS" if result else "❌ FAIL"
            else:
                status = "✅ PASS" if isinstance(result, dict) and result.get('success', False) else "❌ FAIL"
            
            if "PASS" in status:
                passed_tests += 1
            
            print(f"{test_name.replace('_', ' ').title():<30} {status}")
        
        print(f"\nOverall: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests:.1%})")
        
        if passed_tests == total_tests:
            print("🎉 All tests passed! Pattern agents integration is working correctly.")
        elif passed_tests >= 3:
            print("⚠️  Most tests passed. Pattern agents integration is mostly working.")
        else:
            print("❌ Multiple test failures. Pattern agents integration needs attention.")
        
        return results


async def main():
    """Main test runner."""
    tester = PatternAgentsIntegrationTester()
    results = await tester.run_all_tests()
    
    # Optionally save results to file
    with open('pattern_agents_test_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📄 Detailed results saved to: pattern_agents_test_results.json")


if __name__ == "__main__":
    asyncio.run(main())