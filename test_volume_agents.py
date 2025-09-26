#!/usr/bin/env python3
"""
Comprehensive test script for Volume Agents Integration System
Tests all major components without requiring external dependencies
"""

import sys
import os
import traceback
from datetime import datetime
from unittest.mock import Mock, MagicMock
import json

def test_data_structures():
    """Test volume agents data structures"""
    print("=== Testing Volume Agents Data Structures ===")
    
    try:
        # Test importing data structures
        sys.path.insert(0, '/Users/aaryanmanawat/Aaryan/StockAnalyzer Pro/version3.0/3.0/backend')
from agents.volume import VolumeAgentResult, AggregatedVolumeAnalysis
        
        # Test VolumeAgentResult creation
        result = VolumeAgentResult(
            agent_name="test_agent",
            success=True,
            processing_time=1.5,
            confidence_score=0.85
        )
        
        assert result.agent_name == "test_agent"
        assert result.success == True
        assert result.processing_time == 1.5
        assert result.confidence_score == 0.85
        print("✅ VolumeAgentResult data structure works")
        
        # Test AggregatedVolumeAnalysis creation
        aggregated = AggregatedVolumeAnalysis(
            total_processing_time=5.2,
            successful_agents=3,
            failed_agents=2,
            overall_confidence=0.75
        )
        
        assert aggregated.total_processing_time == 5.2
        assert aggregated.successful_agents == 3
        assert aggregated.failed_agents == 2
        assert aggregated.overall_confidence == 0.75
        print("✅ AggregatedVolumeAnalysis data structure works")
        
        return True
        
    except Exception as e:
        print(f"❌ Data structures test failed: {e}")
        traceback.print_exc()
        return False

def test_input_validation():
    """Test input validation methods"""
    print("\n=== Testing Input Validation ===")
    
    try:
from agents.volume import VolumeAgentsOrchestrator
        import pandas as pd
        import numpy as np
        
        # Create orchestrator instance (without gemini client for testing)
        orchestrator = VolumeAgentsOrchestrator(gemini_client=None)
        
        # Test valid inputs
        valid_data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 106, 107], 
            'low': [98, 99, 100],
            'close': [104, 105, 106],
            'volume': [1000, 1500, 1200]
        })
        
        validation_result = orchestrator._validate_inputs(valid_data, "TEST", {})
        assert validation_result is None, f"Valid data should pass validation, got: {validation_result}"
        print("✅ Valid input validation passed")
        
        # Test invalid symbol
        validation_result = orchestrator._validate_inputs(valid_data, "", {})
        assert "Invalid symbol" in validation_result
        print("✅ Invalid symbol validation works")
        
        # Test empty data
        empty_data = pd.DataFrame()
        validation_result = orchestrator._validate_inputs(empty_data, "TEST", {})
        assert "empty" in validation_result.lower()
        print("✅ Empty data validation works")
        
        # Test missing columns
        invalid_data = pd.DataFrame({'price': [100, 101, 102]})
        validation_result = orchestrator._validate_inputs(invalid_data, "TEST", {})
        assert "Missing required columns" in validation_result
        print("✅ Missing columns validation works")
        
        # Test insufficient data points
        small_data = pd.DataFrame({
            'open': [100], 'high': [105], 'low': [98], 'close': [104], 'volume': [1000]
        })
        validation_result = orchestrator._validate_inputs(small_data, "TEST", {})
        assert "Insufficient data points" in validation_result
        print("✅ Insufficient data validation works")
        
        return True
        
    except Exception as e:
        print(f"❌ Input validation test failed: {e}")
        traceback.print_exc()
        return False

def test_fallback_mechanisms():
    """Test fallback and error handling mechanisms"""
    print("\n=== Testing Fallback Mechanisms ===")
    
    try:
from agents.volume import VolumeAgentsOrchestrator
        
        # Create orchestrator
        orchestrator = VolumeAgentsOrchestrator(gemini_client=None)
        
        # Test fallback result creation
        fallback_result = orchestrator._create_fallback_result("TEST", "Test error message", 2.5)
        
        assert fallback_result.successful_agents == 0
        assert fallback_result.failed_agents == len(orchestrator.agent_config)
        assert fallback_result.total_processing_time == 2.5
        assert fallback_result.overall_confidence == 0.0
        assert "Test error message" in fallback_result.unified_analysis.get('error', '')
        print("✅ Fallback result creation works")
        
        # Test integration manager fallback
from agents.volume import VolumeAgentIntegrationManager
        
        integration_manager = VolumeAgentIntegrationManager(gemini_client=None)
        
        # Test degraded analysis result
        degraded_result = integration_manager._create_degraded_analysis_result(
            "TEST", "System failure", 1.8
        )
        
        assert degraded_result['success'] == False
        assert degraded_result['degraded_mode'] == True
        assert degraded_result['processing_time'] == 1.8
        assert "System failure" in degraded_result['error']
        print("✅ Degraded analysis result creation works")
        
        return True
        
    except Exception as e:
        print(f"❌ Fallback mechanisms test failed: {e}")
        traceback.print_exc()
        return False

def test_agent_configuration():
    """Test agent configuration system"""
    print("\n=== Testing Agent Configuration ===")
    
    try:
from agents.volume import VolumeAgentsOrchestrator
        
        orchestrator = VolumeAgentsOrchestrator(gemini_client=None)
        
        # Test agent configuration structure
        config = orchestrator.agent_config
        
        expected_agents = [
            'volume_anomaly', 'institutional_activity', 'volume_confirmation',
            'support_resistance', 'volume_momentum'
        ]
        
        for agent_name in expected_agents:
            assert agent_name in config, f"Agent {agent_name} not found in configuration"
            agent_config = config[agent_name]
            
            # Check required configuration fields
            assert 'enabled' in agent_config
            assert 'weight' in agent_config
            assert 'timeout' in agent_config
            assert 'processor' in agent_config
            assert 'charts' in agent_config
            assert 'prompt_template' in agent_config
            
            # Check weight is reasonable
            assert 0 < agent_config['weight'] <= 1, f"Invalid weight for {agent_name}: {agent_config['weight']}"
            
            # Check timeout is reasonable
            assert agent_config['timeout'] > 0, f"Invalid timeout for {agent_name}: {agent_config['timeout']}"
            
        print("✅ All agents have proper configuration")
        
        # Test weight distribution
        total_weight = sum(config[agent]['weight'] for agent in config)
        assert abs(total_weight - 1.0) < 0.01, f"Weights don't sum to 1.0, got: {total_weight}"
        print("✅ Agent weights properly distributed")
        
        return True
        
    except Exception as e:
        print(f"❌ Agent configuration test failed: {e}")
        traceback.print_exc()
        return False

def test_integration_with_orchestrator():
    """Test integration with main orchestrator"""
    print("\n=== Testing Integration with Main Orchestrator ===")
    
    try:
        # Test that StockAnalysisOrchestrator can import volume agents
        from analysis.orchestrator import StockAnalysisOrchestrator
        
        # Create orchestrator instance (this will test the import)
        orchestrator = StockAnalysisOrchestrator()
        
        # Check that volume agents manager was initialized
        assert hasattr(orchestrator, 'volume_agents_manager'), "Volume agents manager not initialized"
        assert orchestrator.volume_agents_manager is not None, "Volume agents manager is None"
        print("✅ Volume agents manager properly initialized in main orchestrator")
        
        # Test that the manager has the right type
from agents.volume import VolumeAgentIntegrationManager
        assert isinstance(orchestrator.volume_agents_manager, VolumeAgentIntegrationManager)
        print("✅ Volume agents manager has correct type")
        
        # Test health check method exists and works
        health_check_result = orchestrator.volume_agents_manager.is_volume_agents_healthy()
        assert isinstance(health_check_result, bool), "Health check should return boolean"
        print(f"✅ Volume agents health check works (result: {health_check_result})")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration with orchestrator test failed: {e}")
        traceback.print_exc()
        return False

def test_quality_assessment():
    """Test result quality assessment"""
    print("\n=== Testing Quality Assessment ===")
    
    try:
from agents.volume import VolumeAgentIntegrationManager, AggregatedVolumeAnalysis
        
        integration_manager = VolumeAgentIntegrationManager(gemini_client=None)
        
        # Test excellent quality (all agents successful)
        excellent_result = AggregatedVolumeAnalysis(
            successful_agents=5, failed_agents=0, overall_confidence=0.9
        )
        quality = integration_manager._assess_result_quality(excellent_result)
        assert quality == 'excellent', f"Expected 'excellent', got: {quality}"
        print("✅ Excellent quality assessment works")
        
        # Test good quality (80% success)
        good_result = AggregatedVolumeAnalysis(
            successful_agents=4, failed_agents=1, overall_confidence=0.8
        )
        quality = integration_manager._assess_result_quality(good_result)
        assert quality == 'good', f"Expected 'good', got: {quality}"
        print("✅ Good quality assessment works")
        
        # Test poor quality (20% success)
        poor_result = AggregatedVolumeAnalysis(
            successful_agents=1, failed_agents=4, overall_confidence=0.2
        )
        quality = integration_manager._assess_result_quality(poor_result)
        assert quality == 'poor', f"Expected 'poor', got: {quality}"
        print("✅ Poor quality assessment works")
        
        # Test failed quality (no success)
        failed_result = AggregatedVolumeAnalysis(
            successful_agents=0, failed_agents=5, overall_confidence=0.0
        )
        quality = integration_manager._assess_result_quality(failed_result)
        assert quality == 'failed', f"Expected 'failed', got: {quality}"
        print("✅ Failed quality assessment works")
        
        return True
        
    except Exception as e:
        print(f"❌ Quality assessment test failed: {e}")
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all tests and report results"""
    print("🚀 Starting Volume Agents Integration Tests")
    print("=" * 60)
    
    tests = [
        ("Data Structures", test_data_structures),
        ("Input Validation", test_input_validation), 
        ("Fallback Mechanisms", test_fallback_mechanisms),
        ("Agent Configuration", test_agent_configuration),
        ("Integration with Orchestrator", test_integration_with_orchestrator),
        ("Quality Assessment", test_quality_assessment)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results[test_name] = success
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results[test_name] = False
    
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name:<30} {status}")
        if success:
            passed += 1
    
    print("=" * 60)
    print(f"Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All tests passed! Volume agents integration is working correctly.")
        return True
    else:
        print(f"⚠️  {total-passed} test(s) failed. Check the details above.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)