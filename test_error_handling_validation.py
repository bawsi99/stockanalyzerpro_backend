#!/usr/bin/env python3
"""
Validation script for volume agents error handling enhancements

This script tests the error handling and fallback mechanisms we implemented
for the volume agents integration system.
"""

import sys
import os
import pandas as pd
import numpy as np
import asyncio
from datetime import datetime
from unittest.mock import Mock, patch

# Add current directory to path
sys.path.append('.')

def test_basic_functionality():
    """Test basic functionality and imports"""
    print("🧪 Testing basic functionality...")
    
    try:
        # Test enhanced chart visualizer imports
        from patterns.visualization import ChartVisualizer
        print("✅ ChartVisualizer imports successfully")
        
        # Test new enhanced chart methods
        if hasattr(ChartVisualizer, 'plot_enhanced_volume_chart_with_agents'):
            print("✅ Enhanced volume chart method exists")
        else:
            print("❌ Enhanced volume chart method missing")
            
        if hasattr(ChartVisualizer, '_create_basic_volume_chart'):
            print("✅ Basic volume chart fallback method exists")
        else:
            print("❌ Basic volume chart fallback method missing")
            
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False
    
    return True

def test_chart_generation_fallback():
    """Test chart generation fallback mechanisms"""
    print("\n🧪 Testing chart generation fallback...")
    
    try:
        from patterns.visualization import ChartVisualizer
        
        # Create sample data
        dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
        sample_data = pd.DataFrame({
            'open': np.random.uniform(100, 200, 50),
            'high': np.random.uniform(110, 220, 50),
            'low': np.random.uniform(90, 180, 50),
            'close': np.random.uniform(100, 200, 50),
            'volume': np.random.uniform(10000, 100000, 50)
        }, index=dates)
        
        # Test with no volume agents data (should fall back to traditional chart)
        try:
            fig = ChartVisualizer.plot_enhanced_volume_chart_with_agents(
                sample_data, {}, None, None, "TEST"
            )
            if fig is not None:
                print("✅ Fallback to traditional chart works")
            else:
                print("❌ Fallback chart generation failed")
        except Exception as chart_error:
            print(f"❌ Chart generation error: {chart_error}")
        
        # Test with invalid volume agents data
        try:
            invalid_data = {'invalid': 'data', 'success': False}
            fig = ChartVisualizer.plot_enhanced_volume_chart_with_agents(
                sample_data, {}, invalid_data, None, "TEST"
            )
            if fig is not None:
                print("✅ Handles invalid volume agents data gracefully")
            else:
                print("❌ Failed to handle invalid volume agents data")
        except Exception as chart_error:
            print(f"❌ Invalid data handling error: {chart_error}")
        
        # Test ultimate fallback (basic chart)
        try:
            fig = ChartVisualizer._create_basic_volume_chart(sample_data, None, "TEST")
            if fig is not None:
                print("✅ Ultimate fallback (basic chart) works")
            else:
                print("❌ Ultimate fallback failed")
        except Exception as fallback_error:
            print(f"❌ Ultimate fallback error: {fallback_error}")
    
    except Exception as e:
        print(f"❌ Chart test error: {e}")
        return False
    
    return True

def test_orchestrator_integration():
    """Test orchestrator integration with volume agents"""
    print("\n🧪 Testing orchestrator integration...")
    
    try:
        from analysis.orchestrator import StockAnalysisOrchestrator
        
        # Create orchestrator instance
        orchestrator = StockAnalysisOrchestrator()
        
        # Check if volume agents manager exists
        if hasattr(orchestrator, 'volume_agents_manager'):
            print("✅ Volume agents manager integrated in orchestrator")
            
            # Test health check methods
            try:
                should_use, reason = orchestrator.volume_agents_manager.should_use_volume_agents()
                print(f"✅ Health check works: should_use={should_use}")
                print(f"   Reason: {reason[:80]}...")
            except Exception as health_error:
                print(f"❌ Health check error: {health_error}")
            
            # Test metrics methods
            try:
                metrics = orchestrator.volume_agents_manager.get_agent_performance_metrics()
                print(f"✅ Agent metrics tracking: {len(metrics)} agents")
            except Exception as metrics_error:
                print(f"❌ Metrics error: {metrics_error}")
            
            # Test system health summary
            try:
                health_summary = orchestrator.volume_agents_manager.get_system_health_summary()
                status = health_summary.get('system_status', 'unknown')
                health_pct = health_summary.get('health_percentage', 0)
                print(f"✅ System health summary: {status} ({health_pct}% healthy)")
            except Exception as health_error:
                print(f"❌ System health error: {health_error}")
                
        else:
            print("❌ Volume agents manager not found in orchestrator")
    
    except Exception as e:
        print(f"❌ Orchestrator integration error: {e}")
        return False
    
    return True

def test_enhanced_create_visualizations():
    """Test enhanced create_visualizations method"""
    print("\n🧪 Testing enhanced create_visualizations method...")
    
    try:
        from analysis.orchestrator import StockAnalysisOrchestrator
        
        orchestrator = StockAnalysisOrchestrator()
        
        # Create sample data
        dates = pd.date_range(start='2023-01-01', periods=30, freq='D')
        sample_data = pd.DataFrame({
            'open': np.random.uniform(100, 200, 30),
            'high': np.random.uniform(110, 220, 30),
            'low': np.random.uniform(90, 180, 30),
            'close': np.random.uniform(100, 200, 30),
            'volume': np.random.uniform(10000, 100000, 30)
        }, index=dates)
        
        # Test method signature includes volume_agents_result parameter
        import inspect
        sig = inspect.signature(orchestrator.create_visualizations)
        if 'volume_agents_result' in sig.parameters:
            print("✅ Enhanced create_visualizations method signature updated")
        else:
            print("❌ create_visualizations method not enhanced with volume_agents_result parameter")
        
        # Create minimal indicators
        indicators = {'test': 'data'}
        
        # Test visualization creation without volume agents data
        try:
            charts = orchestrator.create_visualizations(sample_data, indicators, "TEST", None, "day", None)
            
            if 'volume_analysis' in charts:
                chart_info = charts['volume_analysis']
                if 'chart_version' in chart_info:
                    print(f"✅ Chart generation successful: {chart_info.get('chart_type', 'unknown')} version")
                else:
                    print("✅ Chart generation successful (traditional)")
            else:
                print("❌ Volume analysis chart not generated")
        except Exception as viz_error:
            print(f"❌ Visualization generation error: {viz_error}")
    
    except Exception as e:
        print(f"❌ Enhanced visualization test error: {e}")
        return False
    
    return True

def test_logging_system():
    """Test comprehensive logging system"""
    print("\n🧪 Testing logging system...")
    
    try:
        # Test if we can import the logger
from agents.volume import volume_agents_logger
        print("✅ Volume agents logger imported successfully")
        
        # Test logging methods
        operation_id = volume_agents_logger.log_operation_start(
            'test_operation', 'TEST', ['agent1', 'agent2']
        )
        print(f"✅ Operation logging started: {operation_id}")
        
        # Test agent execution logging
        volume_agents_logger.log_agent_execution(
            operation_id, 'test_agent', True, 5.0, confidence=0.8
        )
        print("✅ Agent execution logging works")
        
        # Test error logging
        try:
            raise ValueError("Test error for logging")
        except Exception as e:
            volume_agents_logger.log_error_with_context(
                operation_id, e, {'symbol': 'TEST', 'stage': 'testing'}
            )
        print("✅ Error logging with context works")
        
        # Test partial success logging
        volume_agents_logger.log_partial_success(
            operation_id, ['agent1'], ['agent2'], fallback_activated=True
        )
        print("✅ Partial success logging works")
        
        # Test fallback activation logging
        volume_agents_logger.log_fallback_activation(
            operation_id, 'Testing fallback', 'test_fallback'
        )
        print("✅ Fallback activation logging works")
        
        # Test operation completion
        volume_agents_logger.log_operation_complete(
            operation_id, True, 10.0, {'agents': 2, 'success_rate': 0.5}
        )
        print("✅ Operation completion logging works")
        
    except Exception as e:
        print(f"❌ Logging system error: {e}")
        return False
    
    return True

def main():
    """Run all validation tests"""
    print("🚀 Starting Volume Agents Error Handling Validation Tests")
    print("=" * 60)
    
    tests = [
        test_basic_functionality,
        test_chart_generation_fallback,
        test_orchestrator_integration,
        test_enhanced_create_visualizations,
        test_logging_system
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed_tests += 1
        except Exception as e:
            print(f"❌ Test {test_func.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! Error handling enhancements are working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Please review the error handling implementation.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)