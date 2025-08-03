#!/usr/bin/env python3
"""
Test script for Chart Manager functionality.
Run this to verify chart management is working correctly.
"""

import os
import time
import tempfile
import shutil
from pathlib import Path
from chart_manager import ChartManager, get_chart_manager, initialize_chart_manager
from deployment_config import DeploymentConfig, get_deployment_recommendations

def test_chart_manager_basic():
    """Test basic chart manager functionality."""
    print("🧪 Testing Chart Manager Basic Functionality")
    
    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Initialize chart manager with test settings
        chart_manager = ChartManager(
            base_output_dir=temp_dir,
            max_age_hours=1,  # 1 hour for testing
            max_total_size_mb=10,  # 10MB for testing
            cleanup_interval_minutes=1  # 1 minute for testing
        )
        
        # Test directory creation
        symbol = "TEST"
        interval = "1day"
        chart_dir = chart_manager.create_chart_directory(symbol, interval)
        assert chart_dir.exists(), "Chart directory should be created"
        
        # Create a test chart file
        test_chart_path = chart_dir / "test_chart.png"
        test_chart_path.write_bytes(b"fake chart data" * 1000)  # Make it larger
        assert test_chart_path.exists(), "Test chart should be created"
        
        # Test storage stats
        stats = chart_manager.get_storage_stats()
        assert stats['total_files'] == 1, "Should have 1 file"
        assert stats['total_size_mb'] >= 0, "Should have non-negative size"
        
        print("✅ Basic functionality test passed")

def test_chart_manager_cleanup():
    """Test chart cleanup functionality."""
    print("🧪 Testing Chart Manager Cleanup")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        chart_manager = ChartManager(
            base_output_dir=temp_dir,
            max_age_hours=0.01,  # Very short for testing
            max_total_size_mb=1,  # Small size for testing
            cleanup_interval_minutes=1
        )
        
        # Create test files
        symbol = "CLEANUP_TEST"
        interval = "1day"
        chart_dir = chart_manager.create_chart_directory(symbol, interval)
        
        # Create multiple test files
        for i in range(3):
            test_file = chart_dir / f"test_chart_{i}.png"
            test_file.write_bytes(b"fake chart data" * 1000)  # Larger file
        
        # Manually age the files by touching them with old timestamps
        import os
        old_time = time.time() - (2 * 3600)  # 2 hours ago
        for test_file in chart_dir.glob("*.png"):
            os.utime(test_file, (old_time, old_time))
        
        # Test cleanup
        stats = chart_manager.cleanup_old_charts()
        assert stats['files_removed'] > 0, "Should remove old files"
        
        print("✅ Cleanup test passed")

def test_deployment_config():
    """Test deployment configuration."""
    print("🧪 Testing Deployment Configuration")
    
    # Test environment detection
    env_info = DeploymentConfig.get_environment_info()
    assert 'environment' in env_info, "Should have environment info"
    assert 'chart_config' in env_info, "Should have chart config"
    
    # Test recommendations
    recommendations = get_deployment_recommendations()
    assert 'current_environment' in recommendations, "Should have current environment"
    assert 'recommendations' in recommendations, "Should have recommendations"
    
    print("✅ Deployment config test passed")

def test_global_chart_manager():
    """Test global chart manager instance."""
    print("🧪 Testing Global Chart Manager")
    
    # Test global instance
    manager1 = get_chart_manager()
    manager2 = get_chart_manager()
    assert manager1 is manager2, "Should return same instance"
    
    # Test initialization
    with tempfile.TemporaryDirectory() as temp_dir:
        custom_manager = initialize_chart_manager(
            base_output_dir=temp_dir,
            max_age_hours=2,
            max_total_size_mb=50
        )
        assert custom_manager.base_output_dir == Path(temp_dir), "Should use custom directory"
    
    print("✅ Global chart manager test passed")

def test_api_endpoints_simulation():
    """Simulate API endpoint functionality."""
    print("🧪 Testing API Endpoint Simulation")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        chart_manager = ChartManager(
            base_output_dir=temp_dir,
            max_age_hours=1,
            max_total_size_mb=10
        )
        
        # Simulate storage stats endpoint
        stats = chart_manager.get_storage_stats()
        assert isinstance(stats, dict), "Stats should be a dictionary"
        
        # Simulate cleanup endpoint
        cleanup_stats = chart_manager.cleanup_old_charts()
        assert isinstance(cleanup_stats, dict), "Cleanup stats should be a dictionary"
        
        # Simulate specific cleanup
        success = chart_manager.cleanup_specific_charts("TEST", "1day")
        assert isinstance(success, bool), "Cleanup result should be boolean"
        
        print("✅ API endpoint simulation passed")

def main():
    """Run all tests."""
    print("🚀 Starting Chart Manager Tests\n")
    
    try:
        test_chart_manager_basic()
        test_chart_manager_cleanup()
        test_deployment_config()
        test_global_chart_manager()
        test_api_endpoints_simulation()
        
        print("\n🎉 All tests passed!")
        print("\n📊 Current deployment recommendations:")
        recommendations = get_deployment_recommendations()
        print(f"Environment: {recommendations['current_environment']}")
        print(f"Max Age: {recommendations['config']['max_age_hours']} hours")
        print(f"Max Size: {recommendations['config']['max_total_size_mb']} MB")
        print(f"Cleanup Interval: {recommendations['config']['cleanup_interval_minutes']} minutes")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 