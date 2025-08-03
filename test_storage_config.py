#!/usr/bin/env python3
"""
Test script for Storage Configuration functionality.
Run this to verify storage configuration is working correctly.
"""

import os
import tempfile
import shutil
from pathlib import Path
from storage_config import StorageConfig, get_storage_path, get_storage_recommendations

def test_storage_config_basic():
    """Test basic storage configuration functionality."""
    print("🧪 Testing Storage Configuration Basic Functionality")
    
    # Test storage config retrieval
    config = StorageConfig.get_storage_config()
    assert 'charts_path' in config, "Should have charts_path in config"
    assert 'analysis_path' in config, "Should have analysis_path in config"
    assert 'datasets_path' in config, "Should have datasets_path in config"
    
    # Test individual path getters
    charts_path = StorageConfig.get_charts_path()
    analysis_path = StorageConfig.get_analysis_path()
    datasets_path = StorageConfig.get_datasets_path()
    
    assert charts_path, "Charts path should not be empty"
    assert analysis_path, "Analysis path should not be empty"
    assert datasets_path, "Datasets path should not be empty"
    
    print("✅ Basic functionality test passed")

def test_storage_config_environment():
    """Test environment-specific storage configuration."""
    print("🧪 Testing Environment-Specific Configuration")
    
    # Save original environment
    original_env = os.environ.get('ENVIRONMENT', 'development')
    
    # Test development environment
    os.environ['ENVIRONMENT'] = 'development'
    dev_config = StorageConfig.get_storage_config()
    assert './output' in dev_config['base_path'], "Development should use relative paths"
    
    # Test production environment
    os.environ['ENVIRONMENT'] = 'production'
    prod_config = StorageConfig.get_storage_config()
    print(f"Production base_path: {prod_config['base_path']}")
    assert '/app/data' in prod_config['base_path'], "Production should use absolute paths"
    
    # Reset to original environment
    os.environ['ENVIRONMENT'] = original_env
    
    print("✅ Environment-specific configuration test passed")

def test_storage_path_helpers():
    """Test storage path helper functions."""
    print("🧪 Testing Storage Path Helpers")
    
    # Test get_storage_path function
    charts_path = get_storage_path("CHARTS")
    analysis_path = get_storage_path("ANALYSIS")
    
    assert charts_path, "Should get charts path"
    assert analysis_path, "Should get analysis path"
    
    # Test invalid storage type
    try:
        get_storage_path("INVALID")
        assert False, "Should raise ValueError for invalid storage type"
    except ValueError:
        pass  # Expected
    
    print("✅ Storage path helpers test passed")

def test_directory_creation():
    """Test directory creation functionality."""
    print("🧪 Testing Directory Creation")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a temporary storage config
        original_env = os.environ.get('ENVIRONMENT', 'development')
        os.environ['ENVIRONMENT'] = 'development'
        os.environ['STORAGE_BASE_PATH'] = temp_dir
        
        # Test directory creation
        StorageConfig.ensure_directories_exist()
        
        # Check if directories were created
        config = StorageConfig.get_storage_config()
        for key, path in config.items():
            if key.endswith("_path") and path and not path.startswith('./'):
                path_obj = Path(path)
                if path_obj.exists():
                    assert path_obj.is_dir(), f"{path} should be a directory"
        
        # Reset environment
        os.environ['ENVIRONMENT'] = original_env
    
    print("✅ Directory creation test passed")

def test_storage_info():
    """Test storage information functionality."""
    print("🧪 Testing Storage Information")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test directories
        test_charts_dir = Path(temp_dir) / "charts"
        test_analysis_dir = Path(temp_dir) / "analysis"
        test_charts_dir.mkdir()
        test_analysis_dir.mkdir()
        
        # Create test files
        (test_charts_dir / "test.png").write_bytes(b"test data" * 1000)
        (test_analysis_dir / "test.json").write_bytes(b"test data" * 100)
        
        # Set environment to use temp directory
        original_env = os.environ.get('ENVIRONMENT', 'development')
        os.environ['ENVIRONMENT'] = 'development'
        os.environ['STORAGE_BASE_PATH'] = temp_dir
        
        # Get storage info
        storage_info = StorageConfig.get_storage_info()
        
        assert 'environment' in storage_info, "Should have environment info"
        assert 'storage_type' in storage_info, "Should have storage type"
        assert 'paths' in storage_info, "Should have paths"
        assert 'directory_status' in storage_info, "Should have directory status"
        
        # Check directory status
        status = storage_info['directory_status']
        assert len(status) > 0, "Should have directory status entries"
        
        # Reset environment
        os.environ['ENVIRONMENT'] = original_env
    
    print("✅ Storage information test passed")

def test_storage_recommendations():
    """Test storage recommendations functionality."""
    print("🧪 Testing Storage Recommendations")
    
    recommendations = get_storage_recommendations()
    
    assert 'current_environment' in recommendations, "Should have current environment"
    assert 'recommendations' in recommendations, "Should have recommendations"
    assert 'config' in recommendations, "Should have config"
    
    print("✅ Storage recommendations test passed")

def test_environment_variable_override():
    """Test environment variable overrides."""
    print("🧪 Testing Environment Variable Overrides")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Set custom environment variables
        original_env = os.environ.get('ENVIRONMENT', 'development')
        original_charts_path = os.environ.get('STORAGE_CHARTS_PATH', '')
        
        os.environ['ENVIRONMENT'] = 'production'
        os.environ['STORAGE_CHARTS_PATH'] = temp_dir
        
        # Get config
        config = StorageConfig.get_storage_config()
        
        # Should use custom charts path
        assert temp_dir in config['charts_path'], "Should use custom charts path"
        
        # Reset environment
        os.environ['ENVIRONMENT'] = original_env
        if original_charts_path:
            os.environ['STORAGE_CHARTS_PATH'] = original_charts_path
        else:
            os.environ.pop('STORAGE_CHARTS_PATH', None)
    
    print("✅ Environment variable override test passed")

def main():
    """Run all tests."""
    print("🚀 Starting Storage Configuration Tests\n")
    
    try:
        test_storage_config_basic()
        test_storage_config_environment()
        test_storage_path_helpers()
        test_directory_creation()
        test_storage_info()
        test_storage_recommendations()
        test_environment_variable_override()
        
        print("\n🎉 All storage configuration tests passed!")
        print("\n📊 Current storage configuration:")
        config = StorageConfig.get_storage_config()
        print(f"Environment: {StorageConfig.get_environment()}")
        print(f"Storage Type: {config['type']}")
        print(f"Base Path: {config['base_path']}")
        print(f"Charts Path: {config['charts_path']}")
        print(f"Analysis Path: {config['analysis_path']}")
        print(f"Datasets Path: {config['datasets_path']}")
        
        print("\n📋 Storage recommendations:")
        recommendations = get_storage_recommendations()
        print(f"Current Environment: {recommendations['current_environment']}")
        for rec in recommendations['recommendations']['recommendations']:
            print(f"  - {rec}")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 