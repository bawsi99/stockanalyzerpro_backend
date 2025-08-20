#!/usr/bin/env python3
"""
Testing Setup Script for StockAnalyzer Pro Quantitative System

This script sets up the testing environment for the quantitative system,
including installing dependencies, creating test directories, and setting up
basic test configurations.

Usage:
    python setup_testing.py [options]

Options:
    --install-deps: Install testing dependencies
    --create-dirs: Create test directory structure
    --setup-config: Setup test configuration
    --all: Run all setup steps
"""

import argparse
import subprocess
import sys
from pathlib import Path

def install_dependencies():
    """Install testing dependencies."""
    print("📦 Installing testing dependencies...")
    
    dependencies = [
        "pytest",
        "pytest-asyncio", 
        "pytest-cov",
        "pytest-mock",
        "pytest-xdist",
        "aiohttp",
        "psutil"
    ]
    
    for dep in dependencies:
        print(f"Installing {dep}...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", dep], check=True)
            print(f"✅ {dep} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {dep}: {e}")
            return False
    
    return True

def create_test_directories():
    """Create test directory structure."""
    print("📁 Creating test directory structure...")
    
    test_dirs = [
        "tests/unit",
        "tests/integration", 
        "tests/performance",
        "tests/fixtures",
        "tests/reporting"
    ]
    
    for test_dir in test_dirs:
        Path(test_dir).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created {test_dir}")
    
    # Create __init__.py files
    init_files = [
        "tests/__init__.py",
        "tests/unit/__init__.py",
        "tests/integration/__init__.py", 
        "tests/performance/__init__.py"
    ]
    
    for init_file in init_files:
        Path(init_file).touch(exist_ok=True)
        print(f"✅ Created {init_file}")
    
    return True

def create_basic_tests():
    """Create basic test files."""
    print("🧪 Creating basic test files...")
    
    # Basic unit test template
    unit_test_template = '''import pytest
import pandas as pd
import numpy as np

class TestBasicFunctionality:
    """Basic functionality tests."""
    
    def test_imports(self):
        """Test that all modules can be imported."""
        try:
            from technical_indicators import TechnicalIndicators
            from patterns.recognition import PatternRecognition
            from ml.model import load_model
            assert True
        except ImportError as e:
            pytest.fail(f"Import failed: {e}")
    
    def test_basic_data_creation(self):
        """Test basic data creation."""
        data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [95, 96, 97],
            'close': [101, 102, 103],
            'volume': [1000, 1100, 1200]
        })
        assert len(data) == 3
        assert 'close' in data.columns
'''
    
    # Write basic test file
    with open("tests/unit/test_basic.py", "w") as f:
        f.write(unit_test_template)
    
    print("✅ Created tests/unit/test_basic.py")
    
    return True

def setup_test_config():
    """Setup test configuration."""
    print("⚙️ Setting up test configuration...")
    
    # Create pytest configuration
    pytest_config = '''[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = [
    "-v",
    "--tb=short",
    "--disable-warnings",
    "--strict-markers"
]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: marks tests as integration tests",
    "performance: marks tests as performance tests"
]
'''
    
    with open("pyproject.toml", "w") as f:
        f.write(pytest_config)
    
    print("✅ Created pyproject.toml")
    
    # Create .coveragerc
    coverage_config = '''[run]
source = .
omit = 
    */tests/*
    */venv/*
    */env/*
    setup.py

[report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise AssertionError
    raise NotImplementedError
    if 0:
    if __name__ == .__main__.:
    class .*\\bProtocol\\):
    @(abc\\.)?abstractmethod
'''
    
    with open(".coveragerc", "w") as f:
        f.write(coverage_config)
    
    print("✅ Created .coveragerc")
    
    return True

def create_test_runner_script():
    """Create a simple test runner script."""
    print("🚀 Creating test runner script...")
    
    runner_script = '''#!/usr/bin/env python3
"""
Simple Test Runner for StockAnalyzer Pro

Quick test runner for development and validation.
"""

import subprocess
import sys
from pathlib import Path

def run_tests():
    """Run basic tests."""
    print("🧪 Running basic tests...")
    
    # Run basic unit tests
    result = subprocess.run([
        sys.executable, "-m", "pytest", "tests/unit/", "-v"
    ])
    
    if result.returncode == 0:
        print("✅ All tests passed!")
        return True
    else:
        print("❌ Some tests failed!")
        return False

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
'''
    
    with open("run_basic_tests.py", "w") as f:
        f.write(runner_script)
    
    # Make it executable
    Path("run_basic_tests.py").chmod(0o755)
    
    print("✅ Created run_basic_tests.py")
    
    return True

def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description="Setup testing environment")
    parser.add_argument("--install-deps", action="store_true", help="Install testing dependencies")
    parser.add_argument("--create-dirs", action="store_true", help="Create test directory structure")
    parser.add_argument("--setup-config", action="store_true", help="Setup test configuration")
    parser.add_argument("--all", action="store_true", help="Run all setup steps")
    
    args = parser.parse_args()
    
    if not any([args.install_deps, args.create_dirs, args.setup_config, args.all]):
        print("No options specified. Use --help for available options.")
        print("Recommended: python setup_testing.py --all")
        return
    
    print("🔧 Setting up testing environment for StockAnalyzer Pro...")
    print("=" * 60)
    
    success = True
    
    if args.all or args.install_deps:
        success &= install_dependencies()
    
    if args.all or args.create_dirs:
        success &= create_test_directories()
        success &= create_basic_tests()
    
    if args.all or args.setup_config:
        success &= setup_test_config()
        success &= create_test_runner_script()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 Testing environment setup completed successfully!")
        print("\n📋 Next steps:")
        print("1. Run basic tests: python run_basic_tests.py")
        print("2. Run comprehensive tests: python run_quant_tests.py quick")
        print("3. Run backtesting: python run_quant_tests.py backtest --symbols RELIANCE")
        print("4. Run all tests: python run_quant_tests.py all")
        print("\n📚 For more information, see: QUANT_SYSTEM_TESTING_GUIDE.md")
    else:
        print("⚠️  Some setup steps failed. Please check the output above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
