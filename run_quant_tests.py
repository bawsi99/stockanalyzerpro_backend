#!/usr/bin/env python3
"""
Quantitative System Test Runner

This script provides comprehensive testing capabilities for the StockAnalyzer Pro
quantitative system. It includes unit tests, integration tests, backtesting,
performance testing, and system validation.

Usage:
    python run_quant_tests.py [test_type] [options]

Available test types:
    - unit: Run unit tests only
    - integration: Run integration tests only
    - backtest: Run backtesting only
    - performance: Run performance tests only
    - service: Run service endpoint tests only
    - all: Run all tests
    - quick: Run quick validation tests
    - coverage: Run tests with coverage report

Examples:
    python run_quant_tests.py quick
    python run_quant_tests.py backtest --symbols RELIANCE TCS --days 30
    python run_quant_tests.py all --verbose
    python run_quant_tests.py coverage --html
"""

import argparse
import asyncio
import json
import os
import sys
import time
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Add the backend directory to Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

class QuantTestRunner:
    """Comprehensive test runner for the quantitative system."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results = {}
        self.start_time = None
        
    def log(self, message: str):
        """Log message with timestamp."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}")
    
    def run_command(self, command: List[str], description: str) -> bool:
        """Run a shell command and return success status."""
        self.log(f"Running: {description}")
        if self.verbose:
            self.log(f"Command: {' '.join(command)}")
        
        try:
            result = subprocess.run(
                command,
                capture_output=not self.verbose,
                text=True,
                cwd=backend_dir
            )
            
            if result.returncode == 0:
                self.log(f"✅ {description} - PASSED")
                return True
            else:
                self.log(f"❌ {description} - FAILED")
                if not self.verbose and result.stderr:
                    self.log(f"Error: {result.stderr}")
                return False
                
        except Exception as e:
            self.log(f"❌ {description} - ERROR: {e}")
            return False
    
    def run_unit_tests(self) -> bool:
        """Run unit tests."""
        self.log("🧪 Running Unit Tests...")
        
        # Create test directories if they don't exist
        test_dirs = ["tests/unit", "tests/integration", "tests/performance"]
        for test_dir in test_dirs:
            Path(test_dir).mkdir(parents=True, exist_ok=True)
        
        # Run pytest for unit tests
        command = [
            "python", "-m", "pytest", "tests/unit/", "-v",
            "--tb=short", "--disable-warnings"
        ]
        
        return self.run_command(command, "Unit Tests")
    
    def run_integration_tests(self) -> bool:
        """Run integration tests."""
        self.log("🔄 Running Integration Tests...")
        
        command = [
            "python", "-m", "pytest", "tests/integration/", "-v",
            "--tb=short", "--disable-warnings"
        ]
        
        return self.run_command(command, "Integration Tests")
    
    def run_performance_tests(self) -> bool:
        """Run performance tests."""
        self.log("⚡ Running Performance Tests...")
        
        command = [
            "python", "-m", "pytest", "tests/performance/", "-v",
            "--tb=short", "--disable-warnings", "-m", "not slow"
        ]
        
        return self.run_command(command, "Performance Tests")
    
    def run_service_tests(self) -> bool:
        """Run service endpoint tests."""
        self.log("🔧 Running Service Tests...")
        
        # Check if services are running
        self.log("Checking if services are running...")
        
        # Run service endpoint tests
        command = ["python", "test_service_endpoints.py", "all"]
        
        return self.run_command(command, "Service Tests")
    
    def run_backtesting(self, symbols: List[str], days: int = 30) -> bool:
        """Run backtesting."""
        self.log("📊 Running Backtesting...")
        
        if not symbols:
            symbols = ["RELIANCE"]
        
        command = [
            "python", "-m", "backend.run_backtests",
            "--symbols"] + symbols + [
            "--days", str(days),
            "--lookahead", "3",
            "--threshold", "1.0"
        ]
        
        return self.run_command(command, "Backtesting")
    
    def run_ml_model_tests(self) -> bool:
        """Run machine learning model tests."""
        self.log("🤖 Running ML Model Tests...")
        
        try:
            # Test ML model functionality
            from ml.model import load_model, load_registry
            
            # Test model loading
            model = load_model()
            registry = load_registry()
            
            if model is not None:
                self.log("✅ ML Model loaded successfully")
                self.log(f"   Model metrics: {registry.get('metrics', {}) if registry else 'No metrics'}")
                return True
            else:
                self.log("⚠️  No trained ML model found")
                return True  # Not a failure, just no model
                
        except Exception as e:
            self.log(f"❌ ML Model Tests failed: {e}")
            return False
    
    def run_technical_indicators_tests(self) -> bool:
        """Run technical indicators tests."""
        self.log("📈 Running Technical Indicators Tests...")
        
        try:
            import pandas as pd
            from technical_indicators import TechnicalIndicators
            
            # Create test data
            test_data = pd.DataFrame({
                'open': [100, 101, 102, 103, 104],
                'high': [105, 106, 107, 108, 109],
                'low': [95, 96, 97, 98, 99],
                'close': [101, 102, 103, 104, 105],
                'volume': [1000, 1100, 1200, 1300, 1400]
            })
            
            # Test RSI
            rsi = TechnicalIndicators.calculate_rsi(test_data, period=14)
            assert len(rsi) == len(test_data)
            
            # Test MACD
            macd = TechnicalIndicators.calculate_macd(test_data)
            assert 'macd_line' in macd
            
            # Test Bollinger Bands
            bb = TechnicalIndicators.calculate_bollinger_bands(test_data)
            assert 'upper_band' in bb
            
            self.log("✅ Technical Indicators Tests passed")
            return True
            
        except Exception as e:
            self.log(f"❌ Technical Indicators Tests failed: {e}")
            return False
    
    def run_pattern_recognition_tests(self) -> bool:
        """Run pattern recognition tests."""
        self.log("🔍 Running Pattern Recognition Tests...")
        
        try:
            import pandas as pd
            from patterns.recognition import PatternRecognition
            
            # Create test data with known patterns
            test_data = pd.DataFrame({
                'open': [100] * 20,
                'high': [110, 105, 100, 95, 90, 85, 80, 85, 90, 95, 100, 105, 110, 105, 100, 95, 90, 85, 80, 75],
                'low': [90] * 20,
                'close': [105, 100, 95, 90, 85, 80, 85, 90, 95, 100, 105, 100, 95, 90, 85, 80, 75, 70, 65, 60],
                'volume': [1000] * 20
            })
            
            # Test pattern detection
            patterns = PatternRecognition.detect_all_patterns(test_data)
            assert isinstance(patterns, dict)
            
            self.log("✅ Pattern Recognition Tests passed")
            return True
            
        except Exception as e:
            self.log(f"❌ Pattern Recognition Tests failed: {e}")
            return False
    
    def run_quick_validation(self) -> bool:
        """Run quick validation tests."""
        self.log("⚡ Running Quick Validation Tests...")
        
        tests = [
            ("ML Model Tests", self.run_ml_model_tests),
            ("Technical Indicators Tests", self.run_technical_indicators_tests),
            ("Pattern Recognition Tests", self.run_pattern_recognition_tests),
        ]
        
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            self.log(f"\n--- {test_name} ---")
            if test_func():
                passed += 1
        
        self.log(f"\n📊 Quick Validation Results: {passed}/{total} tests passed")
        return passed == total
    
    def run_coverage_tests(self, html: bool = False) -> bool:
        """Run tests with coverage reporting."""
        self.log("📊 Running Coverage Tests...")
        
        command = [
            "python", "-m", "pytest", "tests/", "-v",
            "--cov=.", "--cov-report=term-missing"
        ]
        
        if html:
            command.append("--cov-report=html")
        
        return self.run_command(command, "Coverage Tests")
    
    def run_all_tests(self) -> bool:
        """Run all tests."""
        self.log("🚀 Running All Tests...")
        
        tests = [
            ("Quick Validation", self.run_quick_validation),
            ("Unit Tests", self.run_unit_tests),
            ("Integration Tests", self.run_integration_tests),
            ("Performance Tests", self.run_performance_tests),
            ("Service Tests", self.run_service_tests),
        ]
        
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            self.log(f"\n{'='*50}")
            self.log(f"Running: {test_name}")
            self.log(f"{'='*50}")
            
            if test_func():
                passed += 1
        
        self.log(f"\n{'='*50}")
        self.log(f"📊 ALL TESTS COMPLETE: {passed}/{total} test suites passed")
        self.log(f"{'='*50}")
        
        return passed == total
    
    def generate_report(self) -> Dict:
        """Generate test report."""
        end_time = time.time()
        duration = end_time - self.start_time if self.start_time else 0
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "duration_seconds": duration,
            "results": self.results,
            "summary": {
                "total_tests": len(self.results),
                "passed": sum(1 for r in self.results.values() if r),
                "failed": sum(1 for r in self.results.values() if not r)
            }
        }
        
        # Save report
        report_file = Path("test_report.json")
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.log(f"📄 Test report saved to: {report_file}")
        return report
    
    def run(self, test_type: str, **kwargs) -> bool:
        """Run specified test type."""
        self.start_time = time.time()
        
        self.log(f"🚀 Starting Quantitative System Tests")
        self.log(f"Test Type: {test_type}")
        self.log(f"Timestamp: {datetime.now().isoformat()}")
        self.log(f"{'='*60}")
        
        success = False
        
        if test_type == "quick":
            success = self.run_quick_validation()
        elif test_type == "unit":
            success = self.run_unit_tests()
        elif test_type == "integration":
            success = self.run_integration_tests()
        elif test_type == "performance":
            success = self.run_performance_tests()
        elif test_type == "service":
            success = self.run_service_tests()
        elif test_type == "backtest":
            symbols = kwargs.get('symbols', ["RELIANCE"])
            days = kwargs.get('days', 30)
            success = self.run_backtesting(symbols, days)
        elif test_type == "coverage":
            html = kwargs.get('html', False)
            success = self.run_coverage_tests(html)
        elif test_type == "all":
            success = self.run_all_tests()
        else:
            self.log(f"❌ Unknown test type: {test_type}")
            return False
        
        # Generate report
        report = self.generate_report()
        
        # Final summary
        self.log(f"\n{'='*60}")
        if success:
            self.log("🎉 ALL TESTS PASSED!")
        else:
            self.log("⚠️  SOME TESTS FAILED")
        self.log(f"Duration: {report['duration_seconds']:.2f} seconds")
        self.log(f"{'='*60}")
        
        return success


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Quantitative System Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_quant_tests.py quick                    # Quick validation
  python run_quant_tests.py backtest --symbols RELIANCE TCS --days 30
  python run_quant_tests.py all --verbose           # All tests with verbose output
  python run_quant_tests.py coverage --html         # Coverage with HTML report
        """
    )
    
    parser.add_argument(
        "test_type",
        choices=["quick", "unit", "integration", "performance", "service", "backtest", "coverage", "all"],
        help="Type of tests to run"
    )
    
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["RELIANCE"],
        help="Symbols for backtesting (default: RELIANCE)"
    )
    
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of days for backtesting (default: 30)"
    )
    
    parser.add_argument(
        "--html",
        action="store_true",
        help="Generate HTML coverage report"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # Create test runner
    runner = QuantTestRunner(verbose=args.verbose)
    
    # Run tests
    success = runner.run(
        args.test_type,
        symbols=args.symbols,
        days=args.days,
        html=args.html
    )
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
