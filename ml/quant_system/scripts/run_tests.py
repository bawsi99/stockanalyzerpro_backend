#!/usr/bin/env python3
"""
Quantitative System Test Runner

This script provides a unified interface to run different types of tests
for the quantitative trading system.

Usage:
    python run_quant_tests.py [test_type] [options]

Test Types:
    quick       - Run simplified system test (recommended for quick validation)
    unified     - Run unified ML system test
    advanced    - Run advanced trading system test
    phase2      - Run Phase 2 advanced system test
    realtime    - Run real-time system test
    nbeats      - Run N-BEATS model test
    backtest    - Run backtesting tests
    validation  - Run validation tests
    all         - Run all tests (comprehensive)

Options:
    --symbols SYMBOLS  - Comma-separated list of symbols for backtesting
    --verbose          - Enable verbose output
    --debug            - Enable debug mode
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path

def run_test_script(script_name: str, description: str = None) -> bool:
    """Run a test script and return success status."""
    if description:
        print(f"\n🧪 {description}")
        print("=" * 60)
    
    try:
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=False, 
                              text=True)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Failed to run {script_name}: {e}")
        return False

def run_quick_tests():
    """Run quick validation tests."""
    print("🚀 Running Quick Quant System Tests...")
    print("=" * 60)
    
    tests = [
        ("test_simplified_system.py", "Simplified Advanced Trading System"),
        ("test_unified_ml_system.py", "Unified ML System"),
        ("test_nbeats.py", "N-BEATS Model"),
    ]
    
    results = {}
    for script, desc in tests:
        results[desc] = run_test_script(script, desc)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Quick Test Results Summary:")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for desc, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{desc:35} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    return passed == total

def run_unified_tests():
    """Run unified ML system tests."""
    return run_test_script("test_unified_ml_system.py", "Unified ML System Test")

def run_advanced_tests():
    """Run advanced trading system tests."""
    return run_test_script("test_advanced_trading_system.py", "Advanced Trading System Test")

def run_phase2_tests():
    """Run Phase 2 advanced system tests."""
    return run_test_script("test_phase2_advanced_system.py", "Phase 2 Advanced System Test")

def run_realtime_tests():
    """Run real-time system tests."""
    return run_test_script("test_realtime_system.py", "Real-Time System Test")

def run_nbeats_tests():
    """Run N-BEATS model tests."""
    return run_test_script("test_nbeats.py", "N-BEATS Model Test")

def run_backtest_tests(symbols: str = None):
    """Run backtesting tests."""
    print("🧪 Running Backtesting Tests...")
    print("=" * 60)
    
    tests = [
        ("test_backtest_simple.py", "Simple Backtest"),
        ("test_production_system.py", "Production System"),
    ]
    
    results = {}
    for script, desc in tests:
        results[desc] = run_test_script(script, desc)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Backtest Results Summary:")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for desc, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{desc:25} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    return passed == total

def run_validation_tests():
    """Run validation tests."""
    print("🧪 Running Validation Tests...")
    print("=" * 60)
    
    tests = [
        ("step1_validation.py", "Step 1 Validation"),
        ("validation_unified_ml.py", "Unified ML Validation"),
    ]
    
    results = {}
    for script, desc in tests:
        results[desc] = run_test_script(script, desc)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Validation Results Summary:")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for desc, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{desc:25} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    return passed == total

def run_all_tests():
    """Run all comprehensive tests."""
    print("🚀 Running All Quant System Tests...")
    print("=" * 80)
    print("This will run a comprehensive test suite covering all components.")
    print("=" * 80)
    
    test_suites = [
        ("Quick Tests", run_quick_tests),
        ("Unified ML Tests", run_unified_tests),
        ("Advanced Trading Tests", run_advanced_tests),
        ("Phase 2 Tests", run_phase2_tests),
        ("Real-Time Tests", run_realtime_tests),
        ("N-BEATS Tests", run_nbeats_tests),
        ("Backtest Tests", run_backtest_tests),
        ("Validation Tests", run_validation_tests),
    ]
    
    results = {}
    for suite_name, test_func in test_suites:
        print(f"\n📋 Running {suite_name}...")
        results[suite_name] = test_func()
    
    # Final summary
    print("\n" + "=" * 80)
    print("📊 COMPREHENSIVE TEST RESULTS SUMMARY")
    print("=" * 80)
    
    passed = sum(results.values())
    total = len(results)
    
    for suite_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{suite_name:25} {status}")
    
    print(f"\nOverall: {passed}/{total} test suites passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! The quant system is fully operational.")
        print("✅ Ready for production deployment")
        print("✅ All components working correctly")
        print("✅ System integration verified")
    else:
        print(f"\n⚠️  {total - passed} test suites failed. Please check the issues above.")
    
    return passed == total

def main():
    """Main function to parse arguments and run tests."""
    parser = argparse.ArgumentParser(description="Quantitative System Test Runner")
    parser.add_argument("test_type", nargs="?", default="quick", 
                       choices=["quick", "unified", "advanced", "phase2", "realtime", 
                               "nbeats", "backtest", "validation", "all"],
                       help="Type of test to run")
    parser.add_argument("--symbols", help="Comma-separated list of symbols for backtesting")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Change to quant_system directory
    os.chdir(Path(__file__).parent)
    
    # Run the specified test
    if args.test_type == "quick":
        success = run_quick_tests()
    elif args.test_type == "unified":
        success = run_unified_tests()
    elif args.test_type == "advanced":
        success = run_advanced_tests()
    elif args.test_type == "phase2":
        success = run_phase2_tests()
    elif args.test_type == "realtime":
        success = run_realtime_tests()
    elif args.test_type == "nbeats":
        success = run_nbeats_tests()
    elif args.test_type == "backtest":
        success = run_backtest_tests(args.symbols)
    elif args.test_type == "validation":
        success = run_validation_tests()
    elif args.test_type == "all":
        success = run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
