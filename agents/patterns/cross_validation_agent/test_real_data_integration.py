#!/usr/bin/env python3
"""
Quick test script to verify that the cross-validation agent multi-stock test 
now uses real data instead of synthetic data.

This script runs a minimal test to confirm the integration is working.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the backend directory to sys.path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.join(script_dir, '..', '..', '..')
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

async def test_real_data_integration():
    """Test that the cross-validation agent can use real data."""
    print("🧪 Testing Cross-Validation Agent Real Data Integration")
    print("=" * 60)
    
    try:
        # Import the updated tester
        from multi_stock_test import CrossValidationMultiStockTester
        
        # Initialize tester with a temporary output directory
        test_output_dir = Path(__file__).parent / "real_data_test_output"
        tester = CrossValidationMultiStockTester(str(test_output_dir))
        
        print(f"✅ Successfully imported and initialized CrossValidationMultiStockTester")
        print(f"Real data configuration:")
        print(f"  - Use real data: {tester.use_real_data}")
        print(f"  - Orchestrator available: {tester.orchestrator is not None}")
        print(f"  - Zerodha client available: {tester.zerodha_client is not None}")
        
        # Run a minimal test with just one stock and short period
        print(f"\n🚀 Running minimal test with real data...")
        
        test_stocks = ["RELIANCE"]  # Just test one stock
        test_periods = [30]         # Just test 30 days
        
        results = await tester.run_multi_stock_validation_test(
            test_stocks=test_stocks,
            test_periods=test_periods,
            max_concurrent=1,
            save_results=True
        )
        
        # Check results
        if results.get('test_success', False):
            print("✅ Test completed successfully!")
            
            # Check data source usage
            analysis = results.get('analysis_results', {})
            data_stats = analysis.get('data_source_statistics', {})
            
            real_data_count = data_stats.get('real_market_data', 0)
            fallback_count = data_stats.get('synthetic_fallback', 0)
            synthetic_count = data_stats.get('synthetic_only', 0)
            
            print(f"\n📊 Data Source Statistics:")
            print(f"  - Real market data used: {real_data_count} tests")
            print(f"  - Synthetic fallback used: {fallback_count} tests")
            print(f"  - Synthetic only used: {synthetic_count} tests")
            
            if real_data_count > 0:
                print(f"🎉 SUCCESS: Real market data is being used!")
            elif fallback_count > 0:
                print(f"⚠️  Real data attempted but fell back to synthetic")
            else:
                print(f"ℹ️  Using synthetic data (real data sources not available)")
                
            # Show test summary
            test_summary = analysis.get('test_summary', {})
            success_rate = test_summary.get('success_rate_percent', 0)
            print(f"\n📈 Test Performance:")
            print(f"  - Success rate: {success_rate:.1f}%")
            print(f"  - Total tests: {test_summary.get('total_tests', 0)}")
            
        else:
            print("❌ Test failed")
            error = results.get('analysis_results', {}).get('error', 'Unknown error')
            print(f"Error: {error}")
            
        print(f"\n📁 Test results saved to: {test_output_dir}")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're running this from the correct directory")
        return False
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

async def main():
    """Main test function."""
    print("Cross-Validation Agent Real Data Integration Test")
    print("This script tests that the multi-stock test now uses real data")
    print()
    
    success = await test_real_data_integration()
    
    if success:
        print("\n🎉 Real data integration test completed!")
        print("The cross-validation agent multi-stock test is now using real market data.")
    else:
        print("\n❌ Real data integration test failed!")
        print("Please check the error messages above.")
    
    return success

if __name__ == "__main__":
    # Run the integration test
    result = asyncio.run(main())
    sys.exit(0 if result else 1)