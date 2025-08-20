#!/usr/bin/env python3
"""
Test script to verify correlation analysis functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sector_benchmarking import SectorBenchmarkingProvider
import asyncio
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def test_correlation_analysis():
    """Test the correlation analysis functionality"""
    print("🧪 Testing Correlation Analysis")
    print("=" * 50)
    
    try:
        # Initialize the provider
        provider = SectorBenchmarkingProvider()
        print("✅ SectorBenchmarkingProvider initialized successfully")
        
        # Test synchronous correlation matrix generation
        print("\n🔍 Testing synchronous correlation matrix generation...")
        correlation_result = provider.generate_sector_correlation_matrix("3M")
        
        if correlation_result:
            print("✅ Synchronous correlation matrix generated successfully")
            print(f"   Timeframe: {correlation_result.get('timeframe')}")
            print(f"   Average correlation: {correlation_result.get('average_correlation')}")
            print(f"   High correlation pairs: {len(correlation_result.get('high_correlation_pairs', []))}")
            print(f"   Low correlation pairs: {len(correlation_result.get('low_correlation_pairs', []))}")
            print(f"   Diversification quality: {correlation_result.get('diversification_insights', {}).get('diversification_quality')}")
        else:
            print("❌ Synchronous correlation matrix generation failed")
            return False
        
        # Test asynchronous correlation matrix generation
        print("\n🔍 Testing asynchronous correlation matrix generation...")
        async_correlation_result = await provider.generate_sector_correlation_matrix_async("3M")
        
        if async_correlation_result:
            print("✅ Asynchronous correlation matrix generated successfully")
            print(f"   Timeframe: {async_correlation_result.get('timeframe')}")
            print(f"   Average correlation: {async_correlation_result.get('average_correlation')}")
            print(f"   High correlation pairs: {len(async_correlation_result.get('high_correlation_pairs', []))}")
            print(f"   Low correlation pairs: {len(async_correlation_result.get('low_correlation_pairs', []))}")
            print(f"   Diversification quality: {async_correlation_result.get('diversification_insights', {}).get('diversification_quality')}")
        else:
            print("❌ Asynchronous correlation matrix generation failed")
            return False
        
        # Test sector data retrieval
        print("\n🔍 Testing sector data retrieval...")
        test_sector = "BANKING"
        sector_data = provider._get_sector_data(test_sector, 60)
        
        if sector_data is not None:
            print(f"✅ Sector data retrieved for {test_sector}")
            print(f"   Data points: {len(sector_data)}")
            print(f"   Columns: {list(sector_data.columns)}")
        else:
            print(f"❌ Failed to retrieve sector data for {test_sector}")
            return False
        
        print("\n🎉 All correlation analysis tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error during correlation analysis test: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test function"""
    print("🚀 Starting Correlation Analysis Tests")
    print("=" * 60)
    
    success = await test_correlation_analysis()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ CORRELATION ANALYSIS TESTS COMPLETED SUCCESSFULLY!")
        print("✅ All correlation analysis components are working correctly")
    else:
        print("❌ Some correlation analysis tests failed")
        print("❌ Please review the errors above")
    
    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
