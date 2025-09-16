#!/usr/bin/env python3
"""
Direct test of sector benchmarking provider
"""

import asyncio
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

async def debug_sector_provider():
    """Directly test the sector benchmarking provider"""
    print("🔍 Starting direct sector provider debug...")
    
    try:
        # Import the sector benchmarking provider
        from ml.sector.benchmarking import SectorBenchmarkingProvider
        
        # Create provider
        provider = SectorBenchmarkingProvider()
        print("✅ SectorBenchmarkingProvider created successfully")
        
        # Create sample stock data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=180)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Create realistic stock data with some volatility
        np.random.seed(42)  # For reproducible results
        prices = [2500]  # Starting price for RELIANCE
        for i in range(len(dates) - 1):
            change = np.random.normal(0, 0.02)  # 2% daily volatility
            prices.append(prices[-1] * (1 + change))
        
        stock_data = pd.DataFrame({
            'close': prices,
            'open': [p * 0.999 for p in prices],  # Slightly lower open
            'high': [p * 1.01 for p in prices],   # 1% higher high
            'low': [p * 0.99 for p in prices],    # 1% lower low
            'volume': [10000000 + np.random.randint(-1000000, 1000000) for _ in prices]
        }, index=dates)
        
        print(f"✅ Created sample stock data: {len(stock_data)} rows")
        print(f"Price range: {stock_data['close'].min():.2f} - {stock_data['close'].max():.2f}")
        
        # Test 1: get_optimized_comprehensive_sector_analysis
        print("\n🧪 Test 1: get_optimized_comprehensive_sector_analysis")
        try:
            result1 = await provider.get_optimized_comprehensive_sector_analysis(
                "RELIANCE", stock_data, "OIL_GAS"
            )
            if result1:
                print("✅ get_optimized_comprehensive_sector_analysis succeeded")
                print(f"Keys: {list(result1.keys())}")
                if 'sector_benchmarking' in result1:
                    print("✅ sector_benchmarking found in result")
                else:
                    print("❌ sector_benchmarking NOT found in result")
            else:
                print("❌ get_optimized_comprehensive_sector_analysis returned None")
        except Exception as e:
            print(f"❌ get_optimized_comprehensive_sector_analysis failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Test 2: get_comprehensive_benchmarking (simpler method)
        print("\n🧪 Test 2: get_comprehensive_benchmarking")
        try:
            result2 = provider.get_comprehensive_benchmarking(
                "RELIANCE", stock_data, "OIL_GAS"
            )
            if result2:
                print("✅ get_comprehensive_benchmarking succeeded")
                print(f"Keys: {list(result2.keys())}")
            else:
                print("❌ get_comprehensive_benchmarking returned None")
        except Exception as e:
            print(f"❌ get_comprehensive_benchmarking failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Test 3: Check if Zerodha client is accessible
        print("\n🧪 Test 3: Check Zerodha client")
        try:
            zerodha_client = provider.zerodha_client
            print(f"✅ Zerodha client type: {type(zerodha_client)}")
            
            # Test sync data fetch
            print("Testing sync NIFTY 50 data fetch...")
            nifty_data = provider._get_nifty_data(30)
            if nifty_data is not None:
                print(f"✅ NIFTY 50 data fetched: {len(nifty_data)} rows")
            else:
                print("❌ NIFTY 50 data fetch failed")
            
        except Exception as e:
            print(f"❌ Zerodha client test failed: {e}")
        
        # Test 4: Check sector classification
        print("\n🧪 Test 4: Check sector classification")
        try:
            classifier = provider.sector_classifier
            sector = classifier.get_stock_sector("RELIANCE")
            print(f"✅ RELIANCE sector (auto-detected): {sector}")
            
            sector_display = classifier.get_sector_display_name("OIL_GAS")
            print(f"✅ OIL_GAS display name: {sector_display}")
            
            sector_index = classifier.get_primary_sector_index("OIL_GAS")
            print(f"✅ OIL_GAS primary index: {sector_index}")
            
        except Exception as e:
            print(f"❌ Sector classification test failed: {e}")
        
    except Exception as e:
        print(f"❌ Exception during direct sector provider debug: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_sector_provider())