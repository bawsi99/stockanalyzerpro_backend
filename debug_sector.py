#!/usr/bin/env python3
"""
Debug script to test sector benchmarking functionality
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from analysis.orchestrator import StockAnalysisOrchestrator

async def debug_sector_analysis():
    """Test sector analysis for RELIANCE with OIL_GAS sector"""
    print("🔍 Starting sector analysis debug...")
    
    try:
        # Create orchestrator
        orchestrator = StockAnalysisOrchestrator()
        
        # Test with RELIANCE and OIL_GAS sector
        symbol = "RELIANCE"
        sector = "OIL_GAS"
        
        print(f"📊 Testing enhanced analysis for {symbol} in {sector} sector...")
        
        # Run enhanced analysis
        result = await orchestrator.enhanced_analyze_stock(
            symbol=symbol,
            exchange="NSE",
            period=30,  # Shorter period for faster debugging
            interval="day",
            sector=sector
        )
        
        analysis_results, success_message, error_message = result
        
        if error_message:
            print(f"❌ Error: {error_message}")
            return
        
        if not analysis_results:
            print("❌ No analysis results returned")
            return
        
        print(f"✅ Analysis completed: {success_message}")
        
        # Check sector context
        sector_context = analysis_results.get('sector_context')
        print(f"\n🏭 Sector Context Type: {type(sector_context)}")
        
        if sector_context:
            print("✅ Sector context found!")
            print(f"Keys in sector context: {list(sector_context.keys()) if isinstance(sector_context, dict) else 'Not a dict'}")
            
            # Check for sector_benchmarking
            if isinstance(sector_context, dict):
                sector_benchmarking = sector_context.get('sector_benchmarking')
                if sector_benchmarking:
                    print("✅ Sector benchmarking found!")
                    if isinstance(sector_benchmarking, dict):
                        print(f"Keys in sector_benchmarking: {list(sector_benchmarking.keys())}")
                    else:
                        print(f"Sector benchmarking type: {type(sector_benchmarking)}")
                else:
                    print("❌ No sector_benchmarking in sector_context")
            else:
                print(f"❌ Sector context is not a dict: {sector_context}")
        else:
            print("❌ No sector context found!")
            
        # Debug: Check if sector context contains any data at all
        print(f"\n🔍 Raw sector context: {sector_context}")
        
    except Exception as e:
        print(f"❌ Exception during sector analysis debug: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_sector_analysis())