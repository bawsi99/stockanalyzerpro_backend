#!/usr/bin/env python3
"""
Test script to verify the complete sector rotation data flow from backend to frontend.
This script will help identify where the data pipeline breaks.
"""

import sys
import os
sys.path.append('.')

import asyncio
import json
from analysis.orchestrator import StockAnalysisOrchestrator
from ml.sector.enhanced_classifier import EnhancedSectorClassifier

async def test_full_sector_rotation_flow():
    """Test the complete sector rotation flow end-to-end"""
    print("🧪 Testing Sector Rotation Data Flow")
    print("=" * 50)
    
    # Initialize components
    orchestrator = StockAnalysisOrchestrator()
    sector_classifier = EnhancedSectorClassifier()
    
    # Authenticate
    if not orchestrator.authenticate():
        print("❌ Authentication failed")
        return
    
    # Test symbol
    symbol = "RELIANCE"
    exchange = "NSE"
    
    print(f"📊 Testing with {symbol}")
    
    # Step 1: Get sector classification
    print("\n1️⃣ Getting sector classification...")
    sector = sector_classifier.get_stock_sector(symbol)
    print(f"   Detected sector: {sector}")
    
    # Step 2: Test standalone sector rotation endpoint
    print("\n2️⃣ Testing standalone sector rotation...")
    from ml.sector.benchmarking import SectorBenchmarkingProvider
    sector_provider = SectorBenchmarkingProvider()
    rotation_data = await sector_provider.analyze_sector_rotation_async("1M")
    
    if rotation_data:
        print(f"   ✅ Sector rotation data retrieved")
        print(f"   📈 Leading sectors: {len(rotation_data.get('rotation_patterns', {}).get('leading_sectors', []))}")
        print(f"   📉 Lagging sectors: {len(rotation_data.get('rotation_patterns', {}).get('lagging_sectors', []))}")
        print(f"   💡 Recommendations: {len(rotation_data.get('recommendations', []))}")
        print(f"   💪 Rotation strength: {rotation_data.get('rotation_patterns', {}).get('rotation_strength', 'unknown')}")
        
        # Check if our sector is in the data
        if sector in rotation_data.get('sector_performance', {}):
            sector_perf = rotation_data['sector_performance'][sector]
            print(f"   🎯 {sector} performance: {sector_perf.get('total_return', 'N/A')}%")
            print(f"   📊 {sector} rank: {rotation_data.get('sector_rankings', {}).get(sector, {}).get('rank', 'N/A')}")
        else:
            print(f"   ⚠️ {sector} not found in sector performance data")
    else:
        print("   ❌ Failed to get sector rotation data")
    
    # Step 3: Test full analysis with sector context
    print("\n3️⃣ Testing full analysis with sector context...")
    try:
        analysis_results, success_message, error_message = await orchestrator.enhanced_analyze_stock(
            symbol=symbol,
            exchange=exchange,
            period=90,  # Reduced for faster testing
            interval="day",
            sector=sector
        )
        
        if error_message:
            print(f"   ❌ Analysis failed: {error_message}")
            return
        
        if analysis_results and 'sector_context' in analysis_results:
            sector_context = analysis_results['sector_context']
            print(f"   ✅ Full analysis completed with sector context")
            
            # Check sector rotation in context
            if 'sector_rotation' in sector_context:
                sector_rotation = sector_context['sector_rotation']
                print(f"   📈 Sector rotation included in context")
                print(f"   💪 Rotation strength: {sector_rotation.get('rotation_patterns', {}).get('rotation_strength', 'N/A')}")
                
                # Check if the enhanced sector context is built properly
                if 'enhanced_sector_context' in sector_context:
                    enhanced_context = sector_context['enhanced_sector_context']
                    rotation_insights = enhanced_context.get('rotation_insights', {})
                    print(f"   🎯 Enhanced context sector rank: {rotation_insights.get('sector_rank', 'N/A')}")
                    print(f"   📊 Enhanced context sector performance: {rotation_insights.get('sector_performance', 'N/A')}")
                    print(f"   💪 Enhanced context rotation strength: {rotation_insights.get('rotation_strength', 'N/A')}")
                    print(f"   📈 Leading sectors: {len(rotation_insights.get('leading_sectors', []))}")
                    print(f"   📉 Lagging sectors: {len(rotation_insights.get('lagging_sectors', []))}")
                else:
                    print("   ⚠️ Enhanced sector context not found")
            else:
                print("   ⚠️ Sector rotation not found in sector context")
        else:
            print("   ⚠️ No sector context in analysis results")
            
    except Exception as e:
        print(f"   ❌ Full analysis failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 50)
    print("🏁 Test completed")

if __name__ == "__main__":
    asyncio.run(test_full_sector_rotation_flow())