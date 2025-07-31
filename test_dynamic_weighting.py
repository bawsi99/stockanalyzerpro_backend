#!/usr/bin/env python3
"""
Test script to verify dynamic weighting in enhanced MTF analysis.
This script compares fixed weights vs dynamic weights based on signal quality.
"""

import asyncio
import json
from datetime import datetime
from enhanced_mtf_analysis import enhanced_mtf_analyzer

async def test_dynamic_weighting():
    """Test the dynamic weighting system."""
    print("=" * 80)
    print("DYNAMIC WEIGHTING TEST")
    print("=" * 80)
    
    # Test with a stock
    symbol = "RELIANCE"
    print(f"Testing with: {symbol}")
    
    try:
        # Get MTF analysis results
        result = await enhanced_mtf_analyzer.comprehensive_mtf_analysis(symbol)
        
        if not result.get('success', False):
            print("❌ MTF analysis failed")
            return
        
        print("✅ MTF analysis successful")
        
        # Extract timeframe analyses
        timeframe_analyses = result.get('timeframe_analyses', {})
        cross_validation = result.get('cross_timeframe_validation', {})
        
        print(f"\n📊 OVERALL RESULTS:")
        print(f"   Consensus Trend: {cross_validation.get('consensus_trend', 'Unknown')}")
        print(f"   Signal Strength: {cross_validation.get('signal_strength', 0):.2%}")
        print(f"   Confidence Score: {cross_validation.get('confidence_score', 0):.2%}")
        
        print(f"\n🔄 TIMEFRAME ANALYSIS WITH DYNAMIC WEIGHTING:")
        print("-" * 60)
        
        # Show each timeframe with its dynamic weighting
        for timeframe, analysis in timeframe_analyses.items():
            trend = analysis.get('trend', 'Unknown')
            confidence = analysis.get('confidence', 0)
            data_points = analysis.get('data_points', 0)
            
            # Get base weight from config
            config = enhanced_mtf_analyzer.timeframe_configs[timeframe]
            base_weight = config.weight
            description = config.description
            
            # Calculate dynamic weight (this is what the system actually uses)
            signal_quality = enhanced_mtf_analyzer.calculate_signal_quality(analysis)
            dynamic_weight = base_weight * signal_quality * confidence
            
            # Determine importance level
            if confidence > 0.8:
                importance = "🔥 HIGH"
            elif confidence > 0.6:
                importance = "⚡ MEDIUM-HIGH"
            elif confidence > 0.4:
                importance = "📊 MEDIUM"
            else:
                importance = "⚠️ LOW"
            
            print(f"{timeframe:>6} | {trend:>8} | {confidence:>6.1%} | {base_weight:>5.1%} | {signal_quality:>5.2f} | {dynamic_weight:>6.1%} | {importance} | {description}")
        
        print("-" * 60)
        print("Legend: Base Weight | Signal Quality | Dynamic Weight | Importance")
        
        # Show supporting vs conflicting timeframes
        supporting = cross_validation.get('supporting_timeframes', [])
        conflicting = cross_validation.get('conflicting_timeframes', [])
        
        print(f"\n✅ Supporting Timeframes: {', '.join(supporting) if supporting else 'None'}")
        print(f"⚠️  Conflicting Timeframes: {', '.join(conflicting) if conflicting else 'None'}")
        
        # Show key insights
        print(f"\n💡 KEY INSIGHTS:")
        print(f"   - Dynamic weighting adjusts importance based on signal quality")
        print(f"   - Higher confidence timeframes get more influence")
        print(f"   - Signal quality considers volume, trend consistency, and indicator alignment")
        print(f"   - This makes the analysis more intelligent than fixed weights")
        
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"dynamic_weighting_test_{symbol}_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        print(f"\n📄 Detailed results saved to: {filename}")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_dynamic_weighting()) 