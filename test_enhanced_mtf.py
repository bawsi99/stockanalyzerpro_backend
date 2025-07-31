#!/usr/bin/env python3
"""
Test script for Enhanced Multi-Timeframe Analysis

This script demonstrates the comprehensive multi-timeframe analysis capabilities
including cross-timeframe validation, divergence detection, and confidence-weighted recommendations.
"""

import asyncio
import json
import sys
import os
from datetime import datetime

# Add the backend directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_mtf_analysis import enhanced_mtf_analyzer

async def test_enhanced_mtf_analysis():
    """Test the enhanced multi-timeframe analysis with a sample stock."""
    
    # Test stock symbol
    test_symbol = "RELIANCE"  # Reliance Industries
    exchange = "NSE"
    
    print("=" * 80)
    print("ENHANCED MULTI-TIMEFRAME ANALYSIS TEST")
    print("=" * 80)
    print(f"Testing with: {test_symbol} on {exchange}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()
    
    try:
        # Authenticate with Zerodha
        print("🔐 Authenticating with Zerodha API...")
        auth_success = await enhanced_mtf_analyzer.authenticate()
        if not auth_success:
            print("❌ Authentication failed. Please check your Zerodha credentials.")
            return
        print("✅ Authentication successful")
        print()
        
        # Perform comprehensive multi-timeframe analysis
        print("📊 Starting comprehensive multi-timeframe analysis...")
        print("   This will analyze: 1min, 5min, 15min, 30min, 1hour, 1day timeframes")
        print("   Calculating indicators, patterns, support/resistance, and cross-timeframe validation...")
        print()
        
        start_time = datetime.now()
        results = await enhanced_mtf_analyzer.comprehensive_mtf_analysis(test_symbol, exchange)
        end_time = datetime.now()
        
        analysis_duration = (end_time - start_time).total_seconds()
        
        if not results.get('success', False):
            print(f"❌ Analysis failed: {results.get('error', 'Unknown error')}")
            return
        
        print(f"✅ Analysis completed in {analysis_duration:.2f} seconds")
        print()
        
        # Display results
        print("📈 ANALYSIS RESULTS")
        print("=" * 50)
        
        # Summary
        summary = results.get('summary', {})
        print(f"Overall Signal: {summary.get('overall_signal', 'Unknown')}")
        print(f"Confidence: {summary.get('confidence', 0):.2%}")
        print(f"Risk Level: {summary.get('risk_level', 'Unknown')}")
        print(f"Recommendation: {summary.get('recommendation', 'Unknown')}")
        print(f"Timeframes Analyzed: {summary.get('timeframes_analyzed', 0)}")
        print(f"Signal Alignment: {summary.get('signal_alignment', 'Unknown')}")
        print()
        
        # Cross-timeframe validation
        validation = results.get('cross_timeframe_validation', {})
        print("🔄 CROSS-TIMEFRAME VALIDATION")
        print("-" * 40)
        print(f"Consensus Trend: {validation.get('consensus_trend', 'Unknown')}")
        print(f"Signal Strength: {validation.get('signal_strength', 0):.2%}")
        print(f"Confidence Score: {validation.get('confidence_score', 0):.2%}")
        print()
        
        # Supporting timeframes
        supporting = validation.get('supporting_timeframes', [])
        if supporting:
            print("✅ Supporting Timeframes:")
            for tf in supporting:
                config = enhanced_mtf_analyzer.timeframe_configs.get(tf)
                if config:
                    print(f"   • {tf} ({config.description})")
        print()
        
        # Conflicting timeframes
        conflicting = validation.get('conflicting_timeframes', [])
        if conflicting:
            print("⚠️  Conflicting Timeframes:")
            for tf in conflicting:
                config = enhanced_mtf_analyzer.timeframe_configs.get(tf)
                if config:
                    print(f"   • {tf} ({config.description})")
        print()
        
        # Divergence detection
        if validation.get('divergence_detected', False):
            print("🔍 DIVERGENCE DETECTED")
            print("-" * 25)
            print(f"Type: {validation.get('divergence_type', 'Unknown')}")
            print("⚠️  This indicates potential trend reversal or weakening")
            print()
        
        # Key conflicts
        key_conflicts = validation.get('key_conflicts', [])
        if key_conflicts:
            print("⚠️  KEY CONFLICTS:")
            for conflict in key_conflicts:
                print(f"   • {conflict}")
            print()
        
        # Individual timeframe analysis
        print("📊 INDIVIDUAL TIMEFRAME ANALYSIS")
        print("=" * 40)
        
        timeframe_analyses = results.get('timeframe_analyses', {})
        for timeframe, analysis in timeframe_analyses.items():
            config = enhanced_mtf_analyzer.timeframe_configs.get(timeframe)
            config_desc = config.description if config else 'Unknown'
            print(f"\n{timeframe.upper()} ({config_desc})")
            print("-" * (len(timeframe) + len(config_desc) + 3))
            print(f"Trend: {analysis.get('trend', 'Unknown')}")
            print(f"Confidence: {analysis.get('confidence', 0):.2%}")
            print(f"Data Points: {analysis.get('data_points', 0)}")
            
            # Key indicators
            indicators = analysis.get('key_indicators', {})
            if indicators:
                print("Key Indicators:")
                if indicators.get('rsi'):
                    print(f"   RSI: {indicators['rsi']:.2f}")
                if indicators.get('macd_signal'):
                    print(f"   MACD Signal: {indicators['macd_signal']}")
                if indicators.get('volume_status'):
                    print(f"   Volume: {indicators['volume_status']}")
            
            # Support/Resistance levels
            support_levels = indicators.get('support_levels', [])
            resistance_levels = indicators.get('resistance_levels', [])
            if support_levels:
                print(f"   Support Levels: {[f'{level:.2f}' for level in support_levels]}")
            if resistance_levels:
                print(f"   Resistance Levels: {[f'{level:.2f}' for level in resistance_levels]}")
            
            # Patterns
            patterns = analysis.get('patterns', [])
            if patterns:
                print(f"   Patterns: {', '.join(patterns)}")
            
            # Risk metrics
            risk_metrics = analysis.get('risk_metrics', {})
            if risk_metrics.get('current_price'):
                print(f"   Current Price: ₹{risk_metrics['current_price']:.2f}")
            if risk_metrics.get('volatility'):
                print(f"   Volatility: {risk_metrics['volatility']:.2%}")
            if risk_metrics.get('max_drawdown'):
                print(f"   Max Drawdown: {risk_metrics['max_drawdown']:.2%}")
        
        print()
        print("=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)
        
        # Save results to file
        output_file = f"enhanced_mtf_analysis_{test_symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"📄 Results saved to: {output_file}")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

async def test_multiple_stocks():
    """Test the enhanced multi-timeframe analysis with multiple stocks."""
    
    test_stocks = [
        "RELIANCE",   # Reliance Industries
        "TCS",        # Tata Consultancy Services
        "HDFCBANK",   # HDFC Bank
        "INFY",       # Infosys
        "ICICIBANK"   # ICICI Bank
    ]
    
    print("=" * 80)
    print("MULTIPLE STOCKS ENHANCED MTF ANALYSIS TEST")
    print("=" * 80)
    print(f"Testing {len(test_stocks)} stocks: {', '.join(test_stocks)}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()
    
    # Authenticate once
    print("🔐 Authenticating with Zerodha API...")
    auth_success = await enhanced_mtf_analyzer.authenticate()
    if not auth_success:
        print("❌ Authentication failed. Please check your Zerodha credentials.")
        return
    print("✅ Authentication successful")
    print()
    
    results_summary = []
    
    for i, stock in enumerate(test_stocks, 1):
        print(f"📊 Analyzing {stock} ({i}/{len(test_stocks)})...")
        
        try:
            start_time = datetime.now()
            results = await enhanced_mtf_analyzer.comprehensive_mtf_analysis(stock, "NSE")
            end_time = datetime.now()
            
            duration = (end_time - start_time).total_seconds()
            
            if results.get('success', False):
                summary = results.get('summary', {})
                validation = results.get('cross_timeframe_validation', {})
                
                stock_summary = {
                    'symbol': stock,
                    'overall_signal': summary.get('overall_signal', 'Unknown'),
                    'confidence': summary.get('confidence', 0),
                    'risk_level': summary.get('risk_level', 'Unknown'),
                    'recommendation': summary.get('recommendation', 'Unknown'),
                    'consensus_trend': validation.get('consensus_trend', 'Unknown'),
                    'signal_strength': validation.get('signal_strength', 0),
                    'supporting_timeframes': len(validation.get('supporting_timeframes', [])),
                    'conflicting_timeframes': len(validation.get('conflicting_timeframes', [])),
                    'divergence_detected': validation.get('divergence_detected', False),
                    'analysis_duration': duration,
                    'timeframes_analyzed': summary.get('timeframes_analyzed', 0)
                }
                
                results_summary.append(stock_summary)
                
                print(f"   ✅ {stock}: {summary.get('overall_signal', 'Unknown')} "
                      f"(Confidence: {summary.get('confidence', 0):.2%}, "
                      f"Duration: {duration:.2f}s)")
                
                if validation.get('divergence_detected', False):
                    print(f"   ⚠️  Divergence detected: {validation.get('divergence_type', 'Unknown')}")
                
            else:
                print(f"   ❌ {stock}: Analysis failed - {results.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"   ❌ {stock}: Error - {e}")
        
        print()
    
    # Display summary table
    print("📋 ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"{'Symbol':<12} {'Signal':<8} {'Confidence':<12} {'Risk':<6} {'Supporting':<10} {'Conflicting':<11} {'Divergence':<10}")
    print("-" * 80)
    
    for summary in results_summary:
        print(f"{summary['symbol']:<12} {summary['overall_signal']:<8} "
              f"{summary['confidence']:<12.2%} {summary['risk_level']:<6} "
              f"{summary['supporting_timeframes']:<10} {summary['conflicting_timeframes']:<11} "
              f"{'Yes' if summary['divergence_detected'] else 'No':<10}")
    
    print()
    print("=" * 80)
    print("MULTIPLE STOCKS ANALYSIS COMPLETE")
    print("=" * 80)
    
    # Save summary to file
    summary_file = f"enhanced_mtf_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_file, 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)
    print(f"📄 Summary saved to: {summary_file}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Enhanced Multi-Timeframe Analysis')
    parser.add_argument('--mode', choices=['single', 'multiple'], default='single',
                       help='Test mode: single stock or multiple stocks')
    parser.add_argument('--stock', default='RELIANCE',
                       help='Stock symbol for single mode test')
    
    args = parser.parse_args()
    
    if args.mode == 'single':
        asyncio.run(test_enhanced_mtf_analysis())
    else:
        asyncio.run(test_multiple_stocks()) 