#!/usr/bin/env python3
"""
Quantitative System Output Validator

This script analyzes a stock and shows exactly what your quantitative system provides,
making it easy to validate that everything is working correctly.

Usage:
    python validate_system_output.py [symbol] [period] [interval]

Examples:
    python validate_system_output.py RELIANCE 30 day
    python validate_system_output.py TCS 365 day
    python validate_system_output.py INFY 90 day
"""

import asyncio
import sys
import json
from datetime import datetime
from pathlib import Path

# Add the backend directory to Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

def print_section(title, data, max_items=3):
    """Print a section with formatted output."""
    print(f"\n{'='*60}")
    print(f"📊 {title}")
    print(f"{'='*60}")
    
    if not data:
        print("❌ No data available")
        return
    
    if isinstance(data, dict):
        for key, value in list(data.items())[:max_items]:
            if isinstance(value, (dict, list)):
                print(f"  {key}: {type(value).__name__} ({len(value) if hasattr(value, '__len__') else 'N/A'} items)")
            else:
                print(f"  {key}: {value}")
        
        if len(data) > max_items:
            print(f"  ... and {len(data) - max_items} more items")
    
    elif isinstance(data, list):
        for i, item in enumerate(data[:max_items]):
            if isinstance(item, dict):
                print(f"  Item {i+1}: {type(item).__name__} ({len(item)} keys)")
            else:
                print(f"  Item {i+1}: {item}")
        
        if len(data) > max_items:
            print(f"  ... and {len(data) - max_items} more items")
    
    else:
        print(f"  {data}")

def validate_ai_analysis(ai_analysis):
    """Validate AI analysis output."""
    print_section("AI-Powered Market Analysis", ai_analysis)
    
    # Check for required fields
    required_fields = ['trend', 'confidence_pct']
    missing_fields = [field for field in required_fields if field not in ai_analysis]
    
    if missing_fields:
        print(f"❌ Missing required AI analysis fields: {missing_fields}")
        return False
    
    # Validate confidence percentage
    confidence = ai_analysis.get('confidence_pct', 0)
    if not (0 <= confidence <= 100):
        print(f"❌ Invalid confidence percentage: {confidence} (should be 0-100)")
        return False
    
    print(f"✅ AI Analysis: {ai_analysis.get('trend', 'Unknown')} trend with {confidence}% confidence")
    return True

def validate_technical_indicators(indicators):
    """Validate technical indicators output."""
    print_section("Technical Indicators Analysis", indicators)
    
    # Check for major indicators
    major_indicators = ['rsi', 'macd', 'sma', 'ema', 'bollinger_bands']
    present_indicators = [ind for ind in major_indicators if ind in indicators]
    
    if len(present_indicators) < 3:
        print(f"❌ Too few technical indicators: {present_indicators}")
        return False
    
    print(f"✅ Technical Indicators: {len(present_indicators)} major indicators present")
    
    # Validate RSI if present
    if 'rsi' in indicators:
        rsi_data = indicators['rsi']
        if isinstance(rsi_data, dict) and 'current' in rsi_data:
            rsi_value = rsi_data['current']
            if not (0 <= rsi_value <= 100):
                print(f"❌ Invalid RSI value: {rsi_value} (should be 0-100)")
                return False
            print(f"✅ RSI: {rsi_value} ({rsi_data.get('signal', 'Unknown')})")
    
    return True

def validate_patterns(patterns):
    """Validate pattern recognition output."""
    print_section("Pattern Recognition", patterns)
    
    if not patterns:
        print("❌ No patterns detected")
        return False
    
    # Check for common pattern types
    pattern_types = ['support_levels', 'resistance_levels', 'triangles', 'flags']
    detected_patterns = [pt for pt in pattern_types if pt in patterns]
    
    print(f"✅ Pattern Recognition: {len(detected_patterns)} pattern types detected")
    
    # Show some pattern details
    for pattern_type in detected_patterns[:2]:  # Show first 2 types
        pattern_data = patterns[pattern_type]
        if isinstance(pattern_data, list):
            print(f"  {pattern_type}: {len(pattern_data)} patterns")
        else:
            print(f"  {pattern_type}: {type(pattern_data).__name__}")
    
    return True

def validate_sector_analysis(sector_analysis):
    """Validate sector analysis output."""
    print_section("Sector Analysis & Benchmarking", sector_analysis)
    
    if not sector_analysis:
        print("❌ No sector analysis available")
        return False
    
    # Check for sector classification
    if 'sector' in sector_analysis:
        sector = sector_analysis['sector']
        print(f"✅ Sector Classification: {sector}")
    else:
        print("❌ Missing sector classification")
        return False
    
    # Check for performance metrics
    performance_metrics = ['sector_performance', 'relative_strength', 'sector_rotation']
    present_metrics = [metric for metric in performance_metrics if metric in sector_analysis]
    
    print(f"✅ Sector Metrics: {len(present_metrics)} performance metrics available")
    
    return True

def validate_mtf_analysis(mtf_analysis):
    """Validate multi-timeframe analysis output."""
    print_section("Multi-Timeframe Analysis", mtf_analysis)
    
    if not mtf_analysis:
        print("❌ No multi-timeframe analysis available")
        return False
    
    # Check for timeframe analyses
    if 'timeframe_analyses' in mtf_analysis:
        timeframes = mtf_analysis['timeframe_analyses']
        if isinstance(timeframes, dict):
            print(f"✅ Timeframes Analyzed: {len(timeframes)} timeframes")
            for tf in list(timeframes.keys())[:3]:  # Show first 3
                tf_data = timeframes[tf]
                if isinstance(tf_data, dict) and 'trend' in tf_data:
                    print(f"  {tf}: {tf_data['trend']} trend")
        else:
            print(f"✅ Timeframe Analysis: {type(timeframes).__name__}")
    else:
        print("❌ Missing timeframe analyses")
        return False
    
    # Check for cross-timeframe validation
    if 'cross_timeframe_validation' in mtf_analysis:
        validation = mtf_analysis['cross_timeframe_validation']
        if isinstance(validation, dict) and 'consensus_trend' in validation:
            consensus = validation['consensus_trend']
            confidence = validation.get('confidence_score', 0)
            print(f"✅ Cross-Timeframe Consensus: {consensus} (confidence: {confidence:.2f})")
    
    return True

def validate_ml_predictions(ml_predictions):
    """Validate machine learning predictions output."""
    print_section("Machine Learning Predictions", ml_predictions)
    
    if not ml_predictions:
        print("❌ No ML predictions available")
        return False
    
    # Check for pattern success probability
    if 'pattern_success_probability' in ml_predictions:
        prob = ml_predictions['pattern_success_probability']
        if not (0 <= prob <= 1):
            print(f"❌ Invalid probability: {prob} (should be 0-1)")
            return False
        print(f"✅ Pattern Success Probability: {prob:.2f}")
    
    # Check for signal scoring
    if 'signal_score' in ml_predictions:
        score = ml_predictions['signal_score']
        if not (0 <= score <= 1):
            print(f"❌ Invalid signal score: {score} (should be 0-1)")
            return False
        print(f"✅ Signal Score: {score:.2f}")
    
    # Check for risk metrics
    if 'risk_metrics' in ml_predictions:
        risk_metrics = ml_predictions['risk_metrics']
        print(f"✅ Risk Metrics: {len(risk_metrics)} metrics available")
    
    return True

def validate_risk_management(risk_management):
    """Validate risk management output."""
    print_section("Risk Management & Performance Metrics", risk_management)
    
    if not risk_management:
        print("❌ No risk management data available")
        return False
    
    # Check for risk level
    if 'risk_level' in risk_management:
        risk_level = risk_management['risk_level']
        valid_levels = ['Low', 'Medium', 'High', 'Very High']
        if risk_level not in valid_levels:
            print(f"❌ Invalid risk level: {risk_level}")
            return False
        print(f"✅ Risk Level: {risk_level}")
    
    # Check for stop loss and targets
    if 'stop_loss' in risk_management:
        stop_loss = risk_management['stop_loss']
        print(f"✅ Stop Loss: {stop_loss}")
    
    if 'targets' in risk_management:
        targets = risk_management['targets']
        if isinstance(targets, dict):
            print(f"✅ Targets: {len(targets)} timeframe targets")
        else:
            print(f"✅ Targets: {type(targets).__name__}")
    
    # Check for performance metrics
    if 'performance_metrics' in risk_management:
        perf_metrics = risk_management['performance_metrics']
        print(f"✅ Performance Metrics: {len(perf_metrics)} metrics available")
    
    return True

async def validate_system_output(symbol, period, interval):
    """Validate the complete system output."""
    print(f"🚀 Validating Quantitative System Output")
    print(f"Symbol: {symbol}")
    print(f"Period: {period} days")
    print(f"Interval: {interval}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"{'='*60}")
    
    try:
        # Import and initialize orchestrator
        from agent_capabilities import StockAnalysisOrchestrator
        
        orchestrator = StockAnalysisOrchestrator()
        
        # Authenticate
        print("🔐 Authenticating with Zerodha API...")
        if not orchestrator.authenticate():
            print("❌ Authentication failed")
            return False
        
        print("✅ Authentication successful")
        
        # Perform analysis
        print(f"📊 Analyzing {symbol}...")
        result = await orchestrator.enhanced_analyze_stock(
            symbol=symbol,
            exchange="NSE",
            period=period,
            interval=interval
        )
        
        analysis_results, success_message, error_message = result
        
        if error_message:
            print(f"❌ Analysis failed: {error_message}")
            return False
        
        print("✅ Analysis completed successfully")
        print(f"📝 Message: {success_message}")
        
        # Validate each component
        validation_results = []
        
        # 1. AI Analysis
        ai_analysis = analysis_results.get('ai_analysis', {})
        validation_results.append(("AI Analysis", validate_ai_analysis(ai_analysis)))
        
        # 2. Technical Indicators
        indicators = analysis_results.get('indicators', {})
        validation_results.append(("Technical Indicators", validate_technical_indicators(indicators)))
        
        # 3. Patterns
        patterns = analysis_results.get('patterns', {})
        validation_results.append(("Pattern Recognition", validate_patterns(patterns)))
        
        # 4. Sector Analysis
        sector_analysis = analysis_results.get('sector_analysis', {})
        validation_results.append(("Sector Analysis", validate_sector_analysis(sector_analysis)))
        
        # 5. Multi-Timeframe Analysis
        mtf_analysis = analysis_results.get('multi_timeframe_analysis', {})
        validation_results.append(("Multi-Timeframe Analysis", validate_mtf_analysis(mtf_analysis)))
        
        # 6. ML Predictions
        ml_predictions = analysis_results.get('ml_predictions', {})
        validation_results.append(("ML Predictions", validate_ml_predictions(ml_predictions)))
        
        # 7. Risk Management
        risk_management = analysis_results.get('risk_management', {})
        validation_results.append(("Risk Management", validate_risk_management(risk_management)))
        
        # Summary
        print(f"\n{'='*60}")
        print("📋 VALIDATION SUMMARY")
        print(f"{'='*60}")
        
        passed = 0
        total = len(validation_results)
        
        for component, is_valid in validation_results:
            status = "✅ PASSED" if is_valid else "❌ FAILED"
            print(f"{component:<25} {status}")
            if is_valid:
                passed += 1
        
        print(f"\n📊 Results: {passed}/{total} components validated successfully")
        
        if passed == total:
            print("🎉 All components are working correctly!")
            print("\n📈 Your quantitative system is providing comprehensive analysis including:")
            print("   • AI-powered market analysis with confidence scores")
            print("   • Technical indicators across multiple timeframes")
            print("   • Pattern recognition and chart analysis")
            print("   • Sector benchmarking and relative performance")
            print("   • Multi-timeframe consensus and validation")
            print("   • Machine learning predictions and risk metrics")
            print("   • Risk management and performance analysis")
        else:
            print(f"⚠️  {total - passed} components need attention")
        
        # Save detailed results
        output_file = f"validation_results_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump({
                'symbol': symbol,
                'period': period,
                'interval': interval,
                'timestamp': datetime.now().isoformat(),
                'validation_results': dict(validation_results),
                'summary': {
                    'passed': passed,
                    'total': total,
                    'success_rate': passed / total if total > 0 else 0
                },
                'analysis_results': analysis_results
            }, f, indent=2, default=str)
        
        print(f"\n📄 Detailed results saved to: {output_file}")
        
        return passed == total
        
    except Exception as e:
        print(f"❌ Validation failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function."""
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python validate_system_output.py [symbol] [period] [interval]")
        print("Examples:")
        print("  python validate_system_output.py RELIANCE 30 day")
        print("  python validate_system_output.py TCS 365 day")
        print("  python validate_system_output.py INFY 90 day")
        return
    
    symbol = sys.argv[1].upper()
    period = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    interval = sys.argv[3] if len(sys.argv) > 3 else "day"
    
    # Validate inputs
    if period <= 0:
        print("❌ Period must be positive")
        return
    
    valid_intervals = ["minute", "5minute", "15minute", "30minute", "60minute", "day", "week", "month"]
    if interval not in valid_intervals:
        print(f"❌ Invalid interval. Must be one of: {valid_intervals}")
        return
    
    # Run validation
    success = asyncio.run(validate_system_output(symbol, period, interval))
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
