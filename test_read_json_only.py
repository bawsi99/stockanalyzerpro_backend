"""
Test Script: Read and Display Existing JSON Results
This script reads the existing results.json file and displays its structure
"""

import json
import os
from datetime import datetime

def test_read_json_only():
    """Test reading and displaying the existing JSON results file."""
    print("🧪 Testing Read Existing JSON File")
    print("=" * 60)
    
    # File path
    json_file_path = "output/RELIANCE/results.json"
    
    # Check if file exists
    if not os.path.exists(json_file_path):
        print(f"❌ File not found: {json_file_path}")
        print("Please run main.py first to generate the results.json file")
        return False
    
    print(f"✅ Found file: {json_file_path}")
    
    # Read the JSON file
    print("\n1. Reading JSON file...")
    try:
        with open(json_file_path, 'r') as f:
            analysis_data = json.load(f)
        
        print(f"✅ Successfully read JSON file")
        print(f"   File size: {os.path.getsize(json_file_path)} bytes")
        print(f"   Data keys: {list(analysis_data.keys())}")
        
    except Exception as e:
        print(f"❌ Error reading JSON file: {e}")
        return False
    
    # Display the structure
    print("\n2. JSON Structure Analysis:")
    print("-" * 40)
    
    for key, value in analysis_data.items():
        if isinstance(value, dict):
            print(f"📁 {key}:")
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, dict):
                    print(f"   📁 {sub_key}: {type(sub_value).__name__} with {len(sub_value)} items")
                elif isinstance(sub_value, list):
                    print(f"   📋 {sub_key}: {type(sub_value).__name__} with {len(sub_value)} items")
                else:
                    print(f"   📄 {sub_key}: {type(sub_value).__name__} = {str(sub_value)[:50]}{'...' if len(str(sub_value)) > 50 else ''}")
        elif isinstance(value, list):
            print(f"📋 {key}: {type(value).__name__} with {len(value)} items")
        else:
            print(f"📄 {key}: {type(value).__name__} = {str(value)[:50]}{'...' if len(str(value)) > 50 else ''}")
    
    # Show key summary data
    print("\n3. Key Summary Data:")
    print("-" * 40)
    
    if 'summary' in analysis_data:
        summary = analysis_data['summary']
        print(f"📊 Overall Signal: {summary.get('overall_signal', 'N/A')}")
        print(f"📊 Confidence: {summary.get('confidence', 'N/A')}")
        print(f"📊 Risk Level: {summary.get('risk_level', 'N/A')}")
        print(f"📊 Recommendation: {summary.get('recommendation', 'N/A')}")
    
    if 'metadata' in analysis_data:
        metadata = analysis_data['metadata']
        print(f"📊 Symbol: {metadata.get('symbol', 'N/A')}")
        print(f"📊 Exchange: {metadata.get('exchange', 'N/A')}")
        print(f"📊 Period: {metadata.get('period_days', 'N/A')} days")
        print(f"📊 Interval: {metadata.get('interval', 'N/A')}")
        print(f"📊 Sector: {metadata.get('sector', 'N/A')}")
    
    # Show technical indicators
    print("\n4. Technical Indicators:")
    print("-" * 40)
    
    if 'technical_indicators' in analysis_data:
        indicators = analysis_data['technical_indicators']
        for indicator_name, indicator_data in indicators.items():
            if isinstance(indicator_data, dict):
                value = indicator_data.get('value', 'N/A')
                signal = indicator_data.get('signal', 'N/A')
                print(f"📈 {indicator_name}: {value} ({signal})")
            else:
                print(f"📈 {indicator_name}: {indicator_data}")
    
    # Show patterns if available
    print("\n5. Patterns:")
    print("-" * 40)
    
    if 'patterns' in analysis_data:
        patterns = analysis_data['patterns']
        for pattern_type, pattern_data in patterns.items():
            if isinstance(pattern_data, list):
                print(f"📊 {pattern_type}: {len(pattern_data)} items")
                for i, pattern in enumerate(pattern_data[:3]):  # Show first 3
                    if isinstance(pattern, dict):
                        print(f"   {i+1}. {pattern.get('name', 'Unknown')} - {pattern.get('confidence', 'N/A')}%")
                    else:
                        print(f"   {i+1}. {pattern}")
            else:
                print(f"📊 {pattern_type}: {pattern_data}")
    
    # Show AI analysis if available
    print("\n6. AI Analysis:")
    print("-" * 40)
    
    if 'ai_analysis' in analysis_data:
        ai_analysis = analysis_data['ai_analysis']
        print(f"🤖 Trend: {ai_analysis.get('trend', 'N/A')}")
        print(f"🤖 Confidence: {ai_analysis.get('confidence_pct', 'N/A')}%")
        
        if 'short_term' in ai_analysis:
            short_term = ai_analysis['short_term']
            print(f"🤖 Short Term: {short_term.get('signal', 'N/A')} - {short_term.get('target', 'N/A')}")
        
        if 'medium_term' in ai_analysis:
            medium_term = ai_analysis['medium_term']
            print(f"🤖 Medium Term: {medium_term.get('signal', 'N/A')} - {medium_term.get('target', 'N/A')}")
    
    # Show trading levels if available
    print("\n7. Trading Levels:")
    print("-" * 40)
    
    if 'trading_guidance' in analysis_data:
        guidance = analysis_data['trading_guidance']
        if 'key_levels' in guidance:
            levels = guidance['key_levels']
            if isinstance(levels, list):
                for i, level in enumerate(levels[:5]):  # Show first 5
                    print(f"💰 Level {i+1}: {level}")
    
    # Show sector benchmarking if available
    print("\n8. Sector Benchmarking:")
    print("-" * 40)
    
    if 'sector_benchmarking' in analysis_data and analysis_data['sector_benchmarking'] is not None:
        sector_data = analysis_data['sector_benchmarking']
        print(f"🏭 Sector: {sector_data.get('sector', 'N/A')}")
        print(f"🏭 Beta: {sector_data.get('beta', 'N/A')}")
        print(f"🏭 Sharpe Ratio: {sector_data.get('sharpe_ratio', 'N/A')}")
        print(f"🏭 Volatility: {sector_data.get('volatility', 'N/A')}")
    else:
        print("🏭 Sector Benchmarking: Not available")
    
    print("\n" + "=" * 60)
    print("🎉 JSON File Analysis Complete!")
    print("✅ File structure analyzed")
    print("✅ Key data extracted")
    print("✅ Ready for database storage")
    print("=" * 60)
    print("\n📋 Next Steps:")
    print("1. Set up Supabase environment variables")
    print("2. Run test_store_existing_json.py to store in database")
    print("3. Verify data in Supabase dashboard")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    test_read_json_only() 