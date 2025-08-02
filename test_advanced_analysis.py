#!/usr/bin/env python3
"""
Test script for Advanced Analysis Module
This script tests the advanced analysis functionality to ensure it generates the expected data structure.
"""

import asyncio
import pandas as pd
import numpy as np
from advanced_analysis import advanced_analysis_provider

async def test_advanced_analysis():
    """Test the advanced analysis module with sample data."""
    
    print("🧪 Testing Advanced Analysis Module...")
    
    # Create sample stock data
    print("📊 Creating sample stock data...")
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    np.random.seed(42)
    
    # Generate realistic price data with some volatility
    base_price = 100
    returns = np.random.normal(0.001, 0.02, 100)  # Daily returns with 2% volatility
    prices = [base_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    prices = np.array(prices)
    
    # Create DataFrame
    data = pd.DataFrame({
        'open': prices * 0.99,
        'high': prices * 1.02,
        'low': prices * 0.98,
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, 100)
    }, index=dates)
    
    print(f"✅ Sample data created: {len(data)} days, price range: {prices.min():.2f} - {prices.max():.2f}")
    
    # Create sample indicators
    print("📈 Creating sample indicators...")
    indicators = {
        'sma_20': [prices[-1]] * 20,
        'sma_50': [prices[-1]] * 50,
        'rsi_14': [50] * 14,
        'macd_line': [0] * 20,
        'signal_line': [0] * 20,
        'bollinger_upper': [prices[-1] * 1.05] * 20,
        'bollinger_lower': [prices[-1] * 0.95] * 20
    }
    
    print("✅ Sample indicators created")
    
    # Test advanced analysis generation
    print("🔬 Generating advanced analysis...")
    try:
        result = await advanced_analysis_provider.generate_advanced_analysis(data, 'TEST', indicators)
        
        print("\n📋 Advanced Analysis Results:")
        print("=" * 50)
        
        # Check each component
        components = {
            'Advanced Risk Metrics': result.get('advanced_risk', {}),
            'Advanced Patterns': result.get('advanced_patterns', {}),
            'Stress Testing': result.get('stress_testing', {}),
            'Scenario Analysis': result.get('scenario_analysis', {})
        }
        
        for component_name, component_data in components.items():
            if component_data and not component_data.get('error'):
                print(f"✅ {component_name}: Available")
                if component_name == 'Advanced Risk Metrics':
                    risk_keys = list(component_data.keys())[:5]
                    print(f"   Sample fields: {risk_keys}")
                elif component_name == 'Advanced Patterns':
                    pattern_keys = list(component_data.keys())[:5]
                    print(f"   Sample fields: {pattern_keys}")
                elif component_name == 'Stress Testing':
                    stress_keys = list(component_data.keys())[:5]
                    print(f"   Sample fields: {stress_keys}")
                elif component_name == 'Scenario Analysis':
                    scenario_keys = list(component_data.keys())[:5]
                    print(f"   Sample fields: {scenario_keys}")
            else:
                error_msg = component_data.get('error', 'No data available')
                print(f"❌ {component_name}: {error_msg}")
        
        # Test specific metrics
        print("\n🔍 Detailed Metrics Check:")
        print("=" * 30)
        
        advanced_risk = result.get('advanced_risk', {})
        if advanced_risk and not advanced_risk.get('error'):
            print(f"📊 Risk Score: {advanced_risk.get('risk_score', 'N/A')}")
            print(f"📊 Risk Level: {advanced_risk.get('risk_level', 'N/A')}")
            print(f"📊 Volatility: {advanced_risk.get('annualized_volatility', 'N/A'):.4f}")
            print(f"📊 Sharpe Ratio: {advanced_risk.get('sharpe_ratio', 'N/A'):.4f}")
            print(f"📊 Max Drawdown: {advanced_risk.get('max_drawdown', 'N/A'):.4f}")
        
        stress_testing = result.get('stress_testing', {})
        if stress_testing and not stress_testing.get('error'):
            print(f"📊 Stress Score: {stress_testing.get('stress_score', 'N/A')}")
            print(f"📊 Stress Level: {stress_testing.get('stress_level', 'N/A')}")
            print(f"📊 Worst Case: {stress_testing.get('worst_case_scenario', 'N/A'):.4f}")
        
        scenario_analysis = result.get('scenario_analysis', {})
        if scenario_analysis and not scenario_analysis.get('error'):
            best_case = scenario_analysis.get('best_case', {})
            worst_case = scenario_analysis.get('worst_case', {})
            print(f"📊 Best Case Probability: {best_case.get('probability', 'N/A'):.2f}")
            print(f"📊 Worst Case Probability: {worst_case.get('probability', 'N/A'):.2f}")
        
        print("\n✅ Advanced analysis test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error during advanced analysis: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_frontend_data_structure():
    """Test that the data structure matches frontend expectations."""
    
    print("\n🎯 Testing Frontend Data Structure Compatibility...")
    
    # Create sample data
    dates = pd.date_range('2024-01-01', periods=50, freq='D')
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(50) * 0.5)
    data = pd.DataFrame({
        'open': prices * 0.99,
        'high': prices * 1.02,
        'low': prices * 0.98,
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, 50)
    }, index=dates)
    
    indicators = {
        'sma_20': [prices[-1]] * 20,
        'rsi_14': [50] * 14,
        'macd_line': [0] * 20
    }
    
    # Generate advanced analysis
    result = await advanced_analysis_provider.generate_advanced_analysis(data, 'TEST', indicators)
    
    # Check frontend expected structure
    frontend_structure = {
        'enhanced_metadata': {
            'advanced_risk_metrics': result.get('advanced_risk', {}),
            'stress_testing_metrics': result.get('stress_testing', {}),
            'scenario_analysis_metrics': result.get('scenario_analysis', {})
        },
        'overlays': {
            'advanced_patterns': result.get('advanced_patterns', {})
        }
    }
    
    print("📋 Frontend Data Structure:")
    print("=" * 40)
    
    # Check enhanced_metadata
    enhanced_metadata = frontend_structure['enhanced_metadata']
    for key, value in enhanced_metadata.items():
        if value and not value.get('error'):
            print(f"✅ {key}: Available")
        else:
            print(f"❌ {key}: Not available")
    
    # Check overlays
    overlays = frontend_structure['overlays']
    for key, value in overlays.items():
        if value and not value.get('error'):
            print(f"✅ {key}: Available")
        else:
            print(f"❌ {key}: Not available")
    
    print("\n✅ Frontend data structure test completed!")
    return True

if __name__ == "__main__":
    print("🚀 Starting Advanced Analysis Tests...")
    print("=" * 50)
    
    # Run tests
    async def run_tests():
        test1_success = await test_advanced_analysis()
        test2_success = await test_frontend_data_structure()
        
        print("\n" + "=" * 50)
        if test1_success and test2_success:
            print("🎉 All tests passed! Advanced analysis is working correctly.")
        else:
            print("❌ Some tests failed. Please check the implementation.")
    
    asyncio.run(run_tests()) 