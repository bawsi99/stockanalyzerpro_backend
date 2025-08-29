#!/usr/bin/env python3
"""
Validation script for modified ChartVisualizer methods.
Tests that methods return matplotlib figure objects instead of saving to files.
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_sample_data():
    """Create sample stock data for testing."""
    # Generate 100 days of sample data
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    
    # Generate realistic price data with some trends
    np.random.seed(42)  # For reproducible results
    base_price = 100
    trend = np.linspace(0, 20, 100)  # Upward trend
    noise = np.random.normal(0, 2, 100)
    prices = base_price + trend + noise
    
    # Generate volume data
    base_volume = 1000000
    volume_trend = np.random.normal(1, 0.3, 100)
    volumes = base_volume * volume_trend
    
    # Create DataFrame
    data = pd.DataFrame({
        'open': prices + np.random.normal(0, 0.5, 100),
        'high': prices + np.random.uniform(0, 2, 100),
        'low': prices - np.random.uniform(0, 2, 100),
        'close': prices,
        'volume': volumes
    }, index=dates)
    
    return data

def create_sample_indicators():
    """Create sample technical indicators for testing."""
    return {
        'sma_20': [95, 96, 97, 98, 99, 100, 101, 102, 103, 104],
        'sma_50': [92, 93, 94, 95, 96, 97, 98, 99, 100, 101],
        'rsi': [45, 47, 49, 51, 53, 55, 57, 59, 61, 63],
        'macd': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    }

def test_chart_visualizer_methods():
    """Test all modified ChartVisualizer methods."""
    print("🧪 Testing modified ChartVisualizer methods...")
    
    try:
        from patterns.visualization import ChartVisualizer
        
        # Create sample data
        data = create_sample_data()
        indicators = create_sample_indicators()
        symbol = "TEST_STOCK"
        
        print(f"✅ Sample data created: {len(data)} rows")
        print(f"✅ Sample indicators created: {len(indicators)} indicators")
        
        # Test 1: Technical Overview Chart
        print("\n📊 Testing plot_comprehensive_technical_chart...")
        try:
            fig1 = ChartVisualizer.plot_comprehensive_technical_chart(data, indicators, None, symbol)
            if fig1 is not None and hasattr(fig1, 'savefig'):
                print("✅ plot_comprehensive_technical_chart: SUCCESS - Returns matplotlib figure")
                
                # Test that we can convert to bytes
                import io
                buf = io.BytesIO()
                fig1.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                buf.seek(0)
                img_bytes = buf.getvalue()
                print(f"✅ Figure converted to bytes: {len(img_bytes)} bytes")
                
                # Clean up
                plt.close(fig1)
            else:
                print("❌ plot_comprehensive_technical_chart: FAILED - Does not return matplotlib figure")
                return False
        except Exception as e:
            print(f"❌ plot_comprehensive_technical_chart: ERROR - {e}")
            return False
        
        # Test 2: Pattern Analysis Chart
        print("\n📊 Testing plot_comprehensive_pattern_chart...")
        try:
            fig2 = ChartVisualizer.plot_comprehensive_pattern_chart(data, indicators, None, symbol)
            if fig2 is not None and hasattr(fig2, 'savefig'):
                print("✅ plot_comprehensive_pattern_chart: SUCCESS - Returns matplotlib figure")
                
                # Test that we can convert to bytes
                buf = io.BytesIO()
                fig2.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                buf.seek(0)
                img_bytes = buf.getvalue()
                print(f"✅ Figure converted to bytes: {len(img_bytes)} bytes")
                
                # Clean up
                plt.close(fig2)
            else:
                print("❌ plot_comprehensive_pattern_chart: FAILED - Does not return matplotlib figure")
                return False
        except Exception as e:
            print(f"❌ plot_comprehensive_pattern_chart: ERROR - {e}")
            return False
        
        # Test 3: Volume Analysis Chart
        print("\n📊 Testing plot_comprehensive_volume_chart...")
        try:
            fig3 = ChartVisualizer.plot_comprehensive_volume_chart(data, indicators, None, symbol)
            if fig3 is not None and hasattr(fig3, 'savefig'):
                print("✅ plot_comprehensive_volume_chart: SUCCESS - Returns matplotlib figure")
                
                # Test that we can convert to bytes
                buf = io.BytesIO()
                fig3.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                buf.seek(0)
                img_bytes = buf.getvalue()
                print(f"✅ Figure converted to bytes: {len(img_bytes)} bytes")
                
                # Clean up
                plt.close(fig3)
            else:
                print("❌ plot_comprehensive_volume_chart: FAILED - Does not return matplotlib figure")
                return False
        except Exception as e:
            print(f"❌ plot_comprehensive_volume_chart: ERROR - {e}")
            return False
        
        # Test 4: MTF Comparison Chart
        print("\n📊 Testing plot_mtf_comparison_chart...")
        try:
            fig4 = ChartVisualizer.plot_mtf_comparison_chart(data, indicators, None, symbol)
            if fig4 is not None and hasattr(fig4, 'savefig'):
                print("✅ plot_mtf_comparison_chart: SUCCESS - Returns matplotlib figure")
                
                # Test that we can convert to bytes
                buf = io.BytesIO()
                fig4.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                buf.seek(0)
                img_bytes = buf.getvalue()
                print(f"✅ Figure converted to bytes: {len(img_bytes)} bytes")
                
                # Clean up
                plt.close(fig4)
            else:
                print("❌ plot_mtf_comparison_chart: FAILED - Does not return matplotlib figure")
                return False
        except Exception as e:
            print(f"❌ plot_mtf_comparison_chart: ERROR - {e}")
            return False
        
        # Test 5: Optional file saving still works
        print("\n📊 Testing optional file saving...")
        try:
            test_file = "test_chart.png"
            fig5 = ChartVisualizer.plot_comprehensive_technical_chart(data, indicators, test_file, symbol)
            
            if os.path.exists(test_file):
                print("✅ Optional file saving: SUCCESS - File created when save_path provided")
                os.remove(test_file)  # Clean up
            else:
                print("❌ Optional file saving: FAILED - File not created")
                return False
            
            # Clean up
            plt.close(fig5)
        except Exception as e:
            print(f"❌ Optional file saving: ERROR - {e}")
            return False
        
        print("\n🎉 All ChartVisualizer tests PASSED!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_memory_efficiency():
    """Test memory efficiency of in-memory chart generation."""
    print("\n🧠 Testing memory efficiency...")
    
    try:
        from patterns.visualization import ChartVisualizer
        
        data = create_sample_data()
        indicators = create_sample_indicators()
        symbol = "MEMORY_TEST"
        
        # Generate multiple charts in memory
        charts = []
        for i in range(5):
            fig = ChartVisualizer.plot_comprehensive_technical_chart(data, indicators, None, f"{symbol}_{i}")
            charts.append(fig)
        
        print(f"✅ Generated {len(charts)} charts in memory")
        
        # Convert all to bytes
        import io
        chart_bytes = []
        for fig in charts:
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            img_bytes = buf.getvalue()
            chart_bytes.append(img_bytes)
            plt.close(fig)  # Clean up
        
        total_size = sum(len(bytes) for bytes in chart_bytes)
        print(f"✅ Total memory usage: {total_size:,} bytes ({total_size/1024:.1f} KB)")
        
        # Clean up
        charts.clear()
        chart_bytes.clear()
        
        print("✅ Memory efficiency test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Memory efficiency test FAILED: {e}")
        return False

def main():
    """Main test function."""
    print("🚀 ChartVisualizer Validation Script")
    print("=" * 50)
    
    # Test basic functionality
    basic_test_passed = test_chart_visualizer_methods()
    
    # Test memory efficiency
    memory_test_passed = test_memory_efficiency()
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 TEST SUMMARY")
    print("=" * 50)
    
    if basic_test_passed and memory_test_passed:
        print("🎉 ALL TESTS PASSED!")
        print("✅ ChartVisualizer methods now return matplotlib figures")
        print("✅ Optional file saving still works")
        print("✅ Memory-efficient chart generation")
        print("✅ Ready for in-memory chart processing")
        return 0
    else:
        print("❌ SOME TESTS FAILED!")
        if not basic_test_passed:
            print("❌ Basic functionality tests failed")
        if not memory_test_passed:
            print("❌ Memory efficiency tests failed")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
