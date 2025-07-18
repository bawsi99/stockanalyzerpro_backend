#!/usr/bin/env python3
"""
Test script for Sector Benchmarking Implementation

This script tests the comprehensive sector benchmarking system to ensure
it works correctly with the existing analysis infrastructure.
"""

import asyncio
import sys
import os
import pandas as pd
from datetime import datetime

# Add the backend directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sector_benchmarking import sector_benchmarking_provider
from sector_classifier import sector_classifier
from agent_capabilities import StockAnalysisOrchestrator
from technical_indicators import IndianMarketMetricsProvider

def test_sector_classifier():
    """Test the sector classifier functionality."""
    print("🔍 Testing Sector Classifier...")
    
    # Test stock sector lookup
    test_stocks = ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK', 'TATAMOTORS']
    
    for stock in test_stocks:
        sector = sector_classifier.get_stock_sector(stock)
        sector_name = sector_classifier.get_sector_display_name(sector) if sector else None
        sector_index = sector_classifier.get_primary_sector_index(sector) if sector else None
        
        print(f"  {stock}: {sector} ({sector_name}) - Index: {sector_index}")
    
    # Test sector statistics
    stats = sector_classifier.get_sector_statistics()
    print(f"  Total sectors: {stats.get('total_sectors', 0)}")
    print(f"  Total stocks: {stats.get('total_stocks', 0)}")
    
    print("✅ Sector Classifier tests completed\n")

def test_sector_benchmarking():
    """Test the sector benchmarking functionality."""
    print("📊 Testing Sector Benchmarking...")
    
    # Initialize orchestrator
    orchestrator = StockAnalysisOrchestrator()
    auth_success = orchestrator.authenticate()
    
    if not auth_success:
        print("❌ Failed to authenticate with Zerodha API")
        return False
    
    # Test with a known stock
    test_stock = 'RELIANCE'
    print(f"  Testing with {test_stock}...")
    
    try:
        # Get stock data
        data = orchestrator.retrieve_stock_data(test_stock, "NSE", "day", 365)
        
        if data is None or data.empty:
            print(f"❌ No data available for {test_stock}")
            return False
        
        print(f"  Retrieved {len(data)} data points for {test_stock}")
        
        # Get comprehensive benchmarking
        benchmarking = sector_benchmarking_provider.get_comprehensive_benchmarking(test_stock, data)
        
        # Validate results
        if not benchmarking:
            print("❌ No benchmarking data returned")
            return False
        
        print("  ✅ Benchmarking data structure:")
        print(f"    - Stock Symbol: {benchmarking.get('stock_symbol')}")
        print(f"    - Sector: {benchmarking.get('sector_info', {}).get('sector_name')}")
        print(f"    - Market Beta: {benchmarking.get('market_benchmarking', {}).get('beta', 'N/A')}")
        
        if benchmarking.get('sector_benchmarking'):
            print(f"    - Sector Beta: {benchmarking['sector_benchmarking'].get('sector_beta', 'N/A')}")
            print(f"    - Sector Outperformance: {benchmarking['sector_benchmarking'].get('sector_excess_return', 'N/A')}")
        else:
            print("    - Sector Benchmarking: Not available")
        
        print(f"    - Performance Ranking: {benchmarking.get('relative_performance', {}).get('performance_ranking', {}).get('vs_market', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing sector benchmarking: {e}")
        return False

def test_market_metrics_provider():
    """Test the market metrics provider."""
    print("📈 Testing Market Metrics Provider...")
    
    try:
        provider = IndianMarketMetricsProvider()
        
        # Test NIFTY 50 data
        nifty_data = provider.get_nifty_50_data(30)
        if nifty_data is not None and len(nifty_data) > 0:
            print(f"  ✅ NIFTY 50 data: {len(nifty_data)} points")
        else:
            print("  ❌ NIFTY 50 data not available")
        
        # Test sector index data
        test_sector = 'BANKING'
        sector_index = sector_classifier.get_primary_sector_index(test_sector)
        if sector_index:
            sector_data = provider.get_sector_index_data(test_sector, 30)
            if sector_data is not None and len(sector_data) > 0:
                print(f"  ✅ {test_sector} sector data: {len(sector_data)} points")
            else:
                print(f"  ❌ {test_sector} sector data not available")
        else:
            print(f"  ❌ No index found for {test_sector} sector")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing market metrics provider: {e}")
        return False

def test_full_analysis_integration():
    """Test the full analysis integration with sector benchmarking."""
    print("🔄 Testing Full Analysis Integration...")
    
    # Initialize orchestrator
    orchestrator = StockAnalysisOrchestrator()
    auth_success = orchestrator.authenticate()
    
    if not auth_success:
        print("❌ Failed to authenticate with Zerodha API")
        return False
    
    # Test with a known stock
    test_stock = 'TCS'
    print(f"  Testing full analysis with {test_stock}...")
    
    try:
        # Run full analysis (without charts for speed)
        results, data = asyncio.run(orchestrator.analyze_stock(
            symbol=test_stock,
            exchange="NSE",
            period=365,
            interval="day",
            output_dir=None  # Skip chart generation for speed
        ))
        
        if not results:
            print("❌ No analysis results returned")
            return False
        
        print("  ✅ Analysis completed successfully")
        
        # Check for sector benchmarking in results
        sector_benchmarking = results.get('sector_benchmarking')
        if sector_benchmarking:
            print("  ✅ Sector benchmarking included in results")
            print(f"    - Sector: {sector_benchmarking.get('sector_info', {}).get('sector_name')}")
            print(f"    - Market Beta: {sector_benchmarking.get('market_benchmarking', {}).get('beta', 'N/A')}")
            
            if sector_benchmarking.get('sector_benchmarking'):
                print(f"    - Sector Beta: {sector_benchmarking['sector_benchmarking'].get('sector_beta', 'N/A')}")
        else:
            print("  ❌ Sector benchmarking not found in results")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing full analysis integration: {e}")
        return False

def test_api_endpoints():
    """Test the new API endpoints."""
    print("🌐 Testing API Endpoints...")
    
    import requests
    
    base_url = "http://127.0.0.1:8000"
    
    try:
        # Test sector list endpoint
        response = requests.get(f"{base_url}/sector/list")
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print(f"  ✅ Sector list: {len(data.get('sectors', []))} sectors")
            else:
                print("  ❌ Sector list failed")
        else:
            print(f"  ❌ Sector list endpoint error: {response.status_code}")
        
        # Test stock sector endpoint
        test_stock = 'RELIANCE'
        response = requests.get(f"{base_url}/stock/{test_stock}/sector")
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                sector_info = data.get('sector_info', {})
                print(f"  ✅ {test_stock} sector: {sector_info.get('sector_name')}")
            else:
                print("  ❌ Stock sector lookup failed")
        else:
            print(f"  ❌ Stock sector endpoint error: {response.status_code}")
        
        # Test sector benchmarking endpoint
        payload = {
            "stock": test_stock,
            "exchange": "NSE",
            "period": 365,
            "interval": "day"
        }
        response = requests.post(f"{base_url}/sector/benchmark", json=payload)
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print(f"  ✅ Sector benchmarking for {test_stock}")
            else:
                print("  ❌ Sector benchmarking failed")
        else:
            print(f"  ❌ Sector benchmarking endpoint error: {response.status_code}")
        
        return True
        
    except requests.exceptions.ConnectionError:
        print("  ❌ API server not running (start with: python api.py)")
        return False
    except Exception as e:
        print(f"  ❌ Error testing API endpoints: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Starting Sector Benchmarking Tests")
    print("=" * 50)
    
    test_results = []
    
    # Run tests
    test_results.append(("Sector Classifier", test_sector_classifier()))
    test_results.append(("Market Metrics Provider", test_market_metrics_provider()))
    test_results.append(("Sector Benchmarking", test_sector_benchmarking()))
    test_results.append(("Full Analysis Integration", test_full_analysis_integration()))
    test_results.append(("API Endpoints", test_api_endpoints()))
    
    # Print summary
    print("\n" + "=" * 50)
    print("📋 Test Summary")
    print("=" * 50)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Sector benchmarking implementation is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 