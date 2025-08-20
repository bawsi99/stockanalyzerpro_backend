#!/usr/bin/env python3
"""
quick_test_endpoints.py

Quick test script for service endpoints.
This script provides simple examples of how to test each service endpoint.
"""

import requests
import json
import time
from datetime import datetime

# Service URLs
SERVICE_URLS = {
    "data": "http://localhost:8000",
    "analysis": "http://localhost:8001", 
    "websocket": "http://localhost:8081",
    "service_endpoints": "http://localhost:8002"
}

def test_health_endpoints():
    """Test health endpoints for all services."""
    print("🔍 Testing Health Endpoints...")
    
    for service_name, url in SERVICE_URLS.items():
        try:
            response = requests.get(f"{url}/health", timeout=10)
            if response.status_code == 200:
                print(f"  ✅ {service_name}: Healthy")
            else:
                print(f"  ❌ {service_name}: Unhealthy (Status: {response.status_code})")
        except Exception as e:
            print(f"  ❌ {service_name}: Error - {str(e)}")

def test_data_service():
    """Test data service endpoints."""
    print("\n📊 Testing Data Service Endpoints...")
    
    # Test stock info
    try:
        response = requests.get(f"{SERVICE_URLS['data']}/stock/RELIANCE/info", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"  ✅ Stock Info: {data.get('symbol', 'N/A')} - ₹{data.get('last_price', 'N/A')}")
        else:
            print(f"  ❌ Stock Info: Failed (Status: {response.status_code})")
    except Exception as e:
        print(f"  ❌ Stock Info: Error - {str(e)}")
    
    # Test market status
    try:
        response = requests.get(f"{SERVICE_URLS['data']}/market/status", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"  ✅ Market Status: {data.get('market_status', 'N/A')}")
        else:
            print(f"  ❌ Market Status: Failed (Status: {response.status_code})")
    except Exception as e:
        print(f"  ❌ Market Status: Error - {str(e)}")

def test_analysis_service():
    """Test analysis service endpoints."""
    print("\n🧠 Testing Analysis Service Endpoints...")
    
    # Test sector list
    try:
        response = requests.get(f"{SERVICE_URLS['analysis']}/sector/list", timeout=10)
        if response.status_code == 200:
            data = response.json()
            sectors = data.get('sectors', [])
            print(f"  ✅ Sector List: {len(sectors)} sectors available")
        else:
            print(f"  ❌ Sector List: Failed (Status: {response.status_code})")
    except Exception as e:
        print(f"  ❌ Sector List: Error - {str(e)}")

def test_service_endpoints():
    """Test service endpoints (component testing)."""
    print("\n🔧 Testing Service Endpoints (Component Testing)...")
    
    # Test status
    try:
        response = requests.get(f"{SERVICE_URLS['service_endpoints']}/status", timeout=10)
        if response.status_code == 200:
            data = response.json()
            endpoints = data.get('endpoints', {})
            print(f"  ✅ Status: {len(endpoints)} endpoint categories available")
        else:
            print(f"  ❌ Status: Failed (Status: {response.status_code})")
    except Exception as e:
        print(f"  ❌ Status: Error - {str(e)}")
    
    # Test data fetch
    try:
        payload = {
            "symbol": "RELIANCE",
            "exchange": "NSE",
            "period": 30,
            "interval": "day"
        }
        response = requests.post(f"{SERVICE_URLS['service_endpoints']}/data/fetch", 
                               json=payload, timeout=30)
        if response.status_code == 200:
            data = response.json()
            print(f"  ✅ Data Fetch: {data.get('data_points', 0)} data points for {data.get('symbol', 'N/A')}")
        else:
            print(f"  ❌ Data Fetch: Failed (Status: {response.status_code})")
    except Exception as e:
        print(f"  ❌ Data Fetch: Error - {str(e)}")
    
    # Test technical indicators
    try:
        payload = {
            "symbol": "RELIANCE",
            "exchange": "NSE",
            "period": 30,
            "interval": "day",
            "indicators": "rsi,macd,sma"
        }
        response = requests.post(f"{SERVICE_URLS['service_endpoints']}/technical/indicators", 
                               json=payload, timeout=30)
        if response.status_code == 200:
            data = response.json()
            indicators = data.get('indicators', {})
            print(f"  ✅ Technical Indicators: {len(indicators)} indicators calculated")
        else:
            print(f"  ❌ Technical Indicators: Failed (Status: {response.status_code})")
    except Exception as e:
        print(f"  ❌ Technical Indicators: Error - {str(e)}")
    
    # Test sector info
    try:
        payload = {
            "symbol": "RELIANCE"
        }
        response = requests.post(f"{SERVICE_URLS['service_endpoints']}/sectors/info", 
                               json=payload, timeout=10)
        if response.status_code == 200:
            data = response.json()
            sector_info = data.get('sector_info', {})
            print(f"  ✅ Sector Info: Sector data retrieved for {data.get('symbol', 'N/A')}")
        else:
            print(f"  ❌ Sector Info: Failed (Status: {response.status_code})")
    except Exception as e:
        print(f"  ❌ Sector Info: Error - {str(e)}")

def main():
    """Main test function."""
    print("🧪 Quick Service Endpoints Test")
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Test all services
    test_health_endpoints()
    test_data_service()
    test_analysis_service()
    test_service_endpoints()
    
    print("\n" + "=" * 60)
    print("✅ Quick test completed!")
    print("📄 For detailed testing, use: python test_service_endpoints.py all")

if __name__ == "__main__":
    main()
