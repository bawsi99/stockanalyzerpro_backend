#!/usr/bin/env python3
"""
test_service_endpoints.py

Comprehensive test script for all service endpoints.
This script demonstrates how to test each service component individually.

Usage:
    python test_service_endpoints.py [service_name]

Available services:
    - data: Test data service endpoints
    - analysis: Test analysis service endpoints  
    - websocket: Test WebSocket service endpoints
    - technical: Test technical analysis endpoints
    - patterns: Test pattern recognition endpoints
    - sectors: Test sector analysis endpoints
    - ml: Test machine learning endpoints
    - charts: Test chart generation endpoints
    - all: Test all endpoints
"""

import asyncio
import aiohttp
import json
import sys
import time
from typing import Dict, List, Any
from datetime import datetime

# Service URLs
SERVICE_URLS = {
    "data": "http://localhost:8000",
    "analysis": "http://localhost:8001", 
    "websocket": "http://localhost:8081",
    "service_endpoints": "http://localhost:8002"
}

# Test data
TEST_SYMBOL = "RELIANCE"
TEST_EXCHANGE = "NSE"
TEST_PERIOD = 365
TEST_INTERVAL = "day"

class ServiceEndpointTester:
    """Test class for service endpoints."""
    
    def __init__(self):
        self.session = None
        self.results = {}
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def test_endpoint(self, service: str, endpoint: str, method: str = "GET", 
                          data: Dict = None, expected_status: int = 200) -> Dict:
        """Test a single endpoint."""
        url = f"{SERVICE_URLS[service]}{endpoint}"
        
        try:
            start_time = time.time()
            
            if method.upper() == "GET":
                async with self.session.get(url) as response:
                    result = await response.json()
                    status = response.status
            elif method.upper() == "POST":
                async with self.session.post(url, json=data) as response:
                    result = await response.json()
                    status = response.status
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            end_time = time.time()
            duration = end_time - start_time
            
            success = status == expected_status
            
            return {
                "service": service,
                "endpoint": endpoint,
                "method": method,
                "url": url,
                "status": status,
                "expected_status": expected_status,
                "success": success,
                "duration": duration,
                "result": result if success else None,
                "error": None if success else result.get("detail", "Unknown error")
            }
            
        except Exception as e:
            return {
                "service": service,
                "endpoint": endpoint,
                "method": method,
                "url": url,
                "status": None,
                "expected_status": expected_status,
                "success": False,
                "duration": 0,
                "result": None,
                "error": str(e)
            }
    
    async def test_data_service(self) -> List[Dict]:
        """Test data service endpoints."""
        print("🔍 Testing Data Service Endpoints...")
        
        tests = [
            ("/health", "GET"),
            ("/stock/RELIANCE/info", "GET"),
            ("/market/status", "GET"),
            ("/mapping/token-to-symbol?symbol=RELIANCE", "GET"),
            ("/stock/RELIANCE/history?period=30&interval=day", "GET"),
        ]
        
        results = []
        for endpoint, method in tests:
            result = await self.test_endpoint("data", endpoint, method)
            results.append(result)
            print(f"  {'✅' if result['success'] else '❌'} {endpoint} ({result['duration']:.2f}s)")
        
        return results
    
    async def test_analysis_service(self) -> List[Dict]:
        """Test analysis service endpoints."""
        print("🧠 Testing Analysis Service Endpoints...")
        
        tests = [
            ("/health", "GET"),
            ("/analyze", "POST", {
                "stock": TEST_SYMBOL,
                "exchange": TEST_EXCHANGE,
                "period": TEST_PERIOD,
                "interval": TEST_INTERVAL
            }),
            ("/sector/list", "GET"),
            ("/stock/RELIANCE/indicators", "GET"),
        ]
        
        results = []
        for test in tests:
            if len(test) == 2:
                endpoint, method = test
                data = None
            else:
                endpoint, method, data = test
            
            result = await self.test_endpoint("analysis", endpoint, method, data)
            results.append(result)
            print(f"  {'✅' if result['success'] else '❌'} {endpoint} ({result['duration']:.2f}s)")
        
        return results
    
    async def test_websocket_service(self) -> List[Dict]:
        """Test WebSocket service endpoints."""
        print("🔗 Testing WebSocket Service Endpoints...")
        
        tests = [
            ("/health", "GET"),
            ("/connections", "GET"),
            ("/test", "GET"),
        ]
        
        results = []
        for endpoint, method in tests:
            result = await self.test_endpoint("websocket", endpoint, method)
            results.append(result)
            print(f"  {'✅' if result['success'] else '❌'} {endpoint} ({result['duration']:.2f}s)")
        
        return results
    
    async def test_service_endpoints(self) -> List[Dict]:
        """Test service endpoints (component testing)."""
        print("🔧 Testing Service Endpoints (Component Testing)...")
        
        tests = [
            ("/health", "GET"),
            ("/status", "GET"),
            ("/data/fetch", "POST", {
                "symbol": TEST_SYMBOL,
                "exchange": TEST_EXCHANGE,
                "period": 30,
                "interval": TEST_INTERVAL
            }),
            ("/data/stock-info/RELIANCE", "GET"),
            ("/data/market-status", "GET"),
            ("/technical/indicators", "POST", {
                "symbol": TEST_SYMBOL,
                "exchange": TEST_EXCHANGE,
                "period": 30,
                "interval": TEST_INTERVAL,
                "indicators": "rsi,macd,sma"
            }),
            ("/patterns/detect", "POST", {
                "symbol": TEST_SYMBOL,
                "exchange": TEST_EXCHANGE,
                "period": 30,
                "interval": TEST_INTERVAL,
                "pattern_types": "candlestick,chart"
            }),
            ("/sectors/info", "POST", {
                "symbol": TEST_SYMBOL
            }),
            ("/sectors/rotation", "GET"),
            ("/sectors/correlation", "GET"),
            ("/ml/model-info", "GET"),
        ]
        
        results = []
        for test in tests:
            if len(test) == 2:
                endpoint, method = test
                data = None
            else:
                endpoint, method, data = test
            
            result = await self.test_endpoint("service_endpoints", endpoint, method, data)
            results.append(result)
            print(f"  {'✅' if result['success'] else '❌'} {endpoint} ({result['duration']:.2f}s)")
        
        return results
    
    async def test_all_services(self) -> Dict[str, List[Dict]]:
        """Test all services."""
        print("🚀 Testing All Services...")
        print("=" * 60)
        
        all_results = {}
        
        # Test each service
        all_results["data"] = await self.test_data_service()
        print()
        
        all_results["analysis"] = await self.test_analysis_service()
        print()
        
        all_results["websocket"] = await self.test_websocket_service()
        print()
        
        all_results["service_endpoints"] = await self.test_service_endpoints()
        print()
        
        return all_results
    
    def print_summary(self, results: Dict[str, List[Dict]]):
        """Print test summary."""
        print("=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        
        total_tests = 0
        total_passed = 0
        total_failed = 0
        
        for service, service_results in results.items():
            service_tests = len(service_results)
            service_passed = sum(1 for r in service_results if r['success'])
            service_failed = service_tests - service_passed
            
            total_tests += service_tests
            total_passed += service_passed
            total_failed += service_failed
            
            print(f"{service.upper():<20} | Tests: {service_tests:>3} | Passed: {service_passed:>3} | Failed: {service_failed:>3}")
        
        print("-" * 60)
        print(f"{'TOTAL':<20} | Tests: {total_tests:>3} | Passed: {total_passed:>3} | Failed: {total_failed:>3}")
        
        if total_failed == 0:
            print("🎉 All tests passed!")
        else:
            print(f"⚠️  {total_failed} tests failed. Check the details above.")
        
        # Save results to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"test_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "summary": {
                    "total_tests": total_tests,
                    "total_passed": total_passed,
                    "total_failed": total_failed
                },
                "results": results
            }, f, indent=2)
        
        print(f"📄 Detailed results saved to: {filename}")

async def main():
    """Main test function."""
    if len(sys.argv) < 2:
        print("Usage: python test_service_endpoints.py [service_name]")
        print("Available services: data, analysis, websocket, service_endpoints, all")
        return
    
    service = sys.argv[1].lower()
    
    if service not in ["data", "analysis", "websocket", "service_endpoints", "all"]:
        print(f"Unknown service: {service}")
        print("Available services: data, analysis, websocket, service_endpoints, all")
        return
    
    print(f"🧪 Testing Service: {service.upper()}")
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    async with ServiceEndpointTester() as tester:
        if service == "all":
            results = await tester.test_all_services()
        elif service == "data":
            results = {"data": await tester.test_data_service()}
        elif service == "analysis":
            results = {"analysis": await tester.test_analysis_service()}
        elif service == "websocket":
            results = {"websocket": await tester.test_websocket_service()}
        elif service == "service_endpoints":
            results = {"service_endpoints": await tester.test_service_endpoints()}
        
        tester.print_summary(results)

if __name__ == "__main__":
    asyncio.run(main())
