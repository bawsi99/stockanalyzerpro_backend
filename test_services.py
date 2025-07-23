#!/usr/bin/env python3
"""
test_services.py

Test script to verify that both data service and analysis service are working correctly.
"""

import asyncio
import aiohttp
import json
import time
from typing import Dict, Any

# Service URLs
DATA_SERVICE_URL = "http://localhost:8000"
ANALYSIS_SERVICE_URL = "http://localhost:8001"

async def test_health_endpoint(session: aiohttp.ClientSession, service_name: str, url: str) -> bool:
    """Test health endpoint for a service."""
    try:
        async with session.get(f"{url}/health") as response:
            if response.status == 200:
                data = await response.json()
                print(f"✅ {service_name} Health: {data}")
                return True
            else:
                print(f"❌ {service_name} Health: Status {response.status}")
                return False
    except Exception as e:
        print(f"❌ {service_name} Health: {e}")
        return False

async def test_data_service(session: aiohttp.ClientSession) -> bool:
    """Test data service endpoints."""
    print("\n🔍 Testing Data Service...")
    
    # Test health
    if not await test_health_endpoint(session, "Data Service", DATA_SERVICE_URL):
        return False
    
    # Test market status
    try:
        async with session.get(f"{DATA_SERVICE_URL}/market/status") as response:
            if response.status == 200:
                data = await response.json()
                print(f"✅ Market Status: {data.get('market_status', {}).get('status', 'unknown')}")
            else:
                print(f"⚠️  Market Status: Status {response.status}")
    except Exception as e:
        print(f"⚠️  Market Status: {e}")
    
    # Test WebSocket health
    try:
        async with session.get(f"{DATA_SERVICE_URL}/ws/health") as response:
            if response.status == 200:
                data = await response.json()
                print(f"✅ WebSocket Health: {data.get('websocket_stats', {}).get('total_clients', 0)} clients")
            else:
                print(f"⚠️  WebSocket Health: Status {response.status}")
    except Exception as e:
        print(f"⚠️  WebSocket Health: {e}")
    
    return True

async def test_analysis_service(session: aiohttp.ClientSession) -> bool:
    """Test analysis service endpoints."""
    print("\n🧠 Testing Analysis Service...")
    
    # Test health
    if not await test_health_endpoint(session, "Analysis Service", ANALYSIS_SERVICE_URL):
        return False
    
    # Test sector list
    try:
        async with session.get(f"{ANALYSIS_SERVICE_URL}/sector/list") as response:
            if response.status == 200:
                data = await response.json()
                sectors = data.get('sectors', [])
                print(f"✅ Sector List: {len(sectors)} sectors available")
            else:
                print(f"⚠️  Sector List: Status {response.status}")
    except Exception as e:
        print(f"⚠️  Sector List: {e}")
    
    # Test stock sector info
    try:
        async with session.get(f"{ANALYSIS_SERVICE_URL}/stock/RELIANCE/sector") as response:
            if response.status == 200:
                data = await response.json()
                sector_info = data.get('sector_info', {})
                print(f"✅ Stock Sector: RELIANCE -> {sector_info.get('sector', 'unknown')}")
            else:
                print(f"⚠️  Stock Sector: Status {response.status}")
    except Exception as e:
        print(f"⚠️  Stock Sector: {e}")
    
    return True

async def test_service_communication():
    """Test communication between services."""
    print("\n🔗 Testing Service Communication...")
    
    async with aiohttp.ClientSession() as session:
        # Test data service
        data_service_ok = await test_data_service(session)
        
        # Test analysis service
        analysis_service_ok = await test_analysis_service(session)
        
        # Summary
        print("\n" + "="*50)
        print("📊 TEST SUMMARY")
        print("="*50)
        print(f"Data Service:     {'✅ OK' if data_service_ok else '❌ FAILED'}")
        print(f"Analysis Service: {'✅ OK' if analysis_service_ok else '❌ FAILED'}")
        
        if data_service_ok and analysis_service_ok:
            print("\n🎉 Both services are running correctly!")
            print("\n📝 Next Steps:")
            print("1. Data Service:     http://localhost:8000")
            print("2. Analysis Service: http://localhost:8001")
            print("3. WebSocket:        ws://localhost:8000/ws/stream")
            print("4. Frontend:         Update API endpoints to use both services")
        else:
            print("\n⚠️  Some services are not working correctly.")
            print("Please check:")
            print("1. Are both services running?")
            print("2. Are the ports 8000 and 8001 available?")
            print("3. Are all dependencies installed?")
            print("4. Are environment variables set correctly?")
        
        return data_service_ok and analysis_service_ok

def main():
    """Main test function."""
    print("🧪 Testing Split Backend Services")
    print("="*50)
    
    try:
        result = asyncio.run(test_service_communication())
        return 0 if result else 1
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        return 1

if __name__ == "__main__":
    exit(main()) 