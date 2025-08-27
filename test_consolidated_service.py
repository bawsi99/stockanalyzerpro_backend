#!/usr/bin/env python3
"""
test_consolidated_service.py

Test script to verify that the consolidated service is working properly.
This script tests the basic functionality of the consolidated service.
"""

import asyncio
import websockets
import json
import requests
import time

# Test configuration
BASE_URL = "http://localhost:8000"
WS_URL = "ws://localhost:8000/ws/stream"

async def test_health_endpoints():
    """Test health endpoints."""
    print("🔍 Testing health endpoints...")
    
    try:
        # Test main health endpoint
        response = requests.get(f"{BASE_URL}/health")
        print(f"✅ Main health endpoint: {response.status_code}")
        print(f"   Response: {response.json()}")
        
        # Test data service health
        response = requests.get(f"{BASE_URL}/data/health")
        print(f"✅ Data service health: {response.status_code}")
        print(f"   Response: {response.json()}")
        
        # Test analysis service health
        response = requests.get(f"{BASE_URL}/analysis/health")
        print(f"✅ Analysis service health: {response.status_code}")
        print(f"   Response: {response.json()}")
        
    except Exception as e:
        print(f"❌ Health endpoint test failed: {e}")

async def test_auth_endpoints():
    """Test authentication endpoints."""
    print("\n🔐 Testing authentication endpoints...")
    
    try:
        # Test auth verify endpoint
        response = requests.get(f"{BASE_URL}/auth/verify?token=test_token")
        print(f"✅ Auth verify endpoint: {response.status_code}")
        print(f"   Response: {response.json()}")
        
        # Test auth token endpoint
        response = requests.post(f"{BASE_URL}/auth/token?user_id=test_user")
        print(f"✅ Auth token endpoint: {response.status_code}")
        print(f"   Response: {response.json()}")
        
    except Exception as e:
        print(f"❌ Auth endpoint test failed: {e}")

async def test_websocket_connection():
    """Test WebSocket connection."""
    print("\n🔌 Testing WebSocket connection...")
    
    try:
        # Connect to WebSocket
        async with websockets.connect(WS_URL) as websocket:
            print("✅ WebSocket connection established")
            
            # Send a test message
            test_message = {
                "action": "ping",
                "timestamp": time.time()
            }
            await websocket.send(json.dumps(test_message))
            print("✅ Test message sent")
            
            # Wait for response
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                print(f"✅ Received response: {response}")
            except asyncio.TimeoutError:
                print("⚠️  No response received within 5 seconds")
            
    except Exception as e:
        print(f"❌ WebSocket connection test failed: {e}")

async def test_data_endpoints():
    """Test data service endpoints."""
    print("\n📊 Testing data service endpoints...")
    
    try:
        # Test stock history endpoint
        response = requests.get(f"{BASE_URL}/data/stock/RELIANCE/history?interval=1day&limit=10")
        print(f"✅ Stock history endpoint: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Candles count: {data.get('count', 0)}")
        else:
            print(f"   Error: {response.text}")
        
    except Exception as e:
        print(f"❌ Data endpoint test failed: {e}")

async def test_analysis_endpoints():
    """Test analysis service endpoints."""
    print("\n🔍 Testing analysis service endpoints...")
    
    try:
        # Test sector list endpoint
        response = requests.get(f"{BASE_URL}/analysis/sector/list")
        print(f"✅ Sector list endpoint: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Sectors count: {data.get('total_sectors', 0)}")
        else:
            print(f"   Error: {response.text}")
        
    except Exception as e:
        print(f"❌ Analysis endpoint test failed: {e}")

async def main():
    """Run all tests."""
    print("🚀 Starting consolidated service tests...")
    print("=" * 60)
    
    # Test health endpoints
    await test_health_endpoints()
    
    # Test auth endpoints
    await test_auth_endpoints()
    
    # Test data endpoints
    await test_data_endpoints()
    
    # Test analysis endpoints
    await test_analysis_endpoints()
    
    # Test WebSocket connection
    await test_websocket_connection()
    
    print("\n" + "=" * 60)
    print("✅ All tests completed!")

if __name__ == "__main__":
    asyncio.run(main())
