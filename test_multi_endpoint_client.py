#!/usr/bin/env python3
"""
test_multi_endpoint_client.py

Test client for the Multi-Endpoint WebSocket Stream Service.
This script connects to different WebSocket endpoints and tests their specific functionalities.
"""

import asyncio
import json
import websockets
import time
from datetime import datetime

# WebSocket service configuration
WEBSOCKET_BASE_URL = "ws://localhost:8081"
WEBSOCKET_ENDPOINTS = {
    "main": "/ws/stream",
    "analysis": "/ws/analysis", 
    "alerts": "/ws/alerts",
    "portfolio": "/ws/portfolio"
}

async def test_endpoint(endpoint_name: str, endpoint_path: str):
    """Test a specific WebSocket endpoint."""
    url = WEBSOCKET_BASE_URL + endpoint_path
    print(f"\n🔗 Testing {endpoint_name.upper()} endpoint: {url}")
    print("-" * 60)
    
    try:
        async with websockets.connect(url) as websocket:
            print(f"✅ Connected to {endpoint_name} endpoint")
            
            # Wait for welcome message
            try:
                welcome = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                welcome_data = json.loads(welcome)
                print(f"📨 Welcome: {welcome_data.get('message', 'No message')}")
                print(f"📊 Data types: {welcome_data.get('data_types', [])}")
            except asyncio.TimeoutError:
                print("⚠️  No welcome message received")
            
            # Test ping/pong
            print(f"\n🔍 Testing ping/pong on {endpoint_name}...")
            ping_message = {
                "type": "ping",
                "timestamp": time.time()
            }
            await websocket.send(json.dumps(ping_message))
            
            try:
                pong = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                pong_data = json.loads(pong)
                print(f"✅ Pong received: {pong_data.get('type', 'Unknown')}")
            except asyncio.TimeoutError:
                print("❌ No pong received")
            
            # Test subscription
            print(f"\n📊 Testing subscription on {endpoint_name}...")
            subscription_message = {
                "type": "subscribe",
                "symbol": "RELIANCE"
            }
            await websocket.send(json.dumps(subscription_message))
            
            try:
                subscription_ack = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                subscription_data = json.loads(subscription_ack)
                print(f"✅ Subscription: {subscription_data.get('status', 'Unknown')} for {subscription_data.get('symbol', 'Unknown')}")
            except asyncio.TimeoutError:
                print("❌ No subscription response received")
            
            # Listen for data for a few seconds
            print(f"\n📈 Listening for {endpoint_name} data for 8 seconds...")
            start_time = time.time()
            data_count = 0
            
            while time.time() - start_time < 8:
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                    data = json.loads(message)
                    data_count += 1
                    
                    if data.get('type') == 'data':
                        print(f"📨 Data #{data_count}: {data.get('endpoint', 'Unknown')} - {data.get('symbol', 'Unknown')}")
                        # Print endpoint-specific data structure
                        if endpoint_name == "main":
                            print(f"   Price: {data.get('data', {}).get('price', 'N/A')}")
                        elif endpoint_name == "analysis":
                            print(f"   RSI: {data.get('data', {}).get('technical_indicators', {}).get('rsi', 'N/A')}")
                        elif endpoint_name == "alerts":
                            alerts = data.get('data', {}).get('price_alerts', [])
                            print(f"   Alerts: {len(alerts)} active")
                        elif endpoint_name == "portfolio":
                            pnl = data.get('data', {}).get('pnl', {})
                            print(f"   PnL: {pnl.get('unrealized', 'N/A')}")
                    else:
                        print(f"📨 Message: {data.get('type', 'Unknown')}")
                        
                except asyncio.TimeoutError:
                    # No message received, continue
                    pass
                except Exception as e:
                    print(f"❌ Error receiving message: {e}")
                    break
            
            print(f"✅ Received {data_count} messages from {endpoint_name} endpoint")
            
            # Test unsubscribe
            print(f"\n📊 Testing unsubscribe on {endpoint_name}...")
            unsubscribe_message = {
                "type": "unsubscribe",
                "symbol": "RELIANCE"
            }
            await websocket.send(json.dumps(unsubscribe_message))
            
            try:
                unsubscribe_ack = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                unsubscribe_data = json.loads(unsubscribe_ack)
                print(f"✅ Unsubscribe: {unsubscribe_data.get('status', 'Unknown')}")
            except asyncio.TimeoutError:
                print("❌ No unsubscribe response received")
            
            print(f"✅ {endpoint_name.upper()} endpoint test completed successfully!")
            
    except (websockets.exceptions.ConnectionClosed, ConnectionRefusedError, TimeoutError, websockets.exceptions.InvalidStatus):
        print(f"❌ Connection failed to {endpoint_name} endpoint")
    except Exception as e:
        print(f"❌ Error testing {endpoint_name} endpoint: {e}")

async def test_health_endpoint():
    """Test the health endpoint."""
    import aiohttp
    
    print("\n🏥 Testing health endpoint...")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:8081/health") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Health check passed: {data.get('status', 'Unknown')}")
                    print(f"📊 Total connections: {data.get('total_connections', 0)}")
                    print(f"📈 Active streams: {data.get('active_streams', 0)}")
                    
                    # Print endpoint statistics
                    endpoints = data.get('endpoints', {})
                    for endpoint, stats in endpoints.items():
                        print(f"   {endpoint}: {stats.get('connections', 0)} connections")
                else:
                    print(f"❌ Health check failed: HTTP {response.status}")
    except Exception as e:
        print(f"❌ Error testing health endpoint: {e}")

async def test_connections_endpoint():
    """Test the connections endpoint."""
    import aiohttp
    
    print("\n🔗 Testing connections endpoint...")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:8081/connections") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Connections endpoint: {data.get('total_connections', 0)} total connections")
                    
                    # Print endpoint connections
                    endpoint_connections = data.get('endpoint_connections', {})
                    for endpoint, connections in endpoint_connections.items():
                        print(f"   {endpoint}: {len(connections)} connections")
                else:
                    print(f"❌ Connections endpoint failed: HTTP {response.status}")
    except Exception as e:
        print(f"❌ Error testing connections endpoint: {e}")

async def test_concurrent_connections():
    """Test multiple concurrent connections to different endpoints."""
    print("\n🔄 Testing concurrent connections to multiple endpoints...")
    
    async def connect_to_endpoint(endpoint_name, endpoint_path):
        url = WEBSOCKET_BASE_URL + endpoint_path
        try:
            websocket = await websockets.connect(url)
            
            # Send subscription
            await websocket.send(json.dumps({
                "type": "subscribe",
                "symbol": "RELIANCE"
            }))
            
            # Listen for a few messages
            for i in range(3):
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=3.0)
                    data = json.loads(message)
                    if data.get('type') == 'data':
                        print(f"   {endpoint_name}: Received data #{i+1}")
                except asyncio.TimeoutError:
                    break
                    
            await websocket.close()
            print(f"✅ {endpoint_name} concurrent test completed")
            
        except Exception as e:
            print(f"❌ {endpoint_name} concurrent test failed: {e}")
    
    # Create tasks for all endpoints
    tasks = []
    for endpoint_name, endpoint_path in WEBSOCKET_ENDPOINTS.items():
        task = asyncio.create_task(connect_to_endpoint(endpoint_name, endpoint_path))
        tasks.append(task)
    
    # Wait for all tasks to complete
    await asyncio.gather(*tasks, return_exceptions=True)
    print("✅ All concurrent connection tests completed")

async def main():
    """Main test function."""
    print("=" * 80)
    print("🧪 Multi-Endpoint WebSocket Stream Service Test Client")
    print("=" * 80)
    
    # Test health endpoint first
    await test_health_endpoint()
    
    # Test connections endpoint
    await test_connections_endpoint()
    
    # Test each endpoint individually
    for endpoint_name, endpoint_path in WEBSOCKET_ENDPOINTS.items():
        await test_endpoint(endpoint_name, endpoint_path)
        await asyncio.sleep(1)  # Brief pause between tests
    
    # Test concurrent connections
    await test_concurrent_connections()
    
    print("\n" + "=" * 80)
    print("✅ All multi-endpoint tests completed!")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(main()) 