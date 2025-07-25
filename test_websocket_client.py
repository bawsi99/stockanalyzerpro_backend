#!/usr/bin/env python3
"""
test_websocket_client.py

Test client for the WebSocket Stream Service.
This script connects to the WebSocket service and tests various functionalities.
"""

import asyncio
import json
import websockets
import time
from datetime import datetime

# WebSocket service configuration
WEBSOCKET_URL = "ws://localhost:8081/ws/stream"

async def test_websocket_connection():
    """Test WebSocket connection and basic functionality."""
    print("🧪 Testing WebSocket Stream Service...")
    print(f"🔗 Connecting to: {WEBSOCKET_URL}")
    print("-" * 50)
    
    try:
        async with websockets.connect(WEBSOCKET_URL) as websocket:
            print("✅ Connected to WebSocket service")
            
            # Wait for welcome message
            try:
                welcome = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                welcome_data = json.loads(welcome)
                print(f"📨 Welcome message: {welcome_data.get('message', 'No message')}")
            except asyncio.TimeoutError:
                print("⚠️  No welcome message received within 5 seconds")
            
            # Test ping/pong
            print("\n🔍 Testing ping/pong...")
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
            
            # Test heartbeat
            print("\n💓 Testing heartbeat...")
            heartbeat_message = {
                "type": "heartbeat",
                "timestamp": time.time()
            }
            await websocket.send(json.dumps(heartbeat_message))
            
            try:
                heartbeat_ack = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                heartbeat_data = json.loads(heartbeat_ack)
                print(f"✅ Heartbeat ACK: {heartbeat_data.get('status', 'Unknown')}")
            except asyncio.TimeoutError:
                print("❌ No heartbeat ACK received")
            
            # Test subscription (optional - requires valid symbol)
            print("\n📊 Testing subscription...")
            subscription_message = {
                "type": "subscribe",
                "symbol": "RELIANCE"
            }
            await websocket.send(json.dumps(subscription_message))
            
            try:
                subscription_ack = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                subscription_data = json.loads(subscription_ack)
                print(f"✅ Subscription response: {subscription_data.get('status', 'Unknown')} for {subscription_data.get('symbol', 'Unknown')}")
            except asyncio.TimeoutError:
                print("❌ No subscription response received")
            
            # Test invalid message
            print("\n🚫 Testing invalid message handling...")
            invalid_message = "invalid json message"
            await websocket.send(invalid_message)
            
            try:
                error_response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                error_data = json.loads(error_response)
                print(f"✅ Error handling: {error_data.get('message', 'Unknown error')}")
            except asyncio.TimeoutError:
                print("❌ No error response received")
            
            # Keep connection alive for a few seconds to test data streaming
            print("\n⏳ Keeping connection alive for 10 seconds to test data streaming...")
            start_time = time.time()
            while time.time() - start_time < 10:
                try:
                    # Check for incoming messages
                    message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                    data = json.loads(message)
                    print(f"📨 Received: {data.get('type', 'Unknown')} - {data.get('symbol', 'N/A')}")
                except asyncio.TimeoutError:
                    # No message received, continue
                    pass
                except Exception as e:
                    print(f"❌ Error receiving message: {e}")
                    break
            
            print("\n✅ WebSocket test completed successfully!")
            
    except (websockets.exceptions.ConnectionClosed, ConnectionRefusedError, TimeoutError):
        print("❌ Connection failed. Make sure the WebSocket service is running on port 8081")
    except Exception as e:
        print(f"❌ Error connecting to WebSocket: {e}")

async def test_health_endpoint():
    """Test the health endpoint of the WebSocket service."""
    import aiohttp
    
    print("\n🏥 Testing health endpoint...")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:8081/health") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Health check passed: {data.get('status', 'Unknown')}")
                    print(f"📊 Active connections: {data.get('connections', 0)}")
                    print(f"📈 Active streams: {data.get('active_streams', 0)}")
                else:
                    print(f"❌ Health check failed: HTTP {response.status}")
    except Exception as e:
        print(f"❌ Error testing health endpoint: {e}")

async def test_connections_endpoint():
    """Test the connections endpoint of the WebSocket service."""
    import aiohttp
    
    print("\n🔗 Testing connections endpoint...")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:8081/connections") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Connections endpoint: {data.get('total_connections', 0)} total connections")
                    print(f"📈 Active streams: {data.get('active_streams', [])}")
                else:
                    print(f"❌ Connections endpoint failed: HTTP {response.status}")
    except Exception as e:
        print(f"❌ Error testing connections endpoint: {e}")

async def main():
    """Main test function."""
    print("=" * 60)
    print("🧪 WebSocket Stream Service Test Client")
    print("=" * 60)
    
    # Test health endpoint first
    await test_health_endpoint()
    
    # Test connections endpoint
    await test_connections_endpoint()
    
    # Test WebSocket connection
    await test_websocket_connection()
    
    print("\n" + "=" * 60)
    print("✅ All tests completed!")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main()) 