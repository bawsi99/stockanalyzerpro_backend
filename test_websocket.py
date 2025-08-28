#!/usr/bin/env python3
"""
Test WebSocket connection to debug connection issues.
"""

import asyncio
import websockets
import json

async def test_websocket():
    """Test WebSocket connection to the backend."""
    uri = "ws://localhost:8000/ws/stream"
    
    print(f"🔌 Testing WebSocket connection to: {uri}")
    
    try:
        async with websockets.connect(uri) as websocket:
            print("✅ WebSocket connected successfully!")
            
            # Send a test subscription message
            test_message = {
                "action": "subscribe",
                "symbols": ["RELIANCE"],
                "timeframes": ["1d"]
            }
            
            print(f"📤 Sending test message: {test_message}")
            await websocket.send(json.dumps(test_message))
            
            # Wait for response
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                print(f"📥 Received response: {response}")
            except asyncio.TimeoutError:
                print("⏰ No response received within 5 seconds")
            
            # Send ping
            ping_message = {"action": "ping"}
            print(f"📤 Sending ping: {ping_message}")
            await websocket.send(json.dumps(ping_message))
            
            try:
                pong_response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                print(f"📥 Received pong: {pong_response}")
            except asyncio.TimeoutError:
                print("⏰ No pong response received within 5 seconds")
                
    except Exception as e:
        print(f"❌ WebSocket connection failed: {e}")
        print(f"   Error type: {type(e).__name__}")
        
        # Try to get more details about the error
        if hasattr(e, 'status_code'):
            print(f"   Status code: {e.status_code}")
        if hasattr(e, 'reason'):
            print(f"   Reason: {e.reason}")

if __name__ == "__main__":
    print("🧪 WebSocket Connection Test")
    print("=" * 40)
    
    # Check if backend is running
    import requests
    try:
        response = requests.get("http://localhost:8000/health")
        print(f"✅ Backend health check: {response.status_code}")
        print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"❌ Backend health check failed: {e}")
        print("   Make sure the backend is running on localhost:8000")
        exit(1)
    
    print("\n" + "=" * 40)
    
    # Run WebSocket test
    asyncio.run(test_websocket())
