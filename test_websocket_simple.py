#!/usr/bin/env python3
"""
test_websocket_simple.py

Simple WebSocket test to verify the endpoint is working.
"""

import asyncio
import websockets
import json

async def test_websocket():
    """Test WebSocket connection without authentication."""
    uri = "ws://localhost:8000/data/ws/stream"
    
    try:
        print(f"🔌 Connecting to {uri}")
        async with websockets.connect(uri) as websocket:
            print("✅ WebSocket connection established!")
            
            # Send a simple ping message
            message = {"action": "ping"}
            await websocket.send(json.dumps(message))
            print("✅ Sent ping message")
            
            # Wait for response
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                print(f"✅ Received response: {response}")
            except asyncio.TimeoutError:
                print("⚠️  No response received within 5 seconds")
                
    except Exception as e:
        print(f"❌ WebSocket connection failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_websocket())
