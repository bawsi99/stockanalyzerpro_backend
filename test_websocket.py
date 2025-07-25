#!/usr/bin/env python3
"""
Test script to verify WebSocket streaming functionality
"""

import asyncio
import json
import websockets
import time

async def test_websocket():
    """Test WebSocket connection and subscription"""
    
    # Get authentication token
    import requests
    try:
        response = requests.post("http://localhost:8000/auth/token?user_id=test_user")
        token_data = response.json()
        token = token_data.get('token')
        print(f"✅ Got authentication token: {token[:20]}...")
    except Exception as e:
        print(f"❌ Failed to get authentication token: {e}")
        return
    
    # Connect to WebSocket
    uri = f"ws://localhost:8000/ws/stream?token={token}"
    print(f"🔌 Connecting to WebSocket: {uri}")
    
    try:
        async with websockets.connect(uri) as websocket:
            print("✅ WebSocket connected successfully")
            
            # Wait for welcome message
            welcome_msg = await websocket.recv()
            print(f"📨 Welcome message: {welcome_msg}")
            
            # Send subscription message
            subscription_msg = {
                "action": "subscribe",
                "symbols": ["RELIANCE"],
                "timeframes": ["1d"]
            }
            
            print(f"📤 Sending subscription: {subscription_msg}")
            await websocket.send(json.dumps(subscription_msg))
            
            # Wait for subscription confirmation
            subscription_response = await websocket.recv()
            print(f"📨 Subscription response: {subscription_response}")
            
            # Wait for some data
            print("⏳ Waiting for streaming data...")
            start_time = time.time()
            
            while time.time() - start_time < 30:  # Wait up to 30 seconds
                try:
                    # Set a timeout for receiving messages
                    message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    print(f"📨 Received message: {message}")
                    
                    # Parse the message
                    data = json.loads(message)
                    if data.get('type') in ['tick', 'candle']:
                        print(f"🎉 SUCCESS! Received {data['type']} data: {data}")
                        break
                        
                except asyncio.TimeoutError:
                    print("⏰ No message received in 5 seconds, continuing...")
                    continue
                except Exception as e:
                    print(f"❌ Error receiving message: {e}")
                    break
            
            print("🏁 Test completed")
            
    except Exception as e:
        print(f"❌ WebSocket test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_websocket()) 