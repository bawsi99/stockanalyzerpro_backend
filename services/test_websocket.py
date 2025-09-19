#!/usr/bin/env python3
"""
WebSocket Test Script for StockAnalyzer Pro Backend
Tests the WebSocket streaming functionality
"""

import asyncio
import websockets
import json
import uuid
import time

async def test_websocket_connection():
    """Test WebSocket connection and data streaming."""
    
    # Generate a test user ID
    test_user_id = str(uuid.uuid4())
    
    # WebSocket URL with authentication (auth disabled in config)
    ws_url = "ws://localhost:8000/data/ws/stream"
    
    print("🔌 Connecting to WebSocket...")
    print(f"URL: {ws_url}")
    
    try:
        async with websockets.connect(ws_url) as websocket:
            print("✅ WebSocket connection established!")
            
            # Send subscription message
            subscription_message = {
                "action": "subscribe",
                "symbols": ["TATAMOTORS"],  # Subscribe to TATAMOTORS
                "timeframes": ["1min", "5min"],
                "format": "json"
            }
            
            print(f"📤 Sending subscription: {subscription_message}")
            await websocket.send(json.dumps(subscription_message))
            
            # Listen for messages for 10 seconds
            print("👂 Listening for messages...")
            start_time = time.time()
            message_count = 0
            
            while time.time() - start_time < 10:  # Listen for 10 seconds
                try:
                    # Set timeout to avoid hanging
                    message = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                    message_count += 1
                    
                    try:
                        data = json.loads(message)
                        msg_type = data.get('type', 'unknown')
                        print(f"📨 Message {message_count} ({msg_type}): {json.dumps(data, indent=2)[:200]}...")
                        
                        if msg_type == 'tick':
                            print(f"   💹 Tick: {data.get('token')} - Price: {data.get('price')}")
                        elif msg_type == 'candle':
                            candle_data = data.get('data', {})
                            print(f"   🕯️ Candle: {data.get('timeframe')} - OHLC: {candle_data.get('open')}/{candle_data.get('high')}/{candle_data.get('low')}/{candle_data.get('close')}")
                        
                    except json.JSONDecodeError:
                        print(f"📨 Raw message {message_count}: {message[:100]}...")
                        
                except asyncio.TimeoutError:
                    print("⏱️ No message received in last 2 seconds...")
                    continue
                except websockets.exceptions.ConnectionClosed:
                    print("❌ WebSocket connection closed")
                    break
            
            print(f"📊 Test completed. Received {message_count} messages in 10 seconds.")
            
    except Exception as e:
        print(f"❌ WebSocket test failed: {e}")
        return False
    
    return True

async def test_websocket_endpoints():
    """Test different WebSocket endpoints and functionality."""
    
    print("\n" + "="*50)
    print("🧪 WEBSOCKET FUNCTIONALITY TEST")
    print("="*50)
    
    # Test basic connection
    print("\n1️⃣ Testing basic WebSocket connection...")
    connection_success = await test_websocket_connection()
    
    if connection_success:
        print("✅ WebSocket streaming test completed successfully!")
    else:
        print("❌ WebSocket streaming test failed!")
    
    return connection_success

if __name__ == "__main__":
    asyncio.run(test_websocket_endpoints())