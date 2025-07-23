#!/usr/bin/env python3
"""
Test script to validate Zerodha WebSocket connection and data flow.
This script helps debug WebSocket issues and verify data parsing.
"""

import os
import sys
import time
import json
from typing import List, Dict, Any, Optional

# Try to import optional dependencies
try:
    import dotenv
    dotenv.load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Environment variables may not be loaded.")

try:
    from zerodha_ws_client import zerodha_ws_client, candle_aggregator
except ImportError as e:
    print(f"Error importing zerodha_ws_client: {e}")
    print("Make sure you're running this script from the backend directory")
    sys.exit(1)

def test_websocket_connection() -> bool:
    """Test the WebSocket connection and data flow."""
    print("=== Zerodha WebSocket Connection Test ===")
    
    # Check environment variables
    api_key = os.getenv('ZERODHA_API_KEY')
    access_token = os.getenv('ZERODHA_ACCESS_TOKEN')
    
    print(f"API Key: {api_key[:10] if api_key else 'NOT SET'}...")
    print(f"Access Token: {access_token[:10] if access_token else 'NOT SET'}...")
    
    if not api_key or not access_token:
        print("❌ Environment variables not set!")
        print("Please set ZERODHA_API_KEY and ZERODHA_ACCESS_TOKEN in your .env file")
        return False
    
    # Test tokens (RELIANCE, TCS, HDFC)
    test_tokens = [256265, 11536, 1330]
    
    # Register callback to capture data
    received_data: List[Dict[str, Any]] = []
    
    def data_callback(ticks: List[Dict[str, Any]]) -> None:
        """Callback to capture received data."""
        print(f"📊 Received {len(ticks)} ticks")
        for tick in ticks:
            if isinstance(tick, dict) and 'instrument_token' in tick:
                received_data.append(tick)
                print(f"  Token: {tick['instrument_token']}, Price: {tick.get('last_price', 'N/A')}")
    
    # Register the callback
    zerodha_ws_client.register_tick_hook(data_callback)
    
    try:
        # Connect to WebSocket
        print("\n🔌 Connecting to Zerodha WebSocket...")
        zerodha_ws_client.connect()
        
        # Wait for connection
        time.sleep(3)
        
        if not zerodha_ws_client.running:
            print("❌ Failed to connect to WebSocket")
            return False
        
        print("✅ WebSocket connected successfully")
        
        # Subscribe to test tokens
        print(f"\n📡 Subscribing to tokens: {test_tokens}")
        zerodha_ws_client.subscribe(test_tokens)
        
        # Set mode to quote for OHLCV data
        print("🔧 Setting mode to 'quote' for OHLCV data")
        zerodha_ws_client.set_mode('quote', test_tokens)
        
        # Wait for data
        print("\n⏳ Waiting for data (30 seconds)...")
        start_time = time.time()
        
        while time.time() - start_time < 30:
            if received_data:
                print(f"✅ Received {len(received_data)} data points")
                break
            time.sleep(1)
        
        if not received_data:
            print("❌ No data received within 30 seconds")
            print("This could indicate:")
            print("  - Market is closed")
            print("  - Authentication issues")
            print("  - Network connectivity problems")
            return False
        
        # Test candle aggregation
        print("\n🕯️ Testing candle aggregation...")
        for token in test_tokens:
            candle = candle_aggregator.get_latest_candle(token, '1m')
            if candle:
                print(f"  Token {token}: {candle}")
            else:
                print(f"  Token {token}: No candle data")
        
        print("\n✅ WebSocket test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error during WebSocket test: {e}")
        return False
    
    finally:
        # Cleanup
        try:
            zerodha_ws_client.disconnectWebSocket()
        except Exception as e:
            print(f"Warning: Error during cleanup: {e}")

def test_candle_aggregation() -> None:
    """Test candle aggregation with sample data."""
    print("\n=== Candle Aggregation Test ===")
    
    # Sample tick data
    sample_ticks = [
        {
            'instrument_token': 256265,
            'last_price': 2500.0,
            'volume_traded': 1000,
            'timestamp': int(time.time())
        },
        {
            'instrument_token': 256265,
            'last_price': 2505.0,
            'volume_traded': 500,
            'timestamp': int(time.time()) + 30
        }
    ]
    
    print("Processing sample ticks...")
    for tick in sample_ticks:
        candle_aggregator.process_tick(tick)
        print(f"  Processed tick: {tick['last_price']}")
    
    # Get aggregated candle
    candle = candle_aggregator.get_latest_candle(256265, '1m')
    if candle:
        print(f"✅ Aggregated candle: {candle}")
    else:
        print("❌ No candle aggregated")

def main() -> None:
    """Main function to run all tests."""
    print("Starting Zerodha WebSocket tests...")
    
    # Test candle aggregation first
    test_candle_aggregation()
    
    # Test WebSocket connection
    success = test_websocket_connection()
    
    if success:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print("\n💥 Some tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 