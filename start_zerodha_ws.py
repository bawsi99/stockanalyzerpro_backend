#!/usr/bin/env python3
"""
Start Zerodha WebSocket Client

This script starts the Zerodha WebSocket client independently of the FastAPI service.
It's useful when running the consolidated service which doesn't trigger the data_service startup event.
"""

import os
import asyncio
import time
import logging
from zerodha_ws_client import zerodha_ws_client, set_publish_callback, set_main_event_loop

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Loaded environment variables from .env file")
except ImportError:
    print("⚠️ python-dotenv not available, using system environment variables")

# Check Zerodha credentials
api_key = os.getenv("ZERODHA_API_KEY")
access_token = os.getenv("ZERODHA_ACCESS_TOKEN")

if not api_key or not access_token or api_key == 'your_api_key' or access_token == 'your_access_token':
    print("⚠️ Zerodha credentials not configured or invalid")
    print("📊 Historical data will be available via REST API")
    print("🔴 Live data streaming will be disabled")
    print("💡 To enable live data, set ZERODHA_API_KEY and ZERODHA_ACCESS_TOKEN in .env file")
    exit(1)

# Mock publish callback
async def mock_publish(data):
    """Mock publish callback for testing."""
    logger.info(f"📤 Would publish data: {data['type']} - {data.get('token', '')}")

# Main function
async def main():
    """Main function to start and test the WebSocket client."""
    print("🚀 Starting Zerodha WebSocket Client")
    
    # Set the event loop and publish callback
    loop = asyncio.get_running_loop()
    set_main_event_loop(loop)
    set_publish_callback(mock_publish)
    
    # Connect the WebSocket client
    print(f"🔌 Connecting to Zerodha with API key: {api_key[:5]}...")
    zerodha_ws_client.connect()
    
    # Wait for connection to establish
    print("⏳ Waiting for connection to establish...")
    await asyncio.sleep(3)
    
    if zerodha_ws_client.running:
        print("✅ Zerodha WebSocket client connected successfully!")
        print("✅ Live data streaming is enabled")
        
        # Subscribe to some test tokens
        test_tokens = [738561]  # RELIANCE
        print(f"🔔 Subscribing to test tokens: {test_tokens}")
        zerodha_ws_client.subscribe(test_tokens)
        zerodha_ws_client.set_mode('quote', test_tokens)
        
        print("📈 WebSocket client is now running and will continue in the background")
        print("📊 You can now start the consolidated service")
        print("💡 Press Ctrl+C to stop the WebSocket client")
        
        # Keep the script running
        try:
            while True:
                await asyncio.sleep(5)
                if not zerodha_ws_client.running:
                    print("⚠️ WebSocket client disconnected, attempting to reconnect...")
                    zerodha_ws_client.connect()
                    await asyncio.sleep(3)
                    if zerodha_ws_client.running:
                        print("✅ Reconnected successfully")
                        # Resubscribe to tokens
                        zerodha_ws_client.subscribe(test_tokens)
                        zerodha_ws_client.set_mode('quote', test_tokens)
        except KeyboardInterrupt:
            print("👋 Shutting down WebSocket client...")
            zerodha_ws_client.disconnectWebSocket()
    else:
        print("❌ Failed to connect to Zerodha WebSocket")
        print("📊 Historical data will still be available")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("👋 Shutting down...")
    except Exception as e:
        print(f"❌ Error: {e}")
