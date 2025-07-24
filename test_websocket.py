#!/usr/bin/env python3
"""
Test script to verify WebSocket connection and tick processing
"""
import os
import time
import logging

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from zerodha_ws_client import ZerodhaWSClient

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_websocket():
    """Test WebSocket connection and tick processing"""
    
    # Get credentials from environment
    api_key = os.getenv('ZERODHA_API_KEY')
    access_token = os.getenv('ZERODHA_ACCESS_TOKEN')
    
    if not api_key or not access_token:
        logger.error("ZERODHA_API_KEY and ZERODHA_ACCESS_TOKEN must be set")
        return
    
    # Create WebSocket client
    client = ZerodhaWSClient(api_key, access_token)
    
    # Connect to WebSocket
    logger.info("Connecting to WebSocket...")
    client.connect()
    
    # Wait for connection
    time.sleep(5)
    
    # Subscribe to RELIANCE token (738561)
    test_token = 738561
    logger.info(f"Subscribing to token {test_token}...")
    client.subscribe([test_token])
    
    # Set mode to quote
    client.set_mode('quote', [test_token])
    
    # Monitor for 30 seconds
    logger.info("Monitoring for 30 seconds...")
    start_time = time.time()
    
    while time.time() - start_time < 30:
        # Get latest tick
        tick = client.get_latest_tick(test_token)
        if tick:
            logger.info(f"Latest tick: {tick}")
        
        # Get market status
        market_info = client.get_market_status()
        logger.info(f"Market status: {market_info}")
        
        time.sleep(5)
    
    # Disconnect
    logger.info("Disconnecting...")
    client.disconnectWebSocket()

if __name__ == "__main__":
    test_websocket() 