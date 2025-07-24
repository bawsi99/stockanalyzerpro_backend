#!/usr/bin/env python3
"""
Test different WebSocket modes to find which provides continuous data
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

def test_modes():
    """Test different WebSocket modes"""
    
    # Get credentials from environment
    api_key = os.getenv('ZERODHA_API_KEY')
    access_token = os.getenv('ZERODHA_ACCESS_TOKEN')
    
    if not api_key or not access_token:
        logger.error("ZERODHA_API_KEY and ZERODHA_ACCESS_TOKEN must be set")
        return
    
    # Test different modes
    modes = ['ltp', 'quote', 'full']
    test_token = 738561  # RELIANCE
    
    for mode in modes:
        logger.info(f"\n{'='*50}")
        logger.info(f"Testing mode: {mode}")
        logger.info(f"{'='*50}")
        
        # Create new client for each mode
        client = ZerodhaWSClient(api_key, access_token)
        
        # Connect to WebSocket
        logger.info("Connecting to WebSocket...")
        client.connect()
        
        # Wait for connection
        time.sleep(3)
        
        # Subscribe to token
        logger.info(f"Subscribing to token {test_token}...")
        client.subscribe([test_token])
        
        # Set mode
        logger.info(f"Setting mode to {mode}...")
        client.set_mode(mode, [test_token])
        
        # Monitor for 30 seconds
        logger.info("Monitoring for 30 seconds...")
        start_time = time.time()
        tick_count = 0
        
        while time.time() - start_time < 30:
            tick = client.get_latest_tick(test_token)
            if tick:
                tick_count += 1
                logger.info(f"Tick {tick_count}: {tick}")
            
            time.sleep(5)
        
        logger.info(f"Mode {mode}: Received {tick_count} ticks in 30 seconds")
        
        # Disconnect
        client.disconnectWebSocket()
        time.sleep(2)

if __name__ == "__main__":
    test_modes() 