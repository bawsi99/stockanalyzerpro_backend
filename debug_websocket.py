#!/usr/bin/env python3
"""
Debug script to test WebSocket connection and tick processing
"""
import os
import time
import logging
import json

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

def debug_websocket():
    """Debug WebSocket connection and tick processing"""
    
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
    
    # Subscribe to multiple tokens for more data
    test_tokens = [738561, 11536, 1594]  # RELIANCE, TCS, INFY
    logger.info(f"Subscribing to tokens {test_tokens}...")
    client.subscribe(test_tokens)
    
    # Set mode to quote
    client.set_mode('quote', test_tokens)
    
    # Monitor for 60 seconds with detailed logging
    logger.info("Monitoring for 60 seconds with detailed logging...")
    start_time = time.time()
    tick_count = 0
    
    while time.time() - start_time < 60:
        # Get latest tick for each token
        for token in test_tokens:
            tick = client.get_latest_tick(token)
            if tick:
                tick_count += 1
                logger.info(f"Token {token}: {json.dumps(tick, indent=2)}")
        
        # Get market status
        market_info = client.get_market_status()
        logger.info(f"Market status: {json.dumps(market_info, indent=2)}")
        
        # Get optimization stats
        stats = client.get_optimization_stats()
        logger.info(f"Optimization stats: {json.dumps(stats, indent=2)}")
        
        logger.info(f"Total ticks received so far: {tick_count}")
        logger.info("-" * 50)
        
        time.sleep(10)  # Check every 10 seconds
    
    # Disconnect
    logger.info("Disconnecting...")
    client.disconnectWebSocket()
    
    logger.info(f"Final summary: Received {tick_count} ticks in 60 seconds")

if __name__ == "__main__":
    debug_websocket() 