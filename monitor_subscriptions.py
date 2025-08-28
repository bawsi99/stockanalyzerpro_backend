#!/usr/bin/env python3
"""
Monitor and Subscribe to Tokens

This script monitors the data_service.py's LiveDataPubSub for new token subscriptions
and ensures they are properly subscribed to in the Zerodha WebSocket client.
"""

import os
import sys
import time
import asyncio
import signal

# Try to load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Loaded environment variables from .env file")
except ImportError:
    print("⚠️ python-dotenv not available, using system environment variables")

# Import required modules
try:
    from zerodha_ws_client import zerodha_ws_client
    from data_service import live_pubsub
    print("✅ Successfully imported required modules")
except ImportError as e:
    print(f"❌ Failed to import required modules: {e}")
    print("Make sure zerodha_ws_client.py and data_service.py are in the same directory.")
    sys.exit(1)

# Signal handler for graceful shutdown
def signal_handler(sig, frame):
    print("\n👋 Shutting down subscription monitor...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

async def monitor_subscriptions():
    """Monitor LiveDataPubSub for new token subscriptions and subscribe to them."""
    print("🔍 Starting subscription monitor...")
    
    # Keep track of subscribed tokens
    subscribed_tokens = set()
    
    try:
        while True:
            # Check if there are new tokens to subscribe to
            if hasattr(live_pubsub, 'global_subscribed_tokens'):
                # Convert string tokens to integers
                current_tokens = {int(token) for token in live_pubsub.global_subscribed_tokens}
                
                # Find new tokens
                new_tokens = current_tokens - subscribed_tokens
                if new_tokens:
                    print(f"🔔 Found {len(new_tokens)} new tokens to subscribe to: {list(new_tokens)}")
                    
                    # Check if WebSocket client is running
                    if hasattr(zerodha_ws_client, 'running') and zerodha_ws_client.running:
                        # Subscribe to new tokens
                        zerodha_ws_client.subscribe(list(new_tokens))
                        zerodha_ws_client.set_mode('quote', list(new_tokens))
                        print(f"✅ Successfully subscribed to {len(new_tokens)} new tokens")
                        
                        # Update subscribed tokens
                        subscribed_tokens.update(new_tokens)
                    else:
                        print("⚠️ WebSocket client not running, will try again later")
                
                # Check for tokens to unsubscribe from
                tokens_to_remove = subscribed_tokens - current_tokens
                if tokens_to_remove:
                    print(f"🔕 Found {len(tokens_to_remove)} tokens to unsubscribe from: {list(tokens_to_remove)}")
                    
                    # Check if WebSocket client is running
                    if hasattr(zerodha_ws_client, 'running') and zerodha_ws_client.running:
                        # Unsubscribe from tokens
                        zerodha_ws_client.unsubscribe(list(tokens_to_remove))
                        print(f"✅ Successfully unsubscribed from {len(tokens_to_remove)} tokens")
                        
                        # Update subscribed tokens
                        subscribed_tokens.difference_update(tokens_to_remove)
                    else:
                        print("⚠️ WebSocket client not running, will try again later")
            
            # Wait before checking again
            await asyncio.sleep(2)
    except Exception as e:
        print(f"❌ Error in subscription monitor: {e}")
        return

if __name__ == "__main__":
    try:
        asyncio.run(monitor_subscriptions())
    except KeyboardInterrupt:
        print("👋 Shutting down...")
    except Exception as e:
        print(f"❌ Error: {e}")
