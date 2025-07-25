#!/usr/bin/env python3
"""
verify_fix.py

Simple verification script to test the WebSocket filtering fix.
This script tests the subscription management logic without requiring a full WebSocket connection.
"""

import asyncio
import sys
import os

# Add the backend directory to Python path
sys.path.insert(0, os.path.dirname(__file__))

from data_service import LiveDataPubSub

# Mock the zerodha_ws_client to avoid actual WebSocket connections during testing
class MockZerodhaClient:
    def __init__(self):
        self.subscribed_tokens = set()
    
    def subscribe(self, tokens):
        print(f"🔗 Mock: Subscribing to tokens: {tokens}")
        self.subscribed_tokens.update(tokens)
    
    def unsubscribe(self, tokens):
        print(f"🔗 Mock: Unsubscribing from tokens: {tokens}")
        self.subscribed_tokens.difference_update(tokens)
    
    def set_mode(self, mode, tokens):
        print(f"🔗 Mock: Setting mode {mode} for tokens: {tokens}")

# Replace the global zerodha_ws_client with our mock
import data_service
data_service.zerodha_ws_client = MockZerodhaClient()

async def test_subscription_management():
    """Test the subscription management logic."""
    print("🧪 Testing WebSocket filtering fix...")
    print("-" * 50)
    
    # Create a new LiveDataPubSub instance
    pubsub = LiveDataPubSub()
    
    # Test 1: Subscribe a client to RELIANCE
    print("📡 Test 1: Subscribing client_1 to RELIANCE")
    queue1, filter1 = await pubsub.subscribe({'user_id': 'test_user_1'})
    await pubsub.update_filter(queue1, tokens=['738561'])  # RELIANCE token
    
    # Verify subscription
    assert '738561' in pubsub.token_subscribers
    assert queue1 in pubsub.token_subscribers['738561']
    assert '738561' in pubsub.global_subscribed_tokens
    print("✅ Test 1 PASSED: Client 1 subscribed to RELIANCE")
    
    # Test 2: Subscribe another client to NIFTY DIV OPPS 50
    print("📡 Test 2: Subscribing client_2 to NIFTY DIV OPPS 50")
    queue2, filter2 = await pubsub.subscribe({'user_id': 'test_user_2'})
    await pubsub.update_filter(queue2, tokens=['257033'])  # NIFTY DIV OPPS 50 token
    
    # Verify subscription
    assert '257033' in pubsub.token_subscribers
    assert queue2 in pubsub.token_subscribers['257033']
    assert '257033' in pubsub.global_subscribed_tokens
    assert '738561' in pubsub.global_subscribed_tokens  # Should still be subscribed
    print("✅ Test 2 PASSED: Client 2 subscribed to NIFTY DIV OPPS 50")
    
    # Test 3: Subscribe a third client to both tokens
    print("📡 Test 3: Subscribing client_3 to both tokens")
    queue3, filter3 = await pubsub.subscribe({'user_id': 'test_user_3'})
    await pubsub.update_filter(queue3, tokens=['738561', '257033'])
    
    # Verify subscription
    assert queue3 in pubsub.token_subscribers['738561']
    assert queue3 in pubsub.token_subscribers['257033']
    print("✅ Test 3 PASSED: Client 3 subscribed to both tokens")
    
    # Test 4: Check subscription counts
    print("📊 Test 4: Checking subscription counts")
    stats = await pubsub.get_connection_stats()
    assert stats['total_clients'] == 3
    assert stats['total_tokens'] == 2
    assert len(stats['token_subscribers']) == 2
    assert stats['token_subscribers']['738561'] == 2  # client_1 and client_3
    assert stats['token_subscribers']['257033'] == 2  # client_2 and client_3
    print("✅ Test 4 PASSED: Subscription counts are correct")
    
    # Test 5: Unsubscribe client_1 from RELIANCE
    print("📡 Test 5: Unsubscribing client_1 from RELIANCE")
    await pubsub.update_filter(queue1, tokens=[])
    
    # Verify unsubscription
    assert queue1 not in pubsub.token_subscribers['738561']
    assert '738561' in pubsub.global_subscribed_tokens  # Should still be subscribed (client_3 still needs it)
    
    # Get updated stats
    stats_after_unsub = await pubsub.get_connection_stats()
    assert stats_after_unsub['token_subscribers']['738561'] == 1  # Only client_3
    print("✅ Test 5 PASSED: Client 1 unsubscribed from RELIANCE")
    
    # Test 6: Unsubscribe client_3 from both tokens
    print("📡 Test 6: Unsubscribing client_3 from both tokens")
    await pubsub.update_filter(queue3, tokens=[])
    
    # Verify unsubscription
    assert queue3 not in pubsub.token_subscribers.get('738561', set())
    assert queue3 not in pubsub.token_subscribers.get('257033', set())
    assert '738561' not in pubsub.global_subscribed_tokens  # Should be unsubscribed (no clients need it)
    assert '257033' in pubsub.global_subscribed_tokens  # Should still be subscribed (client_2 still needs it)
    print("✅ Test 6 PASSED: Client 3 unsubscribed from both tokens")
    
    # Test 7: Clean up
    print("🧹 Test 7: Cleaning up connections")
    await pubsub.unsubscribe(queue1)
    await pubsub.unsubscribe(queue2)
    await pubsub.unsubscribe(queue3)
    
    # Verify cleanup
    final_stats = await pubsub.get_connection_stats()
    assert final_stats['total_clients'] == 0
    assert final_stats['total_tokens'] == 0
    print("✅ Test 7 PASSED: All connections cleaned up")
    
    print("\n" + "=" * 50)
    print("🎉 ALL TESTS PASSED! WebSocket filtering fix is working correctly.")
    print("=" * 50)
    
    return True

async def main():
    """Main test function."""
    try:
        success = await test_subscription_management()
        if success:
            print("\n✅ Verification completed successfully!")
            return 0
        else:
            print("\n❌ Verification failed!")
            return 1
    except Exception as e:
        print(f"\n💥 Verification error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 