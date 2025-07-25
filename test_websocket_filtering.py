#!/usr/bin/env python3
"""
test_websocket_filtering.py

Test script to verify that WebSocket filtering is working correctly.
This script will test that clients only receive data for tokens they're subscribed to.
"""

import asyncio
import json
import websockets
import time
from typing import Dict, List, Set

class WebSocketFilteringTester:
    def __init__(self, ws_url: str = "ws://localhost:8000/ws/stream"):
        self.ws_url = ws_url
        self.results = {}
        
    async def test_client(self, client_id: str, symbols: List[str], expected_tokens: Set[str]):
        """Test a single WebSocket client with specific symbol subscriptions."""
        print(f"🧪 Testing client {client_id} with symbols: {symbols}")
        
        try:
            # Connect to WebSocket
            async with websockets.connect(self.ws_url) as websocket:
                print(f"✅ Client {client_id} connected")
                
                # Subscribe to symbols
                subscription_message = {
                    "action": "subscribe",
                    "symbols": symbols,
                    "timeframes": ["1d"]
                }
                await websocket.send(json.dumps(subscription_message))
                print(f"📡 Client {client_id} subscribed to: {symbols}")
                
                # Listen for data for 10 seconds
                received_tokens = set()
                start_time = time.time()
                
                while time.time() - start_time < 10:
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                        data = json.loads(message)
                        
                        if data.get('type') == 'tick':
                            token = data.get('token')
                            if token:
                                received_tokens.add(token)
                                print(f"📨 Client {client_id} received tick for token: {token}")
                        elif data.get('type') == 'candle':
                            token = data.get('token')
                            if token:
                                received_tokens.add(token)
                                print(f"📨 Client {client_id} received candle for token: {token}")
                                
                    except asyncio.TimeoutError:
                        continue
                    except Exception as e:
                        print(f"❌ Error receiving message for client {client_id}: {e}")
                        break
                
                # Store results
                self.results[client_id] = {
                    'expected_tokens': expected_tokens,
                    'received_tokens': received_tokens,
                    'correct': received_tokens.issubset(expected_tokens),
                    'unexpected_tokens': received_tokens - expected_tokens,
                    'missing_tokens': expected_tokens - received_tokens
                }
                
                print(f"📊 Client {client_id} results:")
                print(f"   Expected tokens: {expected_tokens}")
                print(f"   Received tokens: {received_tokens}")
                print(f"   Correct: {self.results[client_id]['correct']}")
                if self.results[client_id]['unexpected_tokens']:
                    print(f"   ❌ Unexpected tokens: {self.results[client_id]['unexpected_tokens']}")
                if self.results[client_id]['missing_tokens']:
                    print(f"   ⚠️  Missing tokens: {self.results[client_id]['missing_tokens']}")
                
        except Exception as e:
            print(f"❌ Error testing client {client_id}: {e}")
            self.results[client_id] = {
                'error': str(e),
                'correct': False
            }
    
    async def run_test(self):
        """Run the complete filtering test."""
        print("🚀 Starting WebSocket filtering test...")
        print(f"🌐 WebSocket URL: {self.ws_url}")
        print("-" * 50)
        
        # Test scenarios
        test_scenarios = [
            {
                'client_id': 'client_1',
                'symbols': ['RELIANCE'],
                'expected_tokens': {'738561'}  # RELIANCE token
            },
            {
                'client_id': 'client_2', 
                'symbols': ['NIFTY DIV OPPS 50'],
                'expected_tokens': {'257033'}  # NIFTY DIV OPPS 50 token
            },
            {
                'client_id': 'client_3',
                'symbols': ['RELIANCE', 'NIFTY DIV OPPS 50'],
                'expected_tokens': {'738561', '257033'}  # Both tokens
            }
        ]
        
        # Run all tests concurrently
        tasks = []
        for scenario in test_scenarios:
            task = self.test_client(
                scenario['client_id'],
                scenario['symbols'],
                scenario['expected_tokens']
            )
            tasks.append(task)
        
        await asyncio.gather(*tasks)
        
        # Print summary
        print("\n" + "=" * 50)
        print("📋 TEST SUMMARY")
        print("=" * 50)
        
        all_correct = True
        for client_id, result in self.results.items():
            if 'error' in result:
                print(f"❌ {client_id}: ERROR - {result['error']}")
                all_correct = False
            elif result['correct']:
                print(f"✅ {client_id}: PASSED")
            else:
                print(f"❌ {client_id}: FAILED")
                if result['unexpected_tokens']:
                    print(f"   Received unexpected tokens: {result['unexpected_tokens']}")
                if result['missing_tokens']:
                    print(f"   Missing expected tokens: {result['missing_tokens']}")
                all_correct = False
        
        print("-" * 50)
        if all_correct:
            print("🎉 ALL TESTS PASSED! WebSocket filtering is working correctly.")
        else:
            print("💥 SOME TESTS FAILED! WebSocket filtering needs attention.")
        
        return all_correct

async def main():
    """Main test function."""
    tester = WebSocketFilteringTester()
    success = await tester.run_test()
    
    if success:
        print("\n✅ WebSocket filtering test completed successfully!")
        return 0
    else:
        print("\n❌ WebSocket filtering test failed!")
        return 1

if __name__ == "__main__":
    import sys
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 