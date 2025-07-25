#!/usr/bin/env python3
"""
test_cors_validation.py

Test script to verify that CORS origin validation is working correctly.
This script tests that WebSocket connections are rejected from unauthorized origins.
"""

import asyncio
import json
import websockets
import time
from typing import Dict, List, Set

class CORSValidationTester:
    def __init__(self, ws_url: str = "ws://localhost:8000/ws/stream"):
        self.ws_url = ws_url
        self.results = {}
        
    async def test_authorized_origin(self, origin: str, expected_success: bool = True):
        """Test WebSocket connection with a specific origin."""
        print(f"🧪 Testing origin: {origin}")
        
        try:
            # Create connection with custom headers
            headers = {'Origin': origin} if origin else {}
            
            async with websockets.connect(self.ws_url, extra_headers=headers) as websocket:
                print(f"✅ Connection successful for origin: {origin}")
                
                # Try to send a simple message
                test_message = {
                    "action": "ping"
                }
                await websocket.send(json.dumps(test_message))
                
                # Wait for response
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                    data = json.loads(response)
                    print(f"📨 Received response: {data}")
                    return True
                except asyncio.TimeoutError:
                    print(f"⏰ Timeout waiting for response from {origin}")
                    return False
                    
        except Exception as e:
            if expected_success:
                print(f"❌ Unexpected error for authorized origin {origin}: {e}")
                return False
            else:
                print(f"✅ Correctly rejected unauthorized origin: {origin}")
                return True
    
    async def run_test(self):
        """Run the complete CORS validation test."""
        print("🚀 Starting CORS validation test...")
        print(f"🌐 WebSocket URL: {self.ws_url}")
        print("-" * 50)
        
        # Test scenarios based on the .env file
        test_scenarios = [
            {
                'origin': 'http://localhost:3000',
                'expected_success': True,
                'description': 'Authorized origin (frontend dev)'
            },
            {
                'origin': 'http://localhost:8080',
                'expected_success': True,
                'description': 'Authorized origin (frontend prod)'
            },
            {
                'origin': 'http://127.0.0.1:3000',
                'expected_success': True,
                'description': 'Authorized origin (localhost IP)'
            },
            {
                'origin': 'http://127.0.0.1:8080',
                'expected_success': True,
                'description': 'Authorized origin (localhost IP prod)'
            },
            {
                'origin': 'http://localhost:8081',
                'expected_success': True,
                'description': 'Authorized origin (additional port)'
            },
            {
                'origin': 'http://localhost:8082',
                'expected_success': True,
                'description': 'Authorized origin (additional port)'
            },
            {
                'origin': 'http://localhost:8083',
                'expected_success': False,
                'description': 'Unauthorized origin (not in CORS_ORIGINS)'
            },
            {
                'origin': 'http://malicious-site.com',
                'expected_success': False,
                'description': 'Unauthorized origin (malicious site)'
            },
            {
                'origin': 'https://localhost:3000',
                'expected_success': False,
                'description': 'Unauthorized origin (HTTPS instead of HTTP)'
            },
            {
                'origin': None,
                'expected_success': True,
                'description': 'No origin header (should be allowed)'
            }
        ]
        
        # Run all tests
        for i, scenario in enumerate(test_scenarios, 1):
            print(f"\n📋 Test {i}: {scenario['description']}")
            success = await self.test_authorized_origin(
                scenario['origin'], 
                scenario['expected_success']
            )
            
            self.results[f"test_{i}"] = {
                'origin': scenario['origin'],
                'expected_success': scenario['expected_success'],
                'actual_success': success,
                'passed': success == scenario['expected_success'],
                'description': scenario['description']
            }
        
        # Print summary
        print("\n" + "=" * 50)
        print("📋 CORS VALIDATION TEST SUMMARY")
        print("=" * 50)
        
        all_passed = True
        for test_id, result in self.results.items():
            if result['passed']:
                print(f"✅ {result['description']}")
                print(f"   Origin: {result['origin']}")
            else:
                print(f"❌ {result['description']}")
                print(f"   Origin: {result['origin']}")
                print(f"   Expected: {'Success' if result['expected_success'] else 'Rejection'}")
                print(f"   Actual: {'Success' if result['actual_success'] else 'Rejection'}")
                all_passed = False
        
        print("-" * 50)
        if all_passed:
            print("🎉 ALL CORS TESTS PASSED! Origin validation is working correctly.")
        else:
            print("💥 SOME CORS TESTS FAILED! Origin validation needs attention.")
        
        return all_passed

async def main():
    """Main test function."""
    tester = CORSValidationTester()
    success = await tester.run_test()
    
    if success:
        print("\n✅ CORS validation test completed successfully!")
        return 0
    else:
        print("\n❌ CORS validation test failed!")
        return 1

if __name__ == "__main__":
    import sys
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 