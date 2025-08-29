#!/usr/bin/env python3
"""
Load Testing Script for Stock Analyzer Pro Service

This script simulates multiple simultaneous users making requests to different endpoints
to stress test the service and measure performance under load.
"""

import asyncio
import aiohttp
import time
import json
import random
from datetime import datetime
from typing import Dict, List, Optional
import statistics
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import threading

@dataclass
class TestResult:
    """Data class for storing test results."""
    endpoint: str
    method: str
    status_code: int
    response_time: float
    timestamp: str
    user_id: str
    success: bool
    error_message: Optional[str] = None

class LoadTester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.results: List[TestResult] = []
        self.lock = threading.Lock()
        
        # Test endpoints with different complexity levels
        self.endpoints = {
            'light': [
                {'path': '/health', 'method': 'GET', 'weight': 30},
                {'path': '/', 'method': 'GET', 'weight': 25},
                {'path': '/data/market-status', 'method': 'GET', 'weight': 20},
                {'path': '/sector/list', 'method': 'GET', 'weight': 15},
                {'path': '/data/token-mapping', 'method': 'GET', 'weight': 10},
            ],
            'medium': [
                {'path': '/stock/RELIANCE/history', 'method': 'GET', 'weight': 25},
                {'path': '/stock/TCS/history', 'method': 'GET', 'weight': 25},
                {'path': '/stock/INFY/history', 'method': 'GET', 'weight': 20},
                {'path': '/stock/HDFCBANK/history', 'method': 'GET', 'weight': 15},
                {'path': '/stock/ICICIBANK/history', 'method': 'GET', 'weight': 15},
            ],
            'heavy': [
                {'path': '/analyze', 'method': 'POST', 'weight': 30, 'data': {'symbol': 'RELIANCE', 'interval': '1D'}},
                {'path': '/analysis/enhanced', 'method': 'POST', 'weight': 25, 'data': {'symbol': 'TCS', 'interval': '1D'}},
                {'path': '/technical/indicators', 'method': 'POST', 'weight': 20, 'data': {'symbol': 'INFY', 'interval': '1D'}},
                {'path': '/patterns/detect', 'method': 'POST', 'weight': 15, 'data': {'symbol': 'HDFCBANK', 'interval': '1D'}},
                {'path': '/sectors/benchmarking', 'method': 'POST', 'weight': 10, 'data': {'sectors': ['NIFTY50', 'BANKNIFTY']}},
            ]
        }
        
        # Stock symbols for dynamic testing
        self.stock_symbols = [
            'RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK', 'HINDUNILVR', 'ITC', 'SBIN',
            'BHARTIARTL', 'AXISBANK', 'ASIANPAINT', 'MARUTI', 'HCLTECH', 'SUNPHARMA', 'TATAMOTORS'
        ]
        
        # Sector names for sector endpoints
        self.sector_names = [
            'NIFTY50', 'BANKNIFTY', 'NIFTYIT', 'NIFTYPHARMA', 'NIFTYAUTO', 'NIFTYFMCG'
        ]
    
    def select_endpoint(self, complexity: str = 'light') -> Dict:
        """Select a random endpoint based on complexity and weight."""
        endpoints = self.endpoints.get(complexity, self.endpoints['light'])
        
        # Calculate total weight
        total_weight = sum(ep['weight'] for ep in endpoints)
        
        # Random selection based on weight
        rand_val = random.uniform(0, total_weight)
        current_weight = 0
        
        for endpoint in endpoints:
            current_weight += endpoint['weight']
            if rand_val <= current_weight:
                return endpoint
        
        return endpoints[0]  # Fallback
    
    def generate_dynamic_data(self, endpoint: Dict) -> Optional[Dict]:
        """Generate dynamic test data for endpoints."""
        if 'data' not in endpoint:
            return None
        
        data = endpoint['data'].copy()
        
        # Replace placeholders with actual values
        if 'symbol' in data:
            data['symbol'] = random.choice(self.stock_symbols)
        
        if 'sectors' in data:
            data['sectors'] = random.sample(self.sector_names, min(2, len(self.sector_names)))
        
        if 'interval' in data:
            data['interval'] = random.choice(['1D', '1W', '1M'])
        
        return data
    
    async def make_request(self, session: aiohttp.ClientSession, endpoint: Dict, user_id: str) -> TestResult:
        """Make a single HTTP request and measure response time."""
        url = f"{self.base_url}{endpoint['path']}"
        method = endpoint['method']
        
        # Generate dynamic data if needed
        data = self.generate_dynamic_data(endpoint)
        
        start_time = time.time()
        success = False
        status_code = 0
        error_message = None
        
        try:
            if method == 'GET':
                async with session.get(url) as response:
                    status_code = response.status
                    await response.text()  # Consume response
                    success = response.status < 400
            elif method == 'POST':
                headers = {'Content-Type': 'application/json'}
                async with session.post(url, json=data, headers=headers) as response:
                    status_code = response.status
                    await response.text()  # Consume response
                    success = response.status < 400
            else:
                error_message = f"Unsupported method: {method}"
                success = False
                
        except Exception as e:
            error_message = str(e)
            success = False
        
        response_time = time.time() - start_time
        
        result = TestResult(
            endpoint=endpoint['path'],
            method=method,
            status_code=status_code,
            response_time=response_time,
            timestamp=datetime.now().isoformat(),
            user_id=user_id,
            success=success,
            error_message=error_message
        )
        
        with self.lock:
            self.results.append(result)
        
        return result
    
    async def simulate_user(self, session: aiohttp.ClientSession, user_id: str, 
                           num_requests: int, complexity: str = 'light') -> List[TestResult]:
        """Simulate a single user making multiple requests."""
        user_results = []
        
        for i in range(num_requests):
            # Vary complexity based on request number
            if i < num_requests * 0.6:
                comp = 'light'
            elif i < num_requests * 0.9:
                comp = 'medium'
            else:
                comp = 'heavy'
            
            endpoint = self.select_endpoint(comp)
            result = await self.make_request(session, endpoint, user_id)
            user_results.append(result)
            
            # Small delay between requests for the same user
            await asyncio.sleep(random.uniform(0.1, 0.5))
        
        return user_results
    
    async def run_load_test(self, num_users: int, requests_per_user: int, 
                           duration_seconds: int = 60) -> Dict:
        """Run the main load test."""
        print(f"🚀 Starting load test...")
        print(f"   Users: {num_users}")
        print(f"   Requests per user: {requests_per_user}")
        print(f"   Total requests: {num_users * requests_per_user}")
        print(f"   Duration: {duration_seconds}s")
        print(f"   Base URL: {self.base_url}")
        
        start_time = time.time()
        
        # Create HTTP session with connection pooling
        connector = aiohttp.TCPConnector(limit=100, limit_per_host=30)
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            # Create tasks for all users
            tasks = []
            for user_id in range(num_users):
                user_id_str = f"user_{user_id:03d}"
                task = self.simulate_user(session, user_id_str, requests_per_user)
                tasks.append(task)
            
            # Run all users concurrently
            print(f"📊 Simulating {num_users} concurrent users...")
            await asyncio.gather(*tasks)
        
        end_time = time.time()
        actual_duration = end_time - start_time
        
        # Calculate statistics
        stats = self.calculate_statistics()
        stats['test_info'] = {
            'num_users': num_users,
            'requests_per_user': requests_per_user,
            'total_requests': num_users * requests_per_user,
            'planned_duration': duration_seconds,
            'actual_duration': actual_duration,
            'requests_per_second': len(self.results) / actual_duration
        }
        
        return stats
    
    def calculate_statistics(self) -> Dict:
        """Calculate comprehensive statistics from test results."""
        if not self.results:
            return {}
        
        # Separate results by endpoint complexity
        light_results = [r for r in self.results if any(ep['path'] == r.endpoint for ep in self.endpoints['light'])]
        medium_results = [r for r in self.results if any(ep['path'] == r.endpoint for ep in self.endpoints['medium'])]
        heavy_results = [r for r in self.results if any(ep['path'] == r.endpoint for ep in self.endpoints['heavy'])]
        
        def calculate_endpoint_stats(results: List[TestResult]) -> Dict:
            if not results:
                return {}
            
            response_times = [r.response_time for r in results]
            success_count = sum(1 for r in results if r.success)
            status_codes = [r.status_code for r in results]
            
            return {
                'count': len(results),
                'success_rate': success_count / len(results),
                'response_time': {
                    'min': min(response_times),
                    'max': max(response_times),
                    'avg': statistics.mean(response_times),
                    'median': statistics.median(response_times),
                    'p95': statistics.quantiles(response_times, n=20)[18] if len(response_times) >= 20 else max(response_times),
                    'p99': statistics.quantiles(response_times, n=100)[98] if len(response_times) >= 100 else max(response_times)
                },
                'status_codes': {
                    '200': status_codes.count(200),
                    '4xx': sum(1 for s in status_codes if 400 <= s < 500),
                    '5xx': sum(1 for s in status_codes if 500 <= s < 600),
                    'other': sum(1 for s in status_codes if s < 200 or (s >= 300 and s < 400) or s >= 600)
                }
            }
        
        # Overall statistics
        all_response_times = [r.response_time for r in self.results]
        all_success_count = sum(1 for r in self.results if r.success)
        
        stats = {
            'overall': {
                'total_requests': len(self.results),
                'success_rate': all_success_count / len(self.results),
                'response_time': {
                    'min': min(all_response_times),
                    'max': max(all_response_times),
                    'avg': statistics.mean(all_response_times),
                    'median': statistics.median(all_response_times),
                    'p95': statistics.quantiles(all_response_times, n=20)[18] if len(all_response_times) >= 20 else max(all_response_times),
                    'p99': statistics.quantiles(all_response_times, n=100)[98] if len(all_response_times) >= 100 else max(all_response_times)
                }
            },
            'by_complexity': {
                'light': calculate_endpoint_stats(light_results),
                'medium': calculate_endpoint_stats(medium_results),
                'heavy': calculate_endpoint_stats(heavy_results)
            },
            'endpoint_breakdown': {}
        }
        
        # Breakdown by individual endpoints
        for endpoint in set(r.endpoint for r in self.results):
            endpoint_results = [r for r in self.results if r.endpoint == endpoint]
            stats['endpoint_breakdown'][endpoint] = calculate_endpoint_stats(endpoint_results)
        
        return stats
    
    def print_summary(self, stats: Dict):
        """Print a comprehensive summary of the load test results."""
        if not stats:
            print("📊 No test results available")
            return
        
        print("\n" + "="*80)
        print("📊 LOAD TEST RESULTS SUMMARY")
        print("="*80)
        
        test_info = stats.get('test_info', {})
        print(f"👥 Test Configuration:")
        print(f"   Users: {test_info.get('num_users', 'N/A')}")
        print(f"   Requests per user: {test_info.get('requests_per_user', 'N/A')}")
        print(f"   Total requests: {test_info.get('total_requests', 'N/A')}")
        print(f"   Duration: {test_info.get('actual_duration', 'N/A'):.2f}s")
        print(f"   Requests/sec: {test_info.get('requests_per_second', 'N/A'):.2f}")
        
        overall = stats.get('overall', {})
        print(f"\n🎯 Overall Performance:")
        print(f"   Success Rate: {overall.get('success_rate', 0):.2%}")
        print(f"   Response Time:")
        print(f"     Min: {overall['response_time'].get('min', 0):.3f}s")
        print(f"     Max: {overall['response_time'].get('max', 0):.3f}s")
        print(f"     Avg: {overall['response_time'].get('avg', 0):.3f}s")
        print(f"     Median: {overall['response_time'].get('median', 0):.3f}s")
        print(f"     P95: {overall['response_time'].get('p95', 0):.3f}s")
        print(f"     P99: {overall['response_time'].get('p99', 0):.3f}s")
        
        complexity_stats = stats.get('by_complexity', {})
        print(f"\n📈 Performance by Complexity:")
        for complexity, comp_stats in complexity_stats.items():
            if comp_stats:
                print(f"   {complexity.upper()}:")
                print(f"     Count: {comp_stats.get('count', 0)}")
                print(f"     Success Rate: {comp_stats.get('success_rate', 0):.2%}")
                print(f"     Avg Response: {comp_stats['response_time'].get('avg', 0):.3f}s")
        
        print("="*80)
    
    def save_results(self, filename: str = "load_test_results.json"):
        """Save test results to a JSON file."""
        try:
            data = {
                'test_results': [vars(result) for result in self.results],
                'statistics': self.calculate_statistics(),
                'timestamp': datetime.now().isoformat()
            }
            
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            print(f"💾 Load test results saved to {filename}")
            
        except Exception as e:
            print(f"❌ Error saving results: {e}")

async def main():
    """Main function for standalone usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Load Tester for Stock Analyzer Pro')
    parser.add_argument('--url', type=str, default='http://localhost:8000', 
                       help='Base URL of the service')
    parser.add_argument('--users', type=int, default=10, 
                       help='Number of concurrent users')
    parser.add_argument('--requests', type=int, default=20, 
                       help='Requests per user')
    parser.add_argument('--duration', type=int, default=60, 
                       help='Test duration in seconds')
    parser.add_argument('--output', type=str, default='load_test_results.json', 
                       help='Output file name')
    
    args = parser.parse_args()
    
    # Create load tester
    tester = LoadTester(base_url=args.url)
    
    try:
        # Run load test
        stats = await tester.run_load_test(
            num_users=args.users,
            requests_per_user=args.requests,
            duration_seconds=args.duration
        )
        
        # Print summary
        tester.print_summary(stats)
        
        # Save results
        tester.save_results(args.output)
        
    except KeyboardInterrupt:
        print("\n👋 Load test interrupted by user")
        stats = tester.calculate_statistics()
        tester.print_summary(stats)
        tester.save_results(args.output)
    except Exception as e:
        print(f"❌ Error during load test: {e}")

if __name__ == "__main__":
    asyncio.run(main())
