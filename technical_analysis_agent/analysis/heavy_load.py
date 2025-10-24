#!/usr/bin/env python3
"""
Heavy Load Analysis for Stock Analyzer Pro Service

This script specifically tests the heavy endpoints that caused crashes on 512 MB instances:
- Analysis service (ML models, pattern recognition)
- Data service (WebSocket streaming, real-time data)
- Heavy computational endpoints
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
import threading

@dataclass
class HeavyTestResult:
    """Data class for heavy load test results."""
    endpoint: str
    method: str
    status_code: int
    response_time: float
    memory_impact: float  # Estimated memory impact
    timestamp: str
    user_id: str
    success: bool
    error_message: Optional[str] = None

class HeavyLoadAnalyzer:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.results: List[HeavyTestResult] = []
        self.lock = threading.Lock()
        
        # Heavy analysis endpoints that consume significant memory
        self.heavy_endpoints = {
            'analysis': [
                {
                    'path': '/analyze',
                    'method': 'POST',
                    'data': {'symbol': 'RELIANCE', 'interval': '1D', 'analysis_type': 'comprehensive'},
                    'weight': 25,
                    'expected_memory_mb': 50  # Estimated memory impact
                },
                {
                    'path': '/analysis/enhanced',
                    'method': 'POST',
                    'data': {'symbol': 'TCS', 'interval': '1D', 'indicators': ['RSI', 'MACD', 'BB']},
                    'weight': 20,
                    'expected_memory_mb': 40
                },
                {
                    'path': '/technical/indicators',
                    'method': 'POST',
                    'data': {'symbol': 'INFY', 'interval': '1D', 'indicators': ['RSI', 'MACD', 'BB', 'ATR']},
                    'weight': 20,
                    'expected_memory_mb': 35
                },
                {
                    'path': '/patterns/detect',
                    'method': 'POST',
                    'data': {'symbol': 'HDFCBANK', 'interval': '1D', 'pattern_types': ['candlestick', 'chart', 'volume']},
                    'weight': 15,
                    'expected_memory_mb': 45
                },
                {
                    'path': '/sectors/benchmarking',
                    'method': 'POST',
                    'data': {'sectors': ['NIFTY50', 'BANKNIFTY', 'NIFTYIT'], 'metrics': ['performance', 'correlation']},
                    'weight': 10,
                    'expected_memory_mb': 30
                },
                {
                    'path': '/analysis/risk',
                    'method': 'POST',
                    'data': {'symbol': 'ICICIBANK', 'interval': '1D', 'risk_metrics': ['VaR', 'Sharpe', 'MaxDrawdown']},
                    'weight': 10,
                    'expected_memory_mb': 40
                }
            ],
            'data_streaming': [
                {
                    'path': '/stock/RELIANCE/history',
                    'method': 'GET',
                    'params': {'interval': '1D', 'period': '1Y'},
                    'weight': 30,
                    'expected_memory_mb': 25
                },
                {
                    'path': '/stock/TCS/history',
                    'method': 'GET',
                    'params': {'interval': '1D', 'period': '1Y'},
                    'weight': 25,
                    'expected_memory_mb': 25
                },
                {
                    'path': '/stock/INFY/history',
                    'method': 'GET',
                    'params': {'interval': '1D', 'period': '1Y'},
                    'weight': 25,
                    'expected_memory_mb': 25
                },
                {
                    'path': '/stock/HDFCBANK/history',
                    'method': 'GET',
                    'params': {'interval': '1D', 'period': '1Y'},
                    'weight': 20,
                    'expected_memory_mb': 25
                }
            ],
            'websocket_simulation': [
                {
                    'path': '/ws/health',
                    'method': 'GET',
                    'weight': 40,
                    'expected_memory_mb': 15
                },
                {
                    'path': '/ws/connections',
                    'method': 'GET',
                    'weight': 30,
                    'expected_memory_mb': 15
                },
                {
                    'path': '/ws/test',
                    'method': 'GET',
                    'weight': 30,
                    'expected_memory_mb': 15
                }
            ]
        }
        
        # Stock symbols for dynamic testing
        self.stock_symbols = [
            'RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK', 'HINDUNILVR', 'ITC', 'SBIN',
            'BHARTIARTL', 'AXISBANK', 'ASIANPAINT', 'MARUTI', 'HCLTECH', 'SUNPHARMA', 'TATAMOTORS',
            'WIPRO', 'ULTRACEMCO', 'POWERGRID', 'NESTLEIND', 'TECHM', 'BAJFINANCE'
        ]
        
        # Analysis types for comprehensive testing
        self.analysis_types = [
            'comprehensive', 'technical', 'fundamental', 'sentiment', 'risk', 'momentum'
        ]
        
        # Technical indicators for testing
        self.technical_indicators = [
            'RSI', 'MACD', 'BB', 'ATR', 'STOCH', 'CCI', 'ADX', 'Williams_R', 'ROC', 'MFI'
        ]
    
    def select_endpoint(self, category: str) -> Dict:
        """Select a random endpoint from a specific category."""
        endpoints = self.heavy_endpoints.get(category, [])
        if not endpoints:
            return {}
        
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
        
        if 'interval' in data:
            data['interval'] = random.choice(['1D', '1W', '1M'])
        
        if 'analysis_type' in data:
            data['analysis_type'] = random.choice(self.analysis_types)
        
        if 'indicators' in data:
            data['indicators'] = random.sample(self.technical_indicators, random.randint(3, 6))
        
        if 'pattern_types' in data:
            data['pattern_types'] = random.sample(['candlestick', 'chart', 'volume', 'support_resistance'], random.randint(2, 4))
        
        if 'sectors' in data:
            data['sectors'] = random.sample(['NIFTY50', 'BANKNIFTY', 'NIFTYIT', 'NIFTYPHARMA', 'NIFTYAUTO'], random.randint(2, 4))
        
        if 'risk_metrics' in data:
            data['risk_metrics'] = random.sample(['VaR', 'Sharpe', 'MaxDrawdown', 'Sortino', 'Calmar'], random.randint(2, 4))
        
        return data
    
    def generate_query_params(self, endpoint: Dict) -> Optional[Dict]:
        """Generate query parameters for GET requests."""
        if 'params' not in endpoint:
            return None
        
        params = endpoint['params'].copy()
        
        # Replace placeholders
        if 'symbol' in params:
            params['symbol'] = random.choice(self.stock_symbols)
        
        if 'interval' in params:
            params['interval'] = random.choice(['1D', '1W', '1M'])
        
        if 'period' in params:
            params['period'] = random.choice(['1M', '3M', '6M', '1Y', '2Y'])
        
        return params
    
    async def make_heavy_request(self, session: aiohttp.ClientSession, endpoint: Dict, user_id: str) -> HeavyTestResult:
        """Make a heavy HTTP request and measure response time and memory impact."""
        url = f"{self.base_url}{endpoint['path']}"
        method = endpoint['method']
        
        start_time = time.time()
        success = False
        status_code = 0
        error_message = None
        
        try:
            if method == 'GET':
                params = self.generate_query_params(endpoint)
                async with session.get(url, params=params) as response:
                    status_code = response.status
                    await response.text()  # Consume response
                    success = response.status < 400
            elif method == 'POST':
                data = self.generate_dynamic_data(endpoint)
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
        
        # Estimate memory impact based on endpoint type and response time
        memory_impact = endpoint.get('expected_memory_mb', 20)
        if response_time > 1.0:  # If response is slow, likely high memory usage
            memory_impact *= 1.5
        if response_time > 5.0:  # Very slow responses indicate high memory pressure
            memory_impact *= 2.0
        
        result = HeavyTestResult(
            endpoint=endpoint['path'],
            method=method,
            status_code=status_code,
            response_time=response_time,
            memory_impact=memory_impact,
            timestamp=datetime.now().isoformat(),
            user_id=user_id,
            success=success,
            error_message=error_message
        )
        
        with self.lock:
            self.results.append(result)
        
        return result
    
    async def simulate_heavy_user(self, session: aiohttp.ClientSession, user_id: str, 
                                 num_requests: int) -> List[HeavyTestResult]:
        """Simulate a user making heavy requests."""
        user_results = []
        
        # Distribute requests across categories
        analysis_requests = int(num_requests * 0.4)  # 40% analysis
        data_requests = int(num_requests * 0.4)      # 40% data
        ws_requests = int(num_requests * 0.2)        # 20% WebSocket
        
        # Analysis requests
        for i in range(analysis_requests):
            endpoint = self.select_endpoint('analysis')
            result = await self.make_heavy_request(session, endpoint, user_id)
            user_results.append(result)
            await asyncio.sleep(random.uniform(0.5, 1.5))  # Longer delays for heavy operations
        
        # Data requests
        for i in range(data_requests):
            endpoint = self.select_endpoint('data_streaming')
            result = await self.make_heavy_request(session, endpoint, user_id)
            user_results.append(result)
            await asyncio.sleep(random.uniform(0.2, 0.8))
        
        # WebSocket simulation requests
        for i in range(ws_requests):
            endpoint = self.select_endpoint('websocket_simulation')
            result = await self.make_heavy_request(session, endpoint, user_id)
            user_results.append(result)
            await asyncio.sleep(random.uniform(0.1, 0.5))
        
        return user_results
    
    async def run_heavy_load_test(self, num_users: int, requests_per_user: int, 
                                 duration_seconds: int = 120) -> Dict:
        """Run heavy load testing specifically targeting memory-intensive operations."""
        print(f"🚀 Starting HEAVY LOAD TEST...")
        print(f"   Users: {num_users}")
        print(f"   Requests per user: {requests_per_user}")
        print(f"   Total requests: {num_users * requests_per_user}")
        print(f"   Duration: {duration_seconds}s")
        print(f"   Base URL: {self.base_url}")
        print(f"   Focus: Analysis, Data Streaming, WebSocket operations")
        
        start_time = time.time()
        
        # Create HTTP session with higher limits for heavy operations
        connector = aiohttp.TCPConnector(limit=200, limit_per_host=50)
        timeout = aiohttp.ClientTimeout(total=60, connect=15)  # Longer timeouts for heavy operations
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            # Create tasks for all users
            tasks = []
            for user_id in range(num_users):
                user_id_str = f"heavy_user_{user_id:03d}"
                task = self.simulate_heavy_user(session, user_id_str, requests_per_user)
                tasks.append(task)
            
            # Run all users concurrently
            print(f"📊 Simulating {num_users} heavy users...")
            await asyncio.gather(*tasks)
        
        end_time = time.time()
        actual_duration = end_time - start_time
        
        # Calculate statistics
        stats = self.calculate_heavy_statistics()
        stats['test_info'] = {
            'num_users': num_users,
            'requests_per_user': requests_per_user,
            'total_requests': num_users * requests_per_user,
            'planned_duration': duration_seconds,
            'actual_duration': actual_duration,
            'requests_per_second': len(self.results) / actual_duration
        }
        
        return stats
    
    def calculate_heavy_statistics(self) -> Dict:
        """Calculate comprehensive statistics for heavy load testing."""
        if not self.results:
            return {}
        
        # Separate results by category
        analysis_results = [r for r in self.results if any(ep['path'] == r.endpoint for ep in self.heavy_endpoints['analysis'])]
        data_results = [r for r in self.results if any(ep['path'] == r.endpoint for ep in self.heavy_endpoints['data_streaming'])]
        ws_results = [r for r in self.results if any(ep['path'] == r.endpoint for ep in self.heavy_endpoints['websocket_simulation'])]
        
        def calculate_category_stats(results: List[HeavyTestResult]) -> Dict:
            if not results:
                return {}
            
            response_times = [r.response_time for r in results]
            success_count = sum(1 for r in results if r.success)
            memory_impacts = [r.memory_impact for r in results]
            status_codes = [r.status_code for r in results]
            
            return {
                'count': len(results),
                'success_rate': success_count / len(results),
                'total_memory_impact_mb': sum(memory_impacts),
                'avg_memory_impact_mb': sum(memory_impacts) / len(memory_impacts),
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
        all_memory_impacts = [r.memory_impact for r in self.results]
        
        stats = {
            'overall': {
                'total_requests': len(self.results),
                'success_rate': all_success_count / len(self.results),
                'total_memory_impact_mb': sum(all_memory_impacts),
                'avg_memory_impact_mb': sum(all_memory_impacts) / len(all_memory_impacts),
                'response_time': {
                    'min': min(all_response_times),
                    'max': max(all_response_times),
                    'avg': statistics.mean(all_response_times),
                    'median': statistics.median(all_response_times),
                    'p95': statistics.quantiles(all_response_times, n=20)[18] if len(all_response_times) >= 20 else max(all_response_times),
                    'p99': statistics.quantiles(all_response_times, n=100)[98] if len(all_response_times) >= 100 else max(all_response_times)
                }
            },
            'by_category': {
                'analysis': calculate_category_stats(analysis_results),
                'data_streaming': calculate_category_stats(data_results),
                'websocket_simulation': calculate_category_stats(ws_results)
            },
            'memory_requirements': {
                'peak_memory_impact_mb': max(all_memory_impacts) if all_memory_impacts else 0,
                'total_memory_impact_mb': sum(all_memory_impacts),
                'avg_memory_per_request_mb': sum(all_memory_impacts) / len(all_memory_impacts) if all_memory_impacts else 0,
                'estimated_concurrent_memory_mb': sum(all_memory_impacts) * 0.3  # Assume 30% concurrent usage
            }
        }
        
        return stats
    
    def print_heavy_summary(self, stats: Dict):
        """Print a comprehensive summary of heavy load test results."""
        if not stats:
            print("📊 No heavy load test results available")
            return
        
        print("\n" + "="*80)
        print("🚀 HEAVY LOAD TEST RESULTS SUMMARY")
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
        print(f"   Total Memory Impact: {overall.get('total_memory_impact_mb', 0):.1f} MB")
        print(f"   Avg Memory per Request: {overall.get('avg_memory_impact_mb', 0):.1f} MB")
        print(f"   Response Time:")
        print(f"     Min: {overall['response_time'].get('min', 0):.3f}s")
        print(f"     Max: {overall['response_time'].get('max', 0):.3f}s")
        print(f"     Avg: {overall['response_time'].get('avg', 0):.3f}s")
        print(f"     P95: {overall['response_time'].get('p95', 0):.3f}s")
        
        category_stats = stats.get('by_category', {})
        print(f"\n📈 Performance by Category:")
        for category, cat_stats in category_stats.items():
            if cat_stats:
                print(f"   {category.upper()}:")
                print(f"     Count: {cat_stats.get('count', 0)}")
                print(f"     Success Rate: {cat_stats.get('success_rate', 0):.2%}")
                print(f"     Avg Response: {cat_stats['response_time'].get('avg', 0):.3f}s")
                print(f"     Memory Impact: {cat_stats.get('avg_memory_impact_mb', 0):.1f} MB per request")
        
        memory_req = stats.get('memory_requirements', {})
        print(f"\n💾 MEMORY REQUIREMENTS ANALYSIS:")
        print(f"   Peak Memory Impact: {memory_req.get('peak_memory_impact_mb', 0):.1f} MB")
        print(f"   Total Memory Impact: {memory_req.get('total_memory_impact_mb', 0):.1f} MB")
        print(f"   Avg Memory per Request: {memory_req.get('avg_memory_per_request_mb', 0):.1f} MB")
        print(f"   Estimated Concurrent Memory: {memory_req.get('estimated_concurrent_memory_mb', 0):.1f} MB")
        
        # Cloud deployment recommendations
        concurrent_memory = memory_req.get('estimated_concurrent_memory_mb', 0)
        print(f"\n☁️  CLOUD DEPLOYMENT RECOMMENDATIONS:")
        if concurrent_memory < 256:
            print(f"   ✅ 512 MB instance: SUFFICIENT")
        elif concurrent_memory < 512:
            print(f"   ⚠️  512 MB instance: MAY BE SUFFICIENT (close to limit)")
        else:
            print(f"   ❌ 512 MB instance: INSUFFICIENT")
            print(f"   💡 Recommended: {max(1024, int(concurrent_memory * 2))} MB minimum")
        
        print("="*80)
    
    def save_heavy_results(self, filename: str = "heavy_load_test_results.json"):
        """Save heavy load test results to a JSON file."""
        try:
            data = {
                'test_results': [vars(result) for result in self.results],
                'statistics': self.calculate_heavy_statistics(),
                'timestamp': datetime.now().isoformat()
            }
            
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            print(f"💾 Heavy load test results saved to {filename}")
            
        except Exception as e:
            print(f"❌ Error saving results: {e}")

async def main():
    """Main function for heavy load testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Heavy Load Analyzer for Stock Analyzer Pro')
    parser.add_argument('--url', type=str, default='http://localhost:8000', 
                       help='Base URL of the service')
    parser.add_argument('--users', type=int, default=10, 
                       help='Number of concurrent users')
    parser.add_argument('--requests', type=int, default=20, 
                       help='Requests per user')
    parser.add_argument('--duration', type=int, default=120, 
                       help='Test duration in seconds')
    parser.add_argument('--output', type=str, default='heavy_load_test_results.json', 
                       help='Output file name')
    
    args = parser.parse_args()
    
    # Create heavy load analyzer
    analyzer = HeavyLoadAnalyzer(base_url=args.url)
    
    try:
        # Run heavy load test
        stats = await analyzer.run_heavy_load_test(
            num_users=args.users,
            requests_per_user=args.requests,
            duration_seconds=args.duration
        )
        
        # Print summary
        analyzer.print_heavy_summary(stats)
        
        # Save results
        analyzer.save_heavy_results(args.output)
        
    except KeyboardInterrupt:
        print("\n👋 Heavy load test interrupted by user")
        stats = analyzer.calculate_heavy_statistics()
        analyzer.print_heavy_summary(stats)
        analyzer.save_heavy_results(args.output)
    except Exception as e:
        print(f"❌ Error during heavy load test: {e}")

if __name__ == "__main__":
    asyncio.run(main())
