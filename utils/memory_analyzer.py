#!/usr/bin/env python3
"""
Service-Specific Memory Analyzer for Stock Analyzer Pro

This script analyzes memory usage of individual service components:
1. Analysis Service (ML models, pattern recognition)
2. Data Service (WebSocket streaming, real-time data)
3. WebSocket Service (live connections, streaming)
4. Combined load testing
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
import psutil
import os

@dataclass
class ServiceMemoryResult:
    """Data class for service-specific memory analysis."""
    service_name: str
    endpoint: str
    method: str
    status_code: int
    response_time: float
    memory_before_mb: float
    memory_after_mb: float
    memory_delta_mb: float
    timestamp: str
    success: bool
    error_message: Optional[str] = None

class ServiceMemoryAnalyzer:
    def __init__(self, base_url: str = "http://localhost:8000", target_pid: Optional[int] = None):
        self.base_url = base_url.rstrip('/')
        self.target_pid = target_pid
        self.results: List[ServiceMemoryResult] = []
        self.lock = threading.Lock()
        
        # Service-specific endpoints for targeted testing
        self.service_endpoints = {
            'analysis_service': [
                {
                    'name': 'Comprehensive Analysis',
                    'path': '/analyze',
                    'method': 'POST',
                    'data': {'symbol': 'RELIANCE', 'interval': '1D', 'analysis_type': 'comprehensive'},
                    'expected_memory_mb': 60,
                    'description': 'Full stock analysis with ML models'
                },
                {
                    'name': 'Enhanced Analysis',
                    'path': '/analysis/enhanced',
                    'method': 'POST',
                    'data': {'symbol': 'TCS', 'interval': '1D', 'indicators': ['RSI', 'MACD', 'BB', 'ATR', 'STOCH']},
                    'expected_memory_mb': 50,
                    'description': 'Technical analysis with multiple indicators'
                },
                {
                    'name': 'Pattern Detection',
                    'path': '/patterns/detect',
                    'method': 'POST',
                    'data': {'symbol': 'INFY', 'interval': '1D', 'pattern_types': ['candlestick', 'chart', 'volume', 'support_resistance']},
                    'expected_memory_mb': 55,
                    'description': 'Complex pattern recognition algorithms'
                },
                {
                    'name': 'Risk Analysis',
                    'path': '/analysis/risk',
                    'method': 'POST',
                    'data': {'symbol': 'HDFCBANK', 'interval': '1D', 'risk_metrics': ['VaR', 'Sharpe', 'MaxDrawdown', 'Sortino', 'Calmar']},
                    'expected_memory_mb': 45,
                    'description': 'Risk calculation and portfolio analysis'
                },
                {
                    'name': 'Sector Benchmarking',
                    'path': '/sectors/benchmarking',
                    'method': 'POST',
                    'data': {'sectors': ['NIFTY50', 'BANKNIFTY', 'NIFTYIT', 'NIFTYPHARMA'], 'metrics': ['performance', 'correlation', 'volatility']},
                    'expected_memory_mb': 40,
                    'description': 'Multi-sector analysis and comparison'
                }
            ],
            'data_service': [
                {
                    'name': 'Stock History (1Y)',
                    'path': '/stock/RELIANCE/history',
                    'method': 'GET',
                    'params': {'interval': '1D', 'period': '1Y'},
                    'expected_memory_mb': 30,
                    'description': 'Large dataset retrieval and processing'
                },
                {
                    'name': 'Stock History (2Y)',
                    'path': '/stock/TCS/history',
                    'method': 'GET',
                    'params': {'interval': '1D', 'period': '2Y'},
                    'expected_memory_mb': 35,
                    'description': 'Extended historical data analysis'
                },
                {
                    'name': 'Multiple Stocks',
                    'path': '/stock/INFY/history',
                    'method': 'GET',
                    'params': {'interval': '1D', 'period': '1Y'},
                    'expected_memory_mb': 30,
                    'description': 'Concurrent stock data processing'
                },
                {
                    'name': 'Market Status',
                    'path': '/data/market-status',
                    'method': 'GET',
                    'expected_memory_mb': 20,
                    'description': 'Real-time market data aggregation'
                },
                {
                    'name': 'Token Mapping',
                    'path': '/data/token-mapping',
                    'method': 'GET',
                    'expected_memory_mb': 25,
                    'description': 'Large mapping table operations'
                }
            ],
            'websocket_service': [
                {
                    'name': 'WebSocket Health',
                    'path': '/ws/health',
                    'method': 'GET',
                    'expected_memory_mb': 20,
                    'description': 'Connection health monitoring'
                },
                {
                    'name': 'WebSocket Connections',
                    'path': '/ws/connections',
                    'method': 'GET',
                    'expected_memory_mb': 25,
                    'description': 'Active connection management'
                },
                {
                    'name': 'WebSocket Test',
                    'path': '/ws/test',
                    'method': 'GET',
                    'expected_memory_mb': 20,
                    'description': 'Connection testing and validation'
                }
            ]
        }
        
        # Stock symbols for dynamic testing
        self.stock_symbols = [
            'RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK', 'HINDUNILVR', 'ITC', 'SBIN',
            'BHARTIARTL', 'AXISBANK', 'ASIANPAINT', 'MARUTI', 'HCLTECH', 'SUNPHARMA', 'TATAMOTORS'
        ]
    
    def get_process_memory(self) -> float:
        """Get current memory usage of the target process."""
        if not self.target_pid:
            return 0.0
        
        try:
            process = psutil.Process(self.target_pid)
            memory_info = process.memory_info()
            return memory_info.rss / 1024 / 1024  # Convert to MB
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return 0.0
    
    def get_system_memory(self) -> Dict:
        """Get current system memory status."""
        try:
            memory = psutil.virtual_memory()
            return {
                'total_mb': memory.total / 1024 / 1024,
                'available_mb': memory.available / 1024 / 1024,
                'used_mb': memory.used / 1024 / 1024,
                'percent': memory.percent
            }
        except Exception:
            return {}
    
    async def test_service_endpoint(self, session: aiohttp.ClientSession, service_name: str, 
                                   endpoint: Dict) -> ServiceMemoryResult:
        """Test a specific service endpoint and measure memory impact."""
        url = f"{self.base_url}{endpoint['path']}"
        method = endpoint['method']
        
        # Get memory before request
        memory_before = self.get_process_memory()
        system_memory_before = self.get_system_memory()
        
        start_time = time.time()
        success = False
        status_code = 0
        error_message = None
        
        try:
            if method == 'GET':
                params = endpoint.get('params', {})
                # Make params dynamic if they contain placeholders
                if 'symbol' in params:
                    params = params.copy()
                    params['symbol'] = random.choice(self.stock_symbols)
                
                async with session.get(url, params=params) as response:
                    status_code = response.status
                    await response.text()  # Consume response
                    success = response.status < 400
                    
            elif method == 'POST':
                data = endpoint.get('data', {}).copy()
                # Make data dynamic
                if 'symbol' in data:
                    data['symbol'] = random.choice(self.stock_symbols)
                
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
        
        # Wait a bit for memory to stabilize
        await asyncio.sleep(0.5)
        
        # Get memory after request
        memory_after = self.get_process_memory()
        memory_delta = memory_after - memory_before
        
        result = ServiceMemoryResult(
            service_name=service_name,
            endpoint=endpoint['path'],
            method=method,
            status_code=status_code,
            response_time=response_time,
            memory_before_mb=memory_before,
            memory_after_mb=memory_after,
            memory_delta_mb=memory_delta,
            timestamp=datetime.now().isoformat(),
            success=success,
            error_message=error_message
        )
        
        with self.lock:
            self.results.append(result)
        
        return result
    
    async def test_service_component(self, session: aiohttp.ClientSession, service_name: str, 
                                   num_requests: int = 5) -> List[ServiceMemoryResult]:
        """Test a specific service component with multiple requests."""
        print(f"🔍 Testing {service_name}...")
        
        endpoints = self.service_endpoints.get(service_name, [])
        if not endpoints:
            print(f"   ❌ No endpoints found for {service_name}")
            return []
        
        results = []
        
        for i in range(num_requests):
            # Select random endpoint from this service
            endpoint = random.choice(endpoints)
            
            print(f"   Request {i+1}/{num_requests}: {endpoint['name']}")
            
            result = await self.test_service_endpoint(session, service_name, endpoint)
            results.append(result)
            
            # Print immediate results
            if result.success:
                print(f"     ✅ Success: {result.response_time:.3f}s, Memory: {result.memory_delta_mb:+.1f} MB")
            else:
                print(f"     ❌ Failed: {result.error_message}")
            
            # Small delay between requests
            await asyncio.sleep(1)
        
        return results
    
    async def run_comprehensive_service_test(self, requests_per_service: int = 5) -> Dict:
        """Run comprehensive testing of all service components."""
        print(f"🚀 Starting Comprehensive Service Memory Analysis...")
        print(f"   Base URL: {self.base_url}")
        print(f"   Target PID: {self.target_pid if self.target_pid else 'System-wide'}")
        print(f"   Requests per service: {requests_per_service}")
        print(f"   Total expected requests: {len(self.service_endpoints) * requests_per_service}")
        
        start_time = time.time()
        
        # Create HTTP session
        connector = aiohttp.TCPConnector(limit=100, limit_per_host=30)
        timeout = aiohttp.ClientTimeout(total=60, connect=15)
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            # Test each service component
            service_results = {}
            
            for service_name in self.service_endpoints.keys():
                print(f"\n📊 Testing {service_name.upper()}...")
                results = await self.test_service_component(session, service_name, requests_per_service)
                service_results[service_name] = results
            
            # Test combined load (all services simultaneously)
            print(f"\n🔥 Testing COMBINED LOAD (all services simultaneously)...")
            combined_results = await self.test_combined_load(session, requests_per_service)
            service_results['combined_load'] = combined_results
        
        end_time = time.time()
        actual_duration = end_time - start_time
        
        # Calculate comprehensive statistics
        stats = self.calculate_service_statistics(service_results)
        stats['test_info'] = {
            'requests_per_service': requests_per_service,
            'total_requests': len(self.results),
            'duration_seconds': actual_duration,
            'target_pid': self.target_pid
        }
        
        return stats
    
    async def test_combined_load(self, session: aiohttp.ClientSession, num_requests: int) -> List[ServiceMemoryResult]:
        """Test all services simultaneously to simulate real-world usage."""
        print(f"   Simulating {num_requests} concurrent requests across all services...")
        
        # Create tasks for all services
        tasks = []
        for service_name in self.service_endpoints.keys():
            endpoints = self.service_endpoints[service_name]
            for i in range(num_requests):
                endpoint = random.choice(endpoints)
                task = self.test_service_endpoint(session, service_name, endpoint)
                tasks.append(task)
        
        # Run all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        valid_results = [r for r in results if isinstance(r, ServiceMemoryResult)]
        
        return valid_results
    
    def calculate_service_statistics(self, service_results: Dict) -> Dict:
        """Calculate comprehensive statistics for service-specific testing."""
        if not self.results:
            return {}
        
        stats = {
            'services': {},
            'overall': {},
            'memory_requirements': {},
            'recommendations': {}
        }
        
        # Calculate statistics for each service
        for service_name, results in service_results.items():
            if not results:
                continue
            
            # Memory statistics
            memory_deltas = [r.memory_delta_mb for r in results]
            memory_before = [r.memory_before_mb for r in results]
            memory_after = [r.memory_after_mb for r in results]
            
            # Response time statistics
            response_times = [r.response_time for r in results]
            success_count = sum(1 for r in results if r.success)
            
            stats['services'][service_name] = {
                'request_count': len(results),
                'success_rate': success_count / len(results),
                'memory': {
                    'baseline_mb': min(memory_before) if memory_before else 0,
                    'peak_mb': max(memory_after) if memory_after else 0,
                    'avg_delta_mb': sum(memory_deltas) / len(memory_deltas) if memory_deltas else 0,
                    'max_delta_mb': max(memory_deltas) if memory_deltas else 0,
                    'total_growth_mb': sum(memory_deltas) if memory_deltas else 0
                },
                'performance': {
                    'avg_response_time': sum(response_times) / len(response_times) if response_times else 0,
                    'max_response_time': max(response_times) if response_times else 0,
                    'min_response_time': min(response_times) if response_times else 0
                }
            }
        
        # Overall statistics
        all_memory_deltas = [r.memory_delta_mb for r in self.results]
        all_response_times = [r.response_time for r in self.results]
        all_success_count = sum(1 for r in self.results if r.success)
        
        stats['overall'] = {
            'total_requests': len(self.results),
            'success_rate': all_success_count / len(self.results),
            'total_memory_growth_mb': sum(all_memory_deltas),
            'avg_memory_growth_mb': sum(all_memory_deltas) / len(all_memory_deltas) if all_memory_deltas else 0,
            'peak_memory_growth_mb': max(all_memory_deltas) if all_memory_deltas else 0,
            'avg_response_time': sum(all_response_times) / len(all_response_times) if all_response_times else 0
        }
        
        # Memory requirements analysis
        baseline_memory = 274  # From previous analysis
        peak_growth = stats['overall']['peak_memory_growth_mb']
        total_growth = stats['overall']['total_memory_growth_mb']
        
        stats['memory_requirements'] = {
            'baseline_memory_mb': baseline_memory,
            'peak_growth_mb': peak_growth,
            'total_growth_mb': total_growth,
            'estimated_peak_memory_mb': baseline_memory + peak_growth,
            'estimated_concurrent_memory_mb': baseline_memory + (total_growth * 0.4),  # Assume 40% concurrent
            'memory_efficiency': 'good' if peak_growth < 100 else 'moderate' if peak_growth < 200 else 'poor'
        }
        
        # Generate recommendations
        peak_memory = baseline_memory + peak_growth
        concurrent_memory = baseline_memory + (total_growth * 0.4)
        
        stats['recommendations'] = {
            'minimum_memory_mb': max(512, int(peak_memory * 1.5)),
            'recommended_memory_mb': max(1024, int(concurrent_memory * 2)),
            'production_memory_mb': max(2048, int(concurrent_memory * 3)),
            'memory_assessment': self._assess_memory_requirements(peak_memory, concurrent_memory),
            'optimization_suggestions': self._generate_optimization_suggestions(stats)
        }
        
        return stats
    
    def _assess_memory_requirements(self, peak_memory: float, concurrent_memory: float) -> str:
        """Assess memory requirements based on analysis results."""
        if peak_memory < 400:
            return "512 MB instance should be sufficient"
        elif peak_memory < 600:
            return "512 MB instance may work but is risky - recommend 1 GB"
        elif peak_memory < 800:
            return "512 MB instance insufficient - recommend 1-2 GB"
        else:
            return "512 MB instance insufficient - recommend 2+ GB"
    
    def _generate_optimization_suggestions(self, stats: Dict) -> List[str]:
        """Generate optimization suggestions based on analysis results."""
        suggestions = []
        
        # Check memory efficiency
        memory_efficiency = stats['memory_requirements'].get('memory_efficiency', 'unknown')
        if memory_efficiency == 'poor':
            suggestions.append("Implement memory cleanup routines for long-running operations")
            suggestions.append("Consider using streaming responses for large datasets")
            suggestions.append("Implement request queuing to limit concurrent heavy operations")
        
        # Check service performance
        for service_name, service_stats in stats['services'].items():
            success_rate = service_stats.get('success_rate', 0)
            if success_rate < 0.8:
                suggestions.append(f"Investigate failures in {service_name} - {success_rate:.1%} success rate")
            
            avg_response = service_stats['performance'].get('avg_response_time', 0)
            if avg_response > 5.0:
                suggestions.append(f"Optimize {service_name} performance - {avg_response:.1f}s average response")
        
        # General suggestions
        suggestions.extend([
            "Implement memory monitoring and alerting",
            "Consider horizontal scaling for high-traffic periods",
            "Use caching for frequently accessed data",
            "Implement request rate limiting for heavy endpoints"
        ])
        
        return suggestions
    
    def print_service_analysis(self, stats: Dict):
        """Print comprehensive service analysis results."""
        print("\n" + "="*80)
        print("🔍 COMPREHENSIVE SERVICE MEMORY ANALYSIS")
        print("="*80)
        
        # Test information
        test_info = stats.get('test_info', {})
        print(f"📊 Test Summary:")
        print(f"   Total Requests: {test_info.get('total_requests', 0)}")
        print(f"   Duration: {test_info.get('duration_seconds', 0):.1f} seconds")
        print(f"   Target PID: {test_info.get('target_pid', 'System-wide')}")
        
        # Service-by-service breakdown
        services = stats.get('services', {})
        print(f"\n📈 Service Performance Breakdown:")
        for service_name, service_stats in services.items():
            print(f"   {service_name.upper()}:")
            print(f"     Requests: {service_stats.get('request_count', 0)}")
            print(f"     Success Rate: {service_stats.get('success_rate', 0):.1%}")
            
            memory = service_stats.get('memory', {})
            print(f"     Memory Growth: {memory.get('avg_delta_mb', 0):+.1f} MB avg, {memory.get('max_delta_mb', 0):+.1f} MB max")
            
            performance = service_stats.get('performance', {})
            print(f"     Response Time: {performance.get('avg_response_time', 0):.3f}s avg")
        
        # Overall statistics
        overall = stats.get('overall', {})
        print(f"\n🎯 Overall Results:")
        print(f"   Success Rate: {overall.get('success_rate', 0):.1%}")
        print(f"   Total Memory Growth: {overall.get('total_memory_growth_mb', 0):.1f} MB")
        print(f"   Peak Memory Growth: {overall.get('peak_memory_growth_mb', 0):.1f} MB")
        print(f"   Average Response Time: {overall.get('avg_response_time', 0):.3f}s")
        
        # Memory requirements
        memory_req = stats.get('memory_requirements', {})
        print(f"\n💾 Memory Requirements Analysis:")
        print(f"   Baseline Memory: {memory_req.get('baseline_memory_mb', 0):.1f} MB")
        print(f"   Peak Growth: {memory_req.get('peak_growth_mb', 0):.1f} MB")
        print(f"   Estimated Peak: {memory_req.get('estimated_peak_memory_mb', 0):.1f} MB")
        print(f"   Concurrent Estimate: {memory_req.get('estimated_concurrent_memory_mb', 0):.1f} MB")
        print(f"   Memory Efficiency: {memory_req.get('memory_efficiency', 'unknown').upper()}")
        
        # Recommendations
        recommendations = stats.get('recommendations', {})
        print(f"\n☁️  Cloud Deployment Recommendations:")
        print(f"   Memory Assessment: {recommendations.get('memory_assessment', 'Unknown')}")
        print(f"   Minimum Memory: {recommendations.get('minimum_memory_mb', 0)} MB")
        print(f"   Recommended Memory: {recommendations.get('recommended_memory_mb', 0)} MB")
        print(f"   Production Memory: {recommendations.get('production_memory_mb', 0)} MB")
        
        # Optimization suggestions
        suggestions = recommendations.get('optimization_suggestions', [])
        if suggestions:
            print(f"\n🔧 Optimization Suggestions:")
            for suggestion in suggestions:
                print(f"   • {suggestion}")
        
        print("="*80)
    
    def save_service_results(self, filename: str = "service_specific_memory_analysis.json"):
        """Save service analysis results to a JSON file."""
        try:
            data = {
                'test_results': [vars(result) for result in self.results],
                'statistics': self.calculate_service_statistics({}),
                'timestamp': datetime.now().isoformat()
            }
            
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            print(f"💾 Service analysis results saved to {filename}")
            
        except Exception as e:
            print(f"❌ Error saving results: {e}")

async def main():
    """Main function for service-specific memory analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Service-Specific Memory Analyzer')
    parser.add_argument('--url', type=str, default='http://localhost:8000', 
                       help='Base URL of the service')
    parser.add_argument('--pid', type=int, help='Target process ID to monitor')
    parser.add_argument('--requests', type=int, default=5, 
                       help='Requests per service component')
    parser.add_argument('--output', type=str, default='service_specific_memory_analysis.json', 
                       help='Output file name')
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = ServiceMemoryAnalyzer(base_url=args.url, target_pid=args.pid)
    
    try:
        # Run comprehensive service test
        stats = await analyzer.run_comprehensive_service_test(
            requests_per_service=args.requests
        )
        
        # Print analysis
        analyzer.print_service_analysis(stats)
        
        # Save results
        analyzer.save_service_results(args.output)
        
    except KeyboardInterrupt:
        print("\n👋 Service analysis interrupted by user")
        stats = analyzer.calculate_service_statistics({})
        analyzer.print_service_analysis(stats)
        analyzer.save_service_results(args.output)
    except Exception as e:
        print(f"❌ Error during service analysis: {e}")

if __name__ == "__main__":
    asyncio.run(main())
