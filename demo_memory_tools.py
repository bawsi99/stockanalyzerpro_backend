#!/usr/bin/env python3
"""
Demo script showing how to use the memory analysis tools individually.
This script demonstrates various ways to monitor memory and test load.
"""

import asyncio
import time
import subprocess
import sys
import os
from memory_monitor import MemoryMonitor
from load_tester import LoadTester

def demo_memory_monitoring():
    """Demonstrate memory monitoring capabilities."""
    print("🔍 DEMO: Memory Monitoring")
    print("=" * 40)
    
    # Create monitor for system-wide monitoring
    monitor = MemoryMonitor()
    
    print("📊 Collecting 5 memory samples...")
    samples = []
    
    for i in range(5):
        sample = monitor.collect_memory_sample()
        samples.append(sample)
        
        # Print current memory status
        system_mem = sample.get('system_memory', {})
        print(f"   Sample {i+1}: System Memory: {system_mem.get('memory_percent', 0):.1f}%")
        
        time.sleep(1)
    
    # Print summary
    print(f"\n📈 Summary:")
    print(f"   Total samples: {len(samples)}")
    
    system_percentages = [s.get('system_memory', {}).get('memory_percent', 0) for s in samples]
    if system_percentages:
        print(f"   Min: {min(system_percentages):.1f}%")
        print(f"   Max: {max(system_percentages):.1f}%")
        print(f"   Avg: {sum(system_percentages) / len(system_percentages):.1f}%")
    
    print("\n✅ Memory monitoring demo completed!\n")

async def demo_load_testing():
    """Demonstrate load testing capabilities."""
    print("🚀 DEMO: Load Testing")
    print("=" * 40)
    
    # Check if service is running
    service_url = "http://localhost:8000"
    
    try:
        import requests
        response = requests.get(f"{service_url}/health", timeout=5)
        if response.status_code != 200:
            print(f"⚠️  Service at {service_url} is not responding properly")
            print("   Starting a quick demo with mock data...")
            return demo_mock_load_testing()
    except:
        print(f"⚠️  Service at {service_url} is not accessible")
        print("   Starting a quick demo with mock data...")
        return demo_mock_load_testing()
    
    # Create load tester
    tester = LoadTester(base_url=service_url)
    
    print("📊 Running light load test...")
    print("   Users: 3")
    print("   Requests per user: 5")
    print("   Duration: 10 seconds")
    
    try:
        stats = await tester.run_load_test(
            num_users=3,
            requests_per_user=5,
            duration_seconds=10
        )
        
        # Print results
        tester.print_summary(stats)
        
    except Exception as e:
        print(f"❌ Load test failed: {e}")
        print("   This might happen if the service is not fully ready")

def demo_mock_load_testing():
    """Demo load testing with mock data when service is not available."""
    print("🎭 DEMO: Mock Load Testing (Service Not Available)")
    print("=" * 50)
    
    print("📊 Simulating load test results...")
    
    # Mock statistics
    mock_stats = {
        'overall': {
            'total_requests': 15,
            'success_rate': 0.93,
            'response_time': {
                'min': 0.05,
                'max': 0.85,
                'avg': 0.23,
                'median': 0.21,
                'p95': 0.45,
                'p99': 0.78
            }
        },
        'by_complexity': {
            'light': {
                'count': 9,
                'success_rate': 1.0,
                'response_time': {'avg': 0.12}
            },
            'medium': {
                'count': 4,
                'success_rate': 0.75,
                'response_time': {'avg': 0.35}
            },
            'heavy': {
                'count': 2,
                'success_rate': 0.5,
                'response_time': {'avg': 0.65}
            }
        }
    }
    
    # Print mock results
    print(f"🎯 Overall Performance:")
    print(f"   Total Requests: {mock_stats['overall']['total_requests']}")
    print(f"   Success Rate: {mock_stats['overall']['success_rate']:.1%}")
    print(f"   Avg Response Time: {mock_stats['overall']['response_time']['avg']:.3f}s")
    
    print(f"\n📈 Performance by Complexity:")
    for complexity, comp_stats in mock_stats['by_complexity'].items():
        print(f"   {complexity.upper()}:")
        print(f"     Count: {comp_stats['count']}")
        print(f"     Success Rate: {comp_stats['success_rate']:.1%}")
        print(f"     Avg Response: {comp_stats['response_time']['avg']:.3f}s")
    
    print("\n✅ Mock load testing demo completed!\n")

def demo_process_monitoring():
    """Demonstrate process-specific memory monitoring."""
    print("🔍 DEMO: Process-Specific Memory Monitoring")
    print("=" * 50)
    
    # Find Python processes
    import psutil
    
    python_processes = []
    for proc in psutil.process_iter(['pid', 'name', 'memory_info']):
        try:
            if 'python' in proc.info['name'].lower():
                memory_mb = proc.info['memory_info'].rss / 1024 / 1024
                python_processes.append({
                    'pid': proc.info['pid'],
                    'name': proc.info['name'],
                    'memory_mb': memory_mb
                })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    
    if python_processes:
        # Sort by memory usage
        python_processes.sort(key=lambda x: x['memory_mb'], reverse=True)
        
        print(f"🐍 Found {len(python_processes)} Python processes:")
        for i, proc in enumerate(python_processes[:5]):  # Top 5
            print(f"   {i+1}. PID {proc['pid']}: {proc['name']} - {proc['memory_mb']:.1f} MB")
        
        # Monitor the largest Python process
        if python_processes:
            largest_proc = python_processes[0]
            print(f"\n📊 Monitoring largest process (PID: {largest_proc['pid']})...")
            
            monitor = MemoryMonitor(pid=largest_proc['pid'])
            
            print("   Collecting 3 samples...")
            for i in range(3):
                sample = monitor.collect_memory_sample()
                process_mem = sample.get('process_memory', {})
                
                if 'memory_rss_mb' in process_mem:
                    print(f"   Sample {i+1}: {process_mem['memory_rss_mb']:.1f} MB")
                else:
                    print(f"   Sample {i+1}: Unable to read memory")
                
                time.sleep(1)
    else:
        print("❌ No Python processes found")
    
    print("\n✅ Process monitoring demo completed!\n")

def demo_memory_breakdown():
    """Demonstrate detailed memory breakdown analysis."""
    print("📊 DEMO: Detailed Memory Breakdown")
    print("=" * 40)
    
    monitor = MemoryMonitor()
    
    print("🔍 Collecting detailed memory information...")
    sample = monitor.collect_memory_sample()
    
    # System memory
    system_mem = sample.get('system_memory', {})
    print(f"\n🖥️  System Memory:")
    print(f"   Total: {system_mem.get('total_memory_mb', 0):.1f} MB")
    print(f"   Available: {system_mem.get('available_memory_mb', 0):.1f} MB")
    print(f"   Used: {system_mem.get('used_memory_mb', 0):.1f} MB")
    print(f"   Usage: {system_mem.get('memory_percent', 0):.1f}%")
    
    # Python processes
    detailed = sample.get('detailed_breakdown', {})
    python_procs = detailed.get('python_processes', [])
    
    if python_procs:
        print(f"\n🐍 Python Processes ({len(python_procs)} total):")
        total_python_memory = detailed.get('total_python_memory_mb', 0)
        print(f"   Total Python Memory: {total_python_memory:.1f} MB")
        
        print(f"   Top 3 by memory usage:")
        for i, proc in enumerate(python_procs[:3]):
            print(f"     {i+1}. PID {proc['pid']}: {proc['name']} - {proc['memory_mb']:.1f} MB")
    
    # Python GC stats
    python_mem = sample.get('python_memory', {})
    if 'gc_count' in python_mem:
        print(f"\n🗑️  Python Garbage Collector:")
        gc_count = python_mem['gc_count']
        print(f"   Generation 0: {gc_count[0]}")
        print(f"   Generation 1: {gc_count[1]}")
        print(f"   Generation 2: {gc_count[2]}")
    
    print("\n✅ Memory breakdown demo completed!\n")

def show_usage_examples():
    """Show various usage examples for the tools."""
    print("📚 USAGE EXAMPLES")
    print("=" * 40)
    
    print("🔍 Memory Monitoring Examples:")
    print("   # Monitor system memory for 1 minute")
    print("   python memory_monitor.py --duration 60")
    print()
    print("   # Monitor specific process with 0.5s interval")
    print("   python memory_monitor.py --pid 12345 --interval 0.5 --duration 300")
    print()
    print("   # Save to custom file")
    print("   python memory_monitor.py --output my_memory_data.json --duration 120")
    print()
    
    print("🚀 Load Testing Examples:")
    print("   # Light load test")
    print("   python load_tester.py --users 5 --requests 10 --duration 60")
    print()
    print("   # Heavy load test")
    print("   python load_tester.py --users 50 --requests 30 --duration 300")
    print()
    print("   # Test custom service")
    print("   python load_tester.py --url http://my-service.com --users 25 --requests 20")
    print()
    
    print("📊 Comprehensive Analysis Examples:")
    print("   # Quick analysis (1 minute)")
    print("   python memory_analysis_runner.py --baseline 15 --load-test 30 --cooldown 15")
    print()
    print("   # Extended analysis (5 minutes)")
    print("   python memory_analysis_runner.py --baseline 60 --load-test 180 --cooldown 60")
    print()
    print("   # High-load analysis")
    print("   python memory_analysis_runner.py --users 50 --requests 25 --interval 0.25")
    print()
    
    print("🔄 Automated Analysis:")
    print("   # Run the automated script")
    print("   ./run_memory_analysis.sh")
    print()

def main():
    """Main demo function."""
    print("🎯 Stock Analyzer Pro - Memory Analysis Tools Demo")
    print("=" * 60)
    print()
    
    # Check if we're in the right directory
    if not os.path.exists("start_with_cors_fix.py"):
        print("❌ Error: Please run this script from the backend directory")
        print("   Expected: start_with_cors_fix.py")
        return
    
    # Check dependencies
    try:
        import psutil
        import aiohttp
        import requests
        print("✅ All required packages are available")
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("   Install with: pip install -r memory_analysis_requirements.txt")
        return
    
    print()
    
    # Run demos
    try:
        demo_memory_monitoring()
        demo_process_monitoring()
        demo_memory_breakdown()
        
        # Load testing demo (async)
        asyncio.run(demo_load_testing())
        
        show_usage_examples()
        
        print("🎉 All demos completed successfully!")
        print()
        print("💡 Next steps:")
        print("   1. Run the automated analysis: ./run_memory_analysis.sh")
        print("   2. Try individual tools with different parameters")
        print("   3. Check the generated JSON files for detailed data")
        print("   4. Use results to plan cloud deployment requirements")
        
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")

if __name__ == "__main__":
    main()
