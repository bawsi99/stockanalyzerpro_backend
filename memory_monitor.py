#!/usr/bin/env python3
"""
Memory Monitor Script for Stock Analyzer Pro Service

This script monitors memory usage of the service and provides detailed breakdowns
of memory consumption by different components and processes.
"""

import psutil
import time
import json
import os
import threading
from datetime import datetime
from typing import Dict, List, Optional
import gc

class MemoryMonitor:
    def __init__(self, pid: Optional[int] = None, output_file: str = "memory_usage.json"):
        self.pid = pid
        self.output_file = output_file
        self.monitoring = False
        self.memory_data = []
        self.lock = threading.Lock()
        
        # Memory thresholds for alerts
        self.memory_thresholds = {
            'warning': 80,  # 80% of available memory
            'critical': 90,  # 90% of available memory
        }
        
    def get_process_memory_info(self) -> Dict:
        """Get detailed memory information for a specific process."""
        if not self.pid:
            return {}
            
        try:
            process = psutil.Process(self.pid)
            
            # Get memory info
            memory_info = process.memory_info()
            memory_percent = process.memory_percent()
            
            # Get CPU info
            cpu_percent = process.cpu_percent()
            
            # Get thread count
            num_threads = process.num_threads()
            
            # Get open files count
            try:
                open_files = len(process.open_files())
            except (psutil.AccessDenied, psutil.ZombieProcess):
                open_files = 0
                
            # Get connections count
            try:
                connections = len(process.connections())
            except (psutil.AccessDenied, psutil.ZombieProcess):
                connections = 0
            
            return {
                'pid': self.pid,
                'memory_rss': memory_info.rss,  # Resident Set Size
                'memory_vms': memory_info.vms,  # Virtual Memory Size
                'memory_percent': memory_percent,
                'cpu_percent': cpu_percent,
                'num_threads': num_threads,
                'open_files': open_files,
                'connections': connections,
                'status': process.status(),
                'create_time': process.create_time(),
                'memory_rss_mb': round(memory_info.rss / 1024 / 1024, 2),
                'memory_vms_mb': round(memory_info.vms / 1024 / 1024, 2)
            }
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess) as e:
            return {'error': str(e)}
    
    def get_system_memory_info(self) -> Dict:
        """Get system-wide memory information."""
        try:
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            return {
                'total_memory_mb': round(memory.total / 1024 / 1024, 2),
                'available_memory_mb': round(memory.available / 1024 / 1024, 2),
                'used_memory_mb': round(memory.used / 1024 / 1024, 2),
                'memory_percent': memory.percent,
                'free_memory_mb': round(memory.free / 1024 / 1024, 2),
                'swap_total_mb': round(swap.total / 1024 / 1024, 2),
                'swap_used_mb': round(swap.used / 1024 / 1024, 2),
                'swap_percent': swap.percent
            }
        except Exception as e:
            return {'error': str(e)}
    
    def get_python_memory_info(self) -> Dict:
        """Get Python-specific memory information."""
        try:
            import gc
            
            # Get garbage collector stats
            gc_stats = gc.get_stats()
            
            # Get memory usage from gc module
            gc.collect()  # Force garbage collection
            
            return {
                'gc_stats': gc_stats,
                'gc_count': gc.get_count(),
                'gc_threshold': gc.get_threshold(),
                'gc_objects': len(gc.get_objects()) if hasattr(gc, 'get_objects') else 0
            }
        except Exception as e:
            return {'error': str(e)}
    
    def get_detailed_memory_breakdown(self) -> Dict:
        """Get detailed memory breakdown by different components."""
        try:
            # Get all Python processes
            python_processes = []
            total_python_memory = 0
            
            for proc in psutil.process_iter(['pid', 'name', 'memory_info', 'memory_percent']):
                try:
                    if 'python' in proc.info['name'].lower():
                        memory_info = proc.info['memory_info']
                        memory_mb = memory_info.rss / 1024 / 1024
                        python_processes.append({
                            'pid': proc.info['pid'],
                            'name': proc.info['name'],
                            'memory_mb': round(memory_mb, 2),
                            'memory_percent': proc.info['memory_percent']
                        })
                        total_python_memory += memory_mb
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            # Sort by memory usage
            python_processes.sort(key=lambda x: x['memory_mb'], reverse=True)
            
            return {
                'python_processes': python_processes,
                'total_python_memory_mb': round(total_python_memory, 2),
                'total_processes': len(python_processes)
            }
        except Exception as e:
            return {'error': str(e)}
    
    def collect_memory_sample(self) -> Dict:
        """Collect a complete memory sample."""
        timestamp = datetime.now().isoformat()
        
        sample = {
            'timestamp': timestamp,
            'system_memory': self.get_system_memory_info(),
            'python_memory': self.get_python_memory_info(),
            'detailed_breakdown': self.get_detailed_memory_breakdown()
        }
        
        # Add process-specific info if PID is provided
        if self.pid:
            sample['process_memory'] = self.get_process_memory_info()
        
        return sample
    
    def start_monitoring(self, interval: float = 1.0):
        """Start continuous memory monitoring."""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, args=(interval,))
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        print(f"🔍 Memory monitoring started with {interval}s interval")
    
    def stop_monitoring(self):
        """Stop memory monitoring."""
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join()
        print("🛑 Memory monitoring stopped")
    
    def _monitor_loop(self, interval: float):
        """Internal monitoring loop."""
        while self.monitoring:
            try:
                sample = self.collect_memory_sample()
                
                with self.lock:
                    self.memory_data.append(sample)
                
                # Check for memory thresholds
                self._check_memory_thresholds(sample)
                
                time.sleep(interval)
                
            except Exception as e:
                print(f"❌ Error in monitoring loop: {e}")
                time.sleep(interval)
    
    def _check_memory_thresholds(self, sample: Dict):
        """Check memory usage against thresholds and alert if necessary."""
        try:
            system_memory = sample.get('system_memory', {})
            memory_percent = system_memory.get('memory_percent', 0)
            
            if memory_percent >= self.memory_thresholds['critical']:
                print(f"🚨 CRITICAL: Memory usage at {memory_percent:.1f}%")
            elif memory_percent >= self.memory_thresholds['warning']:
                print(f"⚠️  WARNING: Memory usage at {memory_percent:.1f}%")
                
        except Exception as e:
            print(f"❌ Error checking memory thresholds: {e}")
    
    def get_memory_statistics(self) -> Dict:
        """Calculate memory usage statistics."""
        if not self.memory_data:
            return {}
        
        try:
            # Extract memory percentages
            memory_percentages = [
                sample.get('system_memory', {}).get('memory_percent', 0)
                for sample in self.memory_data
            ]
            
            # Extract process memory if available
            process_memory_mb = []
            if self.pid:
                for sample in self.memory_data:
                    process_mem = sample.get('process_memory', {})
                    if 'memory_rss_mb' in process_mem:
                        process_memory_mb.append(process_mem['memory_rss_mb'])
            
            stats = {
                'total_samples': len(self.memory_data),
                'monitoring_duration_seconds': len(self.memory_data),
                'system_memory': {
                    'min_percent': min(memory_percentages) if memory_percentages else 0,
                    'max_percent': max(memory_percentages) if memory_percentages else 0,
                    'avg_percent': sum(memory_percentages) / len(memory_percentages) if memory_percentages else 0
                }
            }
            
            if process_memory_mb:
                stats['process_memory'] = {
                    'min_mb': min(process_memory_mb),
                    'max_mb': max(process_memory_mb),
                    'avg_mb': sum(process_memory_mb) / len(process_memory_mb)
                }
            
            return stats
            
        except Exception as e:
            return {'error': str(e)}
    
    def save_data(self):
        """Save memory data to file."""
        try:
            with self.lock:
                data = {
                    'monitoring_info': {
                        'pid': self.pid,
                        'start_time': self.memory_data[0]['timestamp'] if self.memory_data else None,
                        'end_time': self.memory_data[-1]['timestamp'] if self.memory_data else None,
                        'total_samples': len(self.memory_data)
                    },
                    'statistics': self.get_memory_statistics(),
                    'raw_data': self.memory_data
                }
            
            with open(self.output_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            print(f"💾 Memory data saved to {self.output_file}")
            
        except Exception as e:
            print(f"❌ Error saving data: {e}")
    
    def print_summary(self):
        """Print a summary of memory usage."""
        stats = self.get_memory_statistics()
        
        if not stats:
            print("📊 No memory data available")
            return
        
        print("\n" + "="*60)
        print("📊 MEMORY USAGE SUMMARY")
        print("="*60)
        
        print(f"📈 Total Samples: {stats.get('total_samples', 0)}")
        print(f"⏱️  Monitoring Duration: {stats.get('monitoring_duration_seconds', 0)} seconds")
        
        system_memory = stats.get('system_memory', {})
        print(f"\n🖥️  SYSTEM MEMORY:")
        print(f"   Min: {system_memory.get('min_percent', 0):.1f}%")
        print(f"   Max: {system_memory.get('max_percent', 0):.1f}%")
        print(f"   Avg: {system_memory.get('avg_percent', 0):.1f}%")
        
        if 'process_memory' in stats:
            process_memory = stats['process_memory']
            print(f"\n🔍 PROCESS MEMORY (PID: {self.pid}):")
            print(f"   Min: {process_memory.get('min_mb', 0):.1f} MB")
            print(f"   Max: {process_memory.get('max_mb', 0):.1f} MB")
            print(f"   Avg: {process_memory.get('avg_mb', 0):.1f} MB")
        
        print("="*60)

def main():
    """Main function for standalone usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Memory Monitor for Stock Analyzer Pro')
    parser.add_argument('--pid', type=int, help='Process ID to monitor')
    parser.add_argument('--interval', type=float, default=1.0, help='Monitoring interval in seconds')
    parser.add_argument('--duration', type=int, default=60, help='Monitoring duration in seconds')
    parser.add_argument('--output', type=str, default='memory_usage.json', help='Output file name')
    
    args = parser.parse_args()
    
    # Create monitor
    monitor = MemoryMonitor(pid=args.pid, output_file=args.output)
    
    try:
        print(f"🚀 Starting memory monitoring...")
        print(f"   PID: {args.pid if args.pid else 'System-wide'}")
        print(f"   Interval: {args.interval}s")
        print(f"   Duration: {args.duration}s")
        print(f"   Output: {args.output}")
        
        # Start monitoring
        monitor.start_monitoring(interval=args.interval)
        
        # Monitor for specified duration
        time.sleep(args.duration)
        
        # Stop monitoring
        monitor.stop_monitoring()
        
        # Print summary
        monitor.print_summary()
        
        # Save data
        monitor.save_data()
        
    except KeyboardInterrupt:
        print("\n👋 Monitoring interrupted by user")
        monitor.stop_monitoring()
        monitor.print_summary()
        monitor.save_data()
    except Exception as e:
        print(f"❌ Error: {e}")
        monitor.stop_monitoring()

if __name__ == "__main__":
    main()
