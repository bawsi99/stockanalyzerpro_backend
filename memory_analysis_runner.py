#!/usr/bin/env python3
"""
Memory Analysis Runner for Stock Analyzer Pro Service

This script runs both memory monitoring and load testing simultaneously to provide
a comprehensive analysis of memory usage under different load conditions.
"""

import asyncio
import subprocess
import time
import json
import os
import signal
import sys
from datetime import datetime
from typing import Dict, Optional
import threading
import psutil

from memory_monitor import MemoryMonitor
from load_tester import LoadTester

class MemoryAnalysisRunner:
    def __init__(self, service_url: str = "http://localhost:8000"):
        self.service_url = service_url
        self.monitor = None
        self.load_tester = None
        self.service_process = None
        self.monitoring_active = False
        
        # Analysis configuration
        self.analysis_config = {
            'baseline_duration': 30,      # Seconds to monitor baseline
            'load_test_duration': 120,    # Seconds for load testing
            'cooldown_duration': 30,      # Seconds to monitor after load test
            'monitoring_interval': 0.5,   # Memory sampling interval
            'load_test_users': 20,        # Number of concurrent users
            'load_test_requests': 15      # Requests per user
        }
        
        # Results storage
        self.analysis_results = {
            'baseline': [],
            'load_test': [],
            'cooldown': [],
            'summary': {}
        }
    
    def start_service(self) -> Optional[int]:
        """Start the Stock Analyzer Pro service and return its PID."""
        try:
            print("🚀 Starting Stock Analyzer Pro service...")
            
            # Start the service in background
            self.service_process = subprocess.Popen(
                [sys.executable, "start_with_cors_fix.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=os.getcwd()
            )
            
            # Wait a bit for service to start
            time.sleep(5)
            
            # Get the PID
            pid = self.service_process.pid
            print(f"✅ Service started with PID: {pid}")
            
            # Wait for service to be ready
            self._wait_for_service_ready()
            
            return pid
            
        except Exception as e:
            print(f"❌ Error starting service: {e}")
            return None
    
    def _wait_for_service_ready(self, timeout: int = 60):
        """Wait for the service to be ready to accept requests."""
        import requests
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{self.service_url}/health", timeout=5)
                if response.status_code == 200:
                    print("✅ Service is ready to accept requests")
                    return True
            except:
                pass
            
            print("⏳ Waiting for service to be ready...")
            time.sleep(2)
        
        print("⚠️  Service may not be fully ready")
        return False
    
    def stop_service(self):
        """Stop the Stock Analyzer Pro service."""
        if self.service_process:
            try:
                print("🛑 Stopping service...")
                self.service_process.terminate()
                self.service_process.wait(timeout=10)
                print("✅ Service stopped")
            except subprocess.TimeoutExpired:
                print("⚠️  Service didn't stop gracefully, forcing...")
                self.service_process.kill()
            except Exception as e:
                print(f"❌ Error stopping service: {e}")
    
    def start_memory_monitoring(self, pid: int):
        """Start memory monitoring for the service process."""
        try:
            print(f"🔍 Starting memory monitoring for PID: {pid}")
            
            self.monitor = MemoryMonitor(
                pid=pid,
                output_file="memory_analysis.json"
            )
            
            self.monitor.start_monitoring(interval=self.analysis_config['monitoring_interval'])
            self.monitoring_active = True
            
            print("✅ Memory monitoring started")
            
        except Exception as e:
            print(f"❌ Error starting memory monitoring: {e}")
    
    def stop_memory_monitoring(self):
        """Stop memory monitoring."""
        if self.monitor and self.monitoring_active:
            try:
                self.monitor.stop_monitoring()
                self.monitoring_active = False
                print("✅ Memory monitoring stopped")
            except Exception as e:
                print(f"❌ Error stopping memory monitoring: {e}")
    
    async def run_baseline_monitoring(self, duration: int):
        """Run baseline memory monitoring without load."""
        print(f"\n📊 Running baseline monitoring for {duration} seconds...")
        
        start_time = time.time()
        baseline_data = []
        
        while time.time() - start_time < duration:
            if self.monitor:
                sample = self.monitor.collect_memory_sample()
                baseline_data.append(sample)
            
            await asyncio.sleep(self.analysis_config['monitoring_interval'])
        
        self.analysis_results['baseline'] = baseline_data
        print(f"✅ Baseline monitoring completed: {len(baseline_data)} samples")
    
    async def run_load_test(self, duration: int):
        """Run load testing while monitoring memory."""
        print(f"\n🚀 Running load test for {duration} seconds...")
        
        # Start load testing
        self.load_tester = LoadTester(base_url=self.service_url)
        
        # Run load test in background
        load_test_task = asyncio.create_task(
            self.load_tester.run_load_test(
                num_users=self.analysis_config['load_test_users'],
                requests_per_user=self.analysis_config['load_test_requests'],
                duration_seconds=duration
            )
        )
        
        # Monitor memory during load test
        start_time = time.time()
        load_test_data = []
        
        while time.time() - start_time < duration:
            if self.monitor:
                sample = self.monitor.collect_memory_sample()
                sample['load_test_active'] = True
                load_test_data.append(sample)
            
            await asyncio.sleep(self.analysis_config['monitoring_interval'])
        
        # Wait for load test to complete
        load_test_stats = await load_test_task
        
        self.analysis_results['load_test'] = load_test_data
        self.analysis_results['load_test_stats'] = load_test_stats
        
        print(f"✅ Load test completed: {len(load_test_data)} samples")
        return load_test_stats
    
    async def run_cooldown_monitoring(self, duration: int):
        """Run memory monitoring after load test to observe recovery."""
        print(f"\n🔄 Running cooldown monitoring for {duration} seconds...")
        
        start_time = time.time()
        cooldown_data = []
        
        while time.time() - start_time < duration:
            if self.monitor:
                sample = self.monitor.collect_memory_sample()
                sample['load_test_active'] = False
                cooldown_data.append(sample)
            
            await asyncio.sleep(self.analysis_config['monitoring_interval'])
        
        self.analysis_results['cooldown'] = cooldown_data
        print(f"✅ Cooldown monitoring completed: {len(cooldown_data)} samples")
    
    def calculate_memory_analysis(self) -> Dict:
        """Calculate comprehensive memory analysis from all phases."""
        try:
            analysis = {
                'phases': {},
                'memory_growth': {},
                'peak_usage': {},
                'recovery_patterns': {},
                'recommendations': {}
            }
            
            # Analyze baseline phase
            if self.analysis_results['baseline']:
                baseline_stats = self._analyze_phase(self.analysis_results['baseline'], 'baseline')
                analysis['phases']['baseline'] = baseline_stats
            
            # Analyze load test phase
            if self.analysis_results['load_test']:
                load_test_stats = self._analyze_phase(self.analysis_results['load_test'], 'load_test')
                analysis['phases']['load_test'] = load_test_stats
            
            # Analyze cooldown phase
            if self.analysis_results['cooldown']:
                cooldown_stats = self._analyze_phase(self.analysis_results['cooldown'], 'cooldown')
                analysis['phases']['cooldown'] = cooldown_stats
            
            # Calculate memory growth patterns
            analysis['memory_growth'] = self._calculate_memory_growth()
            
            # Calculate peak usage
            analysis['peak_usage'] = self._calculate_peak_usage()
            
            # Calculate recovery patterns
            analysis['recovery_patterns'] = self._calculate_recovery_patterns()
            
            # Generate recommendations
            analysis['recommendations'] = self._generate_recommendations(analysis)
            
            return analysis
            
        except Exception as e:
            print(f"❌ Error calculating memory analysis: {e}")
            return {'error': str(e)}
    
    def _analyze_phase(self, phase_data: list, phase_name: str) -> Dict:
        """Analyze memory data for a specific phase."""
        if not phase_data:
            return {}
        
        try:
            # Extract memory values
            system_memory_percent = [
                sample.get('system_memory', {}).get('memory_percent', 0)
                for sample in phase_data
            ]
            
            process_memory_mb = []
            if self.monitor and self.monitor.pid:
                for sample in phase_data:
                    process_mem = sample.get('process_memory', {})
                    if 'memory_rss_mb' in process_mem:
                        process_memory_mb.append(process_mem['memory_rss_mb'])
            
            # Calculate statistics
            stats = {
                'phase': phase_name,
                'sample_count': len(phase_data),
                'duration_seconds': len(phase_data) * self.analysis_config['monitoring_interval'],
                'system_memory': {
                    'min_percent': min(system_memory_percent) if system_memory_percent else 0,
                    'max_percent': max(system_memory_percent) if system_memory_percent else 0,
                    'avg_percent': sum(system_memory_percent) / len(system_memory_percent) if system_memory_percent else 0,
                    'trend': self._calculate_trend(system_memory_percent)
                }
            }
            
            if process_memory_mb:
                stats['process_memory'] = {
                    'min_mb': min(process_memory_mb),
                    'max_mb': max(process_memory_mb),
                    'avg_mb': sum(process_memory_mb) / len(process_memory_mb),
                    'trend': self._calculate_trend(process_memory_mb)
                }
            
            return stats
            
        except Exception as e:
            return {'error': str(e)}
    
    def _calculate_trend(self, values: list) -> str:
        """Calculate trend of values (increasing, decreasing, stable)."""
        if len(values) < 2:
            return 'insufficient_data'
        
        try:
            # Simple linear regression slope
            n = len(values)
            x_sum = sum(range(n))
            y_sum = sum(values)
            xy_sum = sum(i * val for i, val in enumerate(values))
            x2_sum = sum(i * i for i in range(n))
            
            slope = (n * xy_sum - x_sum * y_sum) / (n * x2_sum - x_sum * x_sum)
            
            if slope > 0.1:
                return 'increasing'
            elif slope < -0.1:
                return 'decreasing'
            else:
                return 'stable'
                
        except:
            return 'unknown'
    
    def _calculate_memory_growth(self) -> Dict:
        """Calculate memory growth patterns across phases."""
        try:
            growth = {}
            
            # Compare baseline vs load test
            if self.analysis_results['baseline'] and self.analysis_results['load_test']:
                baseline_avg = self._get_phase_avg_memory(self.analysis_results['baseline'])
                load_test_avg = self._get_phase_avg_memory(self.analysis_results['load_test'])
                
                if baseline_avg and load_test_avg:
                    growth['baseline_to_load'] = {
                        'system_memory_increase_percent': load_test_avg['system'] - baseline_avg['system'],
                        'process_memory_increase_mb': load_test_avg['process'] - baseline_avg['process'] if 'process' in load_test_avg else 0
                    }
            
            # Compare load test vs cooldown
            if self.analysis_results['load_test'] and self.analysis_results['cooldown']:
                load_test_avg = self._get_phase_avg_memory(self.analysis_results['load_test'])
                cooldown_avg = self._get_phase_avg_memory(self.analysis_results['cooldown'])
                
                if load_test_avg and cooldown_avg:
                    growth['load_to_cooldown'] = {
                        'system_memory_decrease_percent': load_test_avg['system'] - cooldown_avg['system'],
                        'process_memory_decrease_mb': load_test_avg['process'] - cooldown_avg['process'] if 'process' in load_test_avg else 0
                    }
            
            return growth
            
        except Exception as e:
            return {'error': str(e)}
    
    def _get_phase_avg_memory(self, phase_data: list) -> Optional[Dict]:
        """Get average memory usage for a phase."""
        try:
            system_memory_percent = [
                sample.get('system_memory', {}).get('memory_percent', 0)
                for sample in phase_data
            ]
            
            process_memory_mb = []
            if self.monitor and self.monitor.pid:
                for sample in phase_data:
                    process_mem = sample.get('process_memory', {})
                    if 'memory_rss_mb' in process_mem:
                        process_memory_mb.append(process_mem['memory_rss_mb'])
            
            result = {
                'system': sum(system_memory_percent) / len(system_memory_percent) if system_memory_percent else 0
            }
            
            if process_memory_mb:
                result['process'] = sum(process_memory_mb) / len(process_memory_mb)
            
            return result
            
        except Exception:
            return None
    
    def _calculate_peak_usage(self) -> Dict:
        """Calculate peak memory usage across all phases."""
        try:
            all_samples = []
            all_samples.extend(self.analysis_results['baseline'])
            all_samples.extend(self.analysis_results['load_test'])
            all_samples.extend(self.analysis_results['cooldown'])
            
            if not all_samples:
                return {}
            
            # Find peak system memory
            peak_system = max(
                (sample.get('system_memory', {}).get('memory_percent', 0) for sample in all_samples),
                default=0
            )
            
            # Find peak process memory
            peak_process = 0
            if self.monitor and self.monitor.pid:
                peak_process = max(
                    (sample.get('process_memory', {}).get('memory_rss_mb', 0) for sample in all_samples),
                    default=0
                )
            
            return {
                'peak_system_memory_percent': peak_system,
                'peak_process_memory_mb': peak_process,
                'peak_phase': self._identify_peak_phase(all_samples)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _identify_peak_phase(self, all_samples: list) -> str:
        """Identify which phase had the peak memory usage."""
        try:
            peak_sample = max(
                all_samples,
                key=lambda s: s.get('system_memory', {}).get('memory_percent', 0)
            )
            
            if 'load_test_active' in peak_sample:
                return 'load_test' if peak_sample['load_test_active'] else 'cooldown'
            else:
                return 'baseline'
                
        except Exception:
            return 'unknown'
    
    def _calculate_recovery_patterns(self) -> Dict:
        """Calculate memory recovery patterns after load test."""
        try:
            recovery = {}
            
            if self.analysis_results['load_test'] and self.analysis_results['cooldown']:
                # Calculate recovery rate
                load_test_end = self.analysis_results['load_test'][-1]
                cooldown_start = self.analysis_results['cooldown'][0]
                cooldown_end = self.analysis_results['cooldown'][-1]
                
                if 'system_memory' in load_test_end and 'system_memory' in cooldown_start:
                    peak_system = load_test_end['system_memory'].get('memory_percent', 0)
                    start_cooldown = cooldown_start['system_memory'].get('memory_percent', 0)
                    end_cooldown = cooldown_end['system_memory'].get('memory_percent', 0)
                    
                    recovery['system_memory'] = {
                        'peak_usage': peak_system,
                        'immediate_recovery': peak_system - start_cooldown,
                        'full_recovery': peak_system - end_cooldown,
                        'recovery_percentage': ((peak_system - end_cooldown) / (peak_system - start_cooldown)) * 100 if start_cooldown != peak_system else 0
                    }
                
                if self.monitor and self.monitor.pid:
                    if 'process_memory' in load_test_end and 'process_memory' in cooldown_start:
                        peak_process = load_test_end['process_memory'].get('memory_rss_mb', 0)
                        start_cooldown = cooldown_start['process_memory'].get('memory_rss_mb', 0)
                        end_cooldown = cooldown_end['process_memory'].get('memory_rss_mb', 0)
                        
                        recovery['process_memory'] = {
                            'peak_usage_mb': peak_process,
                            'immediate_recovery_mb': peak_process - start_cooldown,
                            'full_recovery_mb': peak_process - end_cooldown,
                            'recovery_percentage': ((peak_process - end_cooldown) / (peak_process - start_cooldown)) * 100 if start_cooldown != peak_process else 0
                        }
            
            return recovery
            
        except Exception as e:
            return {'error': str(e)}
    
    def _generate_recommendations(self, analysis: Dict) -> Dict:
        """Generate recommendations based on memory analysis."""
        try:
            recommendations = {
                'memory_requirements': {},
                'optimization_suggestions': [],
                'deployment_considerations': []
            }
            
            # Memory requirements
            peak_usage = analysis.get('peak_usage', {})
            if 'peak_system_memory_percent' in peak_usage:
                peak_percent = peak_usage['peak_system_memory_percent']
                if peak_percent > 90:
                    recommendations['memory_requirements']['critical'] = f"Peak memory usage was {peak_percent:.1f}% - critical level"
                elif peak_percent > 80:
                    recommendations['memory_requirements']['warning'] = f"Peak memory usage was {peak_percent:.1f}% - monitor closely"
                else:
                    recommendations['memory_requirements']['safe'] = f"Peak memory usage was {peak_percent:.1f}% - within safe limits"
            
            # Optimization suggestions
            memory_growth = analysis.get('memory_growth', {})
            if 'baseline_to_load' in memory_growth:
                growth = memory_growth['baseline_to_load']
                if growth.get('system_memory_increase_percent', 0) > 20:
                    recommendations['optimization_suggestions'].append(
                        f"High memory growth under load: {growth['system_memory_increase_percent']:.1f}% increase"
                    )
                
                if growth.get('process_memory_increase_mb', 0) > 100:
                    recommendations['optimization_suggestions'].append(
                        f"High process memory growth: {growth['process_memory_increase_mb']:.1f} MB increase"
                    )
            
            # Recovery patterns
            recovery = analysis.get('recovery_patterns', {})
            if 'system_memory' in recovery:
                recovery_pct = recovery['system_memory'].get('recovery_percentage', 0)
                if recovery_pct < 80:
                    recommendations['optimization_suggestions'].append(
                        f"Poor memory recovery: only {recovery_pct:.1f}% recovery after load test"
                    )
            
            # Deployment considerations
            recommendations['deployment_considerations'].extend([
                "Monitor memory usage during peak hours",
                "Set up alerts for memory thresholds",
                "Consider horizontal scaling for high traffic periods",
                "Implement memory cleanup routines for long-running processes"
            ])
            
            return recommendations
            
        except Exception as e:
            return {'error': str(e)}
    
    def print_analysis_summary(self, analysis: Dict):
        """Print a comprehensive summary of the memory analysis."""
        print("\n" + "="*80)
        print("📊 COMPREHENSIVE MEMORY ANALYSIS SUMMARY")
        print("="*80)
        
        # Phase summary
        phases = analysis.get('phases', {})
        for phase_name, phase_stats in phases.items():
            if phase_stats:
                print(f"\n📈 {phase_name.upper()} PHASE:")
                print(f"   Duration: {phase_stats.get('duration_seconds', 0):.1f}s")
                print(f"   Samples: {phase_stats.get('sample_count', 0)}")
                
                system_memory = phase_stats.get('system_memory', {})
                print(f"   System Memory: {system_memory.get('avg_percent', 0):.1f}% avg, {system_memory.get('max_percent', 0):.1f}% max")
                print(f"   Trend: {system_memory.get('trend', 'unknown')}")
                
                if 'process_memory' in phase_stats:
                    process_memory = phase_stats['process_memory']
                    print(f"   Process Memory: {process_memory.get('avg_mb', 0):.1f} MB avg, {process_memory.get('max_mb', 0):.1f} MB max")
                    print(f"   Trend: {process_memory.get('trend', 'unknown')}")
        
        # Peak usage
        peak_usage = analysis.get('peak_usage', {})
        if peak_usage:
            print(f"\n🏔️  PEAK MEMORY USAGE:")
            print(f"   System: {peak_usage.get('peak_system_memory_percent', 0):.1f}%")
            print(f"   Process: {peak_usage.get('peak_process_memory_mb', 0):.1f} MB")
            print(f"   Phase: {peak_usage.get('peak_phase', 'unknown')}")
        
        # Memory growth
        memory_growth = analysis.get('memory_growth', {})
        if memory_growth:
            print(f"\n📈 MEMORY GROWTH PATTERNS:")
            if 'baseline_to_load' in memory_growth:
                growth = memory_growth['baseline_to_load']
                print(f"   Baseline → Load Test:")
                print(f"     System: +{growth.get('system_memory_increase_percent', 0):.1f}%")
                print(f"     Process: +{growth.get('process_memory_increase_mb', 0):.1f} MB")
            
            if 'load_to_cooldown' in memory_growth:
                recovery = memory_growth['load_to_cooldown']
                print(f"   Load Test → Cooldown:")
                print(f"     System: -{recovery.get('system_memory_decrease_percent', 0):.1f}%")
                print(f"     Process: -{recovery.get('process_memory_decrease_mb', 0):.1f} MB")
        
        # Recovery patterns
        recovery = analysis.get('recovery_patterns', {})
        if recovery:
            print(f"\n🔄 RECOVERY PATTERNS:")
            if 'system_memory' in recovery:
                sys_recovery = recovery['system_memory']
                print(f"   System Memory Recovery: {sys_recovery.get('recovery_percentage', 0):.1f}%")
            
            if 'process_memory' in recovery:
                proc_recovery = recovery['process_memory']
                print(f"   Process Memory Recovery: {proc_recovery.get('recovery_percentage', 0):.1f}%")
        
        # Recommendations
        recommendations = analysis.get('recommendations', {})
        if recommendations:
            print(f"\n💡 RECOMMENDATIONS:")
            
            memory_req = recommendations.get('memory_requirements', {})
            for level, message in memory_req.items():
                print(f"   {level.upper()}: {message}")
            
            optimizations = recommendations.get('optimization_suggestions', [])
            for opt in optimizations:
                print(f"   🔧 {opt}")
            
            deployments = recommendations.get('deployment_considerations', [])
            for dep in deployments:
                print(f"   🚀 {dep}")
        
        print("="*80)
    
    def save_analysis_results(self, filename: str = "comprehensive_memory_analysis.json"):
        """Save comprehensive analysis results to file."""
        try:
            data = {
                'analysis_timestamp': datetime.now().isoformat(),
                'service_url': self.service_url,
                'analysis_config': self.analysis_config,
                'analysis_results': self.analysis_results,
                'comprehensive_analysis': self.calculate_memory_analysis()
            }
            
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            print(f"💾 Comprehensive analysis saved to {filename}")
            
        except Exception as e:
            print(f"❌ Error saving analysis results: {e}")
    
    async def run_complete_analysis(self):
        """Run the complete memory analysis workflow."""
        try:
            print("🚀 Starting comprehensive memory analysis...")
            
            # Start the service
            pid = self.start_service()
            if not pid:
                print("❌ Failed to start service")
                return
            
            # Start memory monitoring
            self.start_memory_monitoring(pid)
            
            try:
                # Phase 1: Baseline monitoring
                await self.run_baseline_monitoring(self.analysis_config['baseline_duration'])
                
                # Phase 2: Load testing with monitoring
                load_test_stats = await self.run_load_test(self.analysis_config['load_test_duration'])
                
                # Phase 3: Cooldown monitoring
                await self.run_cooldown_monitoring(self.analysis_config['cooldown_duration'])
                
                # Calculate comprehensive analysis
                analysis = self.calculate_memory_analysis()
                
                # Print summary
                self.print_analysis_summary(analysis)
                
                # Save results
                self.save_analysis_results()
                
                # Save load test results separately
                if self.load_tester:
                    self.load_tester.save_results("load_test_during_analysis.json")
                
                print("\n✅ Comprehensive memory analysis completed!")
                
            finally:
                # Stop monitoring
                self.stop_memory_monitoring()
                
                # Stop service
                self.stop_service()
                
        except KeyboardInterrupt:
            print("\n👋 Analysis interrupted by user")
            self.stop_memory_monitoring()
            self.stop_service()
        except Exception as e:
            print(f"❌ Error during analysis: {e}")
            self.stop_memory_monitoring()
            self.stop_service()

async def main():
    """Main function for standalone usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Comprehensive Memory Analysis Runner')
    parser.add_argument('--url', type=str, default='http://localhost:8000', 
                       help='Base URL of the service')
    parser.add_argument('--baseline', type=int, default=30, 
                       help='Baseline monitoring duration in seconds')
    parser.add_argument('--load-test', type=int, default=120, 
                       help='Load test duration in seconds')
    parser.add_argument('--cooldown', type=int, default=30, 
                       help='Cooldown monitoring duration in seconds')
    parser.add_argument('--users', type=int, default=20, 
                       help='Number of concurrent users for load test')
    parser.add_argument('--requests', type=int, default=15, 
                       help='Requests per user for load test')
    parser.add_argument('--interval', type=float, default=0.5, 
                       help='Memory monitoring interval in seconds')
    
    args = parser.parse_args()
    
    # Create runner
    runner = MemoryAnalysisRunner(service_url=args.url)
    
    # Update configuration
    runner.analysis_config.update({
        'baseline_duration': args.baseline,
        'load_test_duration': args.load_test,
        'cooldown_duration': args.cooldown,
        'load_test_users': args.users,
        'load_test_requests': args.requests,
        'monitoring_interval': args.interval
    })
    
    # Run analysis
    await runner.run_complete_analysis()

if __name__ == "__main__":
    asyncio.run(main())
