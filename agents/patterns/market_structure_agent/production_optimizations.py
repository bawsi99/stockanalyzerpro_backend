#!/usr/bin/env python3
"""
Production Optimizations for Integrated Market Structure Agent

This module provides production-ready optimizations including:
- Advanced caching strategies
- Performance monitoring and metrics
- Configuration management
- Resource optimization
- Concurrent processing capabilities
- Memory management
- Error recovery mechanisms
"""

import os
import sys
import json
import asyncio
import logging
import hashlib
import time
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from functools import wraps
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import gc

# Add parent directories to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import the base agent
try:
    from integrated_market_structure_agent import IntegratedMarketStructureAgent
except ImportError:
    print("❌ Failed to import IntegratedMarketStructureAgent")
    sys.exit(1)

# Set up logging
logger = logging.getLogger(__name__)

class ProductionOptimizedAgent(IntegratedMarketStructureAgent):
    """
    Production-optimized version of the integrated market structure agent.
    
    Adds sophisticated caching, performance monitoring, concurrent processing,
    and resource management capabilities for production deployment.
    """
    
    def __init__(self, 
                 charts_output_dir: str = "production_charts",
                 results_output_dir: str = "production_results", 
                 agent_name: str = "pattern_agent",
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the production-optimized agent.
        
        Args:
            charts_output_dir: Directory for generated charts
            results_output_dir: Directory for analysis results
            agent_name: LLM agent configuration name
            config: Production configuration dictionary
        """
        # Initialize base agent
        super().__init__(charts_output_dir, results_output_dir, agent_name)
        
        # Production configuration
        self.config = config or self._load_default_config()
        
        # Performance tracking
        self.performance_metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'average_response_time': 0,
            'memory_usage': [],
            'error_count': 0,
            'start_time': time.time()
        }
        
        # Initialize caching system
        self._init_caching_system()
        
        # Initialize resource monitoring
        self._init_resource_monitoring()
        
        # Initialize concurrent processing
        self._init_concurrent_processing()
        
        logger.info(f"🚀 Production-optimized agent initialized with config: {self.config['cache']['enabled']}")
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default production configuration."""
        return {
            'cache': {
                'enabled': True,
                'ttl_seconds': 3600,  # 1 hour
                'max_size': 1000,
                'compression_enabled': True,
                'persistence_enabled': True,
                'cleanup_interval': 300  # 5 minutes
            },
            'performance': {
                'max_concurrent_requests': 5,
                'timeout_seconds': 120,
                'memory_limit_mb': 2048,
                'chart_quality_level': 'llm_optimized',
                'enable_metrics_collection': True
            },
            'optimization': {
                'enable_parallel_processing': True,
                'chart_reuse_enabled': True,
                'response_compression': True,
                'memory_cleanup_interval': 600  # 10 minutes
            },
            'resilience': {
                'max_retries': 3,
                'retry_delay_seconds': 1,
                'fallback_enabled': True,
                'circuit_breaker_enabled': True
            }
        }
    
    def _init_caching_system(self):
        """Initialize sophisticated caching system."""
        self.cache = {}
        self.cache_metadata = {}
        self.cache_lock = threading.RLock()
        
        # Setup cache directory
        cache_dir = Path("production_cache")
        cache_dir.mkdir(exist_ok=True)
        self.cache_dir = cache_dir
        
        # Load persistent cache if enabled
        if self.config['cache']['persistence_enabled']:
            self._load_persistent_cache()
        
        # Start cache cleanup thread
        if self.config['cache']['enabled']:
            self._start_cache_cleanup_thread()
    
    def _init_resource_monitoring(self):
        """Initialize resource monitoring."""
        self.resource_monitor = {
            'process': psutil.Process(),
            'monitoring_enabled': self.config['performance']['enable_metrics_collection']
        }
        
        # Start resource monitoring thread
        if self.resource_monitor['monitoring_enabled']:
            self._start_resource_monitoring_thread()
    
    def _init_concurrent_processing(self):
        """Initialize concurrent processing capabilities."""
        max_workers = self.config['performance']['max_concurrent_requests']
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.active_requests = {}
        self.request_lock = threading.Lock()
    
    async def analyze_market_structure_optimized(self,
                                               stock_data: Dict[str, Any],
                                               analysis_data: Dict[str, Any],
                                               symbol: str,
                                               scenario_description: str = "Market Analysis",
                                               use_cache: bool = True) -> Dict[str, Any]:
        """
        Production-optimized market structure analysis.
        
        Args:
            stock_data: Stock price and volume data
            analysis_data: Market structure analysis results
            symbol: Stock symbol
            scenario_description: Description of the market scenario
            use_cache: Whether to use caching
            
        Returns:
            Dictionary containing complete analysis results
        """
        start_time = time.time()
        request_id = self._generate_request_id(stock_data, analysis_data, symbol)
        
        try:
            # Update metrics
            self._update_metrics('request_started')
            
            # Check cache first
            if use_cache and self.config['cache']['enabled']:
                cached_result = await self._get_from_cache(request_id)
                if cached_result:
                    self._update_metrics('cache_hit')
                    logger.info(f"🎯 Cache hit for {symbol} - returning cached result")
                    return cached_result
            
            # Log cache miss
            if use_cache:
                self._update_metrics('cache_miss')
            
            # Monitor resource usage
            self._check_resource_limits()
            
            # Execute analysis with circuit breaker
            result = await self._execute_with_circuit_breaker(
                self._perform_analysis_with_optimization,
                stock_data, analysis_data, symbol, scenario_description
            )
            
            # Cache the result if successful
            if result.get('success') and use_cache:
                await self._store_in_cache(request_id, result)
            
            # Update performance metrics
            execution_time = time.time() - start_time
            self._update_metrics('request_completed', execution_time)
            
            # Add production metadata
            result['production_metadata'] = {
                'execution_time': execution_time,
                'cache_used': use_cache,
                'cache_hit': False,  # This was a fresh computation
                'request_id': request_id,
                'resource_usage': self._get_current_resource_usage()
            }
            
            return result
            
        except Exception as e:
            self._update_metrics('error_occurred')
            logger.error(f"❌ Production analysis failed for {symbol}: {e}")
            return self._build_error_result_with_metadata(str(e), symbol, request_id)
    
    async def _perform_analysis_with_optimization(self,
                                                stock_data: Dict[str, Any],
                                                analysis_data: Dict[str, Any],
                                                symbol: str,
                                                scenario_description: str) -> Dict[str, Any]:
        """Perform analysis with production optimizations."""
        
        # Use optimized chart quality setting
        chart_quality = self.config['performance']['chart_quality_level']
        
        # Check if chart can be reused
        chart_reuse_key = None
        if self.config['optimization']['chart_reuse_enabled']:
            chart_reuse_key = self._generate_chart_reuse_key(stock_data, analysis_data)
            existing_chart = await self._get_reusable_chart(chart_reuse_key)
            if existing_chart:
                logger.info(f"📊 Reusing existing chart for {symbol}")
                # Use existing chart path and skip generation
                return await self._analyze_with_existing_chart(
                    existing_chart, analysis_data, symbol, scenario_description
                )
        
        # Execute base analysis with timeout
        timeout = self.config['performance']['timeout_seconds']
        result = await asyncio.wait_for(
            super().analyze_market_structure(
                stock_data=stock_data,
                analysis_data=analysis_data,
                symbol=symbol,
                scenario_description=scenario_description
            ),
            timeout=timeout
        )
        
        # Store chart for reuse if enabled
        if result.get('success') and chart_reuse_key:
            await self._store_reusable_chart(chart_reuse_key, result.get('chart_info', {}))
        
        return result
    
    async def batch_analyze_symbols(self,
                                  symbol_data: List[Dict[str, Any]],
                                  max_concurrent: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Perform batch analysis of multiple symbols with concurrent processing.
        
        Args:
            symbol_data: List of dictionaries containing symbol analysis data
            max_concurrent: Maximum concurrent analyses (uses config default if None)
            
        Returns:
            List of analysis results
        """
        max_concurrent = max_concurrent or self.config['performance']['max_concurrent_requests']
        
        logger.info(f"🚀 Starting batch analysis of {len(symbol_data)} symbols (max concurrent: {max_concurrent})")
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def analyze_single(data):
            async with semaphore:
                return await self.analyze_market_structure_optimized(
                    stock_data=data.get('stock_data', {}),
                    analysis_data=data.get('analysis_data', {}),
                    symbol=data.get('symbol', 'UNKNOWN'),
                    scenario_description=data.get('scenario', 'Batch Analysis')
                )
        
        # Execute all analyses concurrently
        tasks = [analyze_single(data) for data in symbol_data]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions and create proper results
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                final_results.append(self._build_error_result_with_metadata(
                    str(result), symbol_data[i].get('symbol', 'UNKNOWN'), f"batch_{i}"
                ))
            else:
                final_results.append(result)
        
        successful_count = sum(1 for r in final_results if r.get('success', False))
        logger.info(f"✅ Batch analysis completed: {successful_count}/{len(symbol_data)} successful")
        
        return final_results
    
    def _generate_request_id(self, stock_data: Dict[str, Any], analysis_data: Dict[str, Any], symbol: str) -> str:
        """Generate unique request ID for caching."""
        # Create hash of input data for caching key
        data_string = json.dumps({
            'stock_data': stock_data,
            'analysis_data': analysis_data,
            'symbol': symbol
        }, sort_keys=True, default=str)
        
        return hashlib.md5(data_string.encode()).hexdigest()
    
    def _generate_chart_reuse_key(self, stock_data: Dict[str, Any], analysis_data: Dict[str, Any]) -> str:
        """Generate key for chart reuse based on data characteristics."""
        # Create simplified key based on data patterns rather than exact values
        pattern_data = {
            'price_points': len(stock_data.get('prices', [])),
            'swing_count': analysis_data.get('swing_points', {}).get('total_swings', 0),
            'trend_direction': analysis_data.get('trend_analysis', {}).get('trend_direction', ''),
            'bos_count': len(analysis_data.get('bos_choch_analysis', {}).get('bos_events', [])),
            'market_regime': analysis_data.get('market_regime', {}).get('regime', '')
        }
        
        pattern_string = json.dumps(pattern_data, sort_keys=True)
        return hashlib.md5(pattern_string.encode()).hexdigest()[:16]  # Shorter key for chart reuse
    
    async def _get_from_cache(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve result from cache."""
        with self.cache_lock:
            if request_id not in self.cache:
                return None
            
            # Check TTL
            metadata = self.cache_metadata.get(request_id, {})
            created_time = metadata.get('created_time', 0)
            ttl = self.config['cache']['ttl_seconds']
            
            if time.time() - created_time > ttl:
                # Expired, remove from cache
                del self.cache[request_id]
                del self.cache_metadata[request_id]
                return None
            
            # Update access time
            metadata['last_access'] = time.time()
            metadata['access_count'] = metadata.get('access_count', 0) + 1
            
            cached_data = self.cache[request_id]
            
            # Decompress if needed
            if metadata.get('compressed', False):
                try:
                    cached_data = pickle.loads(cached_data)
                except Exception as e:
                    logger.error(f"❌ Cache decompression failed: {e}")
                    return None
            
            return cached_data
    
    async def _store_in_cache(self, request_id: str, result: Dict[str, Any]):
        """Store result in cache."""
        if not self.config['cache']['enabled']:
            return
        
        with self.cache_lock:
            # Check cache size limit
            max_size = self.config['cache']['max_size']
            if len(self.cache) >= max_size:
                self._evict_cache_entries(max_size // 4)  # Evict 25% of entries
            
            # Compress if enabled
            data_to_store = result
            compressed = False
            
            if self.config['cache']['compression_enabled']:
                try:
                    data_to_store = pickle.dumps(result)
                    compressed = True
                except Exception as e:
                    logger.warning(f"⚠️ Cache compression failed, storing uncompressed: {e}")
                    data_to_store = result
            
            # Store in cache
            self.cache[request_id] = data_to_store
            self.cache_metadata[request_id] = {
                'created_time': time.time(),
                'last_access': time.time(),
                'access_count': 0,
                'size_bytes': len(str(data_to_store)),
                'compressed': compressed
            }
    
    async def _get_reusable_chart(self, chart_key: str) -> Optional[str]:
        """Get reusable chart path if available."""
        chart_cache_path = self.cache_dir / f"chart_{chart_key}.json"
        if chart_cache_path.exists():
            try:
                with open(chart_cache_path, 'r') as f:
                    chart_info = json.load(f)
                
                chart_path = chart_info.get('chart_path')
                if chart_path and Path(chart_path).exists():
                    # Check if chart is still fresh
                    created_time = chart_info.get('created_time', 0)
                    if time.time() - created_time < self.config['cache']['ttl_seconds']:
                        return chart_path
            except Exception as e:
                logger.error(f"❌ Chart reuse check failed: {e}")
        
        return None
    
    async def _store_reusable_chart(self, chart_key: str, chart_info: Dict[str, Any]):
        """Store chart information for reuse."""
        try:
            chart_cache_path = self.cache_dir / f"chart_{chart_key}.json"
            chart_data = {
                'chart_path': chart_info.get('chart_path'),
                'created_time': time.time(),
                'metadata': chart_info.get('metadata', {})
            }
            
            with open(chart_cache_path, 'w') as f:
                json.dump(chart_data, f)
        except Exception as e:
            logger.error(f"❌ Chart reuse storage failed: {e}")
    
    async def _analyze_with_existing_chart(self,
                                         chart_path: str,
                                         analysis_data: Dict[str, Any],
                                         symbol: str,
                                         scenario_description: str) -> Dict[str, Any]:
        """Perform analysis using existing chart."""
        try:
            # Create enhanced prompt
            prompt, chart_metadata = await self._create_enhanced_prompt(
                analysis_data, symbol, scenario_description, chart_path
            )
            
            # Execute LLM analysis only (skip chart generation)
            llm_result = await self._execute_llm_analysis(prompt, chart_path)
            if not llm_result['success']:
                return self._build_error_result(f"LLM analysis failed: {llm_result['error']}", symbol)
            
            # Parse and validate response
            structured_result = await self._parse_and_validate_response(
                llm_result['response'], analysis_data, symbol, {'success': True, 'chart_path': chart_path}
            )
            
            # Build final result
            final_result = self._build_final_result(
                structured_result,
                {'success': True, 'chart_path': chart_path, 'reused': True},
                chart_metadata,
                analysis_data,
                symbol,
                scenario_description
            )
            
            # Mark as chart reused
            final_result['chart_info']['chart_reused'] = True
            final_result['performance_metrics']['chart_generation_time'] = 0
            
            return final_result
            
        except Exception as e:
            logger.error(f"❌ Analysis with existing chart failed: {e}")
            return self._build_error_result(str(e), symbol)
    
    def _evict_cache_entries(self, count: int):
        """Evict least recently used cache entries."""
        if not self.cache_metadata:
            return
        
        # Sort by last access time (oldest first)
        sorted_entries = sorted(
            self.cache_metadata.items(),
            key=lambda x: x[1].get('last_access', 0)
        )
        
        for i in range(min(count, len(sorted_entries))):
            entry_id = sorted_entries[i][0]
            if entry_id in self.cache:
                del self.cache[entry_id]
            del self.cache_metadata[entry_id]
        
        logger.info(f"🗑️ Evicted {min(count, len(sorted_entries))} cache entries")
    
    def _start_cache_cleanup_thread(self):
        """Start background cache cleanup thread."""
        def cleanup_thread():
            while True:
                try:
                    time.sleep(self.config['cache']['cleanup_interval'])
                    self._cleanup_expired_cache()
                except Exception as e:
                    logger.error(f"❌ Cache cleanup error: {e}")
        
        thread = threading.Thread(target=cleanup_thread, daemon=True)
        thread.start()
        logger.info("🧹 Cache cleanup thread started")
    
    def _cleanup_expired_cache(self):
        """Clean up expired cache entries."""
        current_time = time.time()
        ttl = self.config['cache']['ttl_seconds']
        expired_keys = []
        
        with self.cache_lock:
            for key, metadata in self.cache_metadata.items():
                created_time = metadata.get('created_time', 0)
                if current_time - created_time > ttl:
                    expired_keys.append(key)
            
            for key in expired_keys:
                if key in self.cache:
                    del self.cache[key]
                del self.cache_metadata[key]
        
        if expired_keys:
            logger.info(f"🗑️ Cleaned up {len(expired_keys)} expired cache entries")
    
    def _start_resource_monitoring_thread(self):
        """Start resource monitoring thread."""
        def monitor_resources():
            while True:
                try:
                    time.sleep(30)  # Monitor every 30 seconds
                    self._collect_resource_metrics()
                except Exception as e:
                    logger.error(f"❌ Resource monitoring error: {e}")
        
        thread = threading.Thread(target=monitor_resources, daemon=True)
        thread.start()
        logger.info("📊 Resource monitoring thread started")
    
    def _collect_resource_metrics(self):
        """Collect current resource usage metrics."""
        try:
            process = self.resource_monitor['process']
            memory_info = process.memory_info()
            
            current_metrics = {
                'timestamp': time.time(),
                'memory_rss_mb': memory_info.rss / 1024 / 1024,
                'memory_vms_mb': memory_info.vms / 1024 / 1024,
                'cpu_percent': process.cpu_percent(),
                'threads': process.num_threads(),
                'open_files': len(process.open_files())
            }
            
            self.performance_metrics['memory_usage'].append(current_metrics)
            
            # Keep only last 100 measurements
            if len(self.performance_metrics['memory_usage']) > 100:
                self.performance_metrics['memory_usage'] = self.performance_metrics['memory_usage'][-100:]
        
        except Exception as e:
            logger.error(f"❌ Resource collection error: {e}")
    
    def _check_resource_limits(self):
        """Check if resource limits are being approached."""
        try:
            memory_limit_mb = self.config['performance']['memory_limit_mb']
            current_memory = self.resource_monitor['process'].memory_info().rss / 1024 / 1024
            
            if current_memory > memory_limit_mb * 0.9:  # 90% threshold
                logger.warning(f"⚠️ Memory usage high: {current_memory:.1f}MB (limit: {memory_limit_mb}MB)")
                # Trigger garbage collection
                gc.collect()
                # Clear some cache if needed
                if len(self.cache) > self.config['cache']['max_size'] * 0.5:
                    self._evict_cache_entries(self.config['cache']['max_size'] // 4)
        
        except Exception as e:
            logger.error(f"❌ Resource limit check error: {e}")
    
    def _get_current_resource_usage(self) -> Dict[str, Any]:
        """Get current resource usage snapshot."""
        try:
            process = self.resource_monitor['process']
            memory_info = process.memory_info()
            
            return {
                'memory_mb': memory_info.rss / 1024 / 1024,
                'cpu_percent': process.cpu_percent(),
                'threads': process.num_threads()
            }
        except Exception:
            return {}
    
    async def _execute_with_circuit_breaker(self, func, *args, **kwargs):
        """Execute function with circuit breaker pattern."""
        max_retries = self.config['resilience']['max_retries']
        retry_delay = self.config['resilience']['retry_delay_seconds']
        
        for attempt in range(max_retries):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"⚠️ Attempt {attempt + 1} failed, retrying in {retry_delay}s: {e}")
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"❌ All {max_retries} attempts failed: {e}")
                    raise
    
    def _update_metrics(self, event_type: str, value: float = None):
        """Update performance metrics."""
        with self.request_lock:
            if event_type == 'request_started':
                self.performance_metrics['total_requests'] += 1
            elif event_type == 'request_completed':
                self.performance_metrics['successful_requests'] += 1
                if value is not None:
                    # Update average response time
                    current_avg = self.performance_metrics['average_response_time']
                    total_requests = self.performance_metrics['successful_requests']
                    new_avg = (current_avg * (total_requests - 1) + value) / total_requests
                    self.performance_metrics['average_response_time'] = new_avg
            elif event_type == 'cache_hit':
                self.performance_metrics['cache_hits'] += 1
            elif event_type == 'cache_miss':
                self.performance_metrics['cache_misses'] += 1
            elif event_type == 'error_occurred':
                self.performance_metrics['error_count'] += 1
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        uptime = time.time() - self.performance_metrics['start_time']
        cache_hit_rate = 0
        
        if self.performance_metrics['cache_hits'] + self.performance_metrics['cache_misses'] > 0:
            cache_hit_rate = self.performance_metrics['cache_hits'] / (
                self.performance_metrics['cache_hits'] + self.performance_metrics['cache_misses']
            )
        
        return {
            'uptime_seconds': uptime,
            'total_requests': self.performance_metrics['total_requests'],
            'successful_requests': self.performance_metrics['successful_requests'],
            'success_rate': self.performance_metrics['successful_requests'] / max(1, self.performance_metrics['total_requests']),
            'error_count': self.performance_metrics['error_count'],
            'average_response_time': self.performance_metrics['average_response_time'],
            'cache_statistics': {
                'cache_hits': self.performance_metrics['cache_hits'],
                'cache_misses': self.performance_metrics['cache_misses'],
                'hit_rate': cache_hit_rate,
                'cache_size': len(self.cache)
            },
            'resource_usage': self._get_current_resource_usage(),
            'configuration': self.config
        }
    
    def _load_persistent_cache(self):
        """Load cache from persistent storage."""
        try:
            cache_file = self.cache_dir / "persistent_cache.pkl"
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    self.cache = cached_data.get('cache', {})
                    self.cache_metadata = cached_data.get('metadata', {})
                logger.info(f"📥 Loaded {len(self.cache)} entries from persistent cache")
        except Exception as e:
            logger.error(f"❌ Failed to load persistent cache: {e}")
    
    def save_persistent_cache(self):
        """Save cache to persistent storage."""
        if not self.config['cache']['persistence_enabled']:
            return
        
        try:
            cache_file = self.cache_dir / "persistent_cache.pkl"
            cache_data = {
                'cache': self.cache,
                'metadata': self.cache_metadata
            }
            
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            
            logger.info(f"💾 Saved {len(self.cache)} entries to persistent cache")
        except Exception as e:
            logger.error(f"❌ Failed to save persistent cache: {e}")
    
    def _build_error_result_with_metadata(self, error_message: str, symbol: str, request_id: str) -> Dict[str, Any]:
        """Build error result with production metadata."""
        result = self._build_error_result(error_message, symbol)
        result['production_metadata'] = {
            'request_id': request_id,
            'error_time': datetime.now().isoformat(),
            'resource_usage': self._get_current_resource_usage()
        }
        return result
    
    def shutdown(self):
        """Graceful shutdown with cleanup."""
        logger.info("🛑 Shutting down production optimized agent...")
        
        # Save persistent cache
        self.save_persistent_cache()
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        # Final performance report
        report = self.get_performance_report()
        logger.info(f"📊 Final performance report: {report['success_rate']:.1%} success rate, {report['cache_statistics']['hit_rate']:.1%} cache hit rate")
        
        logger.info("✅ Production optimized agent shutdown complete")


# Production deployment helper
class ProductionDeploymentManager:
    """Manager for production deployment of the optimized agent."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.agent = None
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load production configuration."""
        default_config = {
            'cache': {'enabled': True, 'ttl_seconds': 3600, 'max_size': 1000},
            'performance': {'max_concurrent_requests': 10, 'timeout_seconds': 120},
            'optimization': {'enable_parallel_processing': True, 'chart_reuse_enabled': True},
            'resilience': {'max_retries': 3, 'circuit_breaker_enabled': True}
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
                    # Merge with defaults
                    for section, values in loaded_config.items():
                        if section in default_config:
                            default_config[section].update(values)
                        else:
                            default_config[section] = values
            except Exception as e:
                logger.error(f"❌ Failed to load config from {config_path}: {e}")
        
        return default_config
    
    def initialize_agent(self) -> ProductionOptimizedAgent:
        """Initialize production-ready agent."""
        self.agent = ProductionOptimizedAgent(config=self.config)
        logger.info("🚀 Production agent initialized")
        return self.agent
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get system health status."""
        if not self.agent:
            return {'status': 'not_initialized', 'healthy': False}
        
        try:
            report = self.agent.get_performance_report()
            memory_usage = report['resource_usage'].get('memory_mb', 0)
            success_rate = report['success_rate']
            
            healthy = (
                memory_usage < self.config['performance']['memory_limit_mb'] and
                success_rate > 0.8 and
                report['error_count'] < 10
            )
            
            return {
                'status': 'running',
                'healthy': healthy,
                'uptime': report['uptime_seconds'],
                'memory_mb': memory_usage,
                'success_rate': success_rate,
                'cache_hit_rate': report['cache_statistics']['hit_rate']
            }
        except Exception as e:
            return {'status': 'error', 'healthy': False, 'error': str(e)}


# Example production usage
async def main():
    """Example production usage."""
    logger.info("🚀 Starting Production Optimized Market Structure Agent Demo")
    
    # Initialize deployment manager
    manager = ProductionDeploymentManager()
    agent = manager.initialize_agent()
    
    try:
        # Example analysis
        stock_data = {
            'prices': [100, 102, 104, 106, 108, 110],
            'volumes': [1000, 1200, 1100, 1500, 1800, 1600],
            'timestamps': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05', '2024-01-06']
        }
        
        analysis_data = {
            'swing_points': {'total_swings': 4, 'swing_density': 0.67},
            'trend_analysis': {'trend_direction': 'uptrend', 'trend_strength': 'strong'},
            'market_regime': {'regime': 'trending', 'confidence': 0.85}
        }
        
        # Single analysis
        result = await agent.analyze_market_structure_optimized(
            stock_data=stock_data,
            analysis_data=analysis_data,
            symbol="PROD_TEST",
            scenario_description="Production Test"
        )
        
        print(f"✅ Single analysis completed: {result['success']}")
        
        # Batch analysis example
        batch_data = [
            {'stock_data': stock_data, 'analysis_data': analysis_data, 'symbol': f'BATCH_{i}'}
            for i in range(3)
        ]
        
        batch_results = await agent.batch_analyze_symbols(batch_data)
        successful_batch = sum(1 for r in batch_results if r.get('success', False))
        print(f"✅ Batch analysis completed: {successful_batch}/3 successful")
        
        # Performance report
        report = agent.get_performance_report()
        print(f"📊 Performance: {report['success_rate']:.1%} success, {report['cache_statistics']['hit_rate']:.1%} cache hit rate")
        
        # Health check
        health = manager.get_health_status()
        print(f"🏥 Health: {'Healthy' if health['healthy'] else 'Unhealthy'}")
        
    finally:
        # Graceful shutdown
        agent.shutdown()

if __name__ == "__main__":
    asyncio.run(main())