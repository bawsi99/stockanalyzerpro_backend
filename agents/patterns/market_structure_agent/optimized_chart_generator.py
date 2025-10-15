#!/usr/bin/env python3
"""
Performance-Optimized Market Structure Chart Generator

This module adds comprehensive performance optimizations:
- Adaptive chart quality based on use case (LLM, display, archive)
- Image compression for LLM without quality loss
- Chart caching with data fingerprinting
- Async/concurrent generation for multiple stocks
- Memory optimization and resource management
- Lazy loading and streaming generation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional, Union
import os
import logging
from pathlib import Path
import json
import hashlib
import asyncio
import concurrent.futures
from dataclasses import dataclass, asdict
from enum import Enum
import time
import pickle
import gzip
import io
from PIL import Image
import threading
from functools import lru_cache, wraps
import weakref
import gc

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChartQuality(Enum):
    """Chart quality levels for different use cases"""
    LLM_OPTIMIZED = "llm_optimized"      # Optimized for LLM analysis (balanced size/quality)
    DISPLAY = "display"                  # For web/app display (medium quality)
    ARCHIVE = "archive"                  # High quality for archival (large files)
    THUMBNAIL = "thumbnail"              # Small thumbnails (low quality)

class CompressionLevel(Enum):
    """Image compression levels"""
    NONE = 0
    LOW = 1
    MEDIUM = 2  
    HIGH = 3

@dataclass
class ChartConfig:
    """Chart generation configuration"""
    quality: ChartQuality = ChartQuality.LLM_OPTIMIZED
    compression: CompressionLevel = CompressionLevel.MEDIUM
    dpi: int = 200
    width: int = 16
    height: int = 12
    max_file_size_mb: float = 5.0
    enable_caching: bool = True
    cache_ttl_hours: int = 24
    
    def get_optimization_params(self) -> Dict[str, Any]:
        """Get optimization parameters based on quality setting"""
        
        if self.quality == ChartQuality.LLM_OPTIMIZED:
            return {
                'dpi': 200,
                'figsize': (14, 10),
                'compression_quality': 85,
                'optimize_png': True,
                'max_file_size_mb': 3.0
            }
        elif self.quality == ChartQuality.DISPLAY:
            return {
                'dpi': 150,
                'figsize': (12, 9),
                'compression_quality': 80,
                'optimize_png': True,
                'max_file_size_mb': 2.0
            }
        elif self.quality == ChartQuality.ARCHIVE:
            return {
                'dpi': 300,
                'figsize': (20, 16),
                'compression_quality': 95,
                'optimize_png': False,
                'max_file_size_mb': 10.0
            }
        else:  # THUMBNAIL
            return {
                'dpi': 100,
                'figsize': (8, 6),
                'compression_quality': 75,
                'optimize_png': True,
                'max_file_size_mb': 1.0
            }

@dataclass
class CacheEntry:
    """Cache entry for chart data"""
    data_hash: str
    chart_path: str
    created_at: datetime
    access_count: int = 0
    file_size: int = 0

class MemoryManager:
    """Manages memory usage and cleanup"""
    
    def __init__(self, max_memory_mb: float = 500):
        self.max_memory_mb = max_memory_mb
        self.figure_cache = weakref.WeakSet()
        
    def cleanup_figures(self):
        """Force cleanup of matplotlib figures"""
        plt.close('all')
        gc.collect()
        
    def monitor_memory(self):
        """Monitor memory usage and cleanup if needed"""
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            if memory_mb > self.max_memory_mb:
                logger.warning(f"High memory usage: {memory_mb:.1f}MB, cleaning up...")
                self.cleanup_figures()
                gc.collect()
                
        except ImportError:
            # Fallback without psutil
            self.cleanup_figures()

class ChartCache:
    """Intelligent caching system for generated charts"""
    
    def __init__(self, cache_dir: str = "chart_cache", max_size_gb: float = 2.0):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_size_gb = max_size_gb
        self.index_file = self.cache_dir / "cache_index.json"
        self.cache_index = self._load_cache_index()
        self._lock = threading.Lock()
        
    def _load_cache_index(self) -> Dict[str, CacheEntry]:
        """Load cache index from disk"""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r') as f:
                    data = json.load(f)
                    return {k: CacheEntry(**v) for k, v in data.items()}
            except Exception as e:
                logger.warning(f"Failed to load cache index: {e}")
        return {}
    
    def _save_cache_index(self):
        """Save cache index to disk"""
        try:
            with open(self.index_file, 'w') as f:
                data = {k: asdict(v) for k, v in self.cache_index.items()}
                # Convert datetime to string for JSON serialization
                for entry in data.values():
                    entry['created_at'] = entry['created_at'].isoformat()
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save cache index: {e}")
    
    def _calculate_data_hash(self, stock_data: pd.DataFrame, analysis_data: Dict, config: ChartConfig) -> str:
        """Calculate hash of input data for caching"""
        try:
            # Create hash from data and config
            data_str = f"{len(stock_data)}_{stock_data['close'].iloc[0]:.2f}_{stock_data['close'].iloc[-1]:.2f}"
            analysis_str = str(len(analysis_data.get('swing_points', {}).get('swing_highs', [])))
            config_str = f"{config.quality.value}_{config.dpi}_{config.width}x{config.height}"
            
            combined = f"{data_str}_{analysis_str}_{config_str}"
            return hashlib.md5(combined.encode()).hexdigest()
        except Exception:
            # Fallback to timestamp-based hash if calculation fails
            return hashlib.md5(str(time.time()).encode()).hexdigest()
    
    def get_cached_chart(self, data_hash: str, ttl_hours: int = 24) -> Optional[str]:
        """Get cached chart if available and not expired"""
        with self._lock:
            entry = self.cache_index.get(data_hash)
            if not entry:
                return None
                
            # Check if expired
            age_hours = (datetime.now() - entry.created_at).total_seconds() / 3600
            if age_hours > ttl_hours:
                logger.debug(f"Cache entry expired: {age_hours:.1f}h > {ttl_hours}h")
                self._remove_cache_entry(data_hash)
                return None
            
            # Check if file still exists
            if not os.path.exists(entry.chart_path):
                logger.warning(f"Cached file missing: {entry.chart_path}")
                self._remove_cache_entry(data_hash)
                return None
            
            # Update access count
            entry.access_count += 1
            logger.debug(f"Cache hit: {entry.chart_path} (accessed {entry.access_count} times)")
            return entry.chart_path
    
    def cache_chart(self, data_hash: str, chart_path: str) -> bool:
        """Cache a generated chart"""
        with self._lock:
            try:
                file_size = os.path.getsize(chart_path)
                entry = CacheEntry(
                    data_hash=data_hash,
                    chart_path=chart_path,
                    created_at=datetime.now(),
                    file_size=file_size
                )
                
                self.cache_index[data_hash] = entry
                self._save_cache_index()
                
                # Check cache size and cleanup if needed
                self._cleanup_cache_if_needed()
                
                logger.debug(f"Cached chart: {chart_path} ({file_size/1024:.1f} KB)")
                return True
                
            except Exception as e:
                logger.warning(f"Failed to cache chart: {e}")
                return False
    
    def _remove_cache_entry(self, data_hash: str):
        """Remove cache entry"""
        if data_hash in self.cache_index:
            entry = self.cache_index[data_hash]
            try:
                if os.path.exists(entry.chart_path):
                    os.remove(entry.chart_path)
            except Exception as e:
                logger.warning(f"Failed to remove cached file: {e}")
            del self.cache_index[data_hash]
    
    def _cleanup_cache_if_needed(self):
        """Clean up cache if it exceeds size limit"""
        total_size_gb = sum(entry.file_size for entry in self.cache_index.values()) / 1024**3
        
        if total_size_gb > self.max_size_gb:
            logger.info(f"Cache size {total_size_gb:.2f}GB exceeds limit, cleaning up...")
            
            # Remove oldest, least accessed entries
            sorted_entries = sorted(
                self.cache_index.items(),
                key=lambda x: (x[1].access_count, x[1].created_at)
            )
            
            # Remove bottom 25%
            to_remove = sorted_entries[:len(sorted_entries)//4]
            for data_hash, _ in to_remove:
                self._remove_cache_entry(data_hash)
                
            self._save_cache_index()
            logger.info(f"Removed {len(to_remove)} cache entries")

class OptimizedMarketStructureCharts:
    """
    Performance-optimized chart generator with adaptive quality and intelligent caching.
    """
    
    def __init__(self, output_dir: str = "optimized_charts", config: Optional[ChartConfig] = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.config = config or ChartConfig()
        
        # Initialize components
        self.memory_manager = MemoryManager()
        self.cache = ChartCache(str(self.output_dir / "cache")) if self.config.enable_caching else None
        
        # Performance tracking
        self.perf_stats = {
            'total_generated': 0,
            'cache_hits': 0,
            'generation_times': [],
            'compression_ratios': [],
            'memory_cleanups': 0
        }
        
        # Thread pool for concurrent operations
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        
        # Enhanced color scheme (optimized for compression)
        self.colors = {
            'price': '#1f77b4',
            'price_fill': '#1f77b4',
            'swing_high': '#ff4444', 
            'swing_low': '#44ff44',
            'bos_bullish': '#00aa00',
            'bos_bearish': '#aa0000',
            'support': '#44ff44',
            'resistance': '#ff4444',
            'trend_up': '#00aa00',
            'trend_down': '#aa0000',
            'neutral': '#888888',
            'volume_up': '#2ca02c',
            'volume_down': '#d62728',
            'fibonacci': '#ffa500',
            'trend_channel': '#9370db'
        }
        
        plt.style.use('seaborn-v0_8-darkgrid')
    
    def create_optimized_chart(self, 
                             stock_data: pd.DataFrame, 
                             analysis_data: Dict[str, Any], 
                             symbol: str, 
                             scenario: str,
                             config: Optional[ChartConfig] = None) -> Tuple[Optional[str], Dict[str, Any]]:
        """
        Create optimized chart with intelligent caching and adaptive quality.
        
        Returns:
            Tuple of (chart_path, performance_metrics)
        """
        start_time = time.time()
        config = config or self.config
        
        try:
            # Check cache first
            data_hash = None
            if self.cache:
                data_hash = self.cache._calculate_data_hash(stock_data, analysis_data, config)
                cached_path = self.cache.get_cached_chart(data_hash, config.cache_ttl_hours)
                
                if cached_path:
                    self.perf_stats['cache_hits'] += 1
                    generation_time = time.time() - start_time
                    
                    return cached_path, {
                        'cache_hit': True,
                        'generation_time': generation_time,
                        'file_size': os.path.getsize(cached_path),
                        'data_hash': data_hash
                    }
            
            # Generate new chart
            logger.info(f"Generating optimized chart for {symbol} - {scenario} ({config.quality.value})")
            
            chart_path = self._generate_chart_optimized(stock_data, analysis_data, symbol, scenario, config)
            
            if not chart_path:
                return None, {'error': 'Chart generation failed'}
            
            # Optimize the generated chart
            optimized_path = self._optimize_chart_file(chart_path, config)
            
            # Cache if enabled
            if self.cache and data_hash:
                self.cache.cache_chart(data_hash, optimized_path)
            
            # Update performance stats
            generation_time = time.time() - start_time
            file_size = os.path.getsize(optimized_path)
            
            self.perf_stats['total_generated'] += 1
            self.perf_stats['generation_times'].append(generation_time)
            
            # Memory cleanup if needed
            self.memory_manager.monitor_memory()
            
            return optimized_path, {
                'cache_hit': False,
                'generation_time': generation_time,
                'file_size': file_size,
                'data_hash': data_hash,
                'quality': config.quality.value
            }
            
        except Exception as e:
            logger.error(f"Optimized chart generation failed: {e}")
            return None, {'error': str(e)}
    
    def _generate_chart_optimized(self, stock_data: pd.DataFrame, analysis_data: Dict[str, Any], 
                                symbol: str, scenario: str, config: ChartConfig) -> Optional[str]:
        """Generate chart with optimized settings"""
        
        try:
            # Get optimization parameters
            params = config.get_optimization_params()
            
            # Create figure with optimized settings
            fig = plt.figure(figsize=params['figsize'], facecolor='white')
            
            if config.quality == ChartQuality.THUMBNAIL:
                # Simplified layout for thumbnails
                ax = fig.add_subplot(111)
                self._plot_thumbnail_chart(ax, stock_data, analysis_data, symbol)
            else:
                # Standard or enhanced layout
                gs = fig.add_gridspec(3, 2, height_ratios=[3, 1, 1], hspace=0.3, wspace=0.2)
                ax_main = fig.add_subplot(gs[0, :])
                ax_volume = fig.add_subplot(gs[1, :], sharex=ax_main)
                ax_summary = fig.add_subplot(gs[2, :])
                
                fig.suptitle(f'{symbol} - {scenario} ({config.quality.value})', 
                            fontsize=14, fontweight='bold')
                
                self._plot_optimized_price_action(ax_main, stock_data, analysis_data, config)
                self._plot_optimized_volume(ax_volume, stock_data, config)
                self._plot_optimized_summary(ax_summary, analysis_data, scenario, config)
                
                self._format_optimized_axes(ax_main, ax_volume, stock_data)
            
            # Save with optimized settings
            filename = f"{symbol}_{scenario}_{config.quality.value}.png"
            filepath = self.output_dir / filename
            
            # Use optimal saving parameters
            plt.savefig(filepath, dpi=params['dpi'], bbox_inches='tight', 
                       facecolor='white', edgecolor='none', 
                       pad_inches=0.1, format='png')
            plt.close(fig)
            
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Optimized chart generation failed: {e}")
            plt.close('all')  # Cleanup on error
            return None
    
    def _plot_thumbnail_chart(self, ax, stock_data: pd.DataFrame, analysis_data: Dict, symbol: str):
        """Plot minimal thumbnail chart"""
        
        dates = pd.to_datetime(stock_data.index)
        ax.plot(dates, stock_data['close'], color=self.colors['price'], linewidth=1.5)
        
        # Add only key swing points
        swing_points = analysis_data.get('swing_points', {})
        for swing in swing_points.get('swing_highs', [])[:3]:  # Max 3 points
            try:
                ax.scatter(pd.to_datetime(swing['date']), swing['price'], 
                         c=self.colors['swing_high'], s=20, marker='^', alpha=0.8)
            except:
                continue
                
        for swing in swing_points.get('swing_lows', [])[:3]:  # Max 3 points
            try:
                ax.scatter(pd.to_datetime(swing['date']), swing['price'], 
                         c=self.colors['swing_low'], s=20, marker='v', alpha=0.8)
            except:
                continue
        
        ax.set_title(symbol, fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Minimal formatting
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, fontsize=8)
    
    def _plot_optimized_price_action(self, ax, stock_data: pd.DataFrame, analysis_data: Dict, config: ChartConfig):
        """Plot price action with quality-based optimizations"""
        
        dates = pd.to_datetime(stock_data.index)
        
        # Adaptive line width based on quality
        line_width = 3 if config.quality == ChartQuality.ARCHIVE else 2
        
        ax.plot(dates, stock_data['close'], color=self.colors['price'], 
               linewidth=line_width, label='Close Price')
        
        # Fill between high and low (reduce alpha for compression)
        ax.fill_between(dates, stock_data['low'], stock_data['high'], 
                       alpha=0.05, color=self.colors['price_fill'])
        
        # Plot swing points with adaptive sizing
        swing_points = analysis_data.get('swing_points', {})
        max_swings = 20 if config.quality == ChartQuality.ARCHIVE else 10  # Limit for performance
        
        for swing in swing_points.get('swing_highs', [])[:max_swings]:
            try:
                size = 80 if config.quality == ChartQuality.ARCHIVE else 50
                ax.scatter(pd.to_datetime(swing['date']), swing['price'], 
                         c=self.colors['swing_high'], s=size, marker='^', 
                         alpha=0.8, edgecolors='black', linewidth=1)
            except:
                continue
        
        for swing in swing_points.get('swing_lows', [])[:max_swings]:
            try:
                size = 80 if config.quality == ChartQuality.ARCHIVE else 50
                ax.scatter(pd.to_datetime(swing['date']), swing['price'], 
                         c=self.colors['swing_low'], s=size, marker='v', 
                         alpha=0.8, edgecolors='black', linewidth=1)
            except:
                continue
        
        # Add basic support/resistance (limited for performance)
        key_levels = analysis_data.get('key_levels', {})
        for level in key_levels.get('support_levels', [])[-3:]:  # Only recent 3
            try:
                ax.axhline(y=level['level'], color=self.colors['support'], 
                          linestyle='--', alpha=0.5, linewidth=1)
            except:
                continue
                
        for level in key_levels.get('resistance_levels', [])[-3:]:
            try:
                ax.axhline(y=level['level'], color=self.colors['resistance'], 
                          linestyle='--', alpha=0.5, linewidth=1)
            except:
                continue
        
        ax.set_title('Price Action & Market Structure', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    def _plot_optimized_volume(self, ax, stock_data: pd.DataFrame, config: ChartConfig):
        """Plot volume with quality-based optimizations"""
        
        if 'volume' not in stock_data.columns:
            ax.text(0.5, 0.5, 'Volume Data Not Available', 
                   transform=ax.transAxes, ha='center', va='center', fontsize=10)
            return
        
        dates = pd.to_datetime(stock_data.index)
        volumes = stock_data['volume']
        
        # Simplified volume bars for better compression
        ax.bar(dates, volumes, alpha=0.6, color=self.colors['neutral'], width=0.8)
        
        # Add moving average only for higher quality
        if config.quality in [ChartQuality.ARCHIVE, ChartQuality.DISPLAY] and len(volumes) > 20:
            vol_ma = volumes.rolling(window=20).mean()
            ax.plot(dates, vol_ma, color='blue', linewidth=1.5, alpha=0.7, label='Vol MA(20)')
            ax.legend(fontsize=9)
        
        ax.set_title('Volume', fontsize=11)
        ax.ticklabel_format(style='plain', axis='y')
    
    def _plot_optimized_summary(self, ax, analysis_data: Dict, scenario: str, config: ChartConfig):
        """Plot summary with adaptive detail level"""
        
        ax.axis('off')
        
        # Extract key metrics
        swing_points = analysis_data.get('swing_points', {})
        total_swings = swing_points.get('total_swings', 0)
        
        trend_analysis = analysis_data.get('trend_analysis', {})
        trend_direction = trend_analysis.get('trend_direction', 'unknown')
        
        bos_choch = analysis_data.get('bos_choch_analysis', {})
        bos_count = len(bos_choch.get('bos_events', []))
        structural_bias = bos_choch.get('structural_bias', 'neutral')
        
        # Adaptive detail level
        if config.quality == ChartQuality.THUMBNAIL:
            summary_text = f"{trend_direction.title()} | {total_swings} swings"
        else:
            summary_text = f"""
MARKET STRUCTURE - {scenario.upper()} ({config.quality.value})

Trend: {trend_direction.title()} | Swings: {total_swings} | BOS: {bos_count}
Bias: {structural_bias.title()}
            """.strip()
        
        font_size = 8 if config.quality == ChartQuality.THUMBNAIL else 10
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
               fontsize=font_size, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))
    
    def _format_optimized_axes(self, ax_main, ax_volume, stock_data):
        """Optimized axis formatting"""
        
        # Simplified date formatting
        ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax_volume.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        
        # Price formatting
        ax_main.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:.1f}'))
        
        # Rotate labels
        plt.setp(ax_main.xaxis.get_majorticklabels(), rotation=45, fontsize=8)
        plt.setp(ax_volume.xaxis.get_majorticklabels(), rotation=45, fontsize=8)
        
        # Optimize grid
        ax_main.grid(True, alpha=0.2)
        ax_volume.grid(True, alpha=0.2)
    
    def _optimize_chart_file(self, chart_path: str, config: ChartConfig) -> str:
        """Optimize chart file size and quality"""
        
        try:
            params = config.get_optimization_params()
            
            if not params.get('optimize_png', False):
                return chart_path  # No optimization needed
            
            # Load and optimize image
            with Image.open(chart_path) as img:
                # Convert to RGB if needed (removes alpha channel for better compression)
                if img.mode in ('RGBA', 'LA'):
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                    img = background
                
                # Apply compression
                optimized_path = chart_path.replace('.png', '_optimized.png')
                
                # Save with optimization
                img.save(optimized_path, 'PNG', optimize=True, compress_level=6)
                
                # Check file size and quality
                original_size = os.path.getsize(chart_path)
                optimized_size = os.path.getsize(optimized_path)
                compression_ratio = optimized_size / original_size
                
                # Use optimized version if it's significantly smaller and within size limit
                max_size_bytes = params['max_file_size_mb'] * 1024 * 1024
                
                if optimized_size < max_size_bytes and compression_ratio < 0.9:
                    os.remove(chart_path)  # Remove original
                    os.rename(optimized_path, chart_path)  # Use optimized version
                    self.perf_stats['compression_ratios'].append(compression_ratio)
                    logger.debug(f"Optimized chart: {compression_ratio:.2f} compression ratio")
                else:
                    os.remove(optimized_path)  # Keep original
            
            return chart_path
            
        except Exception as e:
            logger.warning(f"Chart optimization failed: {e}")
            return chart_path
    
    async def create_charts_batch(self, 
                                chart_requests: List[Tuple[pd.DataFrame, Dict, str, str]],
                                config: Optional[ChartConfig] = None) -> List[Tuple[Optional[str], Dict]]:
        """Create multiple charts concurrently"""
        
        logger.info(f"Starting batch generation of {len(chart_requests)} charts")
        
        config = config or self.config
        
        # Create tasks for concurrent execution
        loop = asyncio.get_event_loop()
        tasks = []
        
        for stock_data, analysis_data, symbol, scenario in chart_requests:
            task = loop.run_in_executor(
                self.thread_pool,
                self.create_optimized_chart,
                stock_data, analysis_data, symbol, scenario, config
            )
            tasks.append(task)
        
        # Execute concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch chart {i} failed: {result}")
                processed_results.append((None, {'error': str(result)}))
            else:
                processed_results.append(result)
        
        logger.info(f"Batch generation completed: {len([r for r in processed_results if r[0]])} successful")
        return processed_results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        
        stats = self.perf_stats.copy()
        
        # Calculate averages
        if stats['generation_times']:
            stats['avg_generation_time'] = sum(stats['generation_times']) / len(stats['generation_times'])
            stats['max_generation_time'] = max(stats['generation_times'])
            stats['min_generation_time'] = min(stats['generation_times'])
        
        if stats['compression_ratios']:
            stats['avg_compression_ratio'] = sum(stats['compression_ratios']) / len(stats['compression_ratios'])
        
        # Cache statistics
        if self.cache:
            cache_stats = {
                'cache_entries': len(self.cache.cache_index),
                'cache_hit_rate': (stats['cache_hits'] / max(stats['total_generated'] + stats['cache_hits'], 1)) * 100
            }
            stats.update(cache_stats)
        
        return stats
    
    def cleanup_resources(self):
        """Clean up resources and caches"""
        
        logger.info("Cleaning up resources...")
        
        # Clean up matplotlib
        self.memory_manager.cleanup_figures()
        
        # Shutdown thread pool
        self.thread_pool.shutdown(wait=True)
        
        # Save cache index
        if self.cache:
            self.cache._save_cache_index()
        
        logger.info("Resource cleanup completed")
    
    def __del__(self):
        """Destructor to ensure cleanup"""
        try:
            self.cleanup_resources()
        except:
            pass


# Test functions
def test_optimized_charts():
    """Test optimized chart generation with different quality settings"""
    
    logger.info("Testing Optimized Market Structure Charts...")
    
    # Import mock data
    from test_chart_generation import create_mock_data_scenarios
    scenarios = create_mock_data_scenarios()
    
    # Test different quality settings
    quality_configs = [
        (ChartQuality.LLM_OPTIMIZED, "LLM Optimized"),
        (ChartQuality.DISPLAY, "Display Quality"), 
        (ChartQuality.THUMBNAIL, "Thumbnail"),
        (ChartQuality.ARCHIVE, "Archive Quality")
    ]
    
    for quality, description in quality_configs:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing {description}")
        logger.info(f"{'='*60}")
        
        config = ChartConfig(quality=quality, enable_caching=True)
        optimizer = OptimizedMarketStructureCharts(
            output_dir=f"optimized_test_{quality.value}",
            config=config
        )
        
        # Test with first scenario
        stock_data, analysis_data, symbol, scenario = scenarios[0]
        
        # Generate chart twice to test caching
        for attempt in range(2):
            chart_path, metrics = optimizer.create_optimized_chart(
                stock_data, analysis_data, symbol, scenario, config
            )
            
            if chart_path:
                cache_status = "CACHE HIT" if metrics.get('cache_hit') else "GENERATED"
                logger.info(f"✅ {cache_status}: {chart_path}")
                logger.info(f"📊 Time: {metrics.get('generation_time', 0):.2f}s, "
                           f"Size: {metrics.get('file_size', 0)/1024:.1f}KB")
            else:
                logger.error(f"❌ Failed: {metrics}")
        
        # Print performance stats
        perf_stats = optimizer.get_performance_stats()
        logger.info(f"📈 Performance: Avg time {perf_stats.get('avg_generation_time', 0):.2f}s, "
                   f"Cache hit rate: {perf_stats.get('cache_hit_rate', 0):.1f}%")
        
        # Cleanup
        optimizer.cleanup_resources()

async def test_batch_generation():
    """Test concurrent batch generation"""
    
    logger.info("\n" + "="*60)
    logger.info("Testing Batch Chart Generation")
    logger.info("="*60)
    
    from test_chart_generation import create_mock_data_scenarios
    scenarios = create_mock_data_scenarios()
    
    # Create batch requests
    batch_requests = [(stock_data, analysis_data, symbol, scenario) 
                     for stock_data, analysis_data, symbol, scenario in scenarios]
    
    config = ChartConfig(quality=ChartQuality.LLM_OPTIMIZED, enable_caching=False)
    optimizer = OptimizedMarketStructureCharts(output_dir="batch_test", config=config)
    
    # Time batch generation
    start_time = time.time()
    results = await optimizer.create_charts_batch(batch_requests, config)
    batch_time = time.time() - start_time
    
    # Analyze results
    successful = [r for r in results if r[0]]
    failed = [r for r in results if not r[0]]
    
    logger.info(f"📊 Batch Results:")
    logger.info(f"  Total requests: {len(batch_requests)}")
    logger.info(f"  Successful: {len(successful)}")
    logger.info(f"  Failed: {len(failed)}")
    logger.info(f"  Total time: {batch_time:.2f}s")
    logger.info(f"  Avg per chart: {batch_time/len(batch_requests):.2f}s")
    
    # Show successful charts
    for chart_path, metrics in successful:
        logger.info(f"✅ Generated: {Path(chart_path).name} ({metrics.get('file_size', 0)/1024:.1f}KB)")
    
    optimizer.cleanup_resources()

def main():
    """Main test function"""
    
    # Test individual optimized charts
    test_optimized_charts()
    
    # Test batch generation
    asyncio.run(test_batch_generation())
    
    logger.info("\n✨ All optimization tests completed!")

if __name__ == "__main__":
    main()