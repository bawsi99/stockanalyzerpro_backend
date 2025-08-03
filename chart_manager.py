import os
import shutil
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import threading
import tempfile

logger = logging.getLogger(__name__)

# Import storage configuration
try:
    from storage_config import StorageConfig
except ImportError:
    # Fallback for when storage_config is not available
    StorageConfig = None

class ChartManager:
    """
    Manages chart lifecycle, storage, and cleanup for deployment environments.
    
    Features:
    - Automatic cleanup of old charts
    - Temporary storage with expiration
    - Memory-efficient chart handling
    - Configurable retention policies
    """
    
    def __init__(self, 
                 base_output_dir: str = None,
                 max_age_hours: int = 24,
                 max_total_size_mb: int = 1000,
                 cleanup_interval_minutes: int = 60,
                 enable_cleanup: bool = True):
        """
        Initialize chart manager.
        
        Args:
            base_output_dir: Base directory for chart storage (auto-detected if None)
            max_age_hours: Maximum age of charts before cleanup (hours)
            max_total_size_mb: Maximum total size of chart directory (MB)
            cleanup_interval_minutes: How often to run cleanup (minutes)
            enable_cleanup: Whether to enable automatic cleanup
        """
        # Use storage config if available, otherwise use provided path
        if base_output_dir is None and StorageConfig is not None:
            base_output_dir = StorageConfig.get_charts_path()
        elif base_output_dir is None:
            base_output_dir = "./output/charts"
        
        self.base_output_dir = Path(base_output_dir)
        self.max_age_hours = max_age_hours
        self.max_total_size_mb = max_total_size_mb
        self.cleanup_interval_minutes = cleanup_interval_minutes
        self.enable_cleanup = enable_cleanup
        
        # Ensure base directory exists
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Start cleanup thread only if enabled
        if self.enable_cleanup:
            self._start_cleanup_thread()
            logger.info(f"ChartManager initialized: {base_output_dir}, max_age={max_age_hours}h, max_size={max_total_size_mb}MB, cleanup=enabled")
        else:
            logger.info(f"ChartManager initialized: {base_output_dir}, max_age={max_age_hours}h, max_size={max_total_size_mb}MB, cleanup=disabled")
    
    def get_chart_directory(self, symbol: str, interval: str) -> Path:
        """Get the directory path for a specific symbol and interval."""
        return self.base_output_dir / f"{symbol}_{interval}"
    
    def create_chart_directory(self, symbol: str, interval: str) -> Path:
        """Create and return chart directory for a symbol and interval."""
        chart_dir = self.get_chart_directory(symbol, interval)
        chart_dir.mkdir(parents=True, exist_ok=True)
        return chart_dir
    
    def cleanup_old_charts(self) -> Dict[str, int]:
        """
        Clean up old charts based on age and size limits.
        
        Returns:
            Dict with cleanup statistics
        """
        stats = {
            'files_removed': 0,
            'directories_removed': 0,
            'bytes_freed': 0,
            'errors': 0
        }
        
        # Check if cleanup is enabled
        if not self.enable_cleanup:
            logger.info("Chart cleanup is disabled")
            return stats
        
        try:
            current_time = time.time()
            max_age_seconds = self.max_age_hours * 3600
            
            # Clean up by age
            for item in self.base_output_dir.rglob('*'):
                if item.is_file():
                    try:
                        file_age = current_time - item.stat().st_mtime
                        if file_age > max_age_seconds:
                            file_size = item.stat().st_size
                            item.unlink()
                            stats['files_removed'] += 1
                            stats['bytes_freed'] += file_size
                            logger.debug(f"Removed old chart file: {item}")
                    except Exception as e:
                        stats['errors'] += 1
                        logger.warning(f"Error removing file {item}: {e}")
                
                elif item.is_dir() and item != self.base_output_dir:
                    try:
                        # Remove empty directories
                        if not any(item.iterdir()):
                            item.rmdir()
                            stats['directories_removed'] += 1
                            logger.debug(f"Removed empty directory: {item}")
                    except Exception as e:
                        stats['errors'] += 1
                        logger.warning(f"Error removing directory {item}: {e}")
            
            # Clean up by size if needed
            total_size = self._get_directory_size(self.base_output_dir)
            if total_size > (self.max_total_size_mb * 1024 * 1024):
                self._cleanup_by_size(stats)
            
            if stats['files_removed'] > 0 or stats['directories_removed'] > 0:
                logger.info(f"Chart cleanup completed: {stats}")
            
        except Exception as e:
            logger.error(f"Error during chart cleanup: {e}")
            stats['errors'] += 1
        
        return stats
    
    def _cleanup_by_size(self, stats: Dict[str, int]) -> None:
        """Clean up charts by removing oldest files when size limit is exceeded."""
        try:
            # Get all files with their modification times
            files_with_time = []
            for file_path in self.base_output_dir.rglob('*.png'):
                try:
                    mtime = file_path.stat().st_mtime
                    size = file_path.stat().st_size
                    files_with_time.append((file_path, mtime, size))
                except Exception:
                    continue
            
            # Sort by modification time (oldest first)
            files_with_time.sort(key=lambda x: x[1])
            
            # Remove oldest files until under size limit
            target_size = self.max_total_size_mb * 1024 * 1024
            current_size = sum(size for _, _, size in files_with_time)
            
            for file_path, _, size in files_with_time:
                if current_size <= target_size:
                    break
                
                try:
                    file_path.unlink()
                    current_size -= size
                    stats['files_removed'] += 1
                    stats['bytes_freed'] += size
                    logger.debug(f"Removed file for size cleanup: {file_path}")
                except Exception as e:
                    stats['errors'] += 1
                    logger.warning(f"Error removing file {file_path}: {e}")
        
        except Exception as e:
            logger.error(f"Error during size-based cleanup: {e}")
            stats['errors'] += 1
    
    def _get_directory_size(self, directory: Path) -> int:
        """Calculate total size of directory in bytes."""
        total_size = 0
        try:
            for file_path in directory.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
        except Exception as e:
            logger.warning(f"Error calculating directory size: {e}")
        return total_size
    
    def _start_cleanup_thread(self) -> None:
        """Start background cleanup thread."""
        def cleanup_worker():
            while True:
                try:
                    time.sleep(self.cleanup_interval_minutes * 60)
                    self.cleanup_old_charts()
                except Exception as e:
                    logger.error(f"Error in cleanup worker: {e}")
        
        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()
        logger.info("Chart cleanup thread started")
    
    def get_storage_stats(self) -> Dict[str, any]:
        """Get current storage statistics."""
        try:
            total_files = 0
            total_size = 0
            oldest_file_age = None
            newest_file_age = None
            
            for file_path in self.base_output_dir.rglob('*.png'):
                if file_path.is_file():
                    total_files += 1
                    file_size = file_path.stat().st_size
                    total_size += file_size
                    
                    file_age = time.time() - file_path.stat().st_mtime
                    if oldest_file_age is None or file_age > oldest_file_age:
                        oldest_file_age = file_age
                    if newest_file_age is None or file_age < newest_file_age:
                        newest_file_age = file_age
            
            return {
                'total_files': total_files,
                'total_size_mb': round(total_size / (1024 * 1024), 2),
                'oldest_file_age_hours': round(oldest_file_age / 3600, 2) if oldest_file_age else None,
                'newest_file_age_hours': round(newest_file_age / 3600, 2) if newest_file_age else None,
                'max_age_hours': self.max_age_hours,
                'max_size_mb': self.max_total_size_mb
            }
        except Exception as e:
            logger.error(f"Error getting storage stats: {e}")
            return {'error': str(e)}
    
    def cleanup_specific_charts(self, symbol: str, interval: str) -> bool:
        """Clean up charts for a specific symbol and interval."""
        try:
            chart_dir = self.get_chart_directory(symbol, interval)
            if chart_dir.exists():
                shutil.rmtree(chart_dir)
                logger.info(f"Cleaned up charts for {symbol}_{interval}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error cleaning up charts for {symbol}_{interval}: {e}")
            return False
    
    def cleanup_all_charts(self) -> Dict[str, int]:
        """Clean up all charts in the output directory."""
        stats = {'files_removed': 0, 'directories_removed': 0, 'bytes_freed': 0, 'errors': 0}
        
        try:
            for item in self.base_output_dir.iterdir():
                if item.is_dir():
                    try:
                        shutil.rmtree(item)
                        stats['directories_removed'] += 1
                        logger.info(f"Removed directory: {item}")
                    except Exception as e:
                        stats['errors'] += 1
                        logger.warning(f"Error removing directory {item}: {e}")
                elif item.is_file():
                    try:
                        file_size = item.stat().st_size
                        item.unlink()
                        stats['files_removed'] += 1
                        stats['bytes_freed'] += file_size
                        logger.info(f"Removed file: {item}")
                    except Exception as e:
                        stats['errors'] += 1
                        logger.warning(f"Error removing file {item}: {e}")
        
        except Exception as e:
            logger.error(f"Error during full cleanup: {e}")
            stats['errors'] += 1
        
        return stats

# Global chart manager instance
_chart_manager: Optional[ChartManager] = None

def get_chart_manager() -> ChartManager:
    """Get or create the global chart manager instance."""
    global _chart_manager
    if _chart_manager is None:
        _chart_manager = ChartManager()
    return _chart_manager

def initialize_chart_manager(**kwargs) -> ChartManager:
    """Initialize the global chart manager with custom settings."""
    global _chart_manager
    _chart_manager = ChartManager(**kwargs)
    return _chart_manager 