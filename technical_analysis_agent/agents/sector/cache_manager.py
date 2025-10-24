#!/usr/bin/env python3
"""
Sector Cache Manager

Manages caching of sector analysis results to avoid redundant calculations.
Implements refresh logic based on time and price changes.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import pandas as pd
import pytz

logger = logging.getLogger(__name__)


class SectorCacheManager:
    """Manages caching of sector analysis results."""
    
    def __init__(self, cache_dir: Optional[Path] = None, config_path: Optional[Path] = None):
        """
        Initialize the cache manager.
        
        Args:
            cache_dir: Path to cache directory (defaults to ./cache)
            config_path: Path to config file (defaults to ./cache_config.json)
        """
        # Setup paths
        self.base_dir = Path(__file__).parent
        self.cache_dir = cache_dir or (self.base_dir / 'cache')
        self.manifest_file = self.cache_dir / 'sector_cache_manifest.json'
        
        # Load configuration
        self.config_path = config_path or (self.base_dir / 'cache_config.json')
        self.config = self._load_config()
        
        # Extract config values
        self.cache_enabled = self.config.get('cache_enabled', True)
        self.price_threshold = self.config.get('price_change_threshold_pct', 2.5)
        self.refresh_days = self.config.get('refresh_interval_days', 7)
        self.sectors = self.config.get('sectors', [])
        
        # Setup timezone
        self.timezone = pytz.timezone(self.config['market_hours_ist']['timezone'])
        
        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize manifest if it doesn't exist
        if not self.manifest_file.exists():
            self._initialize_manifest()
        
        logger.info(f"SectorCacheManager initialized - cache_enabled={self.cache_enabled}, "
                   f"refresh_days={self.refresh_days}, price_threshold={self.price_threshold}%")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            else:
                logger.warning(f"Config file not found: {self.config_path}. Using defaults.")
                return self._get_default_config()
        except Exception as e:
            logger.error(f"Error loading config: {e}. Using defaults.")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'cache_enabled': True,
            'refresh_interval_days': 7,
            'price_change_threshold_pct': 2.5,
            'check_interval_hours': 1,
            'market_hours_ist': {
                'start': '09:15',
                'end': '15:30',
                'timezone': 'Asia/Kolkata'
            },
            'sectors': [],
            'cache_version': '1.0'
        }
    
    def _initialize_manifest(self):
        """Initialize empty manifest file."""
        manifest = {
            'last_updated': datetime.now(self.timezone).isoformat(),
            'sectors': {},
            'config': {
                'refresh_interval_days': self.refresh_days,
                'price_change_threshold_pct': self.price_threshold,
                'market_hours_ist': self.config['market_hours_ist'],
                'check_interval_hours': self.config.get('check_interval_hours', 1)
            }
        }
        self._save_manifest(manifest)
        logger.info("Initialized new cache manifest")
    
    def _load_manifest(self) -> Dict[str, Any]:
        """Load manifest from disk."""
        try:
            if self.manifest_file.exists():
                with open(self.manifest_file, 'r') as f:
                    return json.load(f)
            else:
                self._initialize_manifest()
                return self._load_manifest()
        except Exception as e:
            logger.error(f"Error loading manifest: {e}")
            return {'sectors': {}, 'last_updated': datetime.now(self.timezone).isoformat(), 'config': {}}
    
    def _save_manifest(self, manifest: Dict[str, Any]):
        """Save manifest to disk."""
        try:
            manifest['last_updated'] = datetime.now(self.timezone).isoformat()
            with open(self.manifest_file, 'w') as f:
                json.dump(manifest, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving manifest: {e}")
    
    def _get_sector_dir(self, sector: str) -> Path:
        """Get directory path for a sector."""
        sector_dir = self.cache_dir / sector
        sector_dir.mkdir(parents=True, exist_ok=True)
        return sector_dir
    
    def get_cached_analysis(self, sector: str) -> Optional[Dict[str, Any]]:
        """
        Get cached analysis for a sector.
        
        Args:
            sector: Sector identifier (e.g., 'NIFTY_BANK')
            
        Returns:
            Cached analysis dict or None if not available
        """
        if not self.cache_enabled:
            logger.debug(f"Cache disabled, skipping cache lookup for {sector}")
            return None
        
        try:
            sector_dir = self._get_sector_dir(sector)
            analysis_file = sector_dir / 'analysis.json'
            metadata_file = sector_dir / 'metadata.json'
            
            if not analysis_file.exists() or not metadata_file.exists():
                logger.debug(f"Cache miss for {sector}: files not found")
                return None
            
            # Load metadata
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Check if cache is valid
            should_refresh, reason = self.should_refresh(sector)
            if should_refresh:
                logger.info(f"Cache invalid for {sector}: {reason}")
                return None
            
            # Load analysis
            with open(analysis_file, 'r') as f:
                analysis = json.load(f)
            
            # Calculate age
            analysis_time = datetime.fromisoformat(metadata['analysis_timestamp'])
            now = datetime.now(self.timezone)
            age_hours = (now - analysis_time).total_seconds() / 3600
            
            analysis['cache_metadata'] = {
                'age_hours': round(age_hours, 2),
                'cached_at': metadata['analysis_timestamp'],
                'is_cached': True,
                'status': metadata.get('status', 'valid')
            }
            
            logger.info(f"Cache hit for {sector} (age: {age_hours:.1f}h)")
            return analysis
            
        except Exception as e:
            logger.error(f"Error reading cache for {sector}: {e}")
            return None
    
    def save_analysis(self, sector: str, analysis: Dict[str, Any], current_price: float):
        """
        Save sector analysis to cache.
        
        Args:
            sector: Sector identifier
            analysis: Analysis results to cache
            current_price: Current sector index price
        """
        if not self.cache_enabled:
            logger.debug(f"Cache disabled, skipping save for {sector}")
            return
        
        try:
            sector_dir = self._get_sector_dir(sector)
            analysis_file = sector_dir / 'analysis.json'
            metadata_file = sector_dir / 'metadata.json'
            
            # Save analysis
            with open(analysis_file, 'w') as f:
                json.dump(analysis, f, indent=2)
            
            # Create metadata
            now = datetime.now(self.timezone)
            next_refresh = now + timedelta(days=self.refresh_days)
            
            metadata = {
                'sector': sector,
                'sector_index': analysis.get('sector_info', {}).get('sector_index', 'UNKNOWN'),
                'analysis_timestamp': now.isoformat(),
                'price_at_analysis': current_price,
                'last_price_check': now.isoformat(),
                'current_price': current_price,
                'price_change_pct': 0.0,
                'analysis_period': analysis.get('data_points', {}).get('sector_data_points', 30),
                'data_points': analysis.get('data_points', {}).get('sector_data_points', 0),
                'cache_version': self.config.get('cache_version', '1.0'),
                'status': 'valid',
                'invalidation_reason': None,
                'next_scheduled_refresh': next_refresh.isoformat()
            }
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Update manifest
            manifest = self._load_manifest()
            manifest['sectors'][sector] = {
                'last_analysis': now.isoformat(),
                'last_price': current_price,
                'price_at_analysis': current_price,
                'status': 'valid',
                'cache_file': f'{sector}/analysis.json',
                'next_refresh': next_refresh.isoformat()
            }
            self._save_manifest(manifest)
            
            logger.info(f"Cached analysis for {sector} (price: {current_price})")
            
        except Exception as e:
            logger.error(f"Error saving cache for {sector}: {e}")
    
    def should_refresh(self, sector: str) -> Tuple[bool, str]:
        """
        Check if sector analysis needs refresh.
        
        Args:
            sector: Sector identifier
            
        Returns:
            Tuple of (should_refresh: bool, reason: str)
        """
        try:
            sector_dir = self._get_sector_dir(sector)
            metadata_file = sector_dir / 'metadata.json'
            analysis_file = sector_dir / 'analysis.json'
            
            # Check if cache exists
            if not metadata_file.exists() or not analysis_file.exists():
                return (True, 'cache_missing')
            
            # Load metadata
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Check if manually invalidated
            if metadata.get('status') == 'invalid':
                reason = metadata.get('invalidation_reason', 'manually_invalidated')
                return (True, reason)
            
            # Check cache age (weekly refresh)
            analysis_time = datetime.fromisoformat(metadata['analysis_timestamp'])
            now = datetime.now(self.timezone)
            age_days = (now - analysis_time).days
            
            if age_days >= self.refresh_days:
                return (True, f'cache_expired_age_{age_days}d')
            
            # Check if within next scheduled refresh window
            next_refresh = datetime.fromisoformat(metadata['next_scheduled_refresh'])
            if now >= next_refresh:
                return (True, 'scheduled_refresh')
            
            # Check price change (if we have current price)
            if 'current_price' in metadata and 'price_at_analysis' in metadata:
                current_price = metadata['current_price']
                price_at_analysis = metadata['price_at_analysis']
                
                if price_at_analysis and price_at_analysis > 0:
                    change_pct = abs((current_price - price_at_analysis) / price_at_analysis * 100)
                    
                    if change_pct >= self.price_threshold:
                        return (True, f'price_changed_{change_pct:.2f}%')
            
            # Cache is valid
            return (False, 'cache_valid')
            
        except Exception as e:
            logger.error(f"Error checking refresh status for {sector}: {e}")
            return (True, f'error_{str(e)}')
    
    def update_sector_price(self, sector: str, current_price: float):
        """
        Update current price for a sector in metadata.
        
        Args:
            sector: Sector identifier
            current_price: Current sector index price
        """
        try:
            sector_dir = self._get_sector_dir(sector)
            metadata_file = sector_dir / 'metadata.json'
            
            if not metadata_file.exists():
                logger.warning(f"Cannot update price for {sector}: metadata not found")
                return
            
            # Load and update metadata
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            old_price = metadata.get('current_price', current_price)
            price_at_analysis = metadata.get('price_at_analysis', current_price)
            
            metadata['current_price'] = current_price
            metadata['last_price_check'] = datetime.now(self.timezone).isoformat()
            
            # Calculate price change
            if price_at_analysis and price_at_analysis > 0:
                change_pct = ((current_price - price_at_analysis) / price_at_analysis) * 100
                metadata['price_change_pct'] = round(change_pct, 2)
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Update manifest
            manifest = self._load_manifest()
            if sector in manifest['sectors']:
                manifest['sectors'][sector]['last_price'] = current_price
                self._save_manifest(manifest)
            
            logger.debug(f"Updated price for {sector}: {old_price} -> {current_price}")
            
        except Exception as e:
            logger.error(f"Error updating price for {sector}: {e}")
    
    def invalidate_sector(self, sector: str, reason: str = 'manual_invalidation'):
        """
        Mark sector cache as invalid.
        
        Args:
            sector: Sector identifier
            reason: Reason for invalidation
        """
        try:
            sector_dir = self._get_sector_dir(sector)
            metadata_file = sector_dir / 'metadata.json'
            
            if not metadata_file.exists():
                logger.warning(f"Cannot invalidate {sector}: metadata not found")
                return
            
            # Update metadata
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            metadata['status'] = 'invalid'
            metadata['invalidation_reason'] = reason
            metadata['invalidated_at'] = datetime.now(self.timezone).isoformat()
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Update manifest
            manifest = self._load_manifest()
            if sector in manifest['sectors']:
                manifest['sectors'][sector]['status'] = 'invalid'
                self._save_manifest(manifest)
            
            logger.info(f"Invalidated cache for {sector}: {reason}")
            
        except Exception as e:
            logger.error(f"Error invalidating cache for {sector}: {e}")
    
    def get_cache_status(self) -> Dict[str, Any]:
        """
        Get status of all cached sectors.
        
        Returns:
            Dict with cache status for all sectors
        """
        try:
            manifest = self._load_manifest()
            now = datetime.now(self.timezone)
            
            status = {
                'cache_enabled': self.cache_enabled,
                'total_sectors': len(self.sectors),
                'cached_sectors': len(manifest.get('sectors', {})),
                'valid_caches': 0,
                'invalid_caches': 0,
                'sectors': {}
            }
            
            for sector in self.sectors:
                sector_dir = self._get_sector_dir(sector)
                metadata_file = sector_dir / 'metadata.json'
                
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    analysis_time = datetime.fromisoformat(metadata['analysis_timestamp'])
                    age_hours = (now - analysis_time).total_seconds() / 3600
                    
                    should_refresh, reason = self.should_refresh(sector)
                    cache_status = 'invalid' if should_refresh else 'valid'
                    
                    if cache_status == 'valid':
                        status['valid_caches'] += 1
                    else:
                        status['invalid_caches'] += 1
                    
                    status['sectors'][sector] = {
                        'status': cache_status,
                        'age_hours': round(age_hours, 2),
                        'price_change_pct': metadata.get('price_change_pct', 0.0),
                        'next_refresh': metadata.get('next_scheduled_refresh'),
                        'refresh_reason': reason if should_refresh else None
                    }
                else:
                    status['sectors'][sector] = {
                        'status': 'not_cached',
                        'age_hours': None,
                        'price_change_pct': None,
                        'next_refresh': None,
                        'refresh_reason': 'cache_missing'
                    }
                    status['invalid_caches'] += 1
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting cache status: {e}")
            return {
                'cache_enabled': self.cache_enabled,
                'error': str(e)
            }
    
    def clear_cache(self, sector: Optional[str] = None):
        """
        Clear cache for a specific sector or all sectors.
        
        Args:
            sector: Sector to clear (None = clear all)
        """
        try:
            if sector:
                sector_dir = self._get_sector_dir(sector)
                for file in sector_dir.glob('*.json'):
                    file.unlink()
                logger.info(f"Cleared cache for {sector}")
            else:
                for sector_dir in self.cache_dir.iterdir():
                    if sector_dir.is_dir():
                        for file in sector_dir.glob('*.json'):
                            file.unlink()
                self._initialize_manifest()
                logger.info("Cleared all sector caches")
                
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
