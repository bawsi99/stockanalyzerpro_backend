#!/usr/bin/env python3
"""
Sector Cache API Endpoints

Provides FastAPI endpoints for:
- Cache status monitoring
- Manual cache refresh
- Cache invalidation
- Price monitoring control
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import logging

from cache_manager import SectorCacheManager
from price_monitor import SectorPriceMonitor
from scheduler import SectorAnalysisScheduler

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/agents/sector/cache", tags=["Sector Cache"])

# Initialize cache components (singleton pattern)
cache_manager = None
price_monitor = None
scheduler = None


def get_cache_manager() -> SectorCacheManager:
    """Get or create cache manager instance."""
    global cache_manager
    if cache_manager is None:
        cache_manager = SectorCacheManager()
    return cache_manager


def get_price_monitor() -> SectorPriceMonitor:
    """Get or create price monitor instance."""
    global price_monitor
    if price_monitor is None:
        price_monitor = SectorPriceMonitor(get_cache_manager())
    return price_monitor


def get_scheduler() -> SectorAnalysisScheduler:
    """Get or create scheduler instance."""
    global scheduler
    if scheduler is None:
        scheduler = SectorAnalysisScheduler(get_cache_manager())
    return scheduler


# Pydantic models
class RefreshRequest(BaseModel):
    """Request model for manual refresh."""
    sector: Optional[str] = None  # If None, refresh all sectors
    force: bool = False  # Force refresh even if cache is valid


class InvalidateRequest(BaseModel):
    """Request model for cache invalidation."""
    sector: str
    reason: str = "manual_invalidation"


class CacheStatusResponse(BaseModel):
    """Response model for cache status."""
    cache_enabled: bool
    total_sectors: int
    cached_sectors: int
    valid_caches: int
    invalid_caches: int
    sectors: Dict[str, Any]


class RefreshResponse(BaseModel):
    """Response model for refresh operations."""
    success: bool
    sectors_refreshed: List[str]
    time_taken: Optional[float] = None
    message: str


# API Endpoints

@router.get("/status", response_model=CacheStatusResponse)
async def get_cache_status():
    """
    Get cache status for all sectors.
    
    Returns detailed information about each sector's cache state including:
    - Age of cached data
    - Price change since last analysis
    - Next scheduled refresh time
    - Whether refresh is needed
    """
    try:
        manager = get_cache_manager()
        status = manager.get_cache_status()
        return status
    except Exception as e:
        logger.error(f"Error getting cache status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{sector}")
async def get_sector_cache_status(sector: str):
    """
    Get cache status for a specific sector.
    
    Args:
        sector: Sector identifier (e.g., 'NIFTY_BANK')
    """
    try:
        manager = get_cache_manager()
        
        # Check if sector exists in config
        if sector not in manager.sectors:
            raise HTTPException(
                status_code=404,
                detail=f"Sector '{sector}' not found in configuration"
            )
        
        # Get full status and extract sector-specific info
        full_status = manager.get_cache_status()
        sector_status = full_status['sectors'].get(sector, {})
        
        # Get cached analysis if available
        cached = manager.get_cached_analysis(sector)
        
        return {
            'sector': sector,
            'cache_status': sector_status,
            'has_cached_analysis': cached is not None,
            'cache_metadata': cached.get('cache_metadata') if cached else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting cache status for {sector}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/refresh", response_model=RefreshResponse)
async def refresh_cache(request: RefreshRequest):
    """
    Manually trigger cache refresh for one or all sectors.
    
    Args:
        request: Refresh request with optional sector and force flag
    
    Returns:
        Refresh results including sectors refreshed and time taken
    """
    try:
        import time
        start_time = time.time()
        
        manager = get_cache_manager()
        sched = get_scheduler()
        
        sectors_to_refresh = []
        
        if request.sector:
            # Validate sector
            if request.sector not in manager.sectors:
                raise HTTPException(
                    status_code=404,
                    detail=f"Sector '{request.sector}' not found in configuration"
                )
            
            # Check if refresh is needed (unless forced)
            if not request.force:
                should_refresh, reason = manager.should_refresh(request.sector)
                if not should_refresh:
                    return RefreshResponse(
                        success=True,
                        sectors_refreshed=[],
                        time_taken=0.0,
                        message=f"Sector '{request.sector}' cache is valid, no refresh needed. Use force=true to refresh anyway."
                    )
            
            sectors_to_refresh = [request.sector]
        else:
            # Refresh all sectors
            sectors_to_refresh = manager.sectors
        
        # Trigger immediate refresh
        # Note: This will invalidate the cache. The actual refresh will happen
        # when the sector is next requested via the normal analysis flow.
        results = await sched.trigger_immediate_refresh(sectors=sectors_to_refresh)
        
        successful = [
            r['sector'] for r in results
            if isinstance(r, dict) and r.get('status') in ['success', 'invalidated']
        ]
        
        elapsed = time.time() - start_time
        
        return RefreshResponse(
            success=True,
            sectors_refreshed=successful,
            time_taken=round(elapsed, 2),
            message=f"Successfully refreshed {len(successful)} sector(s)"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error refreshing cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/invalidate")
async def invalidate_cache(request: InvalidateRequest):
    """
    Manually invalidate cache for a specific sector.
    
    Args:
        request: Invalidation request with sector and reason
    
    Returns:
        Invalidation result
    """
    try:
        manager = get_cache_manager()
        
        # Validate sector
        if request.sector not in manager.sectors:
            raise HTTPException(
                status_code=404,
                detail=f"Sector '{request.sector}' not found in configuration"
            )
        
        # Invalidate cache
        manager.invalidate_sector(request.sector, request.reason)
        
        return {
            'success': True,
            'sector': request.sector,
            'message': f"Cache invalidated for '{request.sector}'. Will refresh on next request."
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error invalidating cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/clear")
async def clear_cache(sector: Optional[str] = None):
    """
    Clear cache for a specific sector or all sectors.
    
    Args:
        sector: Optional sector identifier (if None, clears all caches)
    
    Returns:
        Clear operation result
    """
    try:
        manager = get_cache_manager()
        
        if sector:
            # Validate sector
            if sector not in manager.sectors:
                raise HTTPException(
                    status_code=404,
                    detail=f"Sector '{sector}' not found in configuration"
                )
            
            manager.clear_cache(sector)
            message = f"Cache cleared for '{sector}'"
        else:
            manager.clear_cache()
            message = "All sector caches cleared"
        
        return {
            'success': True,
            'message': message
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/config")
async def get_cache_config():
    """
    Get current cache configuration.
    
    Returns:
        Cache configuration settings
    """
    try:
        manager = get_cache_manager()
        return {
            'cache_enabled': manager.cache_enabled,
            'price_threshold_pct': manager.price_threshold,
            'refresh_days': manager.refresh_days,
            'total_sectors': len(manager.sectors),
            'sectors': manager.sectors,
            'config': manager.config
        }
        
    except Exception as e:
        logger.error(f"Error getting cache config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/schedule")
async def get_refresh_schedule():
    """
    Get information about scheduled refreshes.
    
    Returns:
        Schedule information including next refresh time
    """
    try:
        sched = get_scheduler()
        
        next_refresh = sched.get_next_refresh_datetime()
        seconds_until = sched.get_seconds_until_next_refresh()
        
        return {
            'enabled': True,
            'refresh_day': 'Sunday',
            'refresh_time': str(sched.refresh_time),
            'timezone': str(sched.timezone),
            'next_refresh': next_refresh.isoformat(),
            'seconds_until_refresh': round(seconds_until, 2),
            'hours_until_refresh': round(seconds_until / 3600, 2),
            'days_until_refresh': round(seconds_until / 86400, 2)
        }
        
    except Exception as e:
        logger.error(f"Error getting refresh schedule: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/price-check")
async def trigger_price_check():
    """
    Manually trigger a price check for all sectors.
    
    This will check current prices and trigger refresh for any sectors
    where the price has changed more than the threshold.
    
    Returns:
        Price check results
    """
    try:
        monitor = get_price_monitor()
        
        # Perform price check
        results = await monitor.check_once()
        
        # Count sectors that need refresh
        needs_refresh = sum(
            1 for r in results
            if isinstance(r, dict) and r.get('should_refresh', False)
        )
        
        return {
            'success': True,
            'sectors_checked': len(results),
            'sectors_need_refresh': needs_refresh,
            'is_market_hours': monitor.is_market_hours(),
            'is_trading_day': monitor.is_trading_day(),
            'results': results
        }
        
    except Exception as e:
        logger.error(f"Error triggering price check: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Health check endpoint
@router.get("/health")
async def cache_health():
    """
    Health check for cache system.
    
    Returns:
        Health status of cache components
    """
    try:
        manager = get_cache_manager()
        status = manager.get_cache_status()
        
        return {
            'status': 'healthy',
            'cache_manager': 'operational',
            'cache_enabled': manager.cache_enabled,
            'valid_caches': status.get('valid_caches', 0),
            'total_sectors': status.get('total_sectors', 0)
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            'status': 'unhealthy',
            'error': str(e)
        }


# Documentation
@router.get("/")
async def cache_api_info():
    """
    Get information about cache API endpoints.
    """
    return {
        'name': 'Sector Cache API',
        'version': '1.0',
        'description': 'API for managing sector analysis cache',
        'endpoints': {
            'GET /status': 'Get cache status for all sectors',
            'GET /status/{sector}': 'Get cache status for specific sector',
            'POST /refresh': 'Manually trigger cache refresh',
            'POST /invalidate': 'Invalidate cache for specific sector',
            'DELETE /clear': 'Clear cache (specific or all)',
            'GET /config': 'Get cache configuration',
            'GET /schedule': 'Get refresh schedule information',
            'POST /price-check': 'Trigger manual price check',
            'GET /health': 'Health check for cache system'
        }
    }
