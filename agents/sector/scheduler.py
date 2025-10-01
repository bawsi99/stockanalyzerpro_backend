#!/usr/bin/env python3
"""
Sector Analysis Scheduler

Schedules weekly refresh of sector analysis (every Sunday at 3:00 PM IST).
Ensures sector data is kept fresh on a regular schedule.
"""

import asyncio
import logging
from datetime import datetime, time, timedelta
from typing import Optional, Any, Callable, Dict, List
import pytz

from .cache_manager import SectorCacheManager

logger = logging.getLogger(__name__)


class SectorAnalysisScheduler:
    """Schedules weekly refresh of sector analysis."""
    
    def __init__(self, cache_manager: Optional[SectorCacheManager] = None):
        """
        Initialize the scheduler.
        
        Args:
            cache_manager: SectorCacheManager instance (creates new if None)
        """
        self.cache_manager = cache_manager or SectorCacheManager()
        self.running = False
        
        # Market hours configuration
        market_hours = self.cache_manager.config['market_hours_ist']
        self.timezone = pytz.timezone(market_hours['timezone'])
        
        # Parse refresh time (default: Sunday 3:00 PM)
        self.refresh_time = time(15, 0)  # 3:00 PM (15:00)
        self.refresh_day = 6  # Sunday (0=Monday, 6=Sunday)
        
        logger.info(f"SectorAnalysisScheduler initialized - "
                   f"refresh_day=Sunday, refresh_time={self.refresh_time}")
    
    def get_next_refresh_datetime(self) -> datetime:
        """
        Calculate the next scheduled refresh time.
        
        Returns:
            datetime of next refresh
        """
        now = datetime.now(self.timezone)
        
        # Calculate days until next Sunday
        days_until_refresh = (self.refresh_day - now.weekday()) % 7
        
        # If it's Sunday but past refresh time, schedule for next Sunday
        if days_until_refresh == 0:
            refresh_datetime = now.replace(
                hour=self.refresh_time.hour,
                minute=self.refresh_time.minute,
                second=0,
                microsecond=0
            )
            if now >= refresh_datetime:
                days_until_refresh = 7
        
        # Calculate next refresh date
        next_refresh_date = now.date() + timedelta(days=days_until_refresh)
        next_refresh_datetime = datetime.combine(
            next_refresh_date,
            self.refresh_time,
            tzinfo=self.timezone
        )
        
        return next_refresh_datetime
    
    def get_seconds_until_next_refresh(self) -> float:
        """
        Calculate seconds until next scheduled refresh.
        
        Returns:
            Number of seconds until next refresh
        """
        next_refresh = self.get_next_refresh_datetime()
        now = datetime.now(self.timezone)
        return (next_refresh - now).total_seconds()
    
    async def refresh_sector(
        self,
        sector: str,
        refresh_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Refresh analysis for a single sector.
        
        Args:
            sector: Sector identifier
            refresh_callback: Optional async callback function to perform actual refresh
            
        Returns:
            Dict with refresh results
        """
        try:
            logger.info(f"Refreshing sector analysis for {sector}...")
            
            if refresh_callback:
                # Call the provided refresh function
                result = await refresh_callback(sector)
                return {
                    'sector': sector,
                    'status': 'success',
                    'timestamp': datetime.now(self.timezone).isoformat(),
                    'result': result
                }
            else:
                # No callback provided, just invalidate cache
                self.cache_manager.invalidate_sector(sector, 'scheduled_refresh')
                return {
                    'sector': sector,
                    'status': 'invalidated',
                    'timestamp': datetime.now(self.timezone).isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error refreshing sector {sector}: {e}")
            return {
                'sector': sector,
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now(self.timezone).isoformat()
            }
    
    async def refresh_all_sectors(
        self,
        refresh_callback: Optional[Callable] = None,
        max_concurrent: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Refresh analysis for all configured sectors.
        
        Args:
            refresh_callback: Optional async callback function to perform actual refresh
            max_concurrent: Maximum number of concurrent refreshes
            
        Returns:
            List of refresh results for each sector
        """
        sectors = self.cache_manager.sectors
        
        logger.info(f"Starting scheduled refresh for {len(sectors)} sectors...")
        start_time = datetime.now(self.timezone)
        
        # Create semaphore to limit concurrent refreshes
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def refresh_with_semaphore(sector: str):
            async with semaphore:
                return await self.refresh_sector(sector, refresh_callback)
        
        # Refresh all sectors with concurrency limit
        tasks = [refresh_with_semaphore(sector) for sector in sectors]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Calculate statistics
        end_time = datetime.now(self.timezone)
        duration = (end_time - start_time).total_seconds()
        
        successful = sum(
            1 for r in results
            if isinstance(r, dict) and r.get('status') in ['success', 'invalidated']
        )
        failed = len(results) - successful
        
        logger.info(
            f"Scheduled refresh complete: {successful} successful, {failed} failed, "
            f"duration={duration:.1f}s"
        )
        
        return results
    
    async def weekly_refresh_job(self, refresh_callback: Optional[Callable] = None):
        """
        Execute weekly refresh job.
        
        Args:
            refresh_callback: Optional async callback function to perform actual refresh
        """
        logger.info("Executing weekly refresh job...")
        
        try:
            results = await self.refresh_all_sectors(refresh_callback)
            
            # Log summary
            successful = sum(
                1 for r in results
                if isinstance(r, dict) and r.get('status') in ['success', 'invalidated']
            )
            
            logger.info(f"Weekly refresh job completed: {successful}/{len(results)} sectors refreshed")
            
        except Exception as e:
            logger.error(f"Error in weekly refresh job: {e}")
    
    async def scheduling_loop(self, refresh_callback: Optional[Callable] = None):
        """
        Main scheduling loop - waits for next scheduled time and executes refresh.
        
        Args:
            refresh_callback: Optional async callback function to perform actual refresh
        """
        logger.info("Starting scheduling loop...")
        self.running = True
        
        while self.running:
            try:
                # Calculate time until next refresh
                seconds_until_refresh = self.get_seconds_until_next_refresh()
                next_refresh = self.get_next_refresh_datetime()
                
                logger.info(
                    f"Next scheduled refresh: {next_refresh.strftime('%Y-%m-%d %H:%M:%S %Z')} "
                    f"(in {seconds_until_refresh/3600:.1f} hours)"
                )
                
                # Wait until next refresh time
                # Use smaller sleep intervals to allow for graceful shutdown
                while seconds_until_refresh > 0 and self.running:
                    sleep_duration = min(seconds_until_refresh, 3600)  # Max 1 hour sleep
                    await asyncio.sleep(sleep_duration)
                    seconds_until_refresh = self.get_seconds_until_next_refresh()
                
                if not self.running:
                    break
                
                # Execute refresh job
                await self.weekly_refresh_job(refresh_callback)
                
            except asyncio.CancelledError:
                logger.info("Scheduling loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in scheduling loop: {e}")
                # Wait a bit before retrying after error
                await asyncio.sleep(300)  # 5 minutes
        
        logger.info("Scheduling loop stopped")
    
    async def start_scheduler(self, refresh_callback: Optional[Callable] = None):
        """
        Start the scheduler service.
        
        Args:
            refresh_callback: Optional async callback function to perform actual refresh
        """
        if self.running:
            logger.warning("Scheduler already running")
            return
        
        logger.info("Starting sector analysis scheduler...")
        await self.scheduling_loop(refresh_callback)
    
    def stop_scheduler(self):
        """Stop the scheduler service."""
        logger.info("Stopping sector analysis scheduler...")
        self.running = False
    
    async def trigger_immediate_refresh(
        self,
        refresh_callback: Optional[Callable] = None,
        sectors: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Trigger an immediate refresh (for manual/on-demand refresh).
        
        Args:
            refresh_callback: Optional async callback function to perform actual refresh
            sectors: List of sectors to refresh (None = all sectors)
            
        Returns:
            List of refresh results
        """
        if sectors is None:
            sectors = self.cache_manager.sectors
        
        logger.info(f"Triggering immediate refresh for {len(sectors)} sectors...")
        
        # Create tasks for each sector
        tasks = [
            self.refresh_sector(sector, refresh_callback)
            for sector in sectors
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        successful = sum(
            1 for r in results
            if isinstance(r, dict) and r.get('status') in ['success', 'invalidated']
        )
        
        logger.info(f"Immediate refresh complete: {successful}/{len(results)} sectors refreshed")
        
        return results


# Example usage
if __name__ == "__main__":
    import sys
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    async def main():
        """Test the scheduler."""
        scheduler = SectorAnalysisScheduler()
        
        print(f"\n{'='*60}")
        print("SECTOR ANALYSIS SCHEDULER TEST")
        print(f"{'='*60}\n")
        
        print(f"Refresh schedule: Every Sunday at {scheduler.refresh_time}")
        print(f"Timezone: {scheduler.timezone}")
        
        next_refresh = scheduler.get_next_refresh_datetime()
        seconds_until = scheduler.get_seconds_until_next_refresh()
        
        print(f"\nNext scheduled refresh:")
        print(f"  - Date/Time: {next_refresh.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        print(f"  - Time until: {seconds_until/3600:.1f} hours ({seconds_until/86400:.1f} days)")
        
        print(f"\n{'='*60}")
        print("Testing immediate refresh (cache invalidation)...")
        print(f"{'='*60}\n")
        
        # Test immediate refresh without callback (just invalidates)
        test_sectors = ["NIFTY_BANK", "NIFTY_IT"]
        results = await scheduler.trigger_immediate_refresh(sectors=test_sectors)
        
        print(f"\nRefreshed {len(results)} sectors:")
        for result in results:
            if isinstance(result, dict):
                print(f"  - {result.get('sector')}: {result.get('status')}")
        
        print(f"\n✅ Scheduler test complete!")
    
    # Run the test
    asyncio.run(main())
