#!/usr/bin/env python3
"""
Sector Price Monitor

Monitors sector index prices and triggers cache refresh when
prices change significantly (>2.5% threshold).
"""

import asyncio
import logging
from datetime import datetime, time
from typing import Dict, Optional, Any, List
import pytz

from .cache_manager import SectorCacheManager

logger = logging.getLogger(__name__)


class SectorPriceMonitor:
    """Monitors sector index prices and triggers refresh when needed."""
    
    def __init__(self, cache_manager: Optional[SectorCacheManager] = None):
        """
        Initialize the price monitor.
        
        Args:
            cache_manager: SectorCacheManager instance (creates new if None)
        """
        self.cache_manager = cache_manager or SectorCacheManager()
        self.running = False
        self.check_interval = self.cache_manager.config.get('check_interval_hours', 1) * 3600  # Convert to seconds
        
        # Market hours configuration
        market_hours = self.cache_manager.config['market_hours_ist']
        self.timezone = pytz.timezone(market_hours['timezone'])
        
        # Parse market hours
        start_parts = market_hours['start'].split(':')
        end_parts = market_hours['end'].split(':')
        self.market_start = time(int(start_parts[0]), int(start_parts[1]))
        self.market_end = time(int(end_parts[0]), int(end_parts[1]))
        
        logger.info(f"SectorPriceMonitor initialized - check_interval={self.check_interval}s, "
                   f"market_hours={market_hours['start']}-{market_hours['end']}")
    
    def is_market_hours(self) -> bool:
        """
        Check if current time is within market hours (9:15-15:30 IST).
        
        Returns:
            True if within market hours, False otherwise
        """
        now = datetime.now(self.timezone).time()
        return self.market_start <= now <= self.market_end
    
    def is_trading_day(self) -> bool:
        """
        Check if today is a trading day (Monday-Friday).
        
        Returns:
            True if trading day, False otherwise
        """
        now = datetime.now(self.timezone)
        # Monday = 0, Sunday = 6
        return now.weekday() < 5
    
    async def fetch_sector_price(self, sector: str) -> Optional[float]:
        """
        Fetch current price for a sector index.
        
        This is a placeholder that should be replaced with actual Zerodha API calls.
        
        Args:
            sector: Sector identifier (e.g., 'NIFTY_BANK')
            
        Returns:
            Current sector index price or None if unavailable
        """
        try:
            # TODO: Integrate with actual Zerodha client to fetch real prices
            # For now, this is a mock implementation
            
            # Example of what this would look like with real integration:
            # from zerodha.client import ZerodhaDataClient
            # client = ZerodhaDataClient()
            # sector_index = self.cache_manager.sector_classifier.get_primary_sector_index(sector)
            # price_data = await client.get_quote(sector_index)
            # return price_data['last_price']
            
            logger.debug(f"Mock: Fetching price for {sector}")
            return None
            
        except Exception as e:
            logger.error(f"Error fetching price for {sector}: {e}")
            return None
    
    async def check_sector_price(self, sector: str, refresh_callback=None) -> Dict[str, Any]:
        """
        Check price for a sector and determine if refresh is needed.
        
        Args:
            sector: Sector identifier
            refresh_callback: Optional async callback function to refresh sector analysis
            
        Returns:
            Dict with check results
        """
        try:
            # Fetch current price
            current_price = await self.fetch_sector_price(sector)
            
            if current_price is None:
                return {
                    'sector': sector,
                    'status': 'error',
                    'message': 'Failed to fetch price'
                }
            
            # Update price in cache
            self.cache_manager.update_sector_price(sector, current_price)
            
            # Check if refresh is needed
            should_refresh, reason = self.cache_manager.should_refresh(sector)
            
            result = {
                'sector': sector,
                'current_price': current_price,
                'should_refresh': should_refresh,
                'reason': reason,
                'timestamp': datetime.now(self.timezone).isoformat()
            }
            
            # Trigger refresh if needed and callback provided
            if should_refresh and refresh_callback:
                logger.info(f"Triggering refresh for {sector}: {reason}")
                try:
                    await refresh_callback(sector, current_price)
                    result['refresh_triggered'] = True
                except Exception as e:
                    logger.error(f"Error in refresh callback for {sector}: {e}")
                    result['refresh_triggered'] = False
                    result['refresh_error'] = str(e)
            else:
                result['refresh_triggered'] = False
            
            return result
            
        except Exception as e:
            logger.error(f"Error checking sector price for {sector}: {e}")
            return {
                'sector': sector,
                'status': 'error',
                'message': str(e)
            }
    
    async def check_all_sectors(self, refresh_callback=None) -> List[Dict[str, Any]]:
        """
        Check prices for all configured sectors.
        
        Args:
            refresh_callback: Optional async callback function to refresh sector analysis
            
        Returns:
            List of check results for each sector
        """
        sectors = self.cache_manager.sectors
        
        logger.info(f"Checking prices for {len(sectors)} sectors...")
        
        # Check all sectors concurrently
        tasks = [
            self.check_sector_price(sector, refresh_callback)
            for sector in sectors
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Count refreshes triggered
        refreshes_triggered = sum(
            1 for r in results
            if isinstance(r, dict) and r.get('refresh_triggered', False)
        )
        
        if refreshes_triggered > 0:
            logger.info(f"Price check complete: {refreshes_triggered} refresh(es) triggered")
        else:
            logger.debug("Price check complete: no refreshes needed")
        
        return results
    
    async def monitoring_loop(self, refresh_callback=None):
        """
        Main monitoring loop - checks prices at regular intervals.
        
        Args:
            refresh_callback: Optional async callback function to refresh sector analysis
        """
        logger.info("Starting price monitoring loop...")
        self.running = True
        
        while self.running:
            try:
                # Only check during trading days and market hours
                if self.is_trading_day() and self.is_market_hours():
                    logger.debug("Market is open, checking sector prices...")
                    await self.check_all_sectors(refresh_callback)
                else:
                    if not self.is_trading_day():
                        logger.debug("Market closed: not a trading day")
                    else:
                        logger.debug("Market closed: outside trading hours")
                
                # Wait for next check interval
                await asyncio.sleep(self.check_interval)
                
            except asyncio.CancelledError:
                logger.info("Monitoring loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                # Continue after error, wait a bit before retrying
                await asyncio.sleep(60)
        
        logger.info("Price monitoring loop stopped")
    
    async def start_monitoring(self, refresh_callback=None):
        """
        Start the price monitoring service.
        
        Args:
            refresh_callback: Optional async callback function to refresh sector analysis
        """
        if self.running:
            logger.warning("Price monitoring already running")
            return
        
        logger.info("Starting sector price monitoring service...")
        await self.monitoring_loop(refresh_callback)
    
    def stop_monitoring(self):
        """Stop the price monitoring service."""
        logger.info("Stopping sector price monitoring service...")
        self.running = False
    
    async def check_once(self, refresh_callback=None) -> List[Dict[str, Any]]:
        """
        Perform a single price check for all sectors (for testing/manual trigger).
        
        Args:
            refresh_callback: Optional async callback function to refresh sector analysis
            
        Returns:
            List of check results
        """
        logger.info("Performing one-time price check...")
        return await self.check_all_sectors(refresh_callback)


# Example usage
if __name__ == "__main__":
    import sys
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    async def main():
        """Test the price monitor."""
        monitor = SectorPriceMonitor()
        
        print(f"\n{'='*60}")
        print("SECTOR PRICE MONITOR TEST")
        print(f"{'='*60}\n")
        
        print(f"Market hours: {monitor.market_start} - {monitor.market_end}")
        print(f"Check interval: {monitor.check_interval}s ({monitor.check_interval/3600}h)")
        print(f"Is trading day: {monitor.is_trading_day()}")
        print(f"Is market hours: {monitor.is_market_hours()}")
        
        print(f"\n{'='*60}")
        print("Testing single price check...")
        print(f"{'='*60}\n")
        
        # Test a single check
        results = await monitor.check_once()
        
        print(f"\nChecked {len(results)} sectors")
        for result in results[:3]:  # Show first 3 results
            if isinstance(result, dict):
                print(f"  - {result.get('sector')}: {result.get('status', 'checked')}")
        
        print(f"\n✅ Price monitor test complete!")
    
    # Run the test
    asyncio.run(main())
