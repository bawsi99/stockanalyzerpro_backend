#!/usr/bin/env python3
"""
Test script for async index data fetching functionality.
This script tests the new async methods for fetching index data.
"""

import asyncio
import logging
import sys
import os
from datetime import datetime

# Add the backend directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from technical_indicators import IndianMarketMetricsProvider
from sector_benchmarking import SectorBenchmarkingProvider
from agent_capabilities import StockAnalysisOrchestrator
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('AsyncIndexDataTest')

async def test_async_index_data_fetching():
    """Test async index data fetching methods."""
    logger.info("Starting async index data fetching tests...")
    
    try:
        # Test 1: IndianMarketMetricsProvider async methods
        logger.info("Test 1: Testing IndianMarketMetricsProvider async methods...")
        metrics_provider = IndianMarketMetricsProvider()
        
        # Test NIFTY 50 data fetching
        logger.info("Fetching NIFTY 50 data asynchronously...")
        nifty_data = await metrics_provider.get_nifty_50_data_async(period=30)
        if nifty_data is not None and not nifty_data.empty:
            logger.info(f"✅ NIFTY 50 data fetched successfully: {len(nifty_data)} records")
            logger.info(f"   Latest close: {nifty_data['close'].iloc[-1]:.2f}")
        else:
            logger.warning("⚠️ NIFTY 50 data fetch returned None or empty")
        
        # Test INDIA VIX data fetching
        logger.info("Fetching INDIA VIX data asynchronously...")
        vix_data = await metrics_provider.get_india_vix_data_async(period=30)
        if vix_data is not None and not vix_data.empty:
            logger.info(f"✅ INDIA VIX data fetched successfully: {len(vix_data)} records")
            logger.info(f"   Latest close: {vix_data['close'].iloc[-1]:.2f}")
        else:
            logger.warning("⚠️ INDIA VIX data fetch returned None or empty")
        
        # Test sector index data fetching
        logger.info("Fetching NIFTY BANK sector data asynchronously...")
        sector_data = await metrics_provider.get_sector_index_data_async("BANKING", period=30)
        if sector_data is not None and not sector_data.empty:
            logger.info(f"✅ NIFTY BANK data fetched successfully: {len(sector_data)} records")
            logger.info(f"   Latest close: {sector_data['close'].iloc[-1]:.2f}")
        else:
            logger.warning("⚠️ NIFTY BANK data fetch returned None or empty")
        
        # Test 2: Enhanced market metrics with async data
        logger.info("Test 2: Testing enhanced market metrics with async data...")
        
        # Create sample stock data
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
        sample_stock_data = pd.DataFrame({
            'open': [100 + i * 0.1 for i in range(len(dates))],
            'high': [101 + i * 0.1 for i in range(len(dates))],
            'low': [99 + i * 0.1 for i in range(len(dates))],
            'close': [100.5 + i * 0.1 for i in range(len(dates))],
            'volume': [1000000 + i * 1000 for i in range(len(dates))]
        }, index=dates)
        
        enhanced_metrics = await metrics_provider.get_enhanced_market_metrics_async(sample_stock_data, "RELIANCE")
        if enhanced_metrics:
            logger.info("✅ Enhanced market metrics calculated successfully")
            logger.info(f"   Beta: {enhanced_metrics.get('beta', 'N/A')}")
            logger.info(f"   Market correlation: {enhanced_metrics.get('market_correlation', 'N/A')}")
            logger.info(f"   Current VIX: {enhanced_metrics.get('current_vix', 'N/A')}")
        else:
            logger.warning("⚠️ Enhanced market metrics calculation failed")
        
        # Test 3: SectorBenchmarkingProvider async methods
        logger.info("Test 3: Testing SectorBenchmarkingProvider async methods...")
        sector_provider = SectorBenchmarkingProvider()
        
        # Test async sector benchmarking
        logger.info("Testing async sector benchmarking...")
        benchmarking_results = await sector_provider.get_comprehensive_benchmarking_async("RELIANCE", sample_stock_data)
        if benchmarking_results:
            logger.info("✅ Async sector benchmarking completed successfully")
            logger.info(f"   Sector: {benchmarking_results.get('sector', 'N/A')}")
            logger.info(f"   Data points: {benchmarking_results.get('data_points', 'N/A')}")
        else:
            logger.warning("⚠️ Async sector benchmarking failed")
        
        # Test 4: StockAnalysisOrchestrator async methods
        logger.info("Test 4: Testing StockAnalysisOrchestrator async methods...")
        orchestrator = StockAnalysisOrchestrator()
        
        # Authenticate
        auth_success = orchestrator.authenticate()
        if not auth_success:
            logger.warning("⚠️ Authentication failed, skipping orchestrator tests")
        else:
            logger.info("✅ Authentication successful")
            
            # Test async sector context
            logger.info("Testing async sector context...")
            sector_context = await orchestrator.get_sector_context_async("RELIANCE", sample_stock_data, "ENERGY")
            if sector_context:
                logger.info("✅ Async sector context retrieved successfully")
                logger.info(f"   Has sector benchmarking: {'sector_benchmarking' in sector_context}")
                logger.info(f"   Has sector rotation: {'sector_rotation' in sector_context}")
            else:
                logger.warning("⚠️ Async sector context retrieval failed")
        
        logger.info("🎉 All async index data fetching tests completed!")
        
    except Exception as e:
        logger.error(f"❌ Test failed with error: {str(e)}")
        raise

async def test_concurrent_index_data_fetching():
    """Test concurrent fetching of multiple index data sources."""
    logger.info("Testing concurrent index data fetching...")
    
    try:
        metrics_provider = IndianMarketMetricsProvider()
        
        # Create multiple concurrent tasks
        tasks = [
            metrics_provider.get_nifty_50_data_async(30),
            metrics_provider.get_india_vix_data_async(30),
            metrics_provider.get_sector_index_data_async("BANKING", 30),
            metrics_provider.get_sector_index_data_async("IT", 30),
            metrics_provider.get_sector_index_data_async("PHARMA", 30)
        ]
        
        # Execute all tasks concurrently
        start_time = datetime.now()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = datetime.now()
        
        execution_time = (end_time - start_time).total_seconds()
        logger.info(f"⏱️ Concurrent execution time: {execution_time:.2f} seconds")
        
        # Check results
        successful_fetches = 0
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(f"⚠️ Task {i} failed: {result}")
            elif result is not None and not result.empty:
                logger.info(f"✅ Task {i} successful: {len(result)} records")
                successful_fetches += 1
            else:
                logger.warning(f"⚠️ Task {i} returned None or empty data")
        
        logger.info(f"📊 Success rate: {successful_fetches}/{len(tasks)} ({successful_fetches/len(tasks)*100:.1f}%)")
        
    except Exception as e:
        logger.error(f"❌ Concurrent test failed: {str(e)}")
        raise

async def main():
    """Main test function."""
    logger.info("🚀 Starting async index data fetching test suite...")
    
    try:
        # Test individual async methods
        await test_async_index_data_fetching()
        
        # Test concurrent fetching
        await test_concurrent_index_data_fetching()
        
        logger.info("🎉 All tests completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Test suite failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 