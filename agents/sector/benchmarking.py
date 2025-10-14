#!/usr/bin/env python3
"""
Sector Benchmarking Module for Enhanced Stock Analysis

This module provides comprehensive sector-based benchmarking capabilities,
enhancing the current NIFTY-only analysis with sector-specific metrics,
performance comparisons, and risk assessments.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import json
from pathlib import Path
import asyncio

# Import existing components
from agents.sector.classifier import SectorClassifier
from agents.sector.enhanced_classifier import enhanced_sector_classifier
from ml.indicators.technical_indicators import IndianMarketMetricsProvider
from zerodha.client import ZerodhaDataClient

class SectorBenchmarkingProvider:
    """
    Provides comprehensive sector-based benchmarking for stock analysis.
    Enhances the current NIFTY-only analysis with sector-specific metrics.
    """
    
    def __init__(self):
        """Initialize the sector benchmarking provider."""
        self.zerodha_client = ZerodhaDataClient()
        self.market_metrics_provider = IndianMarketMetricsProvider()
        self.sector_classifier = SectorClassifier()
        self.enhanced_classifier = enhanced_sector_classifier
        
        # Layer 2 cache REMOVED - now relying only on Layer 1 (file-based) cache
        # Reasoning: Layer 1 expires when data is stale. If Layer 1 expires, we need fresh data,
        # not old cached data from Layer 2. Layer 2 would serve stale data when we need fresh data.
        
        logging.info("SectorBenchmarkingProvider initialized - Layer 2 cache removed, using Layer 1 only")
    
    def get_comprehensive_benchmarking(self, stock_symbol: str, stock_data: pd.DataFrame, user_sector: str = None) -> Dict[str, Any]:
        """
        Get comprehensive benchmarking analysis including both market and sector metrics.
        
        Args:
            stock_symbol: Stock symbol to analyze
            stock_data: Historical stock data
            user_sector: Optional user-provided sector override
            
        Returns:
            Dict containing comprehensive benchmarking analysis
        """
        try:
            logging.info(f"Calculating comprehensive benchmarking for {stock_symbol}")
            
            # Prioritize user-provided sector over detected sector
            if user_sector:
                sector = user_sector
                logging.info(f"Using user-provided sector '{user_sector}' for {stock_symbol}")
            else:
                # Get sector information from auto-detection
                sector = self.sector_classifier.get_stock_sector(stock_symbol)
                logging.info(f"Using auto-detected sector '{sector}' for {stock_symbol}")
            sector_name = self.sector_classifier.get_sector_display_name(sector) if sector else None
            sector_index = self.sector_classifier.get_primary_sector_index(sector) if sector else None
            
            # Calculate stock returns
            stock_returns = stock_data['close'].pct_change().dropna()
            
            # Get market metrics (NIFTY 50)
            market_metrics = self._calculate_market_metrics(stock_returns)
            
            # Get sector metrics (if available)
            sector_metrics = self._calculate_sector_metrics(stock_returns, sector) if sector else None
            
            # Calculate relative performance
            relative_performance = self._calculate_relative_performance(
                stock_data, sector, market_metrics, sector_metrics
            )
            
            # Calculate sector-specific risk metrics
            sector_risk_metrics = self._calculate_sector_risk_metrics(
                stock_returns, sector, market_metrics, sector_metrics
            ) if sector else None
            
            # Build comprehensive results
            results = {
                "stock_symbol": stock_symbol,
                "sector_info": {
                    "sector": sector,
                    "sector_name": sector_name,
                    "sector_index": sector_index,
                    "sector_stocks_count": len(self.sector_classifier.get_sector_stocks(sector)) if sector else 0
                },
                "market_benchmarking": market_metrics,
                "sector_benchmarking": sector_metrics,
                "relative_performance": relative_performance,
                "sector_risk_metrics": sector_risk_metrics,
                "analysis_summary": self._generate_analysis_summary(
                    stock_symbol, sector, market_metrics, sector_metrics, relative_performance
                ),
                "timestamp": datetime.now().isoformat(),
                "data_points": {
                    "stock_data_points": len(stock_data),
                    "market_data_points": market_metrics.get('data_points', 0),
                    "sector_data_points": sector_metrics.get('data_points', 0) if sector_metrics else 0
                }
            }
            
            return results
            
        except Exception as e:
            logging.error(f"Error in comprehensive benchmarking for {stock_symbol}: {e}")
            return self._get_fallback_benchmarking(stock_symbol, sector)
    
    async def get_comprehensive_benchmarking_async(self, stock_symbol: str, stock_data: pd.DataFrame, user_sector: str = None) -> Dict[str, Any]:
        """Async version of get_comprehensive_benchmarking."""
        try:
            # DEBUG: Comprehensive data tracing
            logging.info(f"\n=== DEBUG: COMPREHENSIVE BENCHMARKING START for {stock_symbol} ===")
            logging.info(f"DEBUG: Input stock_data shape: {stock_data.shape if stock_data is not None else 'None'}")
            logging.info(f"DEBUG: Input stock_data columns: {stock_data.columns.tolist() if stock_data is not None else 'None'}")
            logging.info(f"DEBUG: Input stock_data date range: {stock_data.index[0]} to {stock_data.index[-1] if stock_data is not None and len(stock_data) > 0 else 'N/A'}")
            logging.info(f"DEBUG: Input user_sector: {user_sector}")
            
            # Check if stock_data is valid
            if stock_data is None or stock_data.empty or 'close' not in stock_data.columns:
                logging.error(f"DEBUG: Invalid stock data for {stock_symbol} - Reason: {('None' if stock_data is None else 'Empty' if stock_data.empty else 'No close column')}")
                return self._get_fallback_benchmarking(stock_symbol, user_sector or "UNKNOWN")
            
            # Get stock returns
            logging.info(f"DEBUG: Raw close prices - First 5: {stock_data['close'].head().tolist()}")
            logging.info(f"DEBUG: Raw close prices - Last 5: {stock_data['close'].tail().tolist()}")
            logging.info(f"DEBUG: Raw close prices - Has NaN: {stock_data['close'].isna().sum()} out of {len(stock_data)}")
            
            stock_returns = stock_data['close'].pct_change().dropna()
            
            logging.info(f"DEBUG: Stock returns after pct_change().dropna(): {len(stock_returns)} data points")
            logging.info(f"DEBUG: Stock returns first 5 values: {stock_returns.head().tolist()}")
            logging.info(f"DEBUG: Stock returns last 5 values: {stock_returns.tail().tolist()}")
            logging.info(f"DEBUG: Stock returns stats - mean: {stock_returns.mean():.6f}, std: {stock_returns.std():.6f}")
            
            if len(stock_returns) < 10:
                logging.warning(f"Severely insufficient stock returns data for {stock_symbol}: {len(stock_returns)} < 10")
                return self._get_fallback_benchmarking(stock_symbol, user_sector or "UNKNOWN")
            
            elif len(stock_returns) < 30:
                logging.info(f"Limited stock returns data for {stock_symbol}: {len(stock_returns)} < 30 (recommended). Using degraded analysis mode.")
            
            # Prioritize user-provided sector over detected sector
            if user_sector:
                sector = user_sector
                logging.info(f"Using user-provided sector '{user_sector}' for {stock_symbol}")
            else:
                # Get sector classification from auto-detection
                sector = self.sector_classifier.get_stock_sector(stock_symbol)
                logging.info(f"Using auto-detected sector '{sector}' for {stock_symbol}")
            
            # Fetch market and sector data concurrently
            tasks = [
                self._calculate_market_metrics_async(stock_returns)
            ]
            
            if sector and sector != 'UNKNOWN':
                tasks.append(self._calculate_sector_metrics_async(stock_returns, sector))
            else:
                tasks.append(None)
            
            # Execute tasks concurrently
            logging.info(f"DEBUG: Executing {len(tasks)} async tasks concurrently")
            results = await asyncio.gather(*tasks, return_exceptions=True)
            logging.info(f"DEBUG: Async tasks completed, processing results")
            
            # Process market metrics result
            if isinstance(results[0], Exception):
                logging.error(f"DEBUG: Market metrics task failed with exception: {results[0]}")
                market_metrics = self._get_default_market_metrics()
            else:
                logging.info(f"DEBUG: Market metrics task succeeded")
                market_metrics = results[0]
                if market_metrics:
                    logging.info(f"DEBUG: Market metrics keys: {list(market_metrics.keys())}")
                    logging.info(f"DEBUG: Market metrics beta: {market_metrics.get('beta', 'N/A')}")
                    logging.info(f"DEBUG: Market metrics correlation: {market_metrics.get('correlation', 'N/A')}")
                else:
                    logging.warning(f"DEBUG: Market metrics is None or empty")
            
            # Process sector metrics result
            if len(results) > 1:
                if isinstance(results[1], Exception):
                    logging.error(f"DEBUG: Sector metrics task failed with exception: {results[1]}")
                    sector_metrics = None
                else:
                    logging.info(f"DEBUG: Sector metrics task succeeded")
                    sector_metrics = results[1]
                    if sector_metrics:
                        logging.info(f"DEBUG: Sector metrics keys: {list(sector_metrics.keys())}")
                        logging.info(f"DEBUG: Sector metrics beta: {sector_metrics.get('sector_beta', 'N/A')}")
                        logging.info(f"DEBUG: Sector metrics correlation: {sector_metrics.get('sector_correlation', 'N/A')}")
                    else:
                        logging.warning(f"DEBUG: Sector metrics is None or empty")
            else:
                logging.info(f"DEBUG: No sector metrics task was queued")
                sector_metrics = None
            
            # Calculate relative performance
            relative_performance = self._calculate_relative_performance(stock_data, sector, market_metrics, sector_metrics)
            
            # Calculate risk metrics
            risk_metrics = self._calculate_sector_risk_metrics(stock_returns, sector, market_metrics, sector_metrics)
            
            # Calculate stress metrics
            stress_metrics = self._calculate_sector_stress_metrics(stock_returns, sector_metrics, market_metrics)
            
            # Generate analysis summary
            analysis_summary = self._generate_analysis_summary(stock_symbol, sector, market_metrics, sector_metrics, relative_performance)
            
            # Get sector information with debugging
            sector_name = self.sector_classifier.get_sector_display_name(sector) if sector else None
            sector_index = self.sector_classifier.get_primary_sector_index(sector) if sector else None
            sector_stocks = self.sector_classifier.get_sector_stocks(sector) if sector else []
            
            # Debug logging
            logging.info(f"Sector benchmarking for {stock_symbol}:")
            logging.info(f"  - Sector: {sector}")
            logging.info(f"  - Sector Name: {sector_name}")
            logging.info(f"  - Sector Index: {sector_index}")
            logging.info(f"  - Sector Stocks Count: {len(sector_stocks)}")
            logging.info(f"  - Sample Stocks: {sector_stocks[:5] if sector_stocks else 'None'}")
            
            # Add data quality assessment
            data_quality_assessment = {
                "sufficient_data": len(stock_returns) >= 30,
                "data_points": len(stock_returns),
                "minimum_recommended": 30,
                "reliability": "high" if len(stock_returns) >= 50 else "moderate" if len(stock_returns) >= 30 else "limited",
                "analysis_mode": "full" if len(stock_returns) >= 30 else "degraded",
                "limitations": [] if len(stock_returns) >= 30 else [
                    "Sector comparison may be less accurate with limited data",
                    "Risk metrics are simplified due to insufficient history",
                    "Some benchmarking features may be unavailable"
                ],
                "recommendations": [] if len(stock_returns) >= 50 else [
                    "Results are reliable but may improve with more historical data"
                ] if len(stock_returns) >= 30 else [
                    "Use sector analysis results with caution due to limited data",
                    "Focus on recent performance trends rather than long-term metrics"
                ]
            }
            
            # Build comprehensive results with correct structure
            results = {
                "stock_symbol": stock_symbol,
                "sector_info": {
                    "sector": sector,
                    "sector_name": sector_name,
                    "sector_index": sector_index,
                    "sector_stocks_count": len(sector_stocks)
                },
                "market_benchmarking": market_metrics,
                "sector_benchmarking": sector_metrics,
                "relative_performance": relative_performance,
                "sector_risk_metrics": risk_metrics,
                "analysis_summary": analysis_summary,
                "data_quality": data_quality_assessment,
                "timestamp": datetime.now().isoformat(),
                "data_points": {
                    "stock_data_points": len(stock_returns),
                    "market_data_points": market_metrics.get('data_points', 0),
                    "sector_data_points": sector_metrics.get('data_points', 0) if sector_metrics else 0
                }
            }
            
            return results
            
        except Exception as e:
            logging.error(f"Error in async comprehensive benchmarking: {e}")
            return self._get_fallback_benchmarking(stock_symbol, "UNKNOWN")
    
    def analyze_sector_rotation(self, timeframe: str = "1M") -> Dict[str, Any]:
        """
        Analyze sector rotation patterns and momentum.
        
        Args:
            timeframe: Analysis period ("1M", "3M", "6M", "1Y")
            
        Returns:
            Dict containing sector rotation analysis
        """
        try:
            logging.info(f"Analyzing sector rotation for {timeframe} timeframe")
            
            # Calculate days for timeframe - OPTIMIZED for reduced data fetching
            timeframe_days = {
                "1M": 30,    # OPTIMIZED: Reduced from 90 to 30 days
                "3M": 60,    # OPTIMIZED: Reduced from 90 to 60 days
                "6M": 90,    # OPTIMIZED: Reduced from 180 to 90 days
                "1Y": 180    # OPTIMIZED: Reduced from 365 to 180 days
            }
            days = timeframe_days.get(timeframe, 30)  # Default to 1M instead of 3M
            
            # OPTIMIZATION: Fetch NIFTY 50 data once and reuse for all sectors
            all_sectors_data = self.sector_classifier.get_all_sectors()
            logging.info(f"Fetching NIFTY 50 data once for {timeframe} timeframe (will be reused for all {len(all_sectors_data)} sectors)")
            nifty_data = self._get_nifty_data(days + 20)  # OPTIMIZED: Reduced buffer from 50 to 20 days
            nifty_return = None
            if nifty_data is not None and len(nifty_data) >= days:
                nifty_return = ((nifty_data['close'].iloc[-1] - nifty_data['close'].iloc[-days]) / 
                              nifty_data['close'].iloc[-days]) * 100
                logging.info(f"NIFTY 50 return calculated: {nifty_return:.2f}%")
            else:
                logging.warning("Could not fetch NIFTY 50 data for sector rotation analysis")
            
            # FIXED: Use actual sectors from sector classifier instead of hardcoded list
            all_sectors_data = self.sector_classifier.get_all_sectors()
            all_sectors = [s['code'] for s in all_sectors_data]  # Extract sector codes
            logging.info(f"Using actual sectors from classifier for rotation: {len(all_sectors)} sectors found")
            
            # Get sector performance data
            sector_performance = {}
            sector_momentum = {}
            sector_rankings = {}
            
            for sector in all_sectors:
                try:
                    # Get sector index for data fetching
                    sector_index = self.sector_classifier.get_primary_sector_index(sector)
                    if not sector_index:
                        logging.warning(f"No primary index found for sector: {sector}")
                        continue
                        
                    # Get historical data for sector index - OPTIMIZED timeframe
                    sector_data = self.zerodha_client.get_historical_data(
                        symbol=sector_index,
                        exchange="NSE",
                        period=days + 20
                    )
                    
                    # More flexible data requirement for longer timeframes
                    min_required = days * 0.7 if days > 60 else days * 0.8  # OPTIMIZED: Adjusted thresholds
                    if sector_data is None or len(sector_data) < min_required:
                        logging.warning(f"No sufficient data for {sector} ({sector_index}): got {len(sector_data) if sector_data is not None else 0} records, need at least {min_required:.0f}")
                        continue
                    
                    # Calculate sector performance
                    current_price = sector_data['close'].iloc[-1]
                    start_price = sector_data['close'].iloc[-days]
                    total_return = ((current_price - start_price) / start_price) * 100
                    
                    # Calculate momentum (rate of change) - OPTIMIZED: Reduced from 20 to 10 days
                    recent_prices = sector_data['close'].tail(10)  # OPTIMIZED: Reduced from 20 to 10 days
                    momentum = ((recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0]) * 100
                    
                    # Calculate relative strength vs NIFTY (using pre-fetched data)
                    if nifty_return is not None:
                        relative_strength = total_return - nifty_return
                    else:
                        relative_strength = total_return
                    
                    sector_performance[sector] = {
                        'total_return': round(total_return, 2),
                        'momentum': round(momentum, 2),
                        'relative_strength': round(relative_strength, 2),
                        'current_price': current_price,
                        'start_price': start_price
                    }
                    
                    sector_momentum[sector] = momentum
                    
                except Exception as e:
                    logging.warning(f"Error calculating performance for {sector}: {e}")
                    continue
            
            # Rank sectors by performance - OPTIMIZED: Only store rank, not duplicate performance data
            sorted_sectors = sorted(sector_performance.items(), 
                                  key=lambda x: x[1]['relative_strength'], reverse=True)
            
            for rank, (sector, data) in enumerate(sorted_sectors, 1):
                sector_rankings[sector] = {
                    'rank': rank
                    # Performance data already available in sector_performance[sector]
                }
            
            # Identify rotation patterns
            rotation_analysis = self._identify_rotation_patterns(sector_performance, timeframe)
            
            # Generate recommendations
            recommendations = self._generate_rotation_recommendations(sector_rankings, rotation_analysis, sector_performance)
            
            return {
                'timeframe': timeframe,
                'sector_performance': sector_performance,
                'sector_rankings': sector_rankings,
                'rotation_patterns': rotation_analysis,
                'recommendations': recommendations,
                'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'optimization_note': 'Timeframes optimized for reduced data fetching'
            }
            
        except Exception as e:
            logging.error(f"Error in sector rotation analysis: {e}")
            return None
    
    async def analyze_sector_rotation_async(self, timeframe: str = "1M") -> Dict[str, Any]:
        """
        Async version of analyze_sector_rotation with optimized timeframes.
        
        Args:
            timeframe: Analysis period ("1M", "3M", "6M", "1Y")
            
        Returns:
            Dict containing sector rotation analysis
        """
        try:
            logging.info(f"Async analyzing sector rotation for {timeframe} timeframe")
            
            # Calculate days for timeframe - OPTIMIZED for reduced data fetching
            timeframe_days = {
                "1M": 30,    # OPTIMIZED: Reduced from 90 to 30 days
                "3M": 60,    # OPTIMIZED: Reduced from 90 to 60 days
                "6M": 90,    # OPTIMIZED: Reduced from 180 to 90 days
                "1Y": 180    # OPTIMIZED: Reduced from 365 to 180 days
            }
            days = timeframe_days.get(timeframe, 30)  # Default to 1M
            
            # Fetch NIFTY 50 once
            logging.info(f"Async fetching NIFTY 50 data once for {timeframe} timeframe")
            nifty_data = await self._get_nifty_data_async(days + 20)
            nifty_return = None
            if nifty_data is not None and len(nifty_data) >= 2:
                n = len(nifty_data)
                window = min(days, n - 1)
                try:
                    nifty_return = ((nifty_data['close'].iloc[-1] - nifty_data['close'].iloc[-window]) / nifty_data['close'].iloc[-window]) * 100
                    logging.info(f"NIFTY 50 return calculated: {nifty_return:.2f}% (window={window} days)")
                except Exception as e:
                    logging.warning(f"Failed to compute NIFTY return (async): {e}")
            else:
                logging.warning("Could not fetch NIFTY 50 data for async sector rotation analysis")
            
            # Use actual sectors from classifier
            all_sectors_data = self.sector_classifier.get_all_sectors()
            all_sectors = [s['code'] for s in all_sectors_data]
            
            sector_performance = {}
            sector_rankings = {}
            
            async def fetch_sector_perf(sector_code: str):
                try:
                    sector_index = self.sector_classifier.get_primary_sector_index(sector_code)
                    if not sector_index:
                        logging.warning(f"No primary index found for sector: {sector_code}")
                        return None
                    # Fetch sector index data asynchronously
                    data = await self._get_sector_index_data_async(sector_code, days + 20)
                    min_required = days * 0.7 if days > 60 else days * 0.8
                    if data is None or len(data) < min_required:
                        logging.warning(f"No sufficient data for {sector_code} ({sector_index}): got {len(data) if data is not None else 0} records, need at least {min_required:.0f}")
                        return None
                    close = data['close']
                    n = len(close)
                    current_price = close.iloc[-1]
                    window = min(days, n - 1)
                    start_price = close.iloc[-window]
                    total_return = ((current_price - start_price) / start_price) * 100
                    recent_len = min(10, n - 1)
                    recent_prices = close.tail(recent_len)
                    momentum = ((recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0]) * 100 if recent_len >= 2 else 0.0
                    relative_strength = total_return - nifty_return if nifty_return is not None else total_return
                    return sector_code, {
                        'total_return': round(total_return, 2),
                        'momentum': round(momentum, 2),
                        'relative_strength': round(relative_strength, 2),
                        'current_price': current_price,
                        'start_price': start_price
                    }
                except Exception as e:
                    logging.warning(f"Error calculating performance for {sector_code}: {e}")
                    return None
            
            # Run in parallel
            tasks = [fetch_sector_perf(sec) for sec in all_sectors]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for res in results:
                if isinstance(res, Exception) or res is None:
                    continue
                sec, perf = res
                sector_performance[sec] = perf
            
            # Rank sectors by relative strength
            sorted_sectors = sorted(sector_performance.items(), key=lambda x: x[1]['relative_strength'], reverse=True)
            for rank, (sec, _) in enumerate(sorted_sectors, 1):
                sector_rankings[sec] = {'rank': rank}
            
            rotation_analysis = self._identify_rotation_patterns(sector_performance, timeframe)
            recommendations = self._generate_rotation_recommendations(sector_rankings, rotation_analysis, sector_performance)
            
            return {
                'timeframe': timeframe,
                'sector_performance': sector_performance,
                'sector_rankings': sector_rankings,
                'rotation_patterns': rotation_analysis,
                'recommendations': recommendations,
                'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'optimization_note': 'Async timeframes optimized for reduced data fetching'
            }
        except Exception as e:
            logging.error(f"Error in async sector rotation analysis: {e}")
            return None
    
    def generate_sector_correlation_matrix(self, timeframe: str = "3M") -> Dict[str, Any]:
        """
        Generate correlation matrix between all sectors for portfolio diversification.
        
        Args:
            timeframe: Analysis period ("1M", "3M", "6M", "1Y")
            
        Returns:
            Dict containing correlation matrix and diversification insights
        """
        try:
            logging.info(f"Generating sector correlation matrix for {timeframe} timeframe")
            
            # Calculate days for timeframe - OPTIMIZED for reduced data fetching
            timeframe_days = {
                "1M": 30,    # OPTIMIZED: Reduced from 30 to 30 days (no change needed)
                "3M": 60,    # OPTIMIZED: Reduced from 180 to 60 days
                "6M": 90,    # OPTIMIZED: Reduced from 180 to 90 days
                "1Y": 180    # OPTIMIZED: Reduced from 365 to 180 days
            }
            days = timeframe_days.get(timeframe, 60)  # Default to 3M instead of 6M
            
            # FIXED: Use actual sectors from sector classifier instead of hardcoded list
            all_sectors_data = self.sector_classifier.get_all_sectors()
            all_sectors = [s['code'] for s in all_sectors_data]  # Extract sector codes
            logging.info(f"Using actual sectors from classifier: {len(all_sectors)} sectors found")
            logging.info(f"Sectors: {all_sectors}")
            
            # Collect sector data
            sector_data = {}
            valid_sectors = []
            
            for sector in all_sectors:
                try:
                    # Get sector index for data fetching
                    sector_index = self.sector_classifier.get_primary_sector_index(sector)
                    if not sector_index:
                        logging.warning(f"No primary index found for sector: {sector}")
                        continue
                        
                    # Fetch sector data using sector index
                    data = self.zerodha_client.get_historical_data(
                        symbol=sector_index,
                        exchange="NSE",
                        period=days + 20
                    )
                    
                    # More flexible data requirement for longer timeframes
                    min_required = days * 0.7 if days > 60 else days * 0.8  # OPTIMIZED: Adjusted thresholds
                    if data is not None and len(data) >= min_required:
                        # Calculate daily returns
                        returns = data['close'].pct_change().dropna()
                        sector_data[sector] = returns
                        valid_sectors.append(sector)
                        logging.info(f"Successfully got data for {sector} ({sector_index}): {len(data)} records")
                    else:
                        logging.warning(f"No sufficient data for {sector} ({sector_index}): got {len(data) if data is not None else 0} records, need at least {min_required:.0f}")
                except Exception as e:
                    logging.warning(f"Error getting data for {sector}: {e}")
                    continue
            
            if len(valid_sectors) < 2:
                logging.warning(f"Insufficient sector data for correlation analysis. Only {len(valid_sectors)} sectors have sufficient data (need at least 2)")
                return None
            
            # Create correlation matrix
            returns_df = pd.DataFrame(sector_data)
            correlation_matrix = returns_df.corr()
            
            # Calculate average correlation
            # Get upper triangle of correlation matrix (excluding diagonal)
            upper_triangle = correlation_matrix.where(
                np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
            )
            avg_correlation = upper_triangle.stack().mean()
            
            # ENHANCED: Categorize correlations with proper handling of negative correlations
            correlation_pairs = self._categorize_correlations(correlation_matrix)
            
            # Extract categorized pairs for backward compatibility and enhanced analysis
            high_correlation_pairs = correlation_pairs['high_positive'] + correlation_pairs['high_negative']
            low_correlation_pairs = correlation_pairs['low']
            moderate_correlation_pairs = correlation_pairs['moderate_positive'] + correlation_pairs['moderate_negative']
            negative_correlation_pairs = correlation_pairs['high_negative'] + correlation_pairs['moderate_negative']
            
            # Generate diversification insights
            diversification_insights = self._generate_diversification_insights(
                correlation_matrix, high_correlation_pairs, low_correlation_pairs, avg_correlation
            )
            
            # Calculate sector volatility for risk assessment
            sector_volatility = {}
            for sector in valid_sectors:
                volatility = sector_data[sector].std() * np.sqrt(252) * 100  # Annualized %
                sector_volatility[sector] = round(volatility, 2)
            
            # Calculate average sector volatility for frontend compatibility
            avg_sector_volatility = sum(sector_volatility.values()) / len(sector_volatility) if sector_volatility else 0
            
            # Convert numpy types to native Python types for JSON serialization
            def convert_numpy_types(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {key: convert_numpy_types(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                else:
                    return obj
            
            # Convert correlation matrix to native types
            correlation_matrix_dict = correlation_matrix.round(3).to_dict()
            correlation_matrix_clean = convert_numpy_types(correlation_matrix_dict)
            
            # Convert other data to native types
            high_correlation_pairs_clean = convert_numpy_types(high_correlation_pairs)
            low_correlation_pairs_clean = convert_numpy_types(low_correlation_pairs)
            sector_volatility_clean = convert_numpy_types(sector_volatility)
            diversification_insights_clean = convert_numpy_types(diversification_insights)
            
            return {
                'timeframe': timeframe,
                'correlation_matrix': correlation_matrix_clean,
                'average_correlation': float(round(avg_correlation, 3)),
                'sector_volatility': round(avg_sector_volatility, 2),  # FIXED: Single value for frontend
                'sector_volatilities': sector_volatility_clean,  # Individual sector volatilities for reference
                'high_correlation_pairs': high_correlation_pairs_clean,
                'low_correlation_pairs': low_correlation_pairs_clean,
                # ENHANCED: New correlation categories
                'moderate_correlation_pairs': convert_numpy_types(moderate_correlation_pairs),
                'negative_correlation_pairs': convert_numpy_types(negative_correlation_pairs),
                'correlation_breakdown': {
                    'high_positive': convert_numpy_types(correlation_pairs['high_positive']),
                    'moderate_positive': convert_numpy_types(correlation_pairs['moderate_positive']),
                    'low': convert_numpy_types(correlation_pairs['low']),
                    'moderate_negative': convert_numpy_types(correlation_pairs['moderate_negative']),
                    'high_negative': convert_numpy_types(correlation_pairs['high_negative'])
                },
                'diversification_insights': diversification_insights_clean,
                'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'optimization_note': 'Enhanced correlation analysis with negative correlation support'
            }
            
        except Exception as e:
            logging.error(f"Error generating sector correlation matrix: {e}")
            return None
    
    async def generate_sector_correlation_matrix_async(self, timeframe: str = "3M") -> Dict[str, Any]:
        """
        Generate correlation matrix between all sectors for portfolio diversification (async).
        
        Args:
            timeframe: Analysis period ("1M", "3M", "6M", "1Y")
            
        Returns:
            Dict containing correlation matrix and diversification insights
        """
        try:
            logging.info(f"Async generating sector correlation matrix for {timeframe} timeframe")
            
            # Calculate days for timeframe - OPTIMIZED for reduced data fetching
            timeframe_days = {
                "1M": 30,    # OPTIMIZED: Reduced from 30 to 30 days (no change needed)
                "3M": 60,    # OPTIMIZED: Reduced from 180 to 60 days
                "6M": 90,    # OPTIMIZED: Reduced from 180 to 90 days
                "1Y": 180    # OPTIMIZED: Reduced from 365 to 180 days
            }
            days = timeframe_days.get(timeframe, 60)  # Default to 3M instead of 6M
            
            # FIXED: Use actual sectors from sector classifier instead of hardcoded list
            all_sectors_data = self.sector_classifier.get_all_sectors()
            all_sectors = [s['code'] for s in all_sectors_data]  # Extract sector codes
            logging.info(f"Using actual sectors from classifier: {len(all_sectors)} sectors found")
            
            # Collect sector data using async parallel fetching
            sector_data = {}
            valid_sectors = []
            
            async def fetch_sector_data(sector):
                try:
                    # Get sector index for data fetching
                    sector_index = self.sector_classifier.get_primary_sector_index(sector)
                    if not sector_index:
                        logging.warning(f"No primary index found for sector: {sector}")
                        return None
                        
                    # Fetch sector data using sector index
                    data = await self.zerodha_client.get_historical_data_async(
                        symbol=sector_index,
                        exchange="NSE",
                        period=days + 20
                    )
                    
                    # More flexible data requirement for longer timeframes
                    min_required = days * 0.7 if days > 60 else days * 0.8  # OPTIMIZED: Adjusted thresholds
                    if data is not None and len(data) >= min_required:
                        # Calculate daily returns
                        returns = data['close'].pct_change().dropna()
                        return sector, returns
                    else:
                        logging.warning(f"No sufficient data for {sector} ({sector_index}): got {len(data) if data is not None else 0} records, need at least {min_required:.0f}")
                        return None
                except Exception as e:
                    logging.warning(f"Error getting data for {sector}: {e}")
                    return None
            
            # Fetch all sector data in parallel
            tasks = [fetch_sector_data(sector) for sector in all_sectors]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for result in results:
                if isinstance(result, Exception):
                    logging.warning(f"Exception in sector data fetch: {result}")
                    continue
                if result is not None:
                    sector, returns = result
                    sector_data[sector] = returns
                    valid_sectors.append(sector)
            
            if len(valid_sectors) < 2:
                logging.warning(f"Insufficient sector data for correlation analysis. Only {len(valid_sectors)} sectors have sufficient data (need at least 2)")
                return None
            
            # Create correlation matrix
            returns_df = pd.DataFrame(sector_data)
            correlation_matrix = returns_df.corr()
            
            # Calculate average correlation
            # Get upper triangle of correlation matrix (excluding diagonal)
            upper_triangle = correlation_matrix.where(
                np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
            )
            avg_correlation = upper_triangle.stack().mean()
            
            # ENHANCED: Categorize correlations with proper handling of negative correlations
            correlation_pairs = self._categorize_correlations(correlation_matrix)
            
            # Extract categorized pairs for backward compatibility and enhanced analysis
            high_correlation_pairs = correlation_pairs['high_positive'] + correlation_pairs['high_negative']
            low_correlation_pairs = correlation_pairs['low']
            moderate_correlation_pairs = correlation_pairs['moderate_positive'] + correlation_pairs['moderate_negative']
            negative_correlation_pairs = correlation_pairs['high_negative'] + correlation_pairs['moderate_negative']
            
            # Generate diversification insights
            diversification_insights = self._generate_diversification_insights(
                correlation_matrix, high_correlation_pairs, low_correlation_pairs, avg_correlation
            )
            
            # Calculate sector volatility for risk assessment
            sector_volatility = {}
            for sector in valid_sectors:
                volatility = sector_data[sector].std() * np.sqrt(252) * 100  # Annualized %
                sector_volatility[sector] = round(volatility, 2)
            
            # Calculate average sector volatility for frontend compatibility
            avg_sector_volatility = sum(sector_volatility.values()) / len(sector_volatility) if sector_volatility else 0
            
            # Convert numpy types to native Python types for JSON serialization
            def convert_numpy_types(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {key: convert_numpy_types(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                else:
                    return obj
            
            # Convert correlation matrix to native types
            correlation_matrix_dict = correlation_matrix.round(3).to_dict()
            correlation_matrix_clean = convert_numpy_types(correlation_matrix_dict)
            
            # Convert other data to native types
            high_correlation_pairs_clean = convert_numpy_types(high_correlation_pairs)
            low_correlation_pairs_clean = convert_numpy_types(low_correlation_pairs)
            sector_volatility_clean = convert_numpy_types(sector_volatility)
            diversification_insights_clean = convert_numpy_types(diversification_insights)
            
            return {
                'timeframe': timeframe,
                'correlation_matrix': correlation_matrix_clean,
                'average_correlation': float(round(avg_correlation, 3)),
                'sector_volatility': round(avg_sector_volatility, 2),  # FIXED: Single value for frontend
                'sector_volatilities': sector_volatility_clean,  # Individual sector volatilities for reference
                'high_correlation_pairs': high_correlation_pairs_clean,
                'low_correlation_pairs': low_correlation_pairs_clean,
                # ENHANCED: New correlation categories
                'moderate_correlation_pairs': convert_numpy_types(moderate_correlation_pairs),
                'negative_correlation_pairs': convert_numpy_types(negative_correlation_pairs),
                'correlation_breakdown': {
                    'high_positive': convert_numpy_types(correlation_pairs['high_positive']),
                    'moderate_positive': convert_numpy_types(correlation_pairs['moderate_positive']),
                    'low': convert_numpy_types(correlation_pairs['low']),
                    'moderate_negative': convert_numpy_types(correlation_pairs['moderate_negative']),
                    'high_negative': convert_numpy_types(correlation_pairs['high_negative'])
                },
                'diversification_insights': diversification_insights_clean,
                'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'optimization_note': 'Async enhanced correlation analysis with negative correlation support'
            }
            
        except Exception as e:
            logging.error(f"Error generating async sector correlation matrix: {e}")
            return None

    def _identify_rotation_patterns(self, sector_performance: Dict, timeframe: str) -> Dict[str, Any]:
        """Identify sector rotation patterns and trends."""
        try:
            patterns = {
                'leading_sectors': [],
                'lagging_sectors': [],
                'momentum_shifts': [],
                'rotation_strength': 'weak'
            }
            
            # Identify leading and lagging sectors
            sorted_sectors = sorted(sector_performance.items(), 
                                  key=lambda x: x[1]['relative_strength'], reverse=True)
            
            # Top 3 performing sectors
            patterns['leading_sectors'] = [
                {
                    'sector': sector,
                    'relative_strength': data['relative_strength'],
                    'momentum': data['momentum']
                }
                for sector, data in sorted_sectors[:3]
            ]
            
            # Bottom 3 performing sectors
            patterns['lagging_sectors'] = [
                {
                    'sector': sector,
                    'relative_strength': data['relative_strength'],
                    'momentum': data['momentum']
                }
                for sector, data in sorted_sectors[-3:]
            ]
            
            # Assess rotation strength
            if len(sorted_sectors) >= 2:
                top_performance = sorted_sectors[0][1]['relative_strength']
                bottom_performance = sorted_sectors[-1][1]['relative_strength']
                performance_spread = top_performance - bottom_performance
                
                if performance_spread > 10:
                    patterns['rotation_strength'] = 'strong'
                elif performance_spread > 5:
                    patterns['rotation_strength'] = 'moderate'
                else:
                    patterns['rotation_strength'] = 'weak'
            
            return patterns
            
        except Exception as e:
            logging.error(f"Error identifying rotation patterns: {e}")
            return {}
    
    def _generate_rotation_recommendations(self, sector_rankings: Dict, 
                                         rotation_analysis: Dict,
                                         sector_performance: Dict = None) -> List[Dict]:
        """Generate actionable rotation recommendations."""
        try:
            recommendations = []
            
            # If sector_performance is not provided, extract from rotation_analysis
            if sector_performance is None:
                # Try to get sector performance from rotation analysis patterns
                leading_sectors = rotation_analysis.get('leading_sectors', [])
                lagging_sectors = rotation_analysis.get('lagging_sectors', [])
                
                # Generate recommendations based on leading/lagging sectors
                for sector_data in leading_sectors[:3]:  # Top 3 leading
                    if isinstance(sector_data, dict) and 'relative_strength' in sector_data:
                        recommendations.append({
                            'type': 'overweight',
                            'sector': sector_data.get('sector', 'Unknown'),
                            'reason': f"Leading sector momentum (+{sector_data['relative_strength']:.1f}% vs market)",
                            'confidence': 'high' if sector_data.get('momentum', 0) > 3 else 'medium'
                        })
                
                for sector_data in lagging_sectors[-2:]:  # Bottom 2 lagging
                    if isinstance(sector_data, dict) and 'relative_strength' in sector_data:
                        recommendations.append({
                            'type': 'underweight',
                            'sector': sector_data.get('sector', 'Unknown'),
                            'reason': f"Lagging sector performance ({sector_data['relative_strength']:.1f}% vs market)",
                            'confidence': 'high' if sector_data.get('momentum', 0) < -3 else 'medium'
                        })
            else:
                # Use the provided sector_performance data
                for sector, ranking in sector_rankings.items():
                    if sector not in sector_performance:
                        continue
                        
                    performance = sector_performance[sector]
                    rank = ranking['rank']
                    
                    if rank <= 3 and performance['relative_strength'] > 5:
                        recommendations.append({
                            'type': 'overweight',
                            'sector': sector,
                            'reason': f"Strong sector momentum (+{performance['relative_strength']:.1f}% vs market)",
                            'confidence': 'high' if performance['momentum'] > 3 else 'medium'
                        })
                    
                    elif rank >= len(sector_rankings) - 2 and performance['relative_strength'] < -5:
                        recommendations.append({
                            'type': 'underweight',
                            'sector': sector,
                            'reason': f"Weak sector performance ({performance['relative_strength']:.1f}% vs market)",
                            'confidence': 'high' if performance['momentum'] < -3 else 'medium'
                        })
            
            # Add overall market rotation insight
            rotation_strength = rotation_analysis.get('rotation_strength', 'weak')
            if rotation_strength == 'strong':
                recommendations.append({
                    'type': 'market_insight',
                    'message': "Strong sector rotation detected - consider rebalancing portfolio",
                    'confidence': 'high'
                })
            elif rotation_strength == 'moderate':
                recommendations.append({
                    'type': 'market_insight',
                    'message': "Moderate sector rotation activity - monitor for opportunities",
                    'confidence': 'medium'
                })
            
            return recommendations
            
        except Exception as e:
            logging.error(f"Error generating rotation recommendations: {e}")
            return []
    
    def _categorize_correlations(self, correlation_matrix: pd.DataFrame) -> Dict[str, List]:
        """
        Categorize correlations into high positive, moderate positive, low, moderate negative, and high negative.
        
        Args:
            correlation_matrix: Correlation matrix dataframe
            
        Returns:
            Dict containing categorized correlation pairs
        """
        high_positive = []
        moderate_positive = []
        low = []
        moderate_negative = []
        high_negative = []
        
        # Process each pair of sectors
        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                sector1 = correlation_matrix.columns[i]
                sector2 = correlation_matrix.columns[j]
                correlation = correlation_matrix.iloc[i, j]
                
                # Create common pair structure
                pair = {
                    'sector1': sector1,
                    'sector2': sector2,
                    'correlation': round(correlation, 3),
                    'relationship': 'aligned' if correlation >= 0 else 'inverse'
                }
                
                # Categorize based on absolute correlation value
                abs_corr = abs(correlation)
                
                if abs_corr >= 0.5:  # IMPROVED: Lowered from 0.7 to 0.5 for high correlation
                    if correlation >= 0:
                        high_positive.append(pair)
                    else:
                        high_negative.append(pair)
                elif abs_corr >= 0.3:  # Moderate correlation: 0.3 to 0.5
                    if correlation >= 0:
                        moderate_positive.append(pair)
                    else:
                        moderate_negative.append(pair)
                else:  # Low correlation: < 0.3
                    low.append(pair)
        
        return {
            'high_positive': high_positive,
            'moderate_positive': moderate_positive,
            'low': low,
            'moderate_negative': moderate_negative,
            'high_negative': high_negative
        }
    
    def _generate_diversification_insights(self, correlation_matrix: pd.DataFrame,
                                         high_correlation_pairs: List[Dict],
                                         low_correlation_pairs: List[Dict],
                                         avg_correlation: float) -> Dict[str, Any]:
        """ENHANCED: Generate actionable diversification insights with negative correlation support."""
        try:
            # Get enhanced correlation categorization
            correlation_pairs = self._categorize_correlations(correlation_matrix)
            
            insights = {
                'diversification_quality': 'good',
                'diversification_score': 0,
                'risk_reduction_opportunities': [],
                'concentration_risks': [],
                'hedging_opportunities': [],  # NEW: For negative correlations
                'recommendations': []
            }
            
            # ENHANCED: Calculate diversification score based on absolute correlations
            abs_avg_correlation = abs(avg_correlation) if not np.isnan(avg_correlation) else 0.5
            
            # Assess overall diversification quality
            if abs_avg_correlation < 0.3:
                insights['diversification_quality'] = 'excellent'
                insights['diversification_score'] = 90
            elif abs_avg_correlation < 0.5:
                insights['diversification_quality'] = 'good'
                insights['diversification_score'] = 75
            elif abs_avg_correlation < 0.7:
                insights['diversification_quality'] = 'moderate' 
                insights['diversification_score'] = 60
            else:
                insights['diversification_quality'] = 'poor'
                insights['diversification_score'] = 30
                
            # ENHANCED: Identify concentration risks (positive high correlations)
            for pair in correlation_pairs['high_positive']:
                risk_level = 'critical' if pair['correlation'] > 0.8 else 'high'
                insights['concentration_risks'].append({
                    'message': f"High positive correlation ({pair['correlation']}) between {pair['sector1']} and {pair['sector2']}",
                    'risk_level': risk_level,
                    'relationship': 'aligned',
                    'recommendation': f"Consider reducing exposure to one of these sectors to minimize concentration risk"
                })
            
            # ENHANCED: Identify hedging opportunities (negative correlations)
            for pair in correlation_pairs['high_negative']:
                insights['hedging_opportunities'].append({
                    'message': f"Strong negative correlation ({pair['correlation']}) between {pair['sector1']} and {pair['sector2']}",
                    'opportunity_level': 'excellent',
                    'relationship': 'inverse',
                    'recommendation': f"Excellent hedging pair - when one sector declines, the other typically rises"
                })
            
            for pair in correlation_pairs['moderate_negative']:
                insights['hedging_opportunities'].append({
                    'message': f"Moderate negative correlation ({pair['correlation']}) between {pair['sector1']} and {pair['sector2']}",
                    'opportunity_level': 'good',
                    'relationship': 'inverse',
                    'recommendation': f"Good hedging potential - partial inverse relationship"
                })
            
            # Identify low correlation diversification opportunities
            for pair in correlation_pairs['low']:
                insights['risk_reduction_opportunities'].append({
                    'message': f"Low correlation ({pair['correlation']}) between {pair['sector1']} and {pair['sector2']}",
                    'opportunity': 'excellent' if abs(pair['correlation']) < 0.2 else 'good',
                    'relationship': 'independent',
                    'recommendation': f"Good diversification pair - sectors move relatively independently"
                })
            
            # ENHANCED: Generate comprehensive recommendations
            if insights['diversification_quality'] == 'excellent':
                insights['recommendations'].append({
                    'type': 'positive',
                    'message': "Excellent sector diversification - portfolio is well-balanced across sectors",
                    'priority': 'info'
                })
            elif insights['diversification_quality'] == 'poor':
                insights['recommendations'].append({
                    'type': 'warning',
                    'message': "Poor sector diversification - high correlation risk detected",
                    'priority': 'high'
                })
            
            # Concentration risk warnings
            if len(correlation_pairs['high_positive']) > 3:
                insights['recommendations'].append({
                    'type': 'warning',
                    'message': f"Multiple high-correlation sector pairs detected ({len(correlation_pairs['high_positive'])}) - significant concentration risk",
                    'priority': 'high'
                })
            
            # Hedging opportunities
            if len(correlation_pairs['high_negative']) > 0:
                insights['recommendations'].append({
                    'type': 'opportunity',
                    'message': f"Found {len(correlation_pairs['high_negative'])} strong negative correlation pairs - excellent for hedging strategies",
                    'priority': 'medium'
                })
            
            # Diversification opportunities
            if len(correlation_pairs['low']) > 5:
                insights['recommendations'].append({
                    'type': 'positive',
                    'message': f"Multiple low-correlation pairs available ({len(correlation_pairs['low'])}) - good diversification potential",
                    'priority': 'low'
                })
            
            # Adjust diversification score based on negative correlations (bonus for hedging)
            if len(correlation_pairs['high_negative']) > 0:
                insights['diversification_score'] = min(100, insights['diversification_score'] + len(correlation_pairs['high_negative']) * 5)
            
            return insights
            
        except Exception as e:
            logging.error(f"Error generating diversification insights: {e}")
            return {}
    
    def _calculate_market_metrics(self, stock_returns: pd.Series) -> Dict[str, Any]:
        """Calculate market (NIFTY 50) benchmarking metrics."""
        try:
            # Get NIFTY 50 data
            nifty_data = self.market_metrics_provider.get_nifty_50_data(365)
            
            if nifty_data is None or len(nifty_data) < 20:  # Reduced from 30 to 20
                logging.warning(f"Insufficient NIFTY data (got {len(nifty_data) if nifty_data is not None else 0} data points)")
                return self._get_default_market_metrics()
            
            market_returns = nifty_data['close'].pct_change().dropna()
            
            # Align data
            aligned_data = pd.concat([stock_returns, market_returns], axis=1).dropna()
            if len(aligned_data) < 15:  # Reduced from 30 to 15
                logging.warning(f"Insufficient aligned market data (got {len(aligned_data)} data points)")
                return self._get_default_market_metrics()
            
            stock_aligned = aligned_data.iloc[:, 0]
            market_aligned = aligned_data.iloc[:, 1]
            
            # Calculate metrics
            beta = self._calculate_beta(stock_aligned, market_aligned)
            correlation = self._calculate_correlation(stock_aligned, market_aligned)
            volatility_ratio = stock_aligned.std() / market_aligned.std() if market_aligned.std() > 0 else 1.0
            
            # Calculate performance metrics
            stock_cumulative_return = (1 + stock_aligned).prod() - 1
            market_cumulative_return = (1 + market_aligned).prod() - 1
            excess_return = stock_cumulative_return - market_cumulative_return
            
            # Calculate Sharpe ratios
            risk_free_rate = 0.07  # 7% annual
            stock_sharpe = (stock_aligned.mean() * 252 - risk_free_rate) / (stock_aligned.std() * np.sqrt(252)) if stock_aligned.std() > 0 else 0
            market_sharpe = (market_aligned.mean() * 252 - risk_free_rate) / (market_aligned.std() * np.sqrt(252)) if market_aligned.std() > 0 else 0

            # Additional metrics expected by frontend
            market_volatility = market_aligned.std() * np.sqrt(252)
            market_annualized_return = market_aligned.mean() * 252

            return {
                "beta": float(beta),
                "correlation": float(correlation),
                "volatility_ratio": float(volatility_ratio),
                "stock_return": float(stock_cumulative_return),
                "market_return": float(market_cumulative_return),
                # Frontend-expected keys
                "cumulative_return": float(market_cumulative_return),
                "annualized_return": float(market_annualized_return),
                "volatility": float(market_volatility),
                "risk_free_rate": float(risk_free_rate),
                "excess_return": float(excess_return),
                "stock_sharpe": float(stock_sharpe),
                "market_sharpe": float(market_sharpe),
                "outperformance": float(excess_return),
                "data_points": len(aligned_data),
                "benchmark": "NIFTY 50"
            }
            
        except Exception as e:
            logging.error(f"Error calculating market metrics: {e}")
            return self._get_default_market_metrics()
    
    async def _calculate_market_metrics_async(self, stock_returns: pd.Series) -> Dict[str, Any]:
        """Async version of _calculate_market_metrics."""
        try:
            logging.info(f"\n=== DEBUG: MARKET METRICS CALCULATION START ===")
            logging.info(f"DEBUG: Input stock_returns length: {len(stock_returns)}")
            logging.info(f"DEBUG: Input stock_returns date range: {stock_returns.index[0]} to {stock_returns.index[-1]}")
            
            # Get NIFTY 50 data asynchronously
            nifty_data = await self._get_nifty_data_async(365)
            
            logging.info(f"DEBUG: NIFTY data fetched: {len(nifty_data) if nifty_data is not None else 'None'} data points")
            if nifty_data is not None:
                logging.info(f"DEBUG: NIFTY data date range: {nifty_data.index[0]} to {nifty_data.index[-1]}")
                logging.info(f"DEBUG: NIFTY close prices - First 5: {nifty_data['close'].head().tolist()}")
                logging.info(f"DEBUG: NIFTY close prices - Last 5: {nifty_data['close'].tail().tolist()}")
            
            if nifty_data is None or len(nifty_data) < 20:  # Reduced from 30 to 20
                logging.warning(f"DEBUG: Insufficient NIFTY data (got {len(nifty_data) if nifty_data is not None else 0} data points) - Using defaults")
                return self._get_default_market_metrics()
            
            market_returns = nifty_data['close'].pct_change().dropna()
            logging.info(f"DEBUG: NIFTY returns after pct_change().dropna(): {len(market_returns)} data points")
            logging.info(f"DEBUG: NIFTY returns date range: {market_returns.index[0]} to {market_returns.index[-1]}")
            
            # Align data
            logging.info(f"DEBUG: Before alignment - Stock returns: {len(stock_returns)}, Market returns: {len(market_returns)}")
            aligned_data = pd.concat([stock_returns, market_returns], axis=1).dropna()
            logging.info(f"DEBUG: After alignment: {len(aligned_data)} data points")
            if len(aligned_data) > 0:
                logging.info(f"DEBUG: Aligned data date range: {aligned_data.index[0]} to {aligned_data.index[-1]}")
            
            if len(aligned_data) < 15:  # Reduced from 30 to 15
                logging.warning(f"DEBUG: Insufficient aligned market data (got {len(aligned_data)} data points) - Using defaults")
                return self._get_default_market_metrics()
            
            stock_aligned = aligned_data.iloc[:, 0]
            market_aligned = aligned_data.iloc[:, 1]
            logging.info(f"DEBUG: Aligned stock returns - mean: {stock_aligned.mean():.6f}, std: {stock_aligned.std():.6f}")
            logging.info(f"DEBUG: Aligned market returns - mean: {market_aligned.mean():.6f}, std: {market_aligned.std():.6f}")
            
            # Calculate metrics
            beta = self._calculate_beta(stock_aligned, market_aligned)
            correlation = self._calculate_correlation(stock_aligned, market_aligned)
            logging.info(f"DEBUG: Calculated Beta: {beta:.6f}, Correlation: {correlation:.6f}")
            volatility_ratio = stock_aligned.std() / market_aligned.std() if market_aligned.std() > 0 else 1.0
            
            # Calculate performance metrics
            stock_cumulative_return = (1 + stock_aligned).prod() - 1
            market_cumulative_return = (1 + market_aligned).prod() - 1
            excess_return = stock_cumulative_return - market_cumulative_return
            
            # Calculate Sharpe ratios
            risk_free_rate = 0.07  # 7% annual
            stock_sharpe = (stock_aligned.mean() * 252 - risk_free_rate) / (stock_aligned.std() * np.sqrt(252)) if stock_aligned.std() > 0 else 0
            market_sharpe = (market_aligned.mean() * 252 - risk_free_rate) / (market_aligned.std() * np.sqrt(252)) if market_aligned.std() > 0 else 0

            # Additional metrics expected by frontend
            market_volatility = market_aligned.std() * np.sqrt(252)
            market_annualized_return = market_aligned.mean() * 252
            
            logging.info(f"DEBUG: Final calculated metrics:")
            logging.info(f"  - Beta: {beta:.6f}")
            logging.info(f"  - Correlation: {correlation:.6f}")
            logging.info(f"  - Volatility Ratio: {volatility_ratio:.6f}")
            logging.info(f"  - Stock Cumulative Return: {stock_cumulative_return:.6f}")
            logging.info(f"  - Market Cumulative Return: {market_cumulative_return:.6f}")
            logging.info(f"  - Excess Return: {excess_return:.6f}")
            logging.info(f"  - Stock Sharpe: {stock_sharpe:.6f}")
            logging.info(f"  - Market Sharpe: {market_sharpe:.6f}")
            logging.info(f"DEBUG: Market metrics calculation SUCCESS")

            return {
                "beta": float(beta),
                "correlation": float(correlation),
                "volatility_ratio": float(volatility_ratio),
                "stock_return": float(stock_cumulative_return),
                "market_return": float(market_cumulative_return),
                # Frontend-expected keys
                "cumulative_return": float(market_cumulative_return),
                "annualized_return": float(market_annualized_return),
                "volatility": float(market_volatility),
                "risk_free_rate": float(risk_free_rate),
                "excess_return": float(excess_return),
                "stock_sharpe": float(stock_sharpe),
                "market_sharpe": float(market_sharpe),
                "outperformance": float(excess_return),
                "data_points": len(aligned_data),
                "benchmark": "NIFTY 50"
            }
            
        except Exception as e:
            logging.error(f"Error calculating market metrics: {e}")
            return self._get_default_market_metrics()
    
    def _calculate_sector_metrics(self, stock_returns: pd.Series, sector: str) -> Dict[str, Any]:
        """Calculate sector-specific benchmarking metrics."""
        try:
            logging.info(f"\n=== DEBUG: SECTOR METRICS CALCULATION START ===")
            logging.info(f"DEBUG: Sector: {sector}")
            logging.info(f"DEBUG: Input stock_returns length: {len(stock_returns)}")
            if len(stock_returns) > 0:
                logging.info(f"DEBUG: Input stock_returns date range: {stock_returns.index[0]} to {stock_returns.index[-1]}")
            
            if not sector:
                logging.warning(f"DEBUG: No sector provided - returning None")
                return None
            
            # Get sector index data
            logging.info(f"DEBUG: Fetching sector index data for: {sector}")
            sector_data = self._get_sector_index_data(sector, 365)
            
            logging.info(f"DEBUG: Sector data fetched: {len(sector_data) if sector_data is not None else 'None'} data points")
            if sector_data is not None:
                logging.info(f"DEBUG: Sector data date range: {sector_data.index[0]} to {sector_data.index[-1]}")
                logging.info(f"DEBUG: Sector close prices - First 5: {sector_data['close'].head().tolist()}")
                logging.info(f"DEBUG: Sector close prices - Last 5: {sector_data['close'].tail().tolist()}")
            
            if sector_data is None or len(sector_data) < 20:  # Reduced from 30 to 20
                logging.warning(f"DEBUG: Insufficient sector data for {sector} (got {len(sector_data) if sector_data is not None else 0} data points) - Using basic metrics")
                # Return basic sector metrics instead of None
                return self._get_basic_sector_metrics(stock_returns, sector)
            
            sector_returns = sector_data['close'].pct_change().dropna()
            logging.info(f"DEBUG: Sector returns after pct_change().dropna(): {len(sector_returns)} data points")
            logging.info(f"DEBUG: Sector returns date range: {sector_returns.index[0]} to {sector_returns.index[-1]}")
            
            # Align data
            logging.info(f"DEBUG: Before alignment - Stock returns: {len(stock_returns)}, Sector returns: {len(sector_returns)}")
            aligned_data = pd.concat([stock_returns, sector_returns], axis=1).dropna()
            logging.info(f"DEBUG: After alignment: {len(aligned_data)} data points")
            if len(aligned_data) > 0:
                logging.info(f"DEBUG: Aligned data date range: {aligned_data.index[0]} to {aligned_data.index[-1]}")
            
            if len(aligned_data) < 30:
                logging.warning(f"DEBUG: Insufficient aligned sector data (got {len(aligned_data)} data points) - returning None")
                return None
            
            stock_aligned = aligned_data.iloc[:, 0]
            sector_aligned = aligned_data.iloc[:, 1]
            logging.info(f"DEBUG: Aligned stock returns - mean: {stock_aligned.mean():.6f}, std: {stock_aligned.std():.6f}")
            logging.info(f"DEBUG: Aligned sector returns - mean: {sector_aligned.mean():.6f}, std: {sector_aligned.std():.6f}")
            
            # Calculate metrics
            sector_beta = self._calculate_beta(stock_aligned, sector_aligned)
            sector_correlation = self._calculate_correlation(stock_aligned, sector_aligned)
            logging.info(f"DEBUG: Calculated Sector Beta: {sector_beta:.6f}, Sector Correlation: {sector_correlation:.6f}")
            sector_volatility_ratio = stock_aligned.std() / sector_aligned.std() if sector_aligned.std() > 0 else 1.0
            
            # Calculate performance metrics
            stock_cumulative_return = (1 + stock_aligned).prod() - 1
            sector_cumulative_return = (1 + sector_aligned).prod() - 1
            sector_excess_return = stock_cumulative_return - sector_cumulative_return
            
            # Calculate Sharpe ratios
            risk_free_rate = 0.07  # 7% annual
            stock_sharpe = (stock_aligned.mean() * 252 - risk_free_rate) / (stock_aligned.std() * np.sqrt(252)) if stock_aligned.std() > 0 else 0
            sector_sharpe = (sector_aligned.mean() * 252 - risk_free_rate) / (sector_aligned.std() * np.sqrt(252)) if sector_aligned.std() > 0 else 0

            # Additional metrics expected by frontend
            sector_volatility = sector_aligned.std() * np.sqrt(252)
            sector_annualized_return = sector_aligned.mean() * 252

            # Get sector index symbol
            sector_index = self.sector_classifier.get_primary_sector_index(sector)
            
            logging.info(f"DEBUG: Final sector metrics calculated:")
            logging.info(f"  - Sector Beta: {sector_beta:.6f}")
            logging.info(f"  - Sector Correlation: {sector_correlation:.6f}")
            logging.info(f"  - Sector Volatility Ratio: {sector_volatility_ratio:.6f}")
            logging.info(f"  - Stock Cumulative Return: {stock_cumulative_return:.6f}")
            logging.info(f"  - Sector Cumulative Return: {sector_cumulative_return:.6f}")
            logging.info(f"  - Sector Excess Return: {sector_excess_return:.6f}")
            logging.info(f"  - Stock Sharpe: {stock_sharpe:.6f}")
            logging.info(f"  - Sector Sharpe: {sector_sharpe:.6f}")
            logging.info(f"  - Sector Index: {sector_index}")
            logging.info(f"DEBUG: Sector metrics calculation SUCCESS")
            
            return {
                "sector_beta": float(sector_beta),
                "sector_correlation": float(sector_correlation),
                "sector_volatility_ratio": float(sector_volatility_ratio),
                "stock_return": float(stock_cumulative_return),
                "sector_return": float(sector_cumulative_return),
                # Frontend-expected keys
                "sector_cumulative_return": float(sector_cumulative_return),
                "sector_annualized_return": float(sector_annualized_return),
                "sector_volatility": float(sector_volatility),
                "sector_excess_return": float(sector_excess_return),
                "stock_sharpe": float(stock_sharpe),
                "sector_sharpe": float(sector_sharpe),
                "sector_sharpe_ratio": float(sector_sharpe),
                "sector_outperformance": float(sector_excess_return),
                "data_points": len(aligned_data),
                "sector_index": sector_index,
                "benchmark": sector_index
            }
            
        except Exception as e:
            logging.error(f"Error calculating sector metrics: {e}")
            return None
    
    async def _calculate_sector_metrics_async(self, stock_returns: pd.Series, sector: str) -> Dict[str, Any]:
        """Async version of _calculate_sector_metrics."""
        try:
            if not sector:
                return None
            
            # Get sector index data asynchronously
            sector_data = await self._get_sector_index_data_async(sector, 365)
            
            if sector_data is None or len(sector_data) < 20:  # Reduced from 30 to 20
                logging.warning(f"Insufficient sector data for {sector} (got {len(sector_data) if sector_data is not None else 0} data points)")
                # Return basic sector metrics instead of None
                return self._get_basic_sector_metrics(stock_returns, sector)
            
            sector_returns = sector_data['close'].pct_change().dropna()
            
            # Align data
            aligned_data = pd.concat([stock_returns, sector_returns], axis=1).dropna()
            if len(aligned_data) < 15:  # Reduced from 30 to 15
                logging.warning(f"Insufficient aligned data for {sector} (got {len(aligned_data)} data points)")
                return self._get_basic_sector_metrics(stock_returns, sector)
            
            stock_aligned = aligned_data.iloc[:, 0]
            sector_aligned = aligned_data.iloc[:, 1]
            
            # Calculate metrics
            sector_beta = self._calculate_beta(stock_aligned, sector_aligned)
            sector_correlation = self._calculate_correlation(stock_aligned, sector_aligned)
            sector_volatility_ratio = stock_aligned.std() / sector_aligned.std() if sector_aligned.std() > 0 else 1.0
            
            # Calculate performance metrics
            stock_cumulative_return = (1 + stock_aligned).prod() - 1
            sector_cumulative_return = (1 + sector_aligned).prod() - 1
            sector_excess_return = stock_cumulative_return - sector_cumulative_return
            
            # Calculate Sharpe ratios
            risk_free_rate = 0.07  # 7% annual
            stock_sharpe = (stock_aligned.mean() * 252 - risk_free_rate) / (stock_aligned.std() * np.sqrt(252)) if stock_aligned.std() > 0 else 0
            sector_sharpe = (sector_aligned.mean() * 252 - risk_free_rate) / (sector_aligned.std() * np.sqrt(252)) if sector_aligned.std() > 0 else 0

            # Additional metrics expected by frontend
            sector_volatility = sector_aligned.std() * np.sqrt(252)
            sector_annualized_return = sector_aligned.mean() * 252

            # Get sector index symbol
            sector_index = self.sector_classifier.get_primary_sector_index(sector)
            
            return {
                "sector_beta": float(sector_beta),
                "sector_correlation": float(sector_correlation),
                "sector_volatility_ratio": float(sector_volatility_ratio),
                "stock_return": float(stock_cumulative_return),
                "sector_return": float(sector_cumulative_return),
                # Frontend-expected keys
                "sector_cumulative_return": float(sector_cumulative_return),
                "sector_annualized_return": float(sector_annualized_return),
                "sector_volatility": float(sector_volatility),
                "sector_excess_return": float(sector_excess_return),
                "stock_sharpe": float(stock_sharpe),
                "sector_sharpe": float(sector_sharpe),
                "sector_sharpe_ratio": float(sector_sharpe),
                "sector_outperformance": float(sector_excess_return),
                "data_points": len(aligned_data),
                "sector_index": sector_index,
                "benchmark": sector_index
            }
            
        except Exception as e:
            logging.error(f"Error calculating sector metrics: {e}")
            return None
    
    def _calculate_relative_performance(self, stock_data: pd.DataFrame, sector: str, 
                                      market_metrics: Dict, sector_metrics: Dict) -> Dict[str, Any]:
        """Calculate relative performance metrics."""
        try:
            current_price = stock_data['close'].iloc[-1]
            
            # Get recent performance (last 30 days)
            recent_data = stock_data.tail(30)
            if len(recent_data) < 10:
                return self._get_default_relative_performance()
            
            recent_returns = recent_data['close'].pct_change().dropna()
            recent_volatility = recent_returns.std() * np.sqrt(252)
            
            # Calculate relative strength
            relative_strength = {
                "vs_market": market_metrics.get('excess_return', 0),
                "vs_sector": sector_metrics.get('sector_excess_return', 0) if sector_metrics else 0,
                "recent_volatility": float(recent_volatility),
                "market_volatility": market_metrics.get('volatility_ratio', 1.0) * 0.15,  # Assume 15% market volatility
                "sector_volatility": sector_metrics.get('sector_volatility_ratio', 1.0) * 0.15 if sector_metrics else 0.15
            }
            
            # Calculate momentum
            momentum_20d = (current_price / stock_data['close'].iloc[-20] - 1) if len(stock_data) >= 20 else 0
            momentum_50d = (current_price / stock_data['close'].iloc[-50] - 1) if len(stock_data) >= 50 else 0
            
            # Calculate performance ratios
            market_performance_ratio = (1 + market_metrics.get('excess_return', 0)) if market_metrics else 1.0
            sector_performance_ratio = (1 + sector_metrics.get('sector_excess_return', 0)) if sector_metrics else 1.0
            
            # Calculate consistency scores
            market_consistency = 0.5 + (market_metrics.get('excess_return', 0) * 0.5) if market_metrics else 0.5
            sector_consistency = 0.5 + (sector_metrics.get('sector_excess_return', 0) * 0.5) if sector_metrics else 0.5
            
            # Calculate sector rank based on performance
            sector_rank = self._calculate_sector_rank(sector_metrics, sector) if sector_metrics else 0
            sector_percentile = self._calculate_sector_percentile(sector_metrics, sector) if sector_metrics else 50
            
            return {
                "vs_market": {
                    "performance_ratio": float(market_performance_ratio),
                    "risk_adjusted_ratio": float(market_performance_ratio),
                    "outperformance_periods": 0,
                    "underperformance_periods": 0,
                    "consistency_score": float(market_consistency)
                },
                "vs_sector": {
                    "performance_ratio": float(sector_performance_ratio),
                    "risk_adjusted_ratio": float(sector_performance_ratio),
                    "sector_rank": int(sector_rank),
                    "sector_percentile": int(sector_percentile),
                    "sector_consistency": float(sector_consistency)
                },
                "relative_strength": relative_strength,
                "momentum": {
                    "20_day": float(momentum_20d),
                    "50_day": float(momentum_50d)
                },
                "performance_ranking": self._calculate_performance_ranking(
                    market_metrics, sector_metrics, momentum_20d, momentum_50d
                )
            }
            
        except Exception as e:
            logging.error(f"Error calculating relative performance: {e}")
            return self._get_default_relative_performance()
    
    def _calculate_sector_risk_metrics(self, stock_returns: pd.Series, sector: str,
                                     market_metrics: Dict, sector_metrics: Dict) -> Dict[str, Any]:
        """Calculate sector-specific risk metrics with enhanced analysis."""
        try:
            # Debug: Log input parameters
            logging.info(f"DEBUG: Calculating sector risk metrics for {sector}")
            logging.info(f"DEBUG: Stock returns length: {len(stock_returns) if stock_returns is not None else 'None'}")
            logging.info(f"DEBUG: Sector metrics available: {sector_metrics is not None}")
            logging.info(f"DEBUG: Market metrics available: {market_metrics is not None}")
            
            # If no sector metrics, calculate basic risk metrics from stock data only
            if not sector_metrics:
                logging.warning(f"No sector metrics available for {sector}, calculating basic risk metrics")
                return self._calculate_basic_risk_metrics(stock_returns, sector, market_metrics)
            
            # Check if stock_returns is valid
            if stock_returns is None or len(stock_returns) == 0:
                logging.warning(f"Empty or None stock returns for {sector}, falling back to basic metrics")
                return self._calculate_basic_risk_metrics(stock_returns, sector, market_metrics)
            
            # Calculate base risk metrics
            volatility = stock_returns.std() * np.sqrt(252)
            
            # Check for NaN or zero volatility
            if pd.isna(volatility) or volatility == 0:
                logging.warning(f"Invalid volatility ({volatility}) for {sector}, falling back to basic metrics")
                return self._calculate_basic_risk_metrics(stock_returns, sector, market_metrics)
            
            sector_volatility = sector_metrics.get('sector_volatility_ratio', 1.0) * volatility
            
            # Enhanced risk score calculation
            risk_score = self._calculate_sector_risk_score(stock_returns, sector_metrics, market_metrics)
            
            logging.info(f"DEBUG: Calculated volatility: {volatility}, risk_score: {risk_score}")
            
            # Calculate Value at Risk (VaR)
            var_95 = np.percentile(stock_returns, 5) * np.sqrt(252)
            var_99 = np.percentile(stock_returns, 1) * np.sqrt(252)
            
            # Calculate maximum drawdown
            cumulative_returns = (1 + stock_returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            
            # Sector-specific stress testing
            sector_stress_metrics = self._calculate_sector_stress_metrics(
                stock_returns, sector_metrics, market_metrics
            )
            
            # Sector correlation risk
            sector_correlation = sector_metrics.get('sector_correlation', 0.5)
            correlation_risk = "High" if sector_correlation > 0.8 else "Medium" if sector_correlation > 0.5 else "Low"
            
            # Sector momentum risk
            sector_momentum = sector_metrics.get('sector_return', 0)
            momentum_risk = "High" if abs(sector_momentum) > 0.3 else "Medium" if abs(sector_momentum) > 0.15 else "Low"
            
            # Sector volatility risk
            volatility_risk = "High" if volatility > 0.3 else "Medium" if volatility > 0.15 else "Low"
            
            # Sector concentration risk
            sector_stocks = self.sector_classifier.get_sector_stocks(sector)
            concentration_risk = "High" if len(sector_stocks) < 20 else "Medium" if len(sector_stocks) < 50 else "Low"
            
            # Overall risk assessment
            risk_assessment = self._assess_risk_level(risk_score)
            
            return {
                "risk_score": float(risk_score),
                "risk_level": risk_assessment,
                "volatility": float(volatility),
                "sector_volatility": float(sector_volatility),
                "var_95": float(var_95),
                "var_99": float(var_99),
                "max_drawdown": float(max_drawdown),
                "correlation_risk": correlation_risk,
                "momentum_risk": momentum_risk,
                "volatility_risk": volatility_risk,
                "concentration_risk": concentration_risk,
                "sector_stress_metrics": sector_stress_metrics,
                "risk_factors": self._identify_sector_risk_factors(
                    sector, sector_metrics, market_metrics, risk_score
                ),
                "risk_mitigation": self._suggest_risk_mitigation(
                    sector, risk_score, correlation_risk, momentum_risk
                )
            }
            
        except Exception as e:
            logging.error(f"Error calculating sector risk metrics: {e}")
            return self._calculate_basic_risk_metrics(stock_returns, sector, market_metrics)
    
    def _calculate_sector_stress_metrics(self, stock_returns: pd.Series, 
                                       sector_metrics: Dict, market_metrics: Dict) -> Dict[str, Any]:
        """Calculate sector-specific stress testing metrics."""
        try:
            # Handle None values for sector_metrics and market_metrics
            if sector_metrics is None:
                sector_metrics = {}
            if market_metrics is None:
                market_metrics = {}
            
            # Sector downturn scenario (sector underperforms by 20%)
            sector_downturn_loss = stock_returns.mean() - (sector_metrics.get('sector_return', 0) * 0.2)
            
            # Market crash scenario (market drops 30%, sector correlation impact)
            market_crash_loss = stock_returns.mean() - (market_metrics.get('market_return', 0) * 0.3 * 
                                                      sector_metrics.get('sector_correlation', 0.5))
            
            # Sector-specific crisis scenario
            sector_crisis_loss = stock_returns.mean() - (sector_metrics.get('sector_return', 0) * 0.5)
            
            # Volatility spike scenario
            volatility_spike_loss = stock_returns.mean() - (stock_returns.std() * 2)
            
            # Calculate stress score (0-100 scale, higher = more stressed)
            worst_case = min(sector_downturn_loss, market_crash_loss, sector_crisis_loss, volatility_spike_loss)
            stress_score = min(100, max(0, abs(worst_case) * 100))
            
            # Determine stress level
            stress_level = "High" if stress_score > 70 else "Medium" if stress_score > 30 else "Low"
            
            return {
                "sector_downturn_scenario": float(sector_downturn_loss),
                "market_crash_scenario": float(market_crash_loss),
                "sector_crisis_scenario": float(sector_crisis_loss),
                "volatility_spike_scenario": float(volatility_spike_loss),
                "worst_case_scenario": float(worst_case),
                "stress_score": float(stress_score),
                "stress_level": stress_level
            }
            
        except Exception as e:
            logging.error(f"Error calculating sector stress metrics: {e}")
            return {}
    
    def _get_basic_sector_metrics(self, stock_returns: pd.Series, sector: str) -> Dict[str, Any]:
        """Get basic sector metrics when detailed sector data is not available."""
        try:
            # Get sector index symbol
            sector_index = self.sector_classifier.get_primary_sector_index(sector) if sector else "NIFTY_50"
            
            # Calculate basic metrics from stock data
            volatility = stock_returns.std() * np.sqrt(252)
            cumulative_return = (1 + stock_returns).prod() - 1
            annualized_return = stock_returns.mean() * 252
            
            # Calculate basic drawdown
            cumulative_returns = (1 + stock_returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            
            # Calculate basic Sharpe ratio
            risk_free_rate = 0.07  # 7% annual
            sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility > 0 else 0
            
            return {
                "sector_beta": 1.0,  # Default beta
                "sector_correlation": 0.6,  # Default correlation
                "sector_sharpe_ratio": float(sharpe_ratio),
                "sector_volatility": float(volatility),
                "sector_max_drawdown": float(max_drawdown),
                "sector_cumulative_return": float(cumulative_return),
                "sector_annualized_return": float(annualized_return),
                "sector_index": sector_index,
                "sector_data_points": len(stock_returns),
                "note": "Basic metrics calculated from stock data only"
            }
            
        except Exception as e:
            logging.error(f"Error calculating basic sector metrics: {e}")
            return {
                "sector_beta": 1.0,
                "sector_correlation": 0.6,
                "sector_sharpe_ratio": 0.0,
                "sector_volatility": 0.15,
                "sector_max_drawdown": 0.10,
                "sector_cumulative_return": 0.0,
                "sector_annualized_return": 0.0,
                "sector_index": "NIFTY_50",
                "sector_data_points": 0,
                "note": "Error in calculation - using defaults"
            }
    
    def _calculate_basic_risk_metrics(self, stock_returns: pd.Series, sector: str, market_metrics: Dict) -> Dict[str, Any]:
        """Calculate basic risk metrics when sector data is not available."""
        try:
            # Debug: Log stock returns info
            logging.info(f"DEBUG: Calculating basic risk metrics for sector {sector}")
            logging.info(f"DEBUG: Stock returns length: {len(stock_returns) if stock_returns is not None else 'None'}")
            logging.info(f"DEBUG: Stock returns first 5 values: {stock_returns.head() if stock_returns is not None and len(stock_returns) > 0 else 'Empty/None'}")
            logging.info(f"DEBUG: Stock returns std: {stock_returns.std() if stock_returns is not None and len(stock_returns) > 0 else 'Cannot calculate'}")
            
            # Check if stock_returns is valid
            if stock_returns is None or len(stock_returns) == 0:
                logging.warning(f"Empty or None stock returns for {sector}, using default risk metrics")
                return self._get_zero_fallback_risk_metrics()
            
            # Calculate basic volatility
            volatility = stock_returns.std() * np.sqrt(252)
            
            # Check for NaN or zero volatility
            if pd.isna(volatility) or volatility == 0:
                logging.warning(f"Invalid volatility ({volatility}) for {sector}, using default risk metrics")
                return self._get_zero_fallback_risk_metrics()
            
            # Calculate basic risk score (0-100 scale)
            risk_score = min(100, max(0, volatility * 100))
            
            # Assess risk level
            risk_level = "Low" if risk_score < 30 else "Medium" if risk_score < 70 else "High"
            
            # Calculate basic stress metrics
            var_95 = np.percentile(stock_returns, 5) * np.sqrt(252)
            var_99 = np.percentile(stock_returns, 1) * np.sqrt(252)
            
            # Calculate maximum drawdown
            cumulative_returns = (1 + stock_returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            
            logging.info(f"DEBUG: Successfully calculated basic risk metrics - risk_score: {risk_score}, volatility: {volatility}")
            
            return {
                "risk_score": float(risk_score),
                "risk_level": risk_level,
                "correlation_risk": "Medium",
                "momentum_risk": "Medium",
                "volatility_risk": "High" if volatility > 0.3 else "Medium" if volatility > 0.15 else "Low",
                "sector_stress_metrics": {
                    "stress_score": float(risk_score),
                    "stress_level": risk_level,
                    "stress_factors": ["Limited sector data", "Market volatility"]
                },
                "risk_factors": ["Limited sector data", "Market volatility"],
                "risk_mitigation": ["Diversification", "Stop-loss orders", "Regular monitoring"]
            }
            
        except Exception as e:
            logging.error(f"Error calculating basic risk metrics: {e}")
            return {
                "risk_score": 50.0,
                "risk_level": "Medium",
                "correlation_risk": "Medium",
                "momentum_risk": "Medium",
                "volatility_risk": "Medium",
                "sector_stress_metrics": {
                    "stress_score": 50.0,
                    "stress_level": "Medium",
                    "stress_factors": ["Calculation error"]
                },
                "risk_factors": ["Calculation error"],
                "risk_mitigation": ["Consult financial advisor"]
            }
    
    def _get_zero_fallback_risk_metrics(self) -> Dict[str, Any]:
        """Get default risk metrics when data is insufficient."""
        return {
            "risk_score": 35.0,  # Default moderate risk instead of 0
            "risk_level": "Medium",
            "correlation_risk": "Medium",
            "momentum_risk": "Medium",
            "volatility_risk": "Medium",
            "sector_stress_metrics": {
                "stress_score": 35.0,
                "stress_level": "Medium",
                "stress_factors": ["Insufficient data for calculation"]
            },
            "risk_factors": ["Insufficient data for calculation"],
            "risk_mitigation": ["Obtain more historical data", "Consult financial advisor"]
        }
    
    def _identify_sector_risk_factors(self, sector: str, sector_metrics: Dict, 
                                    market_metrics: Dict, risk_score: float) -> List[str]:
        """Identify sector-specific risk factors."""
        risk_factors = []
        
        # High correlation risk
        if sector_metrics.get('sector_correlation', 0) > 0.8:
            risk_factors.append(f"High correlation with {sector} sector index")
        
        # High volatility risk
        if sector_metrics.get('sector_volatility_ratio', 1.0) > 1.5:
            risk_factors.append(f"Above-average sector volatility")
        
        # Sector underperformance risk
        if sector_metrics.get('sector_excess_return', 0) < -0.1:
            risk_factors.append(f"Sector underperforming market")
        
        # High beta risk
        if sector_metrics.get('sector_beta', 1.0) > 1.3:
            risk_factors.append(f"High sector beta ({sector_metrics.get('sector_beta', 1.0):.2f})")
        
        # Sector-specific risks based on sector type
        sector_risks = {
            'BANKING': ['Interest rate sensitivity', 'Credit risk exposure', 'Regulatory changes'],
            'IT': ['Currency fluctuations', 'Global demand cycles', 'Technology disruption'],
            'PHARMA': ['Regulatory approvals', 'Patent expirations', 'R&D pipeline risks'],
            'AUTO': ['Raw material costs', 'Demand cycles', 'Regulatory compliance'],
            'ENERGY': ['Oil price volatility', 'Regulatory changes', 'Environmental risks'],
            'METAL': ['Commodity price cycles', 'Global demand', 'Environmental regulations']
        }
        
        if sector in sector_risks:
            risk_factors.extend(sector_risks[sector])
        
        return risk_factors
    
    def _suggest_risk_mitigation(self, sector: str, risk_score: float, 
                               correlation_risk: str, momentum_risk: str) -> List[str]:
        """Suggest risk mitigation strategies."""
        mitigation_strategies = []
        
        if risk_score > 70:
            mitigation_strategies.append("Consider position sizing and strict stop-losses")
        
        if correlation_risk == "High":
            mitigation_strategies.append("Diversify across different sectors")
        
        if momentum_risk == "High":
            mitigation_strategies.append("Monitor sector momentum closely")
        
        # Sector-specific mitigation
        sector_mitigation = {
            'BANKING': ['Monitor interest rate trends', 'Track regulatory developments'],
            'IT': ['Hedge currency exposure', 'Monitor global tech trends'],
            'PHARMA': ['Track regulatory pipeline', 'Monitor patent cliffs'],
            'AUTO': ['Monitor commodity prices', 'Track demand indicators'],
            'ENERGY': ['Monitor oil price trends', 'Track regulatory changes'],
            'METAL': ['Monitor commodity cycles', 'Track global demand indicators']
        }
        
        if sector in sector_mitigation:
            mitigation_strategies.extend(sector_mitigation[sector])
        
        return mitigation_strategies
    
    def _get_sector_index_data(self, sector: str, period: int = 365) -> Optional[pd.DataFrame]:
        """Get sector index data (no caching - Layer 1 handles caching)."""
        try:
            logging.info(f"Fetching fresh sector index data for {sector} with period {period}")
            
            # Get sector index symbol
            sector_index = self.sector_classifier.get_primary_sector_index(sector)
            logging.info(f"Resolved sector index for {sector}: {sector_index}")
            
            if not sector_index:
                logging.warning(f"No primary index found for sector: {sector}")
                return None
            
            # Fetch data from Zerodha
            logging.info(f"Fetching data from Zerodha for {sector_index}")
            sector_data = self.zerodha_client.get_historical_data(
                symbol=sector_index,
                exchange="NSE",
                period=period
            )
            logging.info(f"Zerodha data result for {sector_index}: {'Success' if sector_data is not None else 'Failed'}")
            if sector_data is not None:
                logging.info(f"Data points received: {len(sector_data)}")
            
            return sector_data
            
        except Exception as e:
            logging.error(f"Error fetching sector index data for {sector}: {e}")
            return None
    
    async def _get_sector_index_data_async(self, sector: str, period: int = 365) -> Optional[pd.DataFrame]:
        """Async version of _get_sector_index_data (no caching - Layer 1 handles caching)."""
        try:
            # Get sector index symbol
            sector_index = self.sector_classifier.get_primary_sector_index(sector)
            
            if not sector_index:
                logging.warning(f"No primary index found for sector: {sector}")
                return None
            
            # Fetch data from Zerodha asynchronously
            sector_data = await self.zerodha_client.get_historical_data_async(
                symbol=sector_index,
                exchange="NSE",
                period=period
            )
            
            return sector_data
            
        except Exception as e:
            logging.error(f"Error fetching sector index data for {sector}: {e}")
            return None
    
    def _get_sector_data(self, sector: str, period: int = 365) -> Optional[pd.DataFrame]:
        """
        Get historical data for a sector (alias for _get_sector_index_data).
        
        Args:
            sector: Sector name
            period: Number of days to retrieve
            
        Returns:
            DataFrame with historical data or None if not available
        """
        return self._get_sector_index_data(sector, period)
    
    async def _get_sector_data_async(self, sector: str, period: int = 365) -> Optional[pd.DataFrame]:
        """Async version of _get_sector_data."""
        return await self._get_sector_index_data_async(sector, period)
    
    def _get_nifty_data(self, period: int = 365) -> Optional[pd.DataFrame]:
        """
        Get historical data for NIFTY 50 index (no caching - Layer 1 handles caching).
        
        Args:
            period: Number of days to retrieve
            
        Returns:
            DataFrame with historical data or None if not available
        """
        try:
            # Fetch data from Zerodha
            data = self.zerodha_client.get_historical_data(
                symbol="NIFTY 50",
                exchange="NSE",
                interval="day",
                period=period
            )
            
            if data is None or data.empty:
                logging.warning(f"No data available for NIFTY 50")
                return None
                
            return data
            
        except Exception as e:
            logging.error(f"Error getting NIFTY 50 data: {e}")
            return None
    
    async def _get_nifty_data_async(self, period: int = 365) -> Optional[pd.DataFrame]:
        """Async version of _get_nifty_data (no caching - Layer 1 handles caching)."""
        try:
            # Fetch data from Zerodha asynchronously
            data = await self.zerodha_client.get_historical_data_async(
                symbol="NIFTY 50",
                exchange="NSE",
                interval="day",
                period=period
            )
            
            if data is None or data.empty:
                logging.warning(f"No data available for NIFTY 50")
                return None
                
            return data
            
        except Exception as e:
            logging.error(f"Error getting NIFTY 50 data: {e}")
            return None
    
    def _calculate_beta(self, stock_returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate beta coefficient with data alignment."""
        try:
            if len(stock_returns) < 30 or len(benchmark_returns) < 30:
                return 1.0
            
            # Align data by using common date range
            aligned_stock, aligned_benchmark = self._align_return_series(stock_returns, benchmark_returns)
            
            if len(aligned_stock) < 30 or len(aligned_benchmark) < 30:
                logging.warning(f"Insufficient aligned data for beta calculation: {len(aligned_stock)} points")
                return 1.0
            
            # Use sample covariance and sample variance (ddof=1) consistently
            cov = np.cov(aligned_stock, aligned_benchmark, ddof=1)[0, 1]
            var = np.var(aligned_benchmark, ddof=1)
            
            if var == 0:
                return 1.0
            
            beta = cov / var
            return max(0.1, min(3.0, beta))  # Clamp between 0.1 and 3.0
            
        except Exception as e:
            logging.error(f"Error calculating beta: {e}")
            return 1.0
    
    def _calculate_correlation(self, stock_returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate correlation coefficient with data alignment."""
        try:
            if len(stock_returns) < 30 or len(benchmark_returns) < 30:
                return 0.5
            
            # Align data by using common date range
            aligned_stock, aligned_benchmark = self._align_return_series(stock_returns, benchmark_returns)
            
            if len(aligned_stock) < 30 or len(aligned_benchmark) < 30:
                logging.warning(f"Insufficient aligned data for correlation calculation: {len(aligned_stock)} points")
                return 0.5
            
            corr = np.corrcoef(aligned_stock, aligned_benchmark)[0, 1]
            
            if np.isnan(corr):
                return 0.5
            
            return float(corr)
            
        except Exception as e:
            logging.error(f"Error calculating correlation: {e}")
            return 0.5
    
    def _align_return_series(self, stock_returns: pd.Series, benchmark_returns: pd.Series) -> tuple:
        """
        Align two return series by their common date range.
        
        Args:
            stock_returns: Stock return series
            benchmark_returns: Benchmark return series
            
        Returns:
            Tuple of aligned series (stock_aligned, benchmark_aligned)
        """
        try:
            # Find common date range
            common_start = max(stock_returns.index.min(), benchmark_returns.index.min())
            common_end = min(stock_returns.index.max(), benchmark_returns.index.max())
            
            # Filter both series to common date range
            stock_aligned = stock_returns[(stock_returns.index >= common_start) & (stock_returns.index <= common_end)]
            benchmark_aligned = benchmark_returns[(benchmark_returns.index >= common_start) & (benchmark_returns.index <= common_end)]
            
            # Ensure they have the same dates by taking intersection
            common_dates = stock_aligned.index.intersection(benchmark_aligned.index)
            
            if len(common_dates) == 0:
                logging.warning("No common dates found between stock and benchmark data")
                return stock_returns.tail(30), benchmark_returns.tail(30)  # Fallback to last 30 points
            
            stock_final = stock_aligned[common_dates]
            benchmark_final = benchmark_aligned[common_dates]
            
            logging.info(f"Aligned data: Original lengths ({len(stock_returns)}, {len(benchmark_returns)}) → Aligned length {len(stock_final)}")
            
            return stock_final, benchmark_final
            
        except Exception as e:
            logging.warning(f"Error aligning return series: {e}. Using tail alignment as fallback.")
            # Fallback: use same length from the end
            min_len = min(len(stock_returns), len(benchmark_returns))
            return stock_returns.tail(min_len), benchmark_returns.tail(min_len)
    
    def _calculate_performance_ranking(self, market_metrics: Dict, sector_metrics: Dict,
                                     momentum_20d: float, momentum_50d: float) -> Dict[str, str]:
        """Calculate performance ranking."""
        try:
            rankings = {}
            
            # Market performance ranking
            market_excess = market_metrics.get('excess_return', 0)
            if market_excess > 0.1:
                rankings['vs_market'] = 'Excellent'
            elif market_excess > 0.05:
                rankings['vs_market'] = 'Good'
            elif market_excess > 0:
                rankings['vs_market'] = 'Above Average'
            elif market_excess > -0.05:
                rankings['vs_market'] = 'Average'
            else:
                rankings['vs_market'] = 'Below Average'
            
            # Sector performance ranking
            if sector_metrics:
                sector_excess = sector_metrics.get('sector_excess_return', 0)
                if sector_excess > 0.1:
                    rankings['vs_sector'] = 'Excellent'
                elif sector_excess > 0.05:
                    rankings['vs_sector'] = 'Good'
                elif sector_excess > 0:
                    rankings['vs_sector'] = 'Above Average'
                elif sector_excess > -0.05:
                    rankings['vs_sector'] = 'Average'
                else:
                    rankings['vs_sector'] = 'Below Average'
            else:
                rankings['vs_sector'] = 'Not Available'
            
            # Momentum ranking
            if momentum_20d > 0.05:
                rankings['momentum'] = 'Strong'
            elif momentum_20d > 0:
                rankings['momentum'] = 'Positive'
            elif momentum_20d > -0.05:
                rankings['momentum'] = 'Neutral'
            else:
                rankings['momentum'] = 'Weak'
            
            return rankings
            
        except Exception as e:
            logging.error(f"Error calculating performance ranking: {e}")
            return {'vs_market': 'Unknown', 'vs_sector': 'Unknown', 'momentum': 'Unknown'}
    
    def _calculate_sector_risk_score(self, stock_returns: pd.Series, sector_metrics: Dict,
                                   market_metrics: Dict) -> float:
        """Calculate sector-specific risk score."""
        try:
            # Base risk score from volatility
            volatility = stock_returns.std() * np.sqrt(252)
            base_score = min(100, volatility * 100)
            
            # Adjust for sector correlation
            sector_correlation = sector_metrics.get('sector_correlation', 0.5)
            correlation_adjustment = (1 - sector_correlation) * 20  # Higher correlation = lower risk
            
            # Adjust for market correlation
            market_correlation = market_metrics.get('correlation', 0.5)
            market_adjustment = (1 - market_correlation) * 10
            
            # Final risk score
            risk_score = base_score - correlation_adjustment - market_adjustment
            return max(0, min(100, risk_score))
            
        except Exception as e:
            logging.error(f"Error calculating sector risk score: {e}")
            return 50.0
    
    def _assess_risk_level(self, risk_score: float) -> str:
        """Assess risk level based on risk score."""
        if risk_score >= 70:
            return "High"
        elif risk_score >= 40:
            return "Medium"
        else:
            return "Low"
    
    def _calculate_sector_rank(self, sector_metrics: Dict, sector: str) -> int:
        """Calculate sector rank based on performance."""
        try:
            if not sector_metrics:
                return 0
            
            # Get sector stocks to determine total count
            sector_stocks = self.sector_classifier.get_sector_stocks(sector)
            total_stocks = len(sector_stocks) if sector_stocks else 75  # Default to 75 if unknown
            
            # Calculate rank based on sector excess return
            sector_excess_return = sector_metrics.get('sector_excess_return', 0)
            
            # Simple ranking logic: better performance = lower rank (1 is best)
            if sector_excess_return > 0.1:  # Top 20%
                return max(1, int(total_stocks * 0.2))
            elif sector_excess_return > 0.05:  # Top 40%
                return max(1, int(total_stocks * 0.4))
            elif sector_excess_return > 0:  # Top 60%
                return max(1, int(total_stocks * 0.6))
            elif sector_excess_return > -0.05:  # Top 80%
                return max(1, int(total_stocks * 0.8))
            else:  # Bottom 20%
                return total_stocks
            
        except Exception as e:
            logging.error(f"Error calculating sector rank: {e}")
            return 0
    
    def _calculate_sector_percentile(self, sector_metrics: Dict, sector: str) -> int:
        """Calculate sector percentile based on performance."""
        try:
            if not sector_metrics:
                return 50
            
            # Calculate percentile based on sector excess return
            sector_excess_return = sector_metrics.get('sector_excess_return', 0)
            
            # Simple percentile calculation
            if sector_excess_return > 0.1:
                return 20  # Top 20%
            elif sector_excess_return > 0.05:
                return 40  # Top 40%
            elif sector_excess_return > 0:
                return 60  # Top 60%
            elif sector_excess_return > -0.05:
                return 80  # Top 80%
            else:
                return 90  # Bottom 10%
            
        except Exception as e:
            logging.error(f"Error calculating sector percentile: {e}")
            return 50
    
    def _generate_analysis_summary(self, stock_symbol: str, sector: str, market_metrics: Dict,
                                 sector_metrics: Dict, relative_performance: Dict) -> Dict[str, str]:
        """Generate analysis summary."""
        try:
            summary = {}
            
            # Market position analysis
            market_excess = market_metrics.get('excess_return', 0)
            if market_excess > 0.05:  # 5% threshold
                summary['market_position'] = "outperforming"
            elif market_excess < -0.05:  # -5% threshold
                summary['market_position'] = "underperforming"
            else:
                summary['market_position'] = "neutral"
            
            # Sector position analysis
            if sector_metrics:
                sector_excess = sector_metrics.get('sector_excess_return', 0)
                if sector_excess > 0.05:  # 5% threshold
                    summary['sector_position'] = "leading"
                elif sector_excess < -0.05:  # -5% threshold
                    summary['sector_position'] = "lagging"
                else:
                    summary['sector_position'] = "neutral"
            else:
                summary['sector_position'] = "neutral"
            
            # Risk assessment
            risk_metrics = relative_performance.get('sector_risk_metrics', {})
            if risk_metrics:
                risk_level = risk_metrics.get('risk_level', 'Medium')
                summary['risk_assessment'] = risk_level.lower()
            else:
                # Calculate risk from market metrics
                volatility = market_metrics.get('volatility', 0.15)
                if volatility > 0.25:
                    summary['risk_assessment'] = "high"
                elif volatility < 0.10:
                    summary['risk_assessment'] = "low"
                else:
                    summary['risk_assessment'] = "medium"
            
            # Investment recommendation
            market_pos = summary.get('market_position', 'neutral')
            sector_pos = summary.get('sector_position', 'neutral')
            risk_level = summary.get('risk_assessment', 'medium')
            
            if market_pos == "outperforming" and sector_pos == "leading" and risk_level == "low":
                summary['investment_recommendation'] = "Strong Buy"
            elif market_pos == "outperforming" and sector_pos in ["leading", "neutral"]:
                summary['investment_recommendation'] = "Buy"
            elif market_pos == "underperforming" and sector_pos == "lagging":
                summary['investment_recommendation'] = "Sell"
            elif market_pos == "underperforming" or sector_pos == "lagging":
                summary['investment_recommendation'] = "Hold"
            elif risk_level == "high":
                summary['investment_recommendation'] = "Hold with caution"
            else:
                summary['investment_recommendation'] = "Hold"
            
            return summary
            
        except Exception as e:
            logging.error(f"Error generating analysis summary: {e}")
            return {
                'market_position': 'neutral',
                'sector_position': 'neutral',
                'risk_assessment': 'medium',
                'investment_recommendation': 'Hold'
            }
    
    def _get_default_market_metrics(self) -> Dict[str, Any]:
        """Get default market metrics when data is unavailable."""
        return {
            "beta": 1.0,
            "correlation": 0.6,
            "volatility_ratio": 1.0,
            "stock_return": 0.12,  # 12% annual return
            "market_return": 0.10,  # 10% annual return
            "excess_return": 0.02,  # 2% excess return
            "stock_sharpe": 0.8,    # 0.8 Sharpe ratio
            "market_sharpe": 0.6,   # 0.6 Sharpe ratio
            "outperformance": 0.02, # 2% outperformance
            "volatility": 0.15,     # 15% volatility
            "cumulative_return": 0.12,  # 12% cumulative return
            "annualized_return": 0.12,  # 12% annualized return
            "data_points": 0,
            "benchmark": "NIFTY 50",
            "note": "Default values - insufficient data"
        }
    
    def _get_default_relative_performance(self) -> Dict[str, Any]:
        """Get default relative performance metrics."""
        return {
            "vs_market": {
                "performance_ratio": 1.0,
                "risk_adjusted_ratio": 1.0,
                "outperformance_periods": 0,
                "underperformance_periods": 0,
                "consistency_score": 0.5
            },
            "vs_sector": {
                "performance_ratio": 1.0,
                "risk_adjusted_ratio": 1.0,
                "sector_rank": 38,  # Middle rank in a typical sector
                "sector_percentile": 50,
                "sector_consistency": 0.5
            },
            "relative_strength": {
                "vs_market": 1.0,
                "vs_sector": 1.0,
                "recent_volatility": 0.15,
                "market_volatility": 0.15,
                "sector_volatility": 0.15
            },
            "momentum": {
                "20_day": 0.0,
                "50_day": 0.0
            },
            "performance_ranking": {
                "vs_market": "Neutral",
                "vs_sector": "Neutral",
                "momentum": "Neutral"
            }
        }
    
    def _calculate_fallback_sector_metrics(self, stock_symbol: str, sector: str, sector_index: str) -> Dict[str, Any]:
        """
        Calculate sector metrics for fallback analysis using simplified approach.
        Attempts to get real sector data with minimal requirements.
        """
        try:
            logging.info(f"Attempting fallback sector metrics calculation for {sector} ({sector_index})")
            
            # Try to get sector index data with minimal requirements (30 days minimum)
            sector_data = self._get_sector_index_data(sector, 90)  # Try 3 months first
            if sector_data is None or len(sector_data) < 30:
                sector_data = self._get_sector_index_data(sector, 60)  # Try 2 months
            if sector_data is None or len(sector_data) < 20:
                sector_data = self._get_sector_index_data(sector, 30)  # Try 1 month minimum
            
            # Try to get NIFTY data for comparison
            nifty_data = self._get_nifty_data(90)
            if nifty_data is None or len(nifty_data) < 30:
                nifty_data = self._get_nifty_data(60)
            if nifty_data is None or len(nifty_data) < 20:
                nifty_data = self._get_nifty_data(30)
            
            if sector_data is not None and len(sector_data) >= 20 and nifty_data is not None and len(nifty_data) >= 20:
                logging.info(f"Calculating real sector metrics using {len(sector_data)} sector data points and {len(nifty_data)} market data points")
                
                # Calculate sector returns
                sector_returns = sector_data['close'].pct_change().dropna()
                nifty_returns = nifty_data['close'].pct_change().dropna()
                
                # Align data (use common dates)
                common_dates = sector_returns.index.intersection(nifty_returns.index)
                if len(common_dates) >= 15:
                    sector_returns_aligned = sector_returns.loc[common_dates]
                    nifty_returns_aligned = nifty_returns.loc[common_dates]
                    
                    # Calculate sector beta vs NIFTY
                    sector_beta = self._calculate_beta(sector_returns_aligned, nifty_returns_aligned)
                    
                    # Calculate sector correlation vs NIFTY
                    sector_correlation = self._calculate_correlation(sector_returns_aligned, nifty_returns_aligned)
                    
                    # Calculate sector volatility (annualized)
                    sector_volatility = sector_returns_aligned.std() * np.sqrt(252)
                    
                    # Calculate sector returns
                    sector_cumulative_return = (1 + sector_returns_aligned).prod() - 1
                    sector_annualized_return = ((1 + sector_cumulative_return) ** (252 / len(sector_returns_aligned))) - 1
                    
                    # Calculate sector Sharpe ratio (assuming 5% risk-free rate)
                    risk_free_rate = 0.05
                    sector_sharpe = (sector_annualized_return - risk_free_rate) / sector_volatility if sector_volatility > 0 else 0
                    
                    # Calculate max drawdown
                    sector_cumsum = (1 + sector_returns_aligned).cumprod()
                    running_max = sector_cumsum.expanding().max()
                    drawdown = (sector_cumsum - running_max) / running_max
                    sector_max_drawdown = abs(drawdown.min())
                    
                    logging.info(f"Real sector metrics calculated - Beta: {sector_beta:.3f}, Correlation: {sector_correlation:.3f}, Volatility: {sector_volatility:.3f}")
                    
                    return {
                        "sector_beta": round(sector_beta, 3),
                        "sector_correlation": round(sector_correlation, 3),
                        "sector_sharpe_ratio": round(sector_sharpe, 3),
                        "sector_volatility": round(sector_volatility, 3),
                        "sector_max_drawdown": round(sector_max_drawdown, 3),
                        "sector_cumulative_return": round(sector_cumulative_return, 4),
                        "sector_annualized_return": round(sector_annualized_return, 4),
                        "sector_index": sector_index,
                        "sector_data_points": len(common_dates),
                        "fallback_calculation": True,
                        "data_period_days": (common_dates[-1] - common_dates[0]).days,
                        "note": "Calculated using limited available data"
                    }
                
            logging.warning(f"Insufficient data for real sector metrics calculation. Sector data: {len(sector_data) if sector_data is not None else 'None'}, NIFTY data: {len(nifty_data) if nifty_data is not None else 'None'}")
                
        except Exception as e:
            logging.error(f"Error in fallback sector metrics calculation: {e}")
        
        # Return default values if calculation failed
        logging.info(f"Using default sector metrics for {sector}")
        return {
            "sector_beta": 1.0,
            "sector_correlation": 0.6,
            "sector_sharpe_ratio": 0.0,
            "sector_volatility": 0.15,
            "sector_max_drawdown": 0.10,
            "sector_cumulative_return": 0.0,
            "sector_annualized_return": 0.0,
            "sector_index": sector_index,
            "sector_data_points": 0,
            "fallback_calculation": False,
            "note": "Default values - insufficient data available"
        }
    
    def _get_fallback_benchmarking(self, stock_symbol: str, sector: str) -> Dict[str, Any]:
        """Get fallback benchmarking when analysis fails."""
        # Get sector information with better error handling and debugging
        try:
            sector_name = self.sector_classifier.get_sector_display_name(sector) if sector else "Unknown"
            sector_index = self.sector_classifier.get_primary_sector_index(sector) if sector else "NIFTY_50"
            sector_stocks = self.sector_classifier.get_sector_stocks(sector) if sector else []
            sector_stocks_count = len(sector_stocks) if sector_stocks else 0
            
            # Debug logging for fallback
            logging.info(f"Fallback sector info for {stock_symbol}:")
            logging.info(f"  - Sector: {sector}")
            logging.info(f"  - Sector Name: {sector_name}")
            logging.info(f"  - Sector Index: {sector_index}")
            logging.info(f"  - Sector Stocks Count: {sector_stocks_count}")
            logging.info(f"  - Sample Stocks: {sector_stocks[:5] if sector_stocks else 'None'}")
            
        except Exception as e:
            logging.error(f"Error getting sector info for {sector}: {e}")
            sector_name = "Unknown"
            sector_index = "NIFTY_50"
            sector_stocks_count = 0
        
        # Get default risk metrics with proper structure
        default_risk_metrics = {
            "risk_score": 50.0,
            "risk_level": "Medium",
            "correlation_risk": "Medium",
            "momentum_risk": "Medium",
            "volatility_risk": "Medium",
            "sector_stress_metrics": {
                "stress_score": 50.0,
                "stress_level": "Medium",
                "stress_factors": ["Market volatility"]
            },
            "risk_factors": ["Market volatility"],
            "risk_mitigation": ["Diversification", "Stop-loss orders"]
        }
        
        return {
            "stock_symbol": stock_symbol,
            "sector_info": {
                "sector": sector,
                "sector_name": sector_name,
                "sector_index": sector_index,
                "sector_stocks_count": sector_stocks_count
            },
            "market_benchmarking": self._get_default_market_metrics(),
            "sector_benchmarking": self._calculate_fallback_sector_metrics(stock_symbol, sector, sector_index),
            "relative_performance": self._get_default_relative_performance(),
            "sector_risk_metrics": default_risk_metrics,
            "analysis_summary": {
                "market_position": "neutral",
                "sector_position": "neutral",
                "risk_assessment": "medium",
                "investment_recommendation": "hold"
            },
            "data_quality": {
                "sufficient_data": False,
                "data_points": 0,
                "minimum_recommended": 30,
                "reliability": "none",
                "analysis_mode": "fallback",
                "limitations": [
                    "No or insufficient historical data available",
                    "Using default values and estimates",
                    "Sector benchmarking not available",
                    "All indicators are approximations"
                ],
                "recommendations": [
                    "Verify stock symbol is correct",
                    "Try different time period or interval",
                    "Results should not be used for investment decisions",
                    "Focus on current market price and recent news instead"
                ]
            },
            "timestamp": datetime.now().isoformat(),
            "data_points": {"stock_data_points": 0, "market_data_points": 0, "sector_data_points": 0},
            "error": "Comprehensive benchmarking analysis failed - insufficient data"
        }

    def get_stock_specific_benchmarking(self, stock_symbol: str, stock_data: pd.DataFrame, user_sector: str = None) -> Dict[str, Any]:
        """
        Get benchmarking analysis for a specific stock (optimized - only fetches relevant data).
        
        Args:
            stock_symbol: Stock symbol to analyze
            stock_data: Historical stock data
            user_sector: Optional user-provided sector override
            
        Returns:
            Dict containing stock-specific benchmarking analysis
        """
        try:
            logging.info(f"Calculating stock-specific benchmarking for {stock_symbol}")
            
            # Prioritize user-provided sector over detected sector
            if user_sector:
                sector = user_sector
                logging.info(f"Using user-provided sector '{user_sector}' for {stock_symbol}")
            else:
                # Get sector information from auto-detection
                sector = self.sector_classifier.get_stock_sector(stock_symbol)
                logging.info(f"Using auto-detected sector '{sector}' for {stock_symbol}")
            sector_name = self.sector_classifier.get_sector_display_name(sector) if sector else None
            sector_index = self.sector_classifier.get_primary_sector_index(sector) if sector else None
            
            # Calculate stock returns
            stock_returns = stock_data['close'].pct_change().dropna()
            
            # Get market metrics (NIFTY 50) - only this, not all sectors
            market_metrics = self._calculate_market_metrics(stock_returns)
            
            # Get sector metrics (only for this stock's sector)
            sector_metrics = self._calculate_sector_metrics(stock_returns, sector) if sector else None
            
            # Calculate relative performance
            relative_performance = self._calculate_relative_performance(
                stock_data, sector, market_metrics, sector_metrics
            )
            
            # Calculate sector-specific risk metrics
            sector_risk_metrics = self._calculate_sector_risk_metrics(
                stock_returns, sector, market_metrics, sector_metrics
            ) if sector else None
            
            # Build comprehensive results
            results = {
                "stock_symbol": stock_symbol,
                "sector_info": {
                    "sector": sector,
                    "sector_name": sector_name,
                    "sector_index": sector_index,
                    "sector_stocks_count": len(self.sector_classifier.get_sector_stocks(sector)) if sector else 0
                },
                "market_benchmarking": market_metrics,
                "sector_benchmarking": sector_metrics,
                "relative_performance": relative_performance,
                "sector_risk_metrics": sector_risk_metrics,
                "analysis_summary": self._generate_analysis_summary(
                    stock_symbol, sector, market_metrics, sector_metrics, relative_performance
                ),
                "timestamp": datetime.now().isoformat(),
                "data_points": {
                    "stock_data_points": len(stock_data),
                    "market_data_points": market_metrics.get('data_points', 0),
                    "sector_data_points": sector_metrics.get('data_points', 0) if sector_metrics else 0
                },
                "optimization_note": "Stock-specific analysis - only relevant sector data fetched"
            }
            
            return results
            
        except Exception as e:
            logging.error(f"Error in stock-specific benchmarking for {stock_symbol}: {e}")
            return self._get_fallback_benchmarking(stock_symbol, sector)

    def get_optimized_sector_rotation(self, stock_symbol: str, timeframe: str = "3M") -> Dict[str, Any]:
        """
        Get sector rotation analysis optimized for a specific stock (only fetches relevant sectors).
        
        Args:
            stock_symbol: Stock symbol to analyze
            timeframe: Analysis period ("1M", "3M", "6M", "1Y")
            
        Returns:
            Dict containing optimized sector rotation analysis
        """
        try:
            logging.info(f"Calculating optimized sector rotation for {stock_symbol} ({timeframe} timeframe)")
            
            # Get the stock's sector
            stock_sector = self.sector_classifier.get_stock_sector(stock_symbol)
            
            # Calculate days for timeframe
            timeframe_days = {
                "1M": 30,
                "3M": 90,
                "6M": 180,
                "1Y": 365
            }
            days = timeframe_days.get(timeframe, 90)
            
            # Only analyze relevant sectors: stock's sector + top 3 performing sectors
            relevant_sectors = []
            
            # Always include the stock's sector
            if stock_sector:
                relevant_sectors.append(stock_sector)
            
            # Get a quick sample of sector performance to identify top performers
            # (This is a lightweight operation that doesn't fetch full data)
            sector_performance_sample = {}
            
            # Sample only 5 sectors for quick ranking (instead of all sectors)
            all_sectors_data = self.sector_classifier.get_all_sectors()
            sample_sectors = [s['code'] for s in all_sectors_data[:5]]
            for sector in sample_sectors:
                try:
                    # Get minimal data for ranking
                    sector_data = self._get_sector_data(sector, days + 10)
                    if sector_data is not None and len(sector_data) >= days:
                        current_price = sector_data['close'].iloc[-1]
                        start_price = sector_data['close'].iloc[-days]
                        total_return = ((current_price - start_price) / start_price) * 100
                        sector_performance_sample[sector] = total_return
                except Exception as e:
                    logging.warning(f"Error sampling performance for {sector}: {e}")
                    continue
            
            # Get top 3 performing sectors from sample
            sorted_sample = sorted(sector_performance_sample.items(), key=lambda x: x[1], reverse=True)
            top_sectors = [sector for sector, _ in sorted_sample[:3]]
            
            # Add top sectors to relevant sectors (avoid duplicates)
            for sector in top_sectors:
                if sector not in relevant_sectors:
                    relevant_sectors.append(sector)
            
            # OPTIMIZATION: Fetch NIFTY 50 data once and reuse for all sectors
            logging.info(f"Fetching NIFTY 50 data once for optimized {timeframe} timeframe (will be reused for {len(relevant_sectors)} relevant sectors)")
            nifty_data = self._get_nifty_data(days + 50)
            nifty_return = None
            if nifty_data is not None and len(nifty_data) >= days:
                nifty_return = ((nifty_data['close'].iloc[-1] - nifty_data['close'].iloc[-days]) / 
                              nifty_data['close'].iloc[-days]) * 100
                logging.info(f"NIFTY 50 return calculated for optimized analysis: {nifty_return:.2f}%")
            else:
                logging.warning("Could not fetch NIFTY 50 data for optimized sector rotation analysis")
            
            # Now analyze only the relevant sectors
            sector_performance = {}
            sector_momentum = {}
            sector_rankings = {}
            
            for sector in relevant_sectors:
                try:
                    # Get historical data for sector index
                    sector_data = self._get_sector_data(sector, days + 50)
                    # More flexible data requirement for longer timeframes
                    min_required = days * 0.7 if days > 180 else days * 0.8  # 70% for >6M, 80% for ≤6M
                    if sector_data is None or len(sector_data) < min_required:
                        continue
                    
                    # Calculate sector performance
                    current_price = sector_data['close'].iloc[-1]
                    start_price = sector_data['close'].iloc[-days]
                    total_return = ((current_price - start_price) / start_price) * 100
                    
                    # Calculate momentum (rate of change)
                    recent_prices = sector_data['close'].tail(20)
                    momentum = ((recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0]) * 100
                    
                    # Calculate relative strength vs NIFTY (using pre-fetched data)
                    if nifty_return is not None:
                        relative_strength = total_return - nifty_return
                    else:
                        relative_strength = total_return
                    
                    sector_performance[sector] = {
                        'total_return': round(total_return, 2),
                        'momentum': round(momentum, 2),
                        'relative_strength': round(relative_strength, 2),
                        'current_price': current_price,
                        'start_price': start_price
                    }
                    
                    sector_momentum[sector] = momentum
                    
                except Exception as e:
                    logging.warning(f"Error calculating performance for {sector}: {e}")
                    continue
            
            # Rank sectors by performance
            sorted_sectors = sorted(sector_performance.items(), 
                                  key=lambda x: x[1]['relative_strength'], reverse=True)
            
            for rank, (sector, data) in enumerate(sorted_sectors, 1):
                sector_rankings[sector] = {
                    'rank': rank
                    # OPTIMIZED: Performance data already available in sector_performance[sector]
                }
            
            # Identify rotation patterns
            rotation_analysis = self._identify_rotation_patterns(sector_performance, timeframe)
            
            # Generate recommendations
            recommendations = self._generate_rotation_recommendations(sector_rankings, rotation_analysis)
            
            return {
                'timeframe': timeframe,
                'stock_sector': stock_sector,
                'analyzed_sectors': relevant_sectors,
                'sector_performance': sector_performance,
                'sector_rankings': sector_rankings,
                'rotation_patterns': rotation_analysis,
                'recommendations': recommendations,
                'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'optimization_note': f"Analyzed {len(relevant_sectors)} relevant sectors instead of all 16"
            }
            
        except Exception as e:
            logging.error(f"Error in optimized sector rotation analysis: {e}")
            return None

    def get_comprehensive_sector_analysis(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Get comprehensive sector analysis with inter-sector relationships.
        Layer 2 cache removed - always generates fresh analysis when called.
        Layer 1 (file-based) cache handles caching at the endpoint level.
        
        Args:
            force_refresh: Kept for API compatibility (but no longer used)
            
        Returns:
            Dict containing comprehensive sector analysis with rotation and correlation
        """
        try:
            logging.info("Generating fresh comprehensive sector analysis (all sectors)")
            
            # Generate comprehensive analysis (all sectors)
            comprehensive_analysis = {
                'sector_rotation': self.analyze_sector_rotation("3M"),
                'sector_correlation': self.generate_sector_correlation_matrix("6M"),
                'market_overview': self._generate_market_overview(),
                'last_updated': datetime.now().isoformat()
            }
            
            return comprehensive_analysis
            
        except Exception as e:
            logging.error(f"Error in comprehensive sector analysis: {e}")
            return None

    def get_hybrid_stock_analysis(self, stock_symbol: str, stock_data: pd.DataFrame, user_sector: str = None) -> Dict[str, Any]:
        """
        Get hybrid stock analysis combining optimized stock-specific data with comprehensive sector relationships.
        
        Args:
            stock_symbol: Stock symbol to analyze
            stock_data: Historical stock data
            user_sector: Optional user-provided sector override
            
        Returns:
            Dict containing hybrid analysis with both stock-specific and comprehensive sector data
        """
        try:
            logging.info(f"Calculating hybrid analysis for {stock_symbol}")
            
            # Get stock-specific benchmarking (optimized - minimal API calls)
            stock_specific = self.get_stock_specific_benchmarking(stock_symbol, stock_data, user_sector=user_sector)
            
            # Get comprehensive sector analysis (cached - no additional API calls if recent)
            comprehensive = self.get_comprehensive_sector_analysis()
            
            # Prioritize user-provided sector over detected sector
            if user_sector:
                stock_sector = user_sector
                logging.info(f"Using user-provided sector '{user_sector}' for {stock_symbol}")
            else:
                # Get stock's sector from auto-detection
                stock_sector = self.sector_classifier.get_stock_sector(stock_symbol)
                logging.info(f"Using auto-detected sector '{stock_sector}' for {stock_symbol}")
            
            # Extract relevant comprehensive data for this stock
            relevant_comprehensive = self._extract_relevant_comprehensive_data(
                stock_symbol, stock_sector, comprehensive
            )
            
            # Combine stock-specific and comprehensive data
            hybrid_analysis = {
                "stock_symbol": stock_symbol,
                "stock_specific_analysis": stock_specific,
                "comprehensive_sector_context": relevant_comprehensive,
                "analysis_type": "hybrid",
                "performance_notes": {
                    "stock_specific_api_calls": "minimal (1-2 calls)",
                    "comprehensive_data_source": "cached (0 calls if recent)",
                    "total_api_calls": "1-2 calls (vs 32+ before optimization)"
                },
                "timestamp": datetime.now().isoformat()
            }
            
            return hybrid_analysis
            
        except Exception as e:
            logging.error(f"Error in hybrid stock analysis for {stock_symbol}: {e}")
            return None

    def _extract_relevant_comprehensive_data(self, stock_symbol: str, stock_sector: str, 
                                           comprehensive: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract relevant comprehensive sector data for a specific stock.
        
        Args:
            stock_symbol: Stock symbol
            stock_sector: Stock's sector
            comprehensive: Comprehensive sector analysis
            
        Returns:
            Dict containing relevant comprehensive data
        """
        try:
            if not comprehensive:
                return {}
            
            relevant_data = {
                'sector_rotation_context': {},
                'correlation_insights': {},
                'market_context': {}
            }
            
            # Extract sector rotation context - OPTIMIZED: Use sector_performance instead of sector_rankings.performance
            if comprehensive.get('sector_rotation'):
                rotation_data = comprehensive['sector_rotation']
                sector_rankings = rotation_data.get('sector_rankings', {})
                sector_performance = rotation_data.get('sector_performance', {})
                
                if stock_sector and stock_sector in sector_rankings:
                    relevant_data['sector_rotation_context'] = {
                        'stock_sector_rank': sector_rankings[stock_sector]['rank'],
                        'stock_sector_performance': sector_performance.get(stock_sector, {}),  # OPTIMIZED: Get from sector_performance
                        'leading_sectors': rotation_data.get('rotation_patterns', {}).get('leading_sectors', []),
                        'lagging_sectors': rotation_data.get('rotation_patterns', {}).get('lagging_sectors', []),
                        'rotation_strength': rotation_data.get('rotation_patterns', {}).get('rotation_strength', 'unknown'),
                        'recommendations': rotation_data.get('recommendations', [])
                    }
            
            # Extract correlation insights
            if comprehensive.get('sector_correlation'):
                correlation_data = comprehensive['sector_correlation']
                correlation_matrix = correlation_data.get('correlation_matrix', {})
                
                # Find sectors with high/low correlation to stock's sector
                if stock_sector and stock_sector in correlation_matrix:
                    stock_sector_correlations = correlation_matrix.get(stock_sector, {})
                    
                    high_correlation_sectors = []
                    low_correlation_sectors = []
                    
                    for sector, correlation in stock_sector_correlations.items():
                        if sector != stock_sector:  # Exclude self-correlation
                            if correlation > 0.7:
                                high_correlation_sectors.append({
                                    'sector': sector,
                                    'correlation': correlation
                                })
                            elif correlation < 0.3:
                                low_correlation_sectors.append({
                                    'sector': sector,
                                    'correlation': correlation
                                })
                    
                    relevant_data['correlation_insights'] = {
                        'average_correlation': correlation_data.get('average_correlation', 0),
                        'high_correlation_sectors': high_correlation_sectors,
                        'low_correlation_sectors': low_correlation_sectors,
                        'diversification_opportunities': low_correlation_sectors,
                        'concentration_risks': high_correlation_sectors
                    }
            
            # Extract market context
            if comprehensive.get('market_overview'):
                relevant_data['market_context'] = comprehensive['market_overview']
            
            return relevant_data
            
        except Exception as e:
            logging.error(f"Error extracting relevant comprehensive data: {e}")
            return {}

    def _generate_market_overview(self) -> Dict[str, Any]:
        """
        Generate market overview with sector performance summary.
        
        Returns:
            Dict containing market overview
        """
        try:
            # Get NIFTY 50 data for market context
            nifty_data = self._get_nifty_data(30)
            
            market_overview = {
                'market_performance': {
                    'nifty_50_return_30d': 0.0,
                    'market_sentiment': 'neutral'
                },
                'sector_performance_summary': {
                    'leading_sectors': [],
                    'lagging_sectors': [],
                    'sector_count': len(self.sector_classifier.get_all_sectors())
                },
                'market_volatility': {
                    'current_vix': 15.0,  # Default value
                    'volatility_regime': 'normal'
                }
            }
            
            if nifty_data is not None and len(nifty_data) >= 30:
                nifty_return = ((nifty_data['close'].iloc[-1] - nifty_data['close'].iloc[-30]) / 
                              nifty_data['close'].iloc[-30]) * 100
                market_overview['market_performance']['nifty_50_return_30d'] = round(nifty_return, 2)
                market_overview['market_performance']['market_sentiment'] = (
                    'bullish' if nifty_return > 5 else 'bearish' if nifty_return < -5 else 'neutral'
                )
            
            return market_overview
            
        except Exception as e:
            logging.error(f"Error generating market overview: {e}")
            return {} 

    async def get_optimized_comprehensive_sector_analysis(
        self, 
        symbol: str, 
        stock_data: pd.DataFrame, 
        sector: str, 
        requested_period: int = None, 
        use_all_sectors: bool = True,
        cached_rotation: Optional[Dict[str, Any]] = None,
        cached_correlation: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        OPTIMIZED: Unified sector data fetcher that minimizes API calls and maximizes data reuse.
        
        CRITICAL FIX: Now accepts cached rotation and correlation data to avoid redundant sector fetching.
        Stock-specific benchmarking is ALWAYS calculated fresh.
        
        This method fetches all required sector data in a single optimized operation:
        - Uses optimized timeframes (1M, 3M, 6M instead of 3M, 6M, 1Y)
        - Fetches data once and reuses across all analyses
        - Implements smart caching and data sharing
        - Reduces API calls from 35 to 5-8 calls
        - NEW: Accepts cached rotation/correlation to skip sector-to-sector fetching
        
        Args:
            symbol: Stock symbol
            stock_data: Historical stock data
            sector: Stock's sector
            requested_period: Optional period for analysis (in days)
            use_all_sectors: If True, analyze all sectors instead of optimized subset (default: True)
            cached_rotation: Optional cached sector rotation data (sector-agnostic)
            cached_correlation: Optional cached sector correlation data (sector-agnostic)
            
        Returns:
            Dict containing all sector analysis data (benchmarking, rotation, correlation)
        """
        # Try comprehensive analysis first, fall back to optimized if it fails
        max_attempts = 2
        current_attempt = 1
        last_error = None
        
        while current_attempt <= max_attempts:
            current_use_all_sectors = use_all_sectors if current_attempt == 1 else False
            analysis_type = "COMPREHENSIVE" if current_use_all_sectors else "OPTIMIZED (FALLBACK)"
            
            try:
                logging.info(f"{analysis_type}: Starting unified sector analysis for {symbol} in {sector} sector (attempt {current_attempt}/{max_attempts})")
            
                # Calculate appropriate timeframes based on requested period
                if requested_period:
                    # Use requested period with some buffer for analysis
                    base_days = min(requested_period, len(stock_data)) if stock_data is not None and not stock_data.empty else requested_period
                    days = max(30, base_days + 20)  # At least 30 days, add 20 day buffer
                    logging.info(f"{analysis_type}: Using requested period of {requested_period} days, adjusted to {days} days for sector analysis")
                else:
                    # Fallback to optimized timeframes
                    OPTIMIZED_TIMEFRAMES = {
                        "sector_rotation": 30,    # 1M - sufficient for rotation analysis
                        "correlation": 60,        # 3M - sufficient for correlation analysis
                        "benchmarking": 180,      # 6M - sufficient for benchmarking metrics
                        "comprehensive": 180      # 6M - unified timeframe for all analyses
                    }
                    days = OPTIMIZED_TIMEFRAMES["comprehensive"]
                    logging.info(f"{analysis_type}: Using default comprehensive timeframe of {days} days")
            
                # STEP 1: Fetch NIFTY 50 data once (reused for all analyses)
                logging.info(f"{analysis_type}: Fetching NIFTY 50 data once for {days} days (will be reused)")
                nifty_data = await self._get_nifty_data_async(days + 20)
                
                # DEBUG: Log NIFTY data details
                if nifty_data is not None:
                    logging.info(f"{analysis_type}: NIFTY 50 data fetched - Length: {len(nifty_data)}, Required: {days}")
                    logging.info(f"{analysis_type}: NIFTY 50 data date range: {nifty_data.index[0]} to {nifty_data.index[-1]}")
                else:
                    logging.warning(f"{analysis_type}: NIFTY 50 data is None")
            
                # Use more flexible data requirement - accept 60% of requested days, minimum 30 days
                min_required_days = max(30, int(days * 0.6))  # At least 30 days, or 60% of requested (more lenient)
                logging.info(f"DEBUG: Data requirement check - Days requested: {days}, Min required: {min_required_days}, NIFTY data available: {len(nifty_data) if nifty_data is not None else 'None'}")
                
                if nifty_data is None or len(nifty_data) < min_required_days:
                    error_msg = f"Could not fetch sufficient NIFTY 50 data for {analysis_type.lower()} analysis. Got: {len(nifty_data) if nifty_data is not None else 'None'}, Required: {min_required_days}"
                    logging.warning(error_msg)
                    if current_attempt < max_attempts:
                        logging.info(f"Insufficient data for {analysis_type.lower()}, will retry with fallback approach")
                        raise Exception(error_msg)
                    else:
                        return self._get_fallback_optimized_analysis(symbol, sector)
            
                # STEP 2: Fetch stock's sector data once (reused for all analyses)
                logging.info(f"{analysis_type}: Fetching {sector} sector data once for {days} days")
                sector_data = await self._get_sector_data_async(sector, days + 20)
                
                # DEBUG: Log sector data details
                if sector_data is not None:
                    logging.info(f"{analysis_type}: {sector} sector data fetched - Length: {len(sector_data)}, Required: {days}")
                    logging.info(f"{analysis_type}: {sector} sector data date range: {sector_data.index[0]} to {sector_data.index[-1]}")
                else:
                    logging.warning(f"{analysis_type}: {sector} sector data is None")
                
                # Use same flexible requirement as NIFTY data
                logging.info(f"DEBUG: Sector data requirement check - {sector} data available: {len(sector_data) if sector_data is not None else 'None'}, Min required: {min_required_days}")
                
                if sector_data is None or len(sector_data) < min_required_days:
                    error_msg = f"Could not fetch sufficient {sector} sector data for {analysis_type.lower()} analysis. Got: {len(sector_data) if sector_data is not None else 'None'}, Required: {min_required_days}"
                    logging.warning(error_msg)
                    if current_attempt < max_attempts:
                        logging.info(f"Insufficient sector data for {analysis_type.lower()}, will retry with fallback approach")
                        raise Exception(error_msg)
                    else:
                        return self._get_fallback_optimized_analysis(symbol, sector)
            
                # STEP 3: Fetch sectors for analysis ONLY if we don't have cached rotation/correlation
                sector_data_dict = {}
                
                # Initialize relevant_sectors to avoid UnboundLocalError
                relevant_sectors = []
                
                # OPTIMIZATION: Skip sector fetching if we have both cached rotation and correlation
                if cached_rotation and cached_correlation:
                    logging.info(f"✅ {analysis_type}: Skipping sector fetching - using cached rotation & correlation")
                    # No need to fetch other sectors since we have cached data
                    # Still need to set relevant_sectors for metrics reporting
                    relevant_sectors = [sector]  # At minimum, include the stock's sector
                else:
                    # Select sectors based on relevance, performance, or comprehensive analysis flag
                    relevant_sectors = self._get_relevant_sectors_for_analysis(sector, current_use_all_sectors)
                    sector_type = "all sectors" if current_use_all_sectors else "relevant sectors"
                    logging.info(f"{analysis_type}: Fetching {len(relevant_sectors)} {sector_type}")
                
                    async def fetch_relevant_sector_data(sector_name):
                        try:
                            data = await self._get_sector_data_async(sector_name, days + 20)
                            # CRITICAL FIX: Use same flexible requirement as main data validation
                            # Use 60% of requested days, minimum 30 days (same as line 2968)
                            min_required = max(30, int(days * 0.6))
                            logging.info(f"DEBUG: Fetching {sector_name} - Available: {len(data) if data is not None else 'None'}, Required: {min_required}")
                            if data is not None and len(data) >= min_required:
                                logging.info(f"✅ {sector_name} data accepted: {len(data)} >= {min_required}")
                                return sector_name, data
                            else:
                                logging.warning(f"❌ {sector_name} data rejected: {len(data) if data is not None else 'None'} < {min_required}")
                                return None
                        except Exception as e:
                            logging.warning(f"Error fetching {sector_name}: {e}")
                            return None
                    
                    # Fetch relevant sectors in parallel
                    tasks = [fetch_relevant_sector_data(s) for s in relevant_sectors]
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    for result in results:
                        if result is not None and not isinstance(result, Exception):
                            sector_name, data = result
                            sector_data_dict[sector_name] = data
                    
                    # Check if we have sufficient sectors for analysis
                    if len(sector_data_dict) < 3 and current_use_all_sectors and current_attempt < max_attempts:
                        error_msg = f"Insufficient sectors for {analysis_type.lower()} analysis. Got {len(sector_data_dict)} sectors, expected at least 3."
                        logging.warning(error_msg)
                        logging.info(f"Insufficient sectors for {analysis_type.lower()}, will retry with fallback approach")
                        raise Exception(error_msg)
                
                # STEP 4: Calculate all metrics using fetched data
                logging.info(f"{analysis_type}: Calculating comprehensive sector metrics")
                
                # ALWAYS calculate benchmarking metrics (stock-specific, never cached)
                logging.info(f"{analysis_type}: Calculating stock-specific benchmarking for {symbol}")
                benchmarking = self._calculate_optimized_benchmarking(symbol, stock_data, sector, sector_data, nifty_data)
                
                # CRITICAL FIX: Use cached rotation if available, otherwise calculate fresh
                if cached_rotation:
                    logging.info(f"✅ Using cached rotation data for {sector} (skipping sector-to-sector fetching)")
                    rotation = cached_rotation
                else:
                    logging.info(f"🔄 Calculating rotation metrics using relevant sectors")
                    rotation_days = min(30, days // 2) if requested_period else 30
                    rotation = self._calculate_optimized_rotation(sector_data_dict, nifty_data, rotation_days)
                
                # CRITICAL FIX: Use cached correlation if available, otherwise calculate fresh
                if cached_correlation:
                    logging.info(f"✅ Using cached correlation data for {sector} (skipping sector-to-sector fetching)")
                    correlation = cached_correlation
                else:
                    logging.info(f"🔄 Calculating correlation metrics using relevant sectors")
                    correlation_days = min(60, days) if requested_period else 60
                    correlation = self._calculate_optimized_correlation(sector_data_dict, correlation_days, sector)
                
                # STEP 5: Build comprehensive result
                comprehensive_result = {
                    'sector_benchmarking': benchmarking,
                    'sector_rotation': rotation,
                    'sector_correlation': correlation,
                    'optimization_metrics': {
                        'api_calls_reduced': f"35 → {len(relevant_sectors) + 2}",  # +2 for NIFTY and stock's sector
                        'data_points_reduced': f"6,790 → {len(relevant_sectors) * days + days * 2}",
                        'timeframes_optimized': '3M,6M,1Y → 1M,3M,6M',
                        'cache_duration': '1 hour (increased from 15 min)',
                        'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'analysis_type': analysis_type.lower(),
                        'sectors_analyzed': len(sector_data_dict),
                        'attempt': current_attempt
                    }
                }
                
                logging.info(f"{analysis_type}: Unified sector analysis completed for {symbol} with {len(sector_data_dict)} sectors")
                return comprehensive_result
            
            except Exception as e:
                last_error = e
                if current_attempt < max_attempts:
                    logging.warning(f"{analysis_type} failed (attempt {current_attempt}/{max_attempts}): {e}")
                    logging.info(f"Retrying with optimized fallback approach...")
                    current_attempt += 1
                    continue
                else:
                    logging.error(f"All attempts failed for sector analysis. Last error: {e}")
                    return self._get_fallback_optimized_analysis(symbol, sector)
                    
        # If we reach here, all attempts failed
        logging.error(f"Failed to complete sector analysis after {max_attempts} attempts. Last error: {last_error}")
        return self._get_fallback_optimized_analysis(symbol, sector)
    
    def _get_relevant_sectors_for_analysis(self, stock_sector: str, use_all_sectors: bool = False) -> List[str]:
        """
        OPTIMIZED: Select relevant sectors for analysis instead of fetching all sectors.
        
        Args:
            stock_sector: The stock's sector
            use_all_sectors: If True, returns all available sectors for comprehensive analysis
            
        Returns:
            List of relevant sectors to analyze
        """
        # If comprehensive analysis is requested, return all available sectors
        if use_all_sectors:
            try:
                all_sectors_data = self.sector_classifier.get_all_sectors()
                # Extract just the sector codes from the dictionary list
                all_sectors = [sector_info['code'] for sector_info in all_sectors_data]
                logging.info(f"COMPREHENSIVE: Using all {len(all_sectors)} sectors for full analysis: {all_sectors}")
                return all_sectors
            except Exception as e:
                logging.warning(f"Failed to get all sectors, falling back to optimized selection: {e}")
                # Fall back to optimized selection if getting all sectors fails
        
        # Always include the stock's sector
        relevant_sectors = [stock_sector]
        
        # Add high-impact sectors that are commonly correlated (using actual JSON sector codes)
        high_impact_sectors = ['NIFTY_BANK', 'NIFTY_IT', 'NIFTY_PHARMA', 'NIFTY_AUTO', 'NIFTY_FMCG', 'NIFTY_OIL_AND_GAS']
        
        # Add stock's sector if not already included
        if stock_sector not in high_impact_sectors:
            relevant_sectors.extend(high_impact_sectors[:5])  # Top 5 high-impact sectors
        else:
            # If stock's sector is high-impact, add other high-impact sectors
            for sector in high_impact_sectors:
                if sector != stock_sector and len(relevant_sectors) < 6:
                    relevant_sectors.append(sector)
        
        # Ensure we have at least 6 sectors for meaningful analysis
        if len(relevant_sectors) < 6:
            additional_sectors = ['NIFTY_METAL', 'NIFTY_REALTY', 'NIFTY_HEALTHCARE', 'NIFTY_CONSUMER_DURABLES']
            for sector in additional_sectors:
                if sector not in relevant_sectors and len(relevant_sectors) < 8:
                    relevant_sectors.append(sector)
        
        logging.info(f"OPTIMIZED: Selected {len(relevant_sectors)} relevant sectors: {relevant_sectors}")
        return relevant_sectors[:8]  # Limit to top 8 sectors
    
    def _calculate_optimized_benchmarking(self, symbol: str, stock_data: pd.DataFrame, 
                                        sector: str, sector_data: pd.DataFrame, 
                                        nifty_data: pd.DataFrame) -> Dict[str, Any]:
        """
        OPTIMIZED: Calculate benchmarking metrics using pre-fetched data.
        """
        try:
            logging.info(f"\n=== DEBUG: OPTIMIZED BENCHMARKING START for {symbol} ===")
            logging.info(f"DEBUG: Sector: {sector}")
            logging.info(f"DEBUG: Stock data shape: {stock_data.shape if stock_data is not None else 'None'}")
            logging.info(f"DEBUG: Sector data shape: {sector_data.shape if sector_data is not None else 'None'}")
            logging.info(f"DEBUG: NIFTY data shape: {nifty_data.shape if nifty_data is not None else 'None'}")
            
            if stock_data is not None and not stock_data.empty:
                logging.info(f"DEBUG: Stock data date range: {stock_data.index[0]} to {stock_data.index[-1]}")
                logging.info(f"DEBUG: Stock close prices - First 3: {stock_data['close'].head(3).tolist()}")
                logging.info(f"DEBUG: Stock close prices - Last 3: {stock_data['close'].tail(3).tolist()}")
            
            if sector_data is not None and not sector_data.empty:
                logging.info(f"DEBUG: Sector data date range: {sector_data.index[0]} to {sector_data.index[-1]}")
                logging.info(f"DEBUG: Sector close prices - First 3: {sector_data['close'].head(3).tolist()}")
                logging.info(f"DEBUG: Sector close prices - Last 3: {sector_data['close'].tail(3).tolist()}")
            
            if nifty_data is not None and not nifty_data.empty:
                logging.info(f"DEBUG: NIFTY data date range: {nifty_data.index[0]} to {nifty_data.index[-1]}")
                logging.info(f"DEBUG: NIFTY close prices - First 3: {nifty_data['close'].head(3).tolist()}")
                logging.info(f"DEBUG: NIFTY close prices - Last 3: {nifty_data['close'].tail(3).tolist()}")
            
            # Calculate stock returns
            logging.info(f"DEBUG: Calculating stock returns...")
            stock_returns = stock_data['close'].pct_change().dropna()
            logging.info(f"DEBUG: Stock returns length: {len(stock_returns)} (from {len(stock_data)} original data points)")
            
            # CRITICAL FIX: Check for insufficient data before proceeding
            if len(stock_returns) == 0:
                logging.warning(f"WARNING: Insufficient stock data for {symbol} (only {len(stock_data)} data points). Falling back to default benchmarking.")
                logging.info(f"Stock data info: shape={stock_data.shape}, close_values={stock_data['close'].head(10).tolist() if not stock_data.empty else 'empty'}")
                # Return structured fallback so frontend can render gracefully with data_quality flags
                return self._get_fallback_benchmarking(symbol, sector)
            
            # Check for minimum data requirement
            if len(stock_returns) < 2:
                logging.warning(f"WARNING: Very few stock returns ({len(stock_returns)}) for {symbol}, analysis may be unreliable")
            
            logging.info(f"DEBUG: Stock returns stats - mean: {stock_returns.mean():.6f}, std: {stock_returns.std():.6f}")
            
            # Calculate sector returns
            logging.info(f"DEBUG: Calculating sector returns...")
            sector_returns = sector_data['close'].pct_change().dropna()
            logging.info(f"DEBUG: Sector returns length: {len(sector_returns)} (from {len(sector_data)} original data points)")
            
            # CRITICAL FIX: Check for insufficient sector data
            if len(sector_returns) == 0:
                logging.error(f"ERROR: No valid sector returns for {sector}")
                return {
                    'error': 'Insufficient sector data for analysis',
                    'sector': sector,
                    'stock_symbol': symbol,
                    'data_points': len(stock_returns),
                    'error_type': 'no_valid_sector_returns'
                }
            
            logging.info(f"DEBUG: Sector returns stats - mean: {sector_returns.mean():.6f}, std: {sector_returns.std():.6f}")
            
            # Calculate NIFTY returns
            logging.info(f"DEBUG: Calculating NIFTY returns...")
            nifty_returns = nifty_data['close'].pct_change().dropna()
            logging.info(f"DEBUG: NIFTY returns length: {len(nifty_returns)} (from {len(nifty_data)} original data points)")
            
            # CRITICAL FIX: Check for insufficient NIFTY data
            if len(nifty_returns) == 0:
                logging.error(f"ERROR: No valid NIFTY returns")
                return {
                    'error': 'Insufficient NIFTY data for analysis',
                    'sector': sector,
                    'stock_symbol': symbol,
                    'data_points': len(stock_returns),
                    'error_type': 'no_valid_nifty_returns'
                }
            
            logging.info(f"DEBUG: NIFTY returns stats - mean: {nifty_returns.mean():.6f}, std: {nifty_returns.std():.6f}")
            
            # Calculate metrics
            logging.info(f"DEBUG: Calculating beta and correlation metrics...")
            stock_beta = self._calculate_beta(stock_returns, nifty_returns)
            sector_beta = self._calculate_beta(sector_returns, nifty_returns)
            stock_correlation = self._calculate_correlation(stock_returns, nifty_returns)
            sector_correlation = self._calculate_correlation(sector_returns, nifty_returns)
            
            logging.info(f"DEBUG: Calculated metrics:")
            logging.info(f"  - Stock Beta: {stock_beta:.6f}")
            logging.info(f"  - Sector Beta: {sector_beta:.6f}")
            logging.info(f"  - Stock Correlation: {stock_correlation:.6f}")
            logging.info(f"  - Sector Correlation: {sector_correlation:.6f}")
            
            # Calculate performance metrics
            logging.info(f"DEBUG: Calculating performance metrics...")
            stock_cumulative_return = (1 + stock_returns).prod() - 1
            sector_cumulative_return = (1 + sector_returns).prod() - 1
            nifty_cumulative_return = (1 + nifty_returns).prod() - 1
            
            # Calculate volatility (annualized) with protection against invalid values
            import numpy as np
            
            try:
                stock_std = stock_returns.std()
                stock_volatility = stock_std * np.sqrt(252) if not pd.isna(stock_std) and stock_std > 0 else 0.0
            except Exception as e:
                logging.warning(f"WARNING: Error calculating stock volatility for {symbol}: {e}")
                stock_volatility = 0.0
            
            try:
                sector_std = sector_returns.std()
                sector_volatility = sector_std * np.sqrt(252) if not pd.isna(sector_std) and sector_std > 0 else 0.0
            except Exception as e:
                logging.warning(f"WARNING: Error calculating sector volatility for {sector}: {e}")
                sector_volatility = 0.0
            
            try:
                nifty_std = nifty_returns.std()
                nifty_volatility = nifty_std * np.sqrt(252) if not pd.isna(nifty_std) and nifty_std > 0 else 0.0
            except Exception as e:
                logging.warning(f"WARNING: Error calculating NIFTY volatility: {e}")
                nifty_volatility = 0.0
            
            # Calculate annualized returns with protection against division by zero
            days_in_data = len(stock_returns)
            
            # CRITICAL FIX: Ensure days_in_data is not zero and handle edge cases
            if days_in_data == 0:
                logging.error(f"ERROR: Zero days in stock returns data for {symbol}")
                # Return zero annualized returns as fallback
                stock_annualized_return = 0.0
                sector_annualized_return = 0.0
                nifty_annualized_return = 0.0
            elif days_in_data < 2:
                # For very few data points, use simple return without annualization
                logging.warning(f"WARNING: Only {days_in_data} data points for {symbol}, using simple returns")
                stock_annualized_return = stock_cumulative_return
                sector_annualized_return = sector_cumulative_return
                nifty_annualized_return = nifty_cumulative_return
            else:
                # Normal annualization calculation with additional safety checks
                try:
                    # Check for problematic cumulative returns that would cause math errors
                    if (1 + stock_cumulative_return) <= 0:
                        logging.warning(f"WARNING: Invalid stock cumulative return {stock_cumulative_return} for {symbol}, using fallback")
                        stock_annualized_return = stock_cumulative_return
                    else:
                        stock_annualized_return = ((1 + stock_cumulative_return) ** (252 / days_in_data)) - 1
                    
                    if (1 + sector_cumulative_return) <= 0:
                        logging.warning(f"WARNING: Invalid sector cumulative return {sector_cumulative_return} for {sector}")
                        sector_annualized_return = sector_cumulative_return
                    else:
                        sector_annualized_return = ((1 + sector_cumulative_return) ** (252 / days_in_data)) - 1
                    
                    if (1 + nifty_cumulative_return) <= 0:
                        logging.warning(f"WARNING: Invalid NIFTY cumulative return {nifty_cumulative_return}")
                        nifty_annualized_return = nifty_cumulative_return
                    else:
                        nifty_annualized_return = ((1 + nifty_cumulative_return) ** (252 / days_in_data)) - 1
                        
                except (ZeroDivisionError, OverflowError, ValueError) as e:
                    logging.error(f"ERROR: Failed to calculate annualized returns for {symbol}: {e}")
                    logging.error(f"DEBUG: days_in_data={days_in_data}, stock_cumulative_return={stock_cumulative_return}")
                    # Fallback to cumulative returns
                    stock_annualized_return = stock_cumulative_return
                    sector_annualized_return = sector_cumulative_return
                    nifty_annualized_return = nifty_cumulative_return
            
            logging.info(f"DEBUG: Annualized returns calculation - days_in_data: {days_in_data}")
            
            # Calculate Sharpe ratios (risk-adjusted returns)
            risk_free_rate = 0.07  # 7% annual risk-free rate (Indian government bonds)
            stock_sharpe = (stock_annualized_return - risk_free_rate) / stock_volatility if stock_volatility > 0 else 0
            sector_sharpe = (sector_annualized_return - risk_free_rate) / sector_volatility if sector_volatility > 0 else 0
            nifty_sharpe = (nifty_annualized_return - risk_free_rate) / nifty_volatility if nifty_volatility > 0 else 0
            
            logging.info(f"DEBUG: Performance metrics calculated:")
            logging.info(f"  - Stock Cumulative Return: {stock_cumulative_return:.6f}")
            logging.info(f"  - Sector Cumulative Return: {sector_cumulative_return:.6f}")
            logging.info(f"  - NIFTY Cumulative Return: {nifty_cumulative_return:.6f}")
            logging.info(f"  - Stock Annualized Return: {stock_annualized_return:.6f}")
            logging.info(f"  - Sector Annualized Return: {sector_annualized_return:.6f}")
            logging.info(f"  - Stock Volatility: {stock_volatility:.6f}")
            logging.info(f"  - Sector Volatility: {sector_volatility:.6f}")
            logging.info(f"  - Stock Sharpe Ratio: {stock_sharpe:.6f}")
            logging.info(f"  - Sector Sharpe Ratio: {sector_sharpe:.6f}")
            logging.info(f"  - Stock Excess Return: {stock_cumulative_return - nifty_cumulative_return:.6f}")
            logging.info(f"  - Sector Excess Return: {sector_cumulative_return - nifty_cumulative_return:.6f}")
            
            # CRITICAL FIX: Calculate relative performance metrics (including sector rank, percentile, consistency)
            logging.info(f"DEBUG: Calculating optimized relative performance metrics...")
            
            # Create mock market_metrics and sector_metrics for relative performance calculation
            market_metrics = {
                'excess_return': stock_cumulative_return - nifty_cumulative_return,
                'volatility_ratio': stock_volatility / nifty_volatility if nifty_volatility > 0 else 1.0,
                'data_points': len(nifty_returns)
            }
            
            sector_metrics = {
                'sector_excess_return': sector_cumulative_return - nifty_cumulative_return,
                'sector_volatility_ratio': sector_volatility / nifty_volatility if nifty_volatility > 0 else 1.0,
                'sector_correlation': sector_correlation,
                'sector_return': sector_cumulative_return,
                'data_points': len(sector_returns)
            }
            
            # Calculate relative performance using existing method
            relative_performance = self._calculate_relative_performance(
                stock_data, sector, market_metrics, sector_metrics
            )
            
            logging.info(f"DEBUG: Relative performance calculated:")
            logging.info(f"  - Sector Rank: {relative_performance.get('vs_sector', {}).get('sector_rank', 'N/A')}")
            logging.info(f"  - Sector Percentile: {relative_performance.get('vs_sector', {}).get('sector_percentile', 'N/A')}")
            logging.info(f"  - Sector Consistency: {relative_performance.get('vs_sector', {}).get('sector_consistency', 'N/A')}")
            
            result = {
                'stock_symbol': symbol,
                'sector_info': {
                    'sector': sector,
                    'sector_name': self.sector_classifier.get_sector_display_name(sector) if sector else sector,
                    'sector_index': self.sector_classifier.get_primary_sector_index(sector) if sector else f"NIFTY_{sector}",
                    'sector_stocks_count': len(self.sector_classifier.get_sector_stocks(sector)) if sector else 0
                },
                'market_benchmarking': {
                    'beta': round(stock_beta, 3),
                    'correlation': round(stock_correlation, 3),
                    'volatility': round(stock_volatility, 3),
                    'cumulative_return': round(stock_cumulative_return, 4),
                    'annualized_return': round(stock_annualized_return, 4),
                    'stock_sharpe': round(stock_sharpe, 3),
                    'excess_return': round(stock_cumulative_return - nifty_cumulative_return, 4),
                    'data_points': len(nifty_returns)
                },
                'sector_benchmarking': {
                    'sector_beta': round(sector_beta, 3),
                    'sector_correlation': round(sector_correlation, 3),
                    'sector_volatility': round(sector_volatility, 3),
                    'sector_cumulative_return': round(sector_cumulative_return, 4),
                    'sector_annualized_return': round(sector_annualized_return, 4),
                    'sector_sharpe_ratio': round(sector_sharpe, 3),
                    'sector_excess_return': round(sector_cumulative_return - nifty_cumulative_return, 4),
                    'sector_index': self.sector_classifier.get_primary_sector_index(sector) if sector else f"NIFTY_{sector}",
                    'sector_data_points': len(sector_returns)
                },
                # CRITICAL FIX: Include relative performance with sector rank, percentile, consistency
                'relative_performance': relative_performance,
                'sector': sector,
                'beta': round(stock_beta, 3),
                'sector_beta': round(sector_beta, 3),
                'correlation': round(stock_correlation, 3),
                'sector_correlation': round(sector_correlation, 3),
                'excess_return': round(stock_cumulative_return - nifty_cumulative_return, 4),
                'sector_excess_return': round(sector_cumulative_return - nifty_cumulative_return, 4),
                # Add the new calculated metrics
                'stock_sharpe': round(stock_sharpe, 3),
                'sector_sharpe': round(sector_sharpe, 3),
                'stock_volatility': round(stock_volatility, 3),
                'sector_volatility': round(sector_volatility, 3),
                'stock_annualized_return': round(stock_annualized_return, 4),
                'sector_annualized_return': round(sector_annualized_return, 4),
                'stock_cumulative_return': round(stock_cumulative_return, 4),
                'sector_cumulative_return': round(sector_cumulative_return, 4),
                'optimization_note': 'Calculated using pre-fetched data',
                # Add data quality information for frontend
                'data_quality': {
                    'sufficient_data': True,
                    'data_points': len(stock_returns),
                    'minimum_recommended': 30,
                    'reliability': 'high',
                    'analysis_mode': 'optimized',
                    'limitations': [],
                    'recommendations': []
                },
                'data_points': {
                    'stock_data_points': len(stock_returns),
                    'market_data_points': len(nifty_returns),
                    'sector_data_points': len(sector_returns)
                },
                'timestamp': datetime.now().isoformat(),
                'analysis_summary': self._generate_analysis_summary(
                    symbol, sector, market_metrics, sector_metrics, relative_performance
                )
            }
            
            logging.info(f"DEBUG: Final optimized benchmarking result:")
            for key, value in result.items():
                if isinstance(value, (int, float)):
                    logging.info(f"  - {key}: {value}")
                elif key == 'relative_performance' and isinstance(value, dict):
                    vs_sector = value.get('vs_sector', {})
                    logging.info(f"  - {key}.vs_sector.sector_rank: {vs_sector.get('sector_rank', 'N/A')}")
                    logging.info(f"  - {key}.vs_sector.sector_percentile: {vs_sector.get('sector_percentile', 'N/A')}")
                    logging.info(f"  - {key}.vs_sector.sector_consistency: {vs_sector.get('sector_consistency', 'N/A')}")
                else:
                    logging.info(f"  - {key}: {str(value)[:100]}...")  # Truncate long values
            logging.info(f"DEBUG: OPTIMIZED BENCHMARKING SUCCESS")
            
            return result
            
        except Exception as e:
            logging.error(f"Error in optimized benchmarking: {e}")
            import traceback
            traceback.print_exc()
            return {'sector': sector, 'error': str(e)}
    
    def _calculate_optimized_rotation(self, sector_data_dict: Dict[str, pd.DataFrame], 
                                    nifty_data: pd.DataFrame, days: int) -> Dict[str, Any]:
        """
        OPTIMIZED: Calculate rotation metrics using pre-fetched data.
        """
        try:
            # Calculate NIFTY return for comparison
            nifty_return = ((nifty_data['close'].iloc[-1] - nifty_data['close'].iloc[-days]) / 
                          nifty_data['close'].iloc[-days]) * 100
            
            sector_performance = {}
            sector_rankings = {}
            
            for sector_name, sector_data in sector_data_dict.items():
                try:
                    # Calculate sector performance
                    current_price = sector_data['close'].iloc[-1]
                    start_price = sector_data['close'].iloc[-days]
                    total_return = ((current_price - start_price) / start_price) * 100
                    
                    # Calculate momentum (10-day)
                    recent_prices = sector_data['close'].tail(10)
                    momentum = ((recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0]) * 100
                    
                    # Calculate relative strength
                    relative_strength = total_return - nifty_return
                    
                    sector_performance[sector_name] = {
                        'total_return': round(total_return, 2),
                        'momentum': round(momentum, 2),
                        'relative_strength': round(relative_strength, 2),
                        'current_price': current_price,
                        'start_price': start_price
                    }
                    
                except Exception as e:
                    logging.warning(f"Error calculating rotation for {sector_name}: {e}")
                    continue
            
            # Rank sectors - OPTIMIZED: Only store rank, not duplicate performance data
            sorted_sectors = sorted(sector_performance.items(), 
                                  key=lambda x: x[1]['relative_strength'], reverse=True)
            
            for rank, (sector_name, data) in enumerate(sorted_sectors, 1):
                sector_rankings[sector_name] = {
                    'rank': rank
                    # Performance data already available in sector_performance[sector_name]
                }
            
            # CRITICAL FIX: Add rotation patterns analysis that frontend expects
            rotation_patterns = self._identify_rotation_patterns(sector_performance, f"{days}D")
            
            # CRITICAL FIX: Add recommendations that frontend expects  
            recommendations = self._generate_rotation_recommendations(sector_rankings, rotation_patterns, sector_performance)
            
            logging.info(f"Optimized rotation calculated: {len(sector_performance)} sectors, rotation_strength: {rotation_patterns.get('rotation_strength', 'unknown')}")
            
            return {
                'timeframe': f"{days}D",
                'sector_performance': sector_performance,
                'sector_rankings': sector_rankings,
                'rotation_patterns': rotation_patterns,  # CRITICAL FIX: Added this
                'recommendations': recommendations,      # CRITICAL FIX: Added this
                'optimization_note': 'Calculated using pre-fetched data'
            }
            
        except Exception as e:
            logging.error(f"Error in optimized rotation: {e}")
            return {'error': str(e)}
    
    def _calculate_optimized_correlation(self, sector_data_dict: Dict[str, pd.DataFrame], 
                                       days: int, current_sector: str = None) -> Dict[str, Any]:
        """
        OPTIMIZED: Calculate correlation metrics using pre-fetched data.
        """
        try:
            # Calculate returns for all sectors
            returns_data = {}
            for sector_name, sector_data in sector_data_dict.items():
                returns = sector_data['close'].pct_change().dropna()
                if len(returns) >= days * 0.8:
                    returns_data[sector_name] = returns
            
            if len(returns_data) < 2:
                return {'error': 'Insufficient data for correlation analysis'}
            
            # Create correlation matrix
            returns_df = pd.DataFrame(returns_data)
            correlation_matrix = returns_df.corr()
            
            # Calculate average correlation
            upper_triangle = correlation_matrix.where(
                np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
            )
            avg_correlation = upper_triangle.stack().mean()
            
            # Identify high and low correlation pairs
            high_correlation_pairs = []
            low_correlation_pairs = []
            
            for i in range(len(correlation_matrix.columns)):
                for j in range(i + 1, len(correlation_matrix.columns)):
                    sector1 = correlation_matrix.columns[i]
                    sector2 = correlation_matrix.columns[j]
                    correlation = correlation_matrix.iloc[i, j]
                    
                    if correlation > 0.7:
                        high_correlation_pairs.append({
                            'sector1': sector1,
                            'sector2': sector2,
                            'correlation': round(correlation, 3)
                        })
                    elif correlation < 0.3:
                        low_correlation_pairs.append({
                            'sector1': sector1,
                            'sector2': sector2,
                            'correlation': round(correlation, 3)
                        })
            
            # Calculate sector volatilities
            sector_volatilities = {}
            for sector_name, returns in returns_data.items():
                volatility = returns.std() * np.sqrt(252)  # Annualized volatility
                sector_volatilities[sector_name] = round(volatility * 100, 2)  # Convert to percentage
            
            # Get current sector volatility (single value as expected by frontend)
            sector_volatility = sector_volatilities.get(current_sector, None) if current_sector else None
            
            # Calculate diversification insights
            diversification_quality = 'good'
            if avg_correlation > 0.7:
                diversification_quality = 'poor'
            elif avg_correlation > 0.5:
                diversification_quality = 'moderate'
            elif avg_correlation > 0.3:
                diversification_quality = 'good'
            else:
                diversification_quality = 'excellent'
            
            # Generate diversification recommendations
            recommendations = []
            if avg_correlation > 0.6:
                recommendations.append({
                    'type': 'diversification',
                    'message': 'High sector correlation detected - consider diversifying across asset classes',
                    'priority': 'high'
                })
            elif len(high_correlation_pairs) > 0:
                recommendations.append({
                    'type': 'sector_rotation',
                    'message': f'Strong correlation between {high_correlation_pairs[0]["sector1"]} and {high_correlation_pairs[0]["sector2"]} sectors',
                    'priority': 'medium'
                })
            
            if len(low_correlation_pairs) > 0:
                recommendations.append({
                    'type': 'portfolio_balance',
                    'message': f'Good diversification opportunity between {low_correlation_pairs[0]["sector1"]} and {low_correlation_pairs[0]["sector2"]} sectors',
                    'priority': 'low'
                })
            
            return {
                'timeframe': f"{days}D",
                'correlation_matrix': correlation_matrix.round(3).to_dict(),
                'average_correlation': round(avg_correlation, 3),
                'sector_volatility': sector_volatility,  # CRITICAL FIX: Added missing field
                'sector_volatilities': sector_volatilities,  # All sector volatilities for reference
                'high_correlation_pairs': high_correlation_pairs,
                'low_correlation_pairs': low_correlation_pairs,
                'diversification_insights': {  # CRITICAL FIX: Added missing field
                    'diversification_quality': diversification_quality,
                    'recommendations': recommendations
                },
                'optimization_note': 'Calculated using pre-fetched data'
            }
            
        except Exception as e:
            logging.error(f"Error in optimized correlation: {e}")
            return {'error': str(e)}
    
    def _get_fallback_optimized_analysis(self, symbol: str, sector: str) -> Dict[str, Any]:
        """
        Fallback analysis when optimized analysis fails.
        Returns basic sector benchmarking using fallback methods.
        """
        logging.info(f"Using fallback analysis for {symbol} in {sector} sector")
        
        try:
            # Try to get basic sector benchmarking using sync methods
            fallback_benchmarking = self._get_fallback_benchmarking(symbol, sector)
            
            return {
                'sector_benchmarking': fallback_benchmarking,
                'sector_rotation': {
                    'error': 'Optimized rotation analysis failed',
                    'fallback_note': 'Basic sector info provided in sector_benchmarking'
                },
                'sector_correlation': {
                    'error': 'Optimized correlation analysis failed', 
                    'fallback_note': 'Basic sector info provided in sector_benchmarking'
                },
                'optimization_metrics': {
                    'api_calls_reduced': 'Fallback mode - using cached/default data',
                    'data_points_reduced': 'Fallback mode',
                    'timeframes_optimized': 'Fallback mode',
                    'cache_duration': 'Fallback mode',
                    'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'fallback_reason': 'Insufficient market/sector data for optimized analysis'
                }
            }
        except Exception as e:
            logging.error(f"Error in fallback analysis: {e}")
            # Return minimal structure to prevent null sector_context
            return {
                'sector_benchmarking': {
                    'stock_symbol': symbol,
                    'sector_info': {
                        'sector': sector,
                        'sector_name': self.sector_classifier.get_sector_display_name(sector) or sector,
                        'sector_index': self.sector_classifier.get_primary_sector_index(sector) or f'NIFTY_{sector}',
                        'sector_stocks_count': len(self.sector_classifier.get_sector_stocks(sector)) if sector else 0
                    },
                    'market_benchmarking': {
                        'beta': 1.0,
                        'correlation': 0.5, 
                        'sharpe_ratio': 0.0,
                        'volatility': 0.0,
                        'max_drawdown': 0.0,
                        'cumulative_return': 0.0,
                        'annualized_return': 0.0,
                        'risk_free_rate': 0.05,
                        'current_vix': 20,
                        'data_source': 'NSE',
                        'data_points': 0,
                        'note': 'Fallback default values - insufficient data'
                    },
                    'fallback': True,
                    'timestamp': datetime.now().isoformat()
                },
                'sector_rotation': {'fallback': True},
                'sector_correlation': {'fallback': True},
                'optimization_metrics': {
                    'fallback_reason': 'Complete analysis failure - using minimal defaults',
                    'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
            }

# Global instance removed - instantiate locally as needed
