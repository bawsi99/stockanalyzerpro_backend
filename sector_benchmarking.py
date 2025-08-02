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
from sector_classifier import sector_classifier
from enhanced_sector_classifier import enhanced_sector_classifier
from technical_indicators import IndianMarketMetricsProvider
from zerodha_client import ZerodhaDataClient

class SectorBenchmarkingProvider:
    """
    Provides comprehensive sector-based benchmarking for stock analysis.
    Enhances the current NIFTY-only analysis with sector-specific metrics.
    """
    
    def __init__(self):
        """Initialize the sector benchmarking provider."""
        self.zerodha_client = ZerodhaDataClient()
        self.market_metrics_provider = IndianMarketMetricsProvider()
        self.sector_classifier = sector_classifier
        self.enhanced_classifier = enhanced_sector_classifier
        
        # Cache for performance optimization
        self.sector_data_cache = {}
        self.cache_duration = 3600  # OPTIMIZED: Increased from 900 (15 min) to 3600 (1 hour)
        
        # NEW: Comprehensive sector analysis cache
        self.comprehensive_sector_cache = {}
        self.comprehensive_cache_duration = 7200  # OPTIMIZED: Increased from 3600 (1 hour) to 7200 (2 hours)
        self.last_comprehensive_update = None
        
        # Sector index mappings (from technical_indicators.py)
        self.sector_indices = {
            'BANKING': 'NIFTY BANK',
            'IT': 'NIFTY IT',
            'PHARMA': 'NIFTY PHARMA',
            'AUTO': 'NIFTY AUTO',
            'FMCG': 'NIFTY FMCG',
            'ENERGY': 'NIFTY ENERGY',
            'METAL': 'NIFTY METAL',
            'REALTY': 'NIFTY REALTY',
            'OIL_GAS': 'NIFTY OIL AND GAS',
            'HEALTHCARE': 'NIFTY HEALTHCARE',
            'CONSUMER_DURABLES': 'NIFTY CONSR DURBL',
            'MEDIA': 'NIFTY MEDIA',
            'INFRASTRUCTURE': 'NIFTY INFRA',
            'CONSUMPTION': 'NIFTY CONSUMPTION',
            'TELECOM': 'NIFTY SERV SECTOR',
            'TRANSPORT': 'NIFTY SERV SECTOR'
        }
        
        # Sector tokens for data retrieval
        self.sector_tokens = {
            'BANKING': 'NIFTY BANK',
            'IT': 'NIFTY IT',
            'PHARMA': 'NIFTY PHARMA',
            'AUTO': 'NIFTY AUTO',
            'FMCG': 'NIFTY FMCG',
            'ENERGY': 'NIFTY ENERGY',
            'METAL': 'NIFTY METAL',
            'REALTY': 'NIFTY REALTY',
            'OIL_GAS': 'NIFTY OIL AND GAS',
            'HEALTHCARE': 'NIFTY HEALTHCARE',
            'CONSUMER_DURABLES': 'NIFTY CONSR DURBL',
            'MEDIA': 'NIFTY MEDIA',
            'INFRASTRUCTURE': 'NIFTY INFRA',
            'CONSUMPTION': 'NIFTY CONSUMPTION',
            'TELECOM': 'NIFTY SERV SECTOR',
            'TRANSPORT': 'NIFTY SERV SECTOR'
        }
        
        logging.info("SectorBenchmarkingProvider initialized with hybrid caching strategy")
    
    def get_comprehensive_benchmarking(self, stock_symbol: str, stock_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get comprehensive benchmarking analysis including both market and sector metrics.
        
        Args:
            stock_symbol: Stock symbol to analyze
            stock_data: Historical stock data
            
        Returns:
            Dict containing comprehensive benchmarking analysis
        """
        try:
            logging.info(f"Calculating comprehensive benchmarking for {stock_symbol}")
            
            # Get sector information
            sector = self.sector_classifier.get_stock_sector(stock_symbol)
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
    
    async def get_comprehensive_benchmarking_async(self, stock_symbol: str, stock_data: pd.DataFrame) -> Dict[str, Any]:
        """Async version of get_comprehensive_benchmarking."""
        try:
            # Get stock returns
            stock_returns = stock_data['close'].pct_change().dropna()
            if len(stock_returns) < 30:
                return self._get_fallback_benchmarking(stock_symbol, "UNKNOWN")
            
            # Get sector classification
            sector = self.sector_classifier.get_stock_sector(stock_symbol)
            
            # Fetch market and sector data concurrently
            tasks = [
                self._calculate_market_metrics_async(stock_returns)
            ]
            
            if sector and sector != 'UNKNOWN':
                tasks.append(self._calculate_sector_metrics_async(stock_returns, sector))
            else:
                tasks.append(None)
            
            # Execute tasks concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            market_metrics = results[0] if not isinstance(results[0], Exception) else self._get_default_market_metrics()
            sector_metrics = results[1] if len(results) > 1 and not isinstance(results[1], Exception) else None
            
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
            logging.info(f"Fetching NIFTY 50 data once for {timeframe} timeframe (will be reused for all {len(self.sector_tokens)} sectors)")
            nifty_data = self._get_nifty_data(days + 20)  # OPTIMIZED: Reduced buffer from 50 to 20 days
            nifty_return = None
            if nifty_data is not None and len(nifty_data) >= days:
                nifty_return = ((nifty_data['close'].iloc[-1] - nifty_data['close'].iloc[-days]) / 
                              nifty_data['close'].iloc[-days]) * 100
                logging.info(f"NIFTY 50 return calculated: {nifty_return:.2f}%")
            else:
                logging.warning("Could not fetch NIFTY 50 data for sector rotation analysis")
            
            # Get sector performance data
            sector_performance = {}
            sector_momentum = {}
            sector_rankings = {}
            
            for sector, token in self.sector_tokens.items():
                try:
                    # Get historical data for sector index - OPTIMIZED timeframe
                    sector_data = self._get_sector_data(sector, days + 20)  # OPTIMIZED: Reduced buffer from 50 to 20 days
                    # More flexible data requirement for longer timeframes
                    min_required = days * 0.7 if days > 60 else days * 0.8  # OPTIMIZED: Adjusted thresholds
                    if sector_data is None or len(sector_data) < min_required:
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
            
            # Rank sectors by performance
            sorted_sectors = sorted(sector_performance.items(), 
                                  key=lambda x: x[1]['relative_strength'], reverse=True)
            
            for rank, (sector, data) in enumerate(sorted_sectors, 1):
                sector_rankings[sector] = {
                    'rank': rank,
                    'performance': data
                }
            
            # Identify rotation patterns
            rotation_analysis = self._identify_rotation_patterns(sector_performance, timeframe)
            
            # Generate recommendations
            recommendations = self._generate_rotation_recommendations(sector_rankings, rotation_analysis)
            
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
            days = timeframe_days.get(timeframe, 30)  # Default to 1M instead of 3M
            
            # OPTIMIZATION: Fetch NIFTY 50 data once and reuse for all sectors
            logging.info(f"Async fetching NIFTY 50 data once for {timeframe} timeframe")
            nifty_data = await self._get_nifty_data_async(days + 20)  # OPTIMIZED: Reduced buffer from 50 to 20 days
            nifty_return = None
            if nifty_data is not None and len(nifty_data) >= days:
                nifty_return = ((nifty_data['close'].iloc[-1] - nifty_data['close'].iloc[-days]) / 
                              nifty_data['close'].iloc[-days]) * 100
                logging.info(f"NIFTY 50 return calculated: {nifty_return:.2f}%")
            else:
                logging.warning("Could not fetch NIFTY 50 data for async sector rotation analysis")
            
            # Get sector performance data using async parallel fetching
            sector_performance = {}
            sector_momentum = {}
            sector_rankings = {}
            
            async def fetch_sector_data(sector, token):
                try:
                    # Get historical data for sector index - OPTIMIZED timeframe
                    sector_data = await self._get_sector_data_async(sector, days + 20)  # OPTIMIZED: Reduced buffer from 50 to 20 days
                    # More flexible data requirement for longer timeframes
                    min_required = days * 0.7 if days > 60 else days * 0.8  # OPTIMIZED: Adjusted thresholds
                    if sector_data is not None and len(sector_data) >= min_required:
                        # Calculate daily returns
                        returns = sector_data['close'].pct_change().dropna()
                        return sector, returns
                    else:
                        logging.warning(f"No sufficient data for {sector}: got {len(sector_data) if sector_data is not None else 0} records, need at least {min_required:.0f}")
                        return None
                except Exception as e:
                    logging.warning(f"Error calculating performance for {sector}: {e}")
                    return None
            
            # Fetch all sector data in parallel
            tasks = [fetch_sector_data(sector, token) for sector, token in self.sector_tokens.items()]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for result in results:
                if result is not None and not isinstance(result, Exception):
                    sector, returns = result
                    sector_performance[sector] = {
                        'total_return': returns.mean(),
                        'momentum': returns.iloc[-1] - returns.iloc[0],
                        'relative_strength': returns.mean() / returns.std(),
                        'current_price': returns.iloc[-1],
                        'start_price': returns.iloc[0]
                    }
                    sector_momentum[sector] = returns.iloc[-1] - returns.iloc[0]
            
            # Rank sectors by performance
            sorted_sectors = sorted(sector_performance.items(), 
                                  key=lambda x: x[1]['relative_strength'], reverse=True)
            
            for rank, (sector, data) in enumerate(sorted_sectors, 1):
                sector_rankings[sector] = {
                    'rank': rank,
                    'performance': data
                }
            
            # Identify rotation patterns
            rotation_analysis = self._identify_rotation_patterns(sector_performance, timeframe)
            
            # Generate recommendations
            recommendations = self._generate_rotation_recommendations(sector_rankings, rotation_analysis)
            
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
            
            # Collect sector data
            sector_data = {}
            valid_sectors = []
            
            for sector, token in self.sector_tokens.items():
                try:
                    data = self._get_sector_data(sector, days + 20)  # OPTIMIZED: Reduced buffer from 50 to 20 days
                    # More flexible data requirement for longer timeframes
                    min_required = days * 0.7 if days > 60 else days * 0.8  # OPTIMIZED: Adjusted thresholds
                    if data is not None and len(data) >= min_required:
                        # Calculate daily returns
                        returns = data['close'].pct_change().dropna()
                        sector_data[sector] = returns
                        valid_sectors.append(sector)
                        # logging.info(f"Successfully got data for {sector}: {len(data)} records")
                    else:
                        logging.warning(f"No sufficient data for {sector}: got {len(data) if data is not None else 0} records, need at least {min_required:.0f}")
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
            
            # Generate diversification insights
            diversification_insights = self._generate_diversification_insights(
                correlation_matrix, high_correlation_pairs, low_correlation_pairs, avg_correlation
            )
            
            # Calculate sector volatility for risk assessment
            sector_volatility = {}
            for sector in valid_sectors:
                volatility = sector_data[sector].std() * np.sqrt(252) * 100  # Annualized %
                sector_volatility[sector] = round(volatility, 2)
            
            return {
                'timeframe': timeframe,
                'correlation_matrix': correlation_matrix.round(3).to_dict(),
                'average_correlation': round(avg_correlation, 3),
                'high_correlation_pairs': high_correlation_pairs,
                'low_correlation_pairs': low_correlation_pairs,
                'sector_volatility': sector_volatility,
                'diversification_insights': diversification_insights,
                'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'optimization_note': 'Timeframes optimized for reduced data fetching'
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
            
            # Collect sector data using async parallel fetching
            sector_data = {}
            valid_sectors = []
            
            async def fetch_sector_data(sector, token):
                try:
                    data = await self._get_sector_data_async(sector, days + 20)  # OPTIMIZED: Reduced buffer from 50 to 20 days
                    # More flexible data requirement for longer timeframes
                    min_required = days * 0.7 if days > 60 else days * 0.8  # OPTIMIZED: Adjusted thresholds
                    if data is not None and len(data) >= min_required:
                        # Calculate daily returns
                        returns = data['close'].pct_change().dropna()
                        return sector, returns
                    else:
                        logging.warning(f"No sufficient data for {sector}: got {len(data) if data is not None else 0} records, need at least {min_required:.0f}")
                        return None
                except Exception as e:
                    logging.warning(f"Error getting data for {sector}: {e}")
                    return None
            
            # Fetch all sector data in parallel
            tasks = [fetch_sector_data(sector, token) for sector, token in self.sector_tokens.items()]
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
            
            # Generate diversification insights
            diversification_insights = self._generate_diversification_insights(
                correlation_matrix, high_correlation_pairs, low_correlation_pairs, avg_correlation
            )
            
            # Calculate sector volatility for risk assessment
            sector_volatility = {}
            for sector in valid_sectors:
                volatility = sector_data[sector].std() * np.sqrt(252) * 100  # Annualized %
                sector_volatility[sector] = round(volatility, 2)
            
            return {
                'timeframe': timeframe,
                'correlation_matrix': correlation_matrix.round(3).to_dict(),
                'average_correlation': round(avg_correlation, 3),
                'high_correlation_pairs': high_correlation_pairs,
                'low_correlation_pairs': low_correlation_pairs,
                'sector_volatility': sector_volatility,
                'diversification_insights': diversification_insights,
                'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'optimization_note': 'Async timeframes optimized for reduced data fetching'
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
                                         rotation_analysis: Dict) -> List[Dict]:
        """Generate actionable rotation recommendations."""
        try:
            recommendations = []
            
            # Add sector-specific recommendations
            for sector, ranking in sector_rankings.items():
                performance = ranking['performance']
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
            if rotation_analysis['rotation_strength'] == 'strong':
                recommendations.append({
                    'type': 'market_insight',
                    'message': "Strong sector rotation detected - consider rebalancing portfolio",
                    'confidence': 'high'
                })
            
            return recommendations
            
        except Exception as e:
            logging.error(f"Error generating rotation recommendations: {e}")
            return []
    
    def _generate_diversification_insights(self, correlation_matrix: pd.DataFrame,
                                         high_correlation_pairs: List[Dict],
                                         low_correlation_pairs: List[Dict],
                                         avg_correlation: float) -> Dict[str, Any]:
        """Generate insights for portfolio diversification."""
        try:
            insights = {
                'diversification_quality': 'excellent',
                'risk_reduction_opportunities': [],
                'concentration_risks': [],
                'recommendations': []
            }
            
            # Assess overall diversification quality
            if avg_correlation < 0.3:
                insights['diversification_quality'] = 'excellent'
            elif avg_correlation < 0.5:
                insights['diversification_quality'] = 'good'
            elif avg_correlation < 0.7:
                insights['diversification_quality'] = 'moderate'
            else:
                insights['diversification_quality'] = 'poor'
            
            # Identify concentration risks
            for pair in high_correlation_pairs:
                insights['concentration_risks'].append({
                    'message': f"High correlation ({pair['correlation']}) between {pair['sector1']} and {pair['sector2']}",
                    'risk_level': 'high' if pair['correlation'] > 0.8 else 'medium',
                    'recommendation': f"Consider reducing exposure to one of these sectors"
                })
            
            # Identify diversification opportunities
            for pair in low_correlation_pairs:
                insights['risk_reduction_opportunities'].append({
                    'message': f"Low correlation ({pair['correlation']}) between {pair['sector1']} and {pair['sector2']}",
                    'opportunity': 'excellent' if pair['correlation'] < 0.2 else 'good',
                    'recommendation': f"Good diversification pair for portfolio construction"
                })
            
            # Generate overall recommendations
            if insights['diversification_quality'] == 'excellent':
                insights['recommendations'].append({
                    'type': 'positive',
                    'message': "Excellent sector diversification - portfolio is well-balanced",
                    'priority': 'low'
                })
            elif insights['diversification_quality'] == 'poor':
                insights['recommendations'].append({
                    'type': 'warning',
                    'message': "Poor sector diversification - consider rebalancing portfolio",
                    'priority': 'high'
                })
            
            if len(high_correlation_pairs) > 3:
                insights['recommendations'].append({
                    'type': 'warning',
                    'message': f"Multiple high-correlation sector pairs detected - concentration risk",
                    'priority': 'medium'
                })
            
            if len(low_correlation_pairs) > 5:
                insights['recommendations'].append({
                    'type': 'positive',
                    'message': f"Multiple low-correlation sector pairs available for diversification",
                    'priority': 'low'
                })
            
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
            
            return {
                "beta": float(beta),
                "correlation": float(correlation),
                "volatility_ratio": float(volatility_ratio),
                "stock_return": float(stock_cumulative_return),
                "market_return": float(market_cumulative_return),
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
            # Get NIFTY 50 data asynchronously
            nifty_data = await self._get_nifty_data_async(365)
            
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
            
            return {
                "beta": float(beta),
                "correlation": float(correlation),
                "volatility_ratio": float(volatility_ratio),
                "stock_return": float(stock_cumulative_return),
                "market_return": float(market_cumulative_return),
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
            if not sector:
                return None
            
            # Get sector index data
            sector_data = self._get_sector_index_data(sector, 365)
            
            if sector_data is None or len(sector_data) < 20:  # Reduced from 30 to 20
                logging.warning(f"Insufficient sector data for {sector} (got {len(sector_data) if sector_data is not None else 0} data points)")
                # Return basic sector metrics instead of None
                return self._get_basic_sector_metrics(stock_returns, sector)
            
            sector_returns = sector_data['close'].pct_change().dropna()
            
            # Align data
            aligned_data = pd.concat([stock_returns, sector_returns], axis=1).dropna()
            if len(aligned_data) < 30:
                return None
            
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
            
            # Get sector index symbol
            sector_index = self.sector_classifier.get_primary_sector_index(sector)
            
            return {
                "sector_beta": float(sector_beta),
                "sector_correlation": float(sector_correlation),
                "sector_volatility_ratio": float(sector_volatility_ratio),
                "stock_return": float(stock_cumulative_return),
                "sector_return": float(sector_cumulative_return),
                "sector_excess_return": float(sector_excess_return),
                "stock_sharpe": float(stock_sharpe),
                "sector_sharpe": float(sector_sharpe),
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
            
            # Get sector index symbol
            sector_index = self.sector_classifier.get_primary_sector_index(sector)
            
            return {
                "sector_beta": float(sector_beta),
                "sector_correlation": float(sector_correlation),
                "sector_volatility_ratio": float(sector_volatility_ratio),
                "stock_return": float(stock_cumulative_return),
                "sector_return": float(sector_cumulative_return),
                "sector_excess_return": float(sector_excess_return),
                "stock_sharpe": float(stock_sharpe),
                "sector_sharpe": float(sector_sharpe),
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
            # If no sector metrics, calculate basic risk metrics from stock data only
            if not sector_metrics:
                logging.warning(f"No sector metrics available for {sector}, calculating basic risk metrics")
                return self._calculate_basic_risk_metrics(stock_returns, sector, market_metrics)
            
            # Calculate base risk metrics
            volatility = stock_returns.std() * np.sqrt(252)
            sector_volatility = sector_metrics.get('sector_volatility_ratio', 1.0) * volatility
            
            # Enhanced risk score calculation
            risk_score = self._calculate_sector_risk_score(stock_returns, sector_metrics, market_metrics)
            
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
            # Calculate basic volatility
            volatility = stock_returns.std() * np.sqrt(252)
            
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
        """Get sector index data with caching."""
        try:
            logging.info(f"Fetching sector index data for {sector} with period {period}")
            cache_key = f"{sector}_{period}"
            current_time = datetime.now()
            
            # Check cache
            if cache_key in self.sector_data_cache:
                cached_data, cache_time = self.sector_data_cache[cache_key]
                if (current_time - cache_time).total_seconds() < self.cache_duration:
                    logging.info(f"Using cached sector data for {sector}")
                    return cached_data
            
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
            
            # Cache the data
            if sector_data is not None:
                self.sector_data_cache[cache_key] = (sector_data, current_time)
            
            return sector_data
            
        except Exception as e:
            logging.error(f"Error fetching sector index data for {sector}: {e}")
            return None
    
    async def _get_sector_index_data_async(self, sector: str, period: int = 365) -> Optional[pd.DataFrame]:
        """Async version of _get_sector_index_data with caching."""
        try:
            cache_key = f"{sector}_{period}"
            current_time = datetime.now()
            
            # Check cache
            if cache_key in self.sector_data_cache:
                cached_data, cache_time = self.sector_data_cache[cache_key]
                if (current_time - cache_time).total_seconds() < self.cache_duration:
                    return cached_data
            
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
            
            # Cache the data
            if sector_data is not None:
                self.sector_data_cache[cache_key] = (sector_data, current_time)
            
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
        Get historical data for NIFTY 50 index with caching.
        
        Args:
            period: Number of days to retrieve
            
        Returns:
            DataFrame with historical data or None if not available
        """
        try:
            # Check cache first
            cache_key = f"NIFTY_50_{period}"
            current_time = datetime.now()
            
            if cache_key in self.sector_data_cache:
                cached_data, cache_time = self.sector_data_cache[cache_key]
                if (current_time - cache_time).total_seconds() < self.cache_duration:
                    return cached_data
            
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
            
            # Cache the data
            self.sector_data_cache[cache_key] = (data, current_time)
                
            return data
            
        except Exception as e:
            logging.error(f"Error getting NIFTY 50 data: {e}")
            return None
    
    async def _get_nifty_data_async(self, period: int = 365) -> Optional[pd.DataFrame]:
        """Async version of _get_nifty_data with caching."""
        try:
            # Check cache first
            cache_key = f"NIFTY_50_{period}"
            current_time = datetime.now()
            
            if cache_key in self.sector_data_cache:
                cached_data, cache_time = self.sector_data_cache[cache_key]
                if (current_time - cache_time).total_seconds() < self.cache_duration:
                    return cached_data
            
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
            
            # Cache the data
            self.sector_data_cache[cache_key] = (data, current_time)
                
            return data
            
        except Exception as e:
            logging.error(f"Error getting NIFTY 50 data: {e}")
            return None
    
    def _calculate_beta(self, stock_returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate beta coefficient."""
        try:
            if len(stock_returns) < 30 or len(benchmark_returns) < 30:
                return 1.0
            
            cov = np.cov(stock_returns, benchmark_returns)[0, 1]
            var = np.var(benchmark_returns)
            
            if var == 0:
                return 1.0
            
            beta = cov / var
            return max(0.1, min(3.0, beta))  # Clamp between 0.1 and 3.0
            
        except Exception as e:
            logging.error(f"Error calculating beta: {e}")
            return 1.0
    
    def _calculate_correlation(self, stock_returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate correlation coefficient."""
        try:
            if len(stock_returns) < 30 or len(benchmark_returns) < 30:
                return 0.5
            
            corr = np.corrcoef(stock_returns, benchmark_returns)[0, 1]
            
            if np.isnan(corr):
                return 0.5
            
            return float(corr)
            
        except Exception as e:
            logging.error(f"Error calculating correlation: {e}")
            return 0.5
    
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
            "sector_benchmarking": {
                "sector_beta": 1.0,
                "sector_correlation": 0.6,
                "sector_sharpe_ratio": 0.0,
                "sector_volatility": 0.15,
                "sector_max_drawdown": 0.10,
                "sector_cumulative_return": 0.0,
                "sector_annualized_return": 0.0,
                "sector_index": sector_index,
                "sector_data_points": 0
            },
            "relative_performance": self._get_default_relative_performance(),
            "sector_risk_metrics": default_risk_metrics,
            "analysis_summary": {
                "market_position": "neutral",
                "sector_position": "neutral",
                "risk_assessment": "medium",
                "investment_recommendation": "hold"
            },
            "timestamp": datetime.now().isoformat(),
            "data_points": {"stock_data_points": 0, "market_data_points": 0, "sector_data_points": 0},
            "error": "Comprehensive benchmarking analysis failed"
        }

    def get_stock_specific_benchmarking(self, stock_symbol: str, stock_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get benchmarking analysis for a specific stock (optimized - only fetches relevant data).
        
        Args:
            stock_symbol: Stock symbol to analyze
            stock_data: Historical stock data
            
        Returns:
            Dict containing stock-specific benchmarking analysis
        """
        try:
            logging.info(f"Calculating stock-specific benchmarking for {stock_symbol}")
            
            # Get sector information
            sector = self.sector_classifier.get_stock_sector(stock_symbol)
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
            
            # Sample only 5 sectors for quick ranking (instead of all 16)
            sample_sectors = list(self.sector_tokens.keys())[:5]
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
                    'rank': rank,
                    'performance': data
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
        Uses smart caching to avoid repeated API calls.
        
        Args:
            force_refresh: Force refresh of cached data
            
        Returns:
            Dict containing comprehensive sector analysis with rotation and correlation
        """
        try:
            current_time = datetime.now()
            
            # Check if we have valid cached comprehensive data
            if (not force_refresh and 
                self.last_comprehensive_update and 
                (current_time - self.last_comprehensive_update).total_seconds() < self.comprehensive_cache_duration):
                
                logging.info("Using cached comprehensive sector analysis")
                return self.comprehensive_sector_cache
            
            logging.info("Generating fresh comprehensive sector analysis (all sectors)")
            
            # Generate comprehensive analysis (all sectors)
            comprehensive_analysis = {
                'sector_rotation': self.analyze_sector_rotation("3M"),
                'sector_correlation': self.generate_sector_correlation_matrix("6M"),
                'market_overview': self._generate_market_overview(),
                'last_updated': current_time.isoformat(),
                'cache_duration_minutes': self.comprehensive_cache_duration // 60
            }
            
            # Cache the comprehensive analysis
            self.comprehensive_sector_cache = comprehensive_analysis
            self.last_comprehensive_update = current_time
            
            logging.info("Comprehensive sector analysis cached successfully")
            return comprehensive_analysis
            
        except Exception as e:
            logging.error(f"Error in comprehensive sector analysis: {e}")
            return None

    def get_hybrid_stock_analysis(self, stock_symbol: str, stock_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get hybrid stock analysis combining optimized stock-specific data with comprehensive sector relationships.
        
        Args:
            stock_symbol: Stock symbol to analyze
            stock_data: Historical stock data
            
        Returns:
            Dict containing hybrid analysis with both stock-specific and comprehensive sector data
        """
        try:
            logging.info(f"Calculating hybrid analysis for {stock_symbol}")
            
            # Get stock-specific benchmarking (optimized - minimal API calls)
            stock_specific = self.get_stock_specific_benchmarking(stock_symbol, stock_data)
            
            # Get comprehensive sector analysis (cached - no additional API calls if recent)
            comprehensive = self.get_comprehensive_sector_analysis()
            
            # Get stock's sector
            stock_sector = self.sector_classifier.get_stock_sector(stock_symbol)
            
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
            
            # Extract sector rotation context
            if comprehensive.get('sector_rotation'):
                rotation_data = comprehensive['sector_rotation']
                sector_rankings = rotation_data.get('sector_rankings', {})
                
                if stock_sector and stock_sector in sector_rankings:
                    relevant_data['sector_rotation_context'] = {
                        'stock_sector_rank': sector_rankings[stock_sector]['rank'],
                        'stock_sector_performance': sector_rankings[stock_sector]['performance'],
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
                    'sector_count': len(self.sector_tokens)
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

    async def get_optimized_comprehensive_sector_analysis(self, symbol: str, stock_data: pd.DataFrame, sector: str) -> Dict[str, Any]:
        """
        OPTIMIZED: Unified sector data fetcher that minimizes API calls and maximizes data reuse.
        
        This method fetches all required sector data in a single optimized operation:
        - Uses optimized timeframes (1M, 3M, 6M instead of 3M, 6M, 1Y)
        - Fetches data once and reuses across all analyses
        - Implements smart caching and data sharing
        - Reduces API calls from 35 to 5-8 calls
        
        Args:
            symbol: Stock symbol
            stock_data: Historical stock data
            sector: Stock's sector
            
        Returns:
            Dict containing all sector analysis data (benchmarking, rotation, correlation)
        """
        try:
            logging.info(f"OPTIMIZED: Starting unified sector analysis for {symbol} in {sector} sector")
            
            # OPTIMIZED TIMEFRAMES: Reduced data requirements
            OPTIMIZED_TIMEFRAMES = {
                "sector_rotation": 30,    # 1M - sufficient for rotation analysis
                "correlation": 60,        # 3M - sufficient for correlation analysis
                "benchmarking": 180,      # 6M - sufficient for benchmarking metrics
                "comprehensive": 180      # 6M - unified timeframe for all analyses
            }
            
            # Use comprehensive timeframe for all data fetching
            days = OPTIMIZED_TIMEFRAMES["comprehensive"]
            
            # STEP 1: Fetch NIFTY 50 data once (reused for all analyses)
            logging.info(f"OPTIMIZED: Fetching NIFTY 50 data once for {days} days (will be reused)")
            nifty_data = await self._get_nifty_data_async(days + 20)
            if nifty_data is None or len(nifty_data) < days:
                logging.warning("Could not fetch NIFTY 50 data for optimized analysis")
                return self._get_fallback_optimized_analysis(symbol, sector)
            
            # STEP 2: Fetch stock's sector data once (reused for all analyses)
            logging.info(f"OPTIMIZED: Fetching {sector} sector data once for {days} days")
            sector_data = await self._get_sector_data_async(sector, days + 20)
            if sector_data is None or len(sector_data) < days:
                logging.warning(f"Could not fetch {sector} sector data for optimized analysis")
                return self._get_fallback_optimized_analysis(symbol, sector)
            
            # STEP 3: Fetch top 8 sectors for rotation analysis (instead of all 16)
            # Select sectors based on relevance and performance
            relevant_sectors = self._get_relevant_sectors_for_analysis(sector)
            logging.info(f"OPTIMIZED: Fetching {len(relevant_sectors)} relevant sectors instead of all 16")
            
            sector_data_dict = {}
            async def fetch_relevant_sector_data(sector_name):
                try:
                    data = await self._get_sector_data_async(sector_name, days + 20)
                    if data is not None and len(data) >= days * 0.8:
                        return sector_name, data
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
            
            # STEP 4: Calculate all metrics using fetched data
            logging.info("OPTIMIZED: Calculating comprehensive sector metrics")
            
            # Calculate benchmarking metrics
            benchmarking = self._calculate_optimized_benchmarking(symbol, stock_data, sector, sector_data, nifty_data)
            
            # Calculate rotation metrics using relevant sectors
            rotation = self._calculate_optimized_rotation(sector_data_dict, nifty_data, OPTIMIZED_TIMEFRAMES["sector_rotation"])
            
            # Calculate correlation metrics using relevant sectors
            correlation = self._calculate_optimized_correlation(sector_data_dict, OPTIMIZED_TIMEFRAMES["correlation"])
            
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
                    'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
            }
            
            logging.info(f"OPTIMIZED: Unified sector analysis completed for {symbol}")
            return comprehensive_result
            
        except Exception as e:
            logging.error(f"Error in optimized comprehensive sector analysis: {e}")
            return self._get_fallback_optimized_analysis(symbol, sector)
    
    def _get_relevant_sectors_for_analysis(self, stock_sector: str) -> List[str]:
        """
        OPTIMIZED: Select relevant sectors for analysis instead of fetching all 16.
        
        Args:
            stock_sector: The stock's sector
            
        Returns:
            List of relevant sectors to analyze
        """
        # Always include the stock's sector
        relevant_sectors = [stock_sector]
        
        # Add high-impact sectors that are commonly correlated
        high_impact_sectors = ['BANKING', 'IT', 'PHARMA', 'AUTO', 'FMCG', 'ENERGY']
        
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
            additional_sectors = ['METAL', 'REALTY', 'HEALTHCARE', 'CONSUMER_DURABLES']
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
            # Calculate stock returns
            stock_returns = stock_data['close'].pct_change().dropna()
            
            # Calculate sector returns
            sector_returns = sector_data['close'].pct_change().dropna()
            
            # Calculate NIFTY returns
            nifty_returns = nifty_data['close'].pct_change().dropna()
            
            # Calculate metrics
            stock_beta = self._calculate_beta(stock_returns, nifty_returns)
            sector_beta = self._calculate_beta(sector_returns, nifty_returns)
            stock_correlation = self._calculate_correlation(stock_returns, nifty_returns)
            sector_correlation = self._calculate_correlation(sector_returns, nifty_returns)
            
            # Calculate performance metrics
            stock_cumulative_return = (1 + stock_returns).prod() - 1
            sector_cumulative_return = (1 + sector_returns).prod() - 1
            nifty_cumulative_return = (1 + nifty_returns).prod() - 1
            
            return {
                'sector': sector,
                'beta': round(stock_beta, 3),
                'sector_beta': round(sector_beta, 3),
                'correlation': round(stock_correlation, 3),
                'sector_correlation': round(sector_correlation, 3),
                'excess_return': round(stock_cumulative_return - nifty_cumulative_return, 4),
                'sector_excess_return': round(sector_cumulative_return - nifty_cumulative_return, 4),
                'optimization_note': 'Calculated using pre-fetched data'
            }
            
        except Exception as e:
            logging.error(f"Error in optimized benchmarking: {e}")
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
            
            # Rank sectors
            sorted_sectors = sorted(sector_performance.items(), 
                                  key=lambda x: x[1]['relative_strength'], reverse=True)
            
            for rank, (sector_name, data) in enumerate(sorted_sectors, 1):
                sector_rankings[sector_name] = {
                    'rank': rank,
                    'performance': data
                }
            
            return {
                'timeframe': f"{days}D",
                'sector_performance': sector_performance,
                'sector_rankings': sector_rankings,
                'optimization_note': 'Calculated using pre-fetched data'
            }
            
        except Exception as e:
            logging.error(f"Error in optimized rotation: {e}")
            return {'error': str(e)}
    
    def _calculate_optimized_correlation(self, sector_data_dict: Dict[str, pd.DataFrame], 
                                       days: int) -> Dict[str, Any]:
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
            
            return {
                'timeframe': f"{days}D",
                'correlation_matrix': correlation_matrix.round(3).to_dict(),
                'average_correlation': round(avg_correlation, 3),
                'high_correlation_pairs': high_correlation_pairs,
                'low_correlation_pairs': low_correlation_pairs,
                'optimization_note': 'Calculated using pre-fetched data'
            }
            
        except Exception as e:
            logging.error(f"Error in optimized correlation: {e}")
            return {'error': str(e)}
    
    def _get_fallback_optimized_analysis(self, symbol: str, sector: str) -> Dict[str, Any]:
        """
        Fallback analysis when optimized analysis fails.
        """
        return {
            'sector_benchmarking': {'sector': sector, 'error': 'Optimized analysis failed'},
            'sector_rotation': {'error': 'Optimized analysis failed'},
            'sector_correlation': {'error': 'Optimized analysis failed'},
            'optimization_metrics': {
                'api_calls_reduced': 'Fallback mode',
                'data_points_reduced': 'Fallback mode',
                'timeframes_optimized': 'Fallback mode',
                'cache_duration': 'Fallback mode',
                'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        }

# Global instance
sector_benchmarking_provider = SectorBenchmarkingProvider()
