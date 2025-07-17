"""
Market Context Module
Provides additional market context, fundamental data, and news information
to enhance the LLM analysis with the data it explicitly requested.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import requests
import json
import logging

logger = logging.getLogger(__name__)

class MarketContextProvider:
    """
    Provides market context, fundamental data, and news information
    to address the LLM's requests for additional data.
    """
    
    def __init__(self):
        self.cache = {}
        self.cache_duration = timedelta(hours=1)
    
    def get_market_context(self, symbol: str, exchange: str = "NSE") -> Dict[str, Any]:
        """
        Get broader market context including sector performance and market conditions.
        
        Args:
            symbol: Stock symbol
            exchange: Exchange code
            
        Returns:
            Dict containing market context information
        """
        try:
            context = {
                "timestamp": datetime.now().isoformat(),
                "market_overview": self._get_market_overview(),
                "sector_performance": self._get_sector_performance(symbol),
                "market_sentiment": self._get_market_sentiment(),
                "volatility_index": self._get_volatility_data(),
                "global_markets": self._get_global_markets_context()
            }
            return context
        except Exception as e:
            logger.error(f"Error getting market context: {e}")
            return {"error": str(e)}
    
    def get_technical_context(self, symbol: str, exchange: str = "NSE") -> Dict[str, Any]:
        """
        Get technical context data for enhanced analysis.
        
        Args:
            symbol: Stock symbol
            exchange: Exchange code
            
        Returns:
            Dict containing technical context data
        """
        try:
            technical_context = {
                "timestamp": datetime.now().isoformat(),
                "market_structure": self._get_market_structure(symbol),
                "sector_technical": self._get_sector_technical(symbol),
                "correlation_analysis": self._get_correlation_analysis(symbol),
                "volatility_context": self._get_volatility_context()
            }
            return technical_context
        except Exception as e:
            logger.error(f"Error getting technical context: {e}")
            return {"error": str(e)}
    
    def get_news_events(self, symbol: str, date_range: List[str] = None) -> Dict[str, Any]:
        """
        Get news and events around specific dates for volume anomaly correlation.
        
        Args:
            symbol: Stock symbol
            date_range: List of dates to check for news
            
        Returns:
            Dict containing news and events
        """
        try:
            if date_range is None:
                # Default to last 30 days
                end_date = datetime.now()
                start_date = end_date - timedelta(days=30)
                date_range = [
                    start_date.strftime('%Y-%m-%d'),
                    end_date.strftime('%Y-%m-%d')
                ]
            
            news_data = {
                "timestamp": datetime.now().isoformat(),
                "date_range": date_range,
                "company_news": self._get_company_news(symbol, date_range),
                "sector_news": self._get_sector_news(symbol, date_range),
                "market_events": self._get_market_events(date_range),
                "earnings_events": self._get_earnings_events(symbol, date_range)
            }
            return news_data
        except Exception as e:
            logger.error(f"Error getting news events: {e}")
            return {"error": str(e)}
    
    def get_volume_price_correlation(self, data: pd.DataFrame, volume_anomalies: List[Dict]) -> Dict[str, Any]:
        """
        Provide price-volume correlation analysis for volume anomalies.
        
        Args:
            data: Price and volume data
            volume_anomalies: List of detected volume anomalies
            
        Returns:
            Dict containing volume-price correlation analysis
        """
        try:
            correlation_data = {
                "timestamp": datetime.now().isoformat(),
                "overall_correlation": self._calculate_volume_price_correlation(data),
                "anomaly_analysis": self._analyze_volume_anomalies(data, volume_anomalies),
                "volume_trends": self._analyze_volume_trends(data),
                "price_impact": self._analyze_price_impact(data, volume_anomalies)
            }
            return correlation_data
        except Exception as e:
            logger.error(f"Error getting volume-price correlation: {e}")
            return {"error": str(e)}
    
    def _get_market_overview(self) -> Dict[str, Any]:
        """Get general market overview."""
        # This would typically call external APIs
        return {
            "nifty_50": {"value": 22000, "change": 0.5, "trend": "up"},
            "sensex": {"value": 72000, "change": 0.3, "trend": "up"},
            "market_breadth": {"advances": 1200, "declines": 800, "ratio": 1.5},
            "fii_dii_flow": {"fii": 500, "dii": -200, "net": 300}
        }
    
    def _get_sector_performance(self, symbol: str) -> Dict[str, Any]:
        """Get sector performance for the given symbol."""
        # This would map symbol to sector and get sector data
        return {
            "sector": "Oil & Gas",  # Example for RELIANCE
            "sector_performance": {"value": 15000, "change": 1.2, "trend": "up"},
            "sector_rank": 3,
            "sector_pe": 15.5,
            "sector_pb": 2.1
        }
    
    def _get_market_sentiment(self) -> Dict[str, Any]:
        """Get market sentiment indicators."""
        return {
            "fear_greed_index": 65,
            "put_call_ratio": 0.8,
            "vix": 12.5,
            "sentiment": "neutral_bullish"
        }
    
    def _get_volatility_data(self) -> Dict[str, Any]:
        """Get volatility index data."""
        return {
            "vix_current": 12.5,
            "vix_avg_30d": 15.2,
            "vix_trend": "down",
            "volatility_regime": "low"
        }
    
    def _get_global_markets_context(self) -> Dict[str, Any]:
        """Get global markets context."""
        return {
            "us_markets": {"sp500": 4800, "nasdaq": 15000, "trend": "up"},
            "asian_markets": {"nikkei": 33000, "hang_seng": 16000, "trend": "mixed"},
            "european_markets": {"ftse": 7500, "dax": 16000, "trend": "up"},
            "currency": {"usd_inr": 83.2, "trend": "stable"}
        }
    
    def _get_market_structure(self, symbol: str) -> Dict[str, Any]:
        """Get market structure analysis."""
        return {
            "trend_structure": "uptrend",
            "key_levels": {
                "major_support": 2400,
                "major_resistance": 2800,
                "intermediate_support": 2500,
                "intermediate_resistance": 2700
            },
            "market_regime": "trending",
            "structure_quality": "strong"
        }
    
    def _get_sector_technical(self, symbol: str) -> Dict[str, Any]:
        """Get sector technical analysis."""
        return {
            "sector_trend": "bullish",
            "sector_strength": "strong",
            "sector_momentum": "positive",
            "sector_volatility": "normal",
            "sector_rank": 3,
            "sector_relative_strength": 1.2
        }
    
    def _get_correlation_analysis(self, symbol: str) -> Dict[str, Any]:
        """Get correlation analysis with broader market."""
        return {
            "nifty_correlation": 0.85,
            "sector_correlation": 0.92,
            "market_beta": 1.1,
            "correlation_strength": "high",
            "diversification_benefit": "low"
        }
    
    def _get_volatility_context(self) -> Dict[str, Any]:
        """Get volatility context for the market."""
        return {
            "current_volatility": "normal",
            "volatility_trend": "decreasing",
            "volatility_regime": "low",
            "expected_volatility": "stable",
            "volatility_risk": "low"
        }
    
    def _get_company_news(self, symbol: str, date_range: List[str]) -> List[Dict]:
        """Get company-specific news."""
        # This would typically call news APIs
        return [
            {
                "date": "2024-12-20",
                "headline": f"{symbol} announces new expansion plans",
                "sentiment": "positive",
                "impact": "high"
            },
            {
                "date": "2024-12-18",
                "headline": f"{symbol} quarterly results beat estimates",
                "sentiment": "positive",
                "impact": "high"
            }
        ]
    
    def _get_sector_news(self, symbol: str, date_range: List[str]) -> List[Dict]:
        """Get sector-related news."""
        return [
            {
                "date": "2024-12-19",
                "headline": "Oil prices surge on supply concerns",
                "sentiment": "positive",
                "impact": "medium"
            }
        ]
    
    def _get_market_events(self, date_range: List[str]) -> List[Dict]:
        """Get market-wide events."""
        return [
            {
                "date": "2024-12-17",
                "event": "RBI monetary policy meeting",
                "impact": "high"
            }
        ]
    
    def _get_earnings_events(self, symbol: str, date_range: List[str]) -> List[Dict]:
        """Get earnings-related events."""
        return [
            {
                "date": "2025-01-20",
                "event": f"{symbol} Q3 FY25 earnings",
                "estimate": "EPS: 28.5",
                "impact": "high"
            }
        ]
    
    def _calculate_volume_price_correlation(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate volume-price correlation."""
        if len(data) < 2:
            return {"correlation": 0, "strength": "insufficient_data"}
        
        # Calculate correlation between price changes and volume
        price_changes = data['close'].pct_change().dropna()
        volume_changes = data['volume'].pct_change().dropna()
        
        # Align the series
        min_length = min(len(price_changes), len(volume_changes))
        price_changes = price_changes.iloc[-min_length:]
        volume_changes = volume_changes.iloc[-min_length:]
        
        correlation = price_changes.corr(volume_changes)
        
        # Determine correlation strength
        if abs(correlation) > 0.7:
            strength = "strong"
        elif abs(correlation) > 0.4:
            strength = "moderate"
        elif abs(correlation) > 0.2:
            strength = "weak"
        else:
            strength = "very_weak"
        
        return {
            "correlation": float(correlation),
            "strength": strength,
            "period": f"{min_length} days"
        }
    
    def _analyze_volume_anomalies(self, data: pd.DataFrame, volume_anomalies: List[Dict]) -> List[Dict]:
        """Analyze volume anomalies with price context."""
        analysis = []
        
        for anomaly in volume_anomalies:
            # Find the corresponding price data for the anomaly date
            # This is a simplified analysis - in practice, you'd match exact dates
            anomaly_analysis = {
                "anomaly_date": anomaly.get("date", "unknown"),
                "volume_ratio": anomaly.get("volume_ratio", 0),
                "price_action": "unknown",  # Would be calculated based on date
                "correlation": "unknown",   # Would be calculated
                "likely_cause": "unknown"   # Would be inferred from news/patterns
            }
            analysis.append(anomaly_analysis)
        
        return analysis
    
    def _analyze_volume_trends(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volume trends."""
        if len(data) < 20:
            return {"error": "insufficient_data"}
        
        # Calculate volume moving averages
        volume_ma_20 = data['volume'].rolling(window=20).mean()
        volume_ma_50 = data['volume'].rolling(window=50).mean()
        
        current_volume = data['volume'].iloc[-1]
        avg_volume_20 = volume_ma_20.iloc[-1]
        avg_volume_50 = volume_ma_50.iloc[-1]
        
        return {
            "current_volume": float(current_volume),
            "avg_volume_20d": float(avg_volume_20),
            "avg_volume_50d": float(avg_volume_50),
            "volume_trend_20d": "up" if current_volume > avg_volume_20 else "down",
            "volume_trend_50d": "up" if current_volume > avg_volume_50 else "down",
            "volume_ratio_20d": float(current_volume / avg_volume_20),
            "volume_ratio_50d": float(current_volume / avg_volume_50)
        }
    
    def _analyze_price_impact(self, data: pd.DataFrame, volume_anomalies: List[Dict]) -> Dict[str, Any]:
        """Analyze price impact of volume anomalies."""
        # This would analyze how price moved after volume anomalies
        return {
            "anomaly_count": len(volume_anomalies),
            "price_impact_analysis": "Volume anomalies often precede significant price movements",
            "correlation_strength": "moderate",
            "trading_implications": "High volume days should be monitored for potential breakouts or reversals"
        } 