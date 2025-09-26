#!/usr/bin/env python3
"""
Volume Confirmation Agent - Data Processing Module

This module handles all data processing specific to the Volume Confirmation Agent,
including price-volume correlation analysis, trend validation, and context formatting.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Try to import scipy for advanced statistical analysis
try:
    from scipy import stats
    from scipy.signal import find_peaks
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

class VolumeConfirmationProcessor:
    """
    Specialized data processor for Volume Confirmation Agent
    
    Focuses on price-volume relationship validation and trend confirmation analysis
    """
    
    def __init__(self):
        self.min_data_points = 20  # Minimum data points for reliable analysis
        
    def calculate_price_volume_correlation(self, data: pd.DataFrame, window: int = 30) -> Dict[str, Any]:
        """
        Calculate price-volume correlation metrics
        
        Args:
            data: DataFrame with OHLCV data
            window: Rolling window for correlation (default 30 days)
            
        Returns:
            Dict containing correlation analysis
        """
        if len(data) < self.min_data_points:
            return {"error": "Insufficient data for correlation analysis"}
        
        try:
            # Calculate price changes and volume ratios
            data = data.copy()
            data['price_change'] = data['close'].pct_change()
            data['volume_change'] = data['volume'].pct_change()
            
            # Calculate rolling correlation
            rolling_corr = data['price_change'].rolling(window=window).corr(data['volume_change'])
            
            # Overall correlation
            if HAS_SCIPY:
                overall_corr, p_value = stats.pearsonr(
                    data['price_change'].dropna(), 
                    data['volume_change'].dropna()
                )
                significance = "high" if p_value < 0.05 else "medium" if p_value < 0.1 else "low"
            else:
                overall_corr = data['price_change'].corr(data['volume_change'])
                significance = "medium"  # Default when p-value not available
            
            # Current correlation (last 10 days)
            recent_corr = rolling_corr.tail(10).mean()
            
            # Correlation trend
            if len(rolling_corr.dropna()) > 5:
                recent_corr_values = rolling_corr.dropna().tail(5)
                if len(recent_corr_values) >= 3:
                    trend_slope = np.polyfit(range(len(recent_corr_values)), recent_corr_values, 1)[0]
                    correlation_trend = "increasing" if trend_slope > 0.01 else "decreasing" if trend_slope < -0.01 else "stable"
                else:
                    correlation_trend = "stable"
            else:
                correlation_trend = "insufficient_data"
            
            return {
                "correlation_coefficient": round(overall_corr, 3),
                "recent_correlation": round(recent_corr, 3),
                "significance_level": significance,
                "correlation_trend": correlation_trend,
                "correlation_strength": "strong" if abs(overall_corr) > 0.5 else "medium" if abs(overall_corr) > 0.3 else "weak",
                "correlation_direction": "positive" if overall_corr > 0.1 else "negative" if overall_corr < -0.1 else "neutral"
            }
            
        except Exception as e:
            return {"error": f"Correlation calculation failed: {str(e)}"}
    
    def analyze_recent_confirmations(self, data: pd.DataFrame, lookback: int = 14) -> List[Dict[str, Any]]:
        """
        Analyze recent price movements and their volume confirmation
        
        Args:
            data: DataFrame with OHLCV data
            lookback: Number of recent days to analyze
            
        Returns:
            List of recent confirmation signals
        """
        if len(data) < lookback:
            return []
        
        try:
            recent_data = data.tail(lookback).copy()
            confirmations = []
            
            # Calculate volume moving average for context
            volume_ma = data['volume'].rolling(window=20).mean()
            
            for i in range(1, len(recent_data)):
                current = recent_data.iloc[i]
                previous = recent_data.iloc[i-1]
                
                # Price movement analysis
                price_change_pct = ((current['close'] - previous['close']) / previous['close']) * 100
                
                if abs(price_change_pct) > 1.0:  # Significant price movement (>1%)
                    # Volume analysis
                    volume_ratio = current['volume'] / volume_ma.iloc[current.name] if volume_ma.iloc[current.name] > 0 else 1.0
                    
                    # Determine confirmation
                    if price_change_pct > 1.0 and volume_ratio > 1.2:
                        volume_response = "confirming"
                        significance = "high" if volume_ratio > 2.0 else "medium"
                    elif price_change_pct < -1.0 and volume_ratio > 1.2:
                        volume_response = "confirming"
                        significance = "high" if volume_ratio > 2.0 else "medium"
                    elif abs(price_change_pct) > 2.0 and volume_ratio < 0.8:
                        volume_response = "diverging"
                        significance = "medium"
                    else:
                        continue  # Skip insignificant movements
                    
                    confirmations.append({
                        "date": current.name.strftime('%Y-%m-%d'),
                        "price_change_pct": round(price_change_pct, 2),
                        "volume_ratio": round(volume_ratio, 2),
                        "price_move": "up" if price_change_pct > 0 else "down",
                        "volume_response": volume_response,
                        "significance": significance
                    })
            
            return confirmations[-5:]  # Return last 5 significant confirmations
            
        except Exception as e:
            return [{"error": f"Recent confirmation analysis failed: {str(e)}"}]
    
    def analyze_trend_support(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze volume support for price trends
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dict containing trend support analysis
        """
        if len(data) < 30:
            return {"error": "Insufficient data for trend analysis"}
        
        try:
            data = data.copy()
            
            # Identify trend periods
            data['price_ma_20'] = data['close'].rolling(window=20).mean()
            data['price_trend'] = np.where(data['close'] > data['price_ma_20'], 1, -1)
            
            # Volume analysis during trends
            volume_ma = data['volume'].rolling(window=20).mean()
            data['volume_ratio'] = data['volume'] / volume_ma
            
            # Uptrend analysis
            uptrend_data = data[data['price_trend'] == 1].tail(20)  # Recent uptrend periods
            if len(uptrend_data) > 5:
                uptrend_volume_avg = uptrend_data['volume_ratio'].mean()
                uptrend_support = "strong" if uptrend_volume_avg > 1.2 else "weak" if uptrend_volume_avg < 0.8 else "medium"
            else:
                uptrend_support = "none"
            
            # Downtrend analysis
            downtrend_data = data[data['price_trend'] == -1].tail(20)  # Recent downtrend periods
            if len(downtrend_data) > 5:
                downtrend_volume_avg = downtrend_data['volume_ratio'].mean()
                downtrend_support = "strong" if downtrend_volume_avg > 1.2 else "weak" if downtrend_volume_avg < 0.8 else "medium"
            else:
                downtrend_support = "none"
            
            # Consolidation analysis (price near moving average)
            consolidation_data = data[abs(data['close'] - data['price_ma_20']) / data['price_ma_20'] < 0.02].tail(20)
            if len(consolidation_data) > 5:
                consolidation_volume_avg = consolidation_data['volume_ratio'].mean()
                if consolidation_volume_avg < 0.8:
                    consolidation_pattern = "contracting"
                elif consolidation_volume_avg > 1.2:
                    consolidation_pattern = "expanding"
                else:
                    consolidation_pattern = "irregular"
            else:
                consolidation_pattern = "insufficient_data"
            
            return {
                "uptrend_volume_support": uptrend_support,
                "downtrend_volume_support": downtrend_support,
                "consolidation_volume_pattern": consolidation_pattern,
                "current_trend": "uptrend" if data['price_trend'].iloc[-1] == 1 else "downtrend",
                "trend_consistency": len(data[data['price_trend'] == data['price_trend'].iloc[-1]].tail(10)) / 10
            }
            
        except Exception as e:
            return {"error": f"Trend support analysis failed: {str(e)}"}
    
    def calculate_volume_moving_averages(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate volume moving averages for context
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dict containing volume averages
        """
        try:
            volume_10d = data['volume'].rolling(window=10).mean().iloc[-1]
            volume_20d = data['volume'].rolling(window=20).mean().iloc[-1]
            volume_50d = data['volume'].rolling(window=50).mean().iloc[-1] if len(data) >= 50 else volume_20d
            
            current_volume = data['volume'].iloc[-1]
            
            return {
                "volume_10d_avg": int(volume_10d) if not pd.isna(volume_10d) else 0,
                "volume_20d_avg": int(volume_20d) if not pd.isna(volume_20d) else 0,
                "volume_50d_avg": int(volume_50d) if not pd.isna(volume_50d) else 0,
                "current_volume": int(current_volume),
                "volume_vs_10d": round(current_volume / volume_10d, 2) if volume_10d > 0 else 1.0,
                "volume_vs_20d": round(current_volume / volume_20d, 2) if volume_20d > 0 else 1.0,
                "volume_vs_50d": round(current_volume / volume_50d, 2) if volume_50d > 0 else 1.0
            }
        except Exception as e:
            return {"error": f"Volume averages calculation failed: {str(e)}"}
    
    def process_volume_confirmation_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Main processing function for Volume Confirmation Agent
        
        Args:
            data: DataFrame with OHLCV data (indexed by date)
            
        Returns:
            Dict containing all volume confirmation analysis data
        """
        if len(data) < self.min_data_points:
            return {
                "error": f"Insufficient data: need at least {self.min_data_points} data points, got {len(data)}",
                "data_length": len(data)
            }
        
        try:
            # Core analysis components
            correlation_analysis = self.calculate_price_volume_correlation(data)
            recent_confirmations = self.analyze_recent_confirmations(data)
            trend_support = self.analyze_trend_support(data)
            volume_averages = self.calculate_volume_moving_averages(data)
            
            # Overall assessment
            overall_assessment = self._determine_overall_assessment(
                correlation_analysis, recent_confirmations, trend_support
            )
            
            return {
                "analysis_timestamp": datetime.now().isoformat(),
                "data_period": f"{len(data)} days",
                "data_range": f"{data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}",
                "price_volume_correlation": correlation_analysis,
                "recent_movements": recent_confirmations,
                "trend_support": trend_support,
                "volume_averages": volume_averages,
                "overall_assessment": overall_assessment,
                "data_quality": "excellent" if len(data) > 60 else "good" if len(data) > 30 else "limited"
            }
            
        except Exception as e:
            return {
                "error": f"Volume confirmation processing failed: {str(e)}",
                "data_length": len(data)
            }
    
    def _determine_overall_assessment(self, correlation: Dict, confirmations: List, trends: Dict) -> Dict[str, Any]:
        """
        Determine overall volume confirmation assessment
        
        Args:
            correlation: Correlation analysis results
            confirmations: Recent confirmation signals
            trends: Trend support analysis
            
        Returns:
            Dict containing overall assessment
        """
        try:
            # Correlation scoring
            corr_score = 0
            if not correlation.get('error'):
                corr_strength = correlation.get('correlation_strength', 'weak')
                if corr_strength == 'strong':
                    corr_score = 3
                elif corr_strength == 'medium':
                    corr_score = 2
                else:
                    corr_score = 1
            
            # Recent confirmations scoring
            conf_score = 0
            confirming_signals = len([c for c in confirmations if c.get('volume_response') == 'confirming'])
            diverging_signals = len([c for c in confirmations if c.get('volume_response') == 'diverging'])
            
            if confirming_signals > diverging_signals:
                conf_score = 2
            elif confirming_signals == diverging_signals:
                conf_score = 1
            else:
                conf_score = 0
            
            # Trend support scoring
            trend_score = 0
            if not trends.get('error'):
                current_trend = trends.get('current_trend', '')
                if current_trend == 'uptrend' and trends.get('uptrend_volume_support') == 'strong':
                    trend_score = 2
                elif current_trend == 'downtrend' and trends.get('downtrend_volume_support') == 'strong':
                    trend_score = 2
                else:
                    trend_score = 1
            
            # Overall assessment
            total_score = corr_score + conf_score + trend_score
            
            if total_score >= 6:
                status = "volume_confirms_price"
                strength = "strong"
            elif total_score >= 4:
                status = "volume_confirms_price"
                strength = "medium"
            elif total_score >= 2:
                status = "mixed_signals"
                strength = "weak"
            else:
                status = "volume_diverges"
                strength = "weak"
            
            confidence = min(95, max(30, (total_score / 7) * 100))
            
            return {
                "confirmation_status": status,
                "confirmation_strength": strength,
                "confidence_score": int(confidence),
                "scoring_breakdown": {
                    "correlation_score": corr_score,
                    "confirmations_score": conf_score,
                    "trend_score": trend_score,
                    "total_score": total_score
                }
            }
            
        except Exception as e:
            return {
                "confirmation_status": "analysis_error",
                "confirmation_strength": "unknown",
                "confidence_score": 0,
                "error": f"Assessment calculation failed: {str(e)}"
            }

def test_volume_confirmation_processor():
    """Test function for Volume Confirmation Processor"""
    print("🧪 Testing Volume Confirmation Processor")
    print("=" * 50)
    
    # Create sample data
    dates = pd.date_range(start='2024-07-01', end='2024-09-20', freq='D')
    np.random.seed(42)
    
    # Generate realistic price and volume data
    base_price = 2400
    price_trend = np.cumsum(np.random.normal(0.5, 15, len(dates)))
    prices = base_price + price_trend
    
    sample_data = pd.DataFrame({
        'open': prices + np.random.normal(0, 5, len(dates)),
        'high': prices + np.abs(np.random.normal(10, 8, len(dates))),
        'low': prices - np.abs(np.random.normal(10, 8, len(dates))),
        'close': prices,
        'volume': np.abs(np.random.lognormal(14.5, 0.6, len(dates)))  # Log-normal volume
    }, index=dates)
    
    # Ensure realistic OHLC relationships
    sample_data['high'] = np.maximum(sample_data[['open', 'close']].max(axis=1), sample_data['high'])
    sample_data['low'] = np.minimum(sample_data[['open', 'close']].min(axis=1), sample_data['low'])
    
    print(f"✅ Created sample data: {len(sample_data)} days")
    print(f"   Price range: ₹{sample_data['close'].min():.2f} - ₹{sample_data['close'].max():.2f}")
    print(f"   Volume range: {sample_data['volume'].min():,.0f} - {sample_data['volume'].max():,.0f}")
    
    # Test processor
    processor = VolumeConfirmationProcessor()
    result = processor.process_volume_confirmation_data(sample_data)
    
    if 'error' in result:
        print(f"❌ Processing failed: {result['error']}")
        return False
    
    print(f"\n📊 Volume Confirmation Analysis Results:")
    print(f"   Data period: {result['data_period']}")
    print(f"   Data quality: {result['data_quality']}")
    
    # Correlation results
    correlation = result['price_volume_correlation']
    if 'error' not in correlation:
        print(f"   Correlation: {correlation['correlation_coefficient']:.3f} ({correlation['correlation_strength']})")
        print(f"   Trend: {correlation['correlation_trend']}")
    
    # Recent confirmations
    confirmations = result['recent_movements']
    print(f"   Recent signals: {len(confirmations)} confirmations analyzed")
    
    # Overall assessment
    assessment = result['overall_assessment']
    print(f"   Assessment: {assessment['confirmation_status']}")
    print(f"   Strength: {assessment['confirmation_strength']}")
    print(f"   Confidence: {assessment['confidence_score']}%")
    
    print("\n✅ Volume Confirmation Processor test completed successfully!")
    return True

if __name__ == "__main__":
    test_volume_confirmation_processor()