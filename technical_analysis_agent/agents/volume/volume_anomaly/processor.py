#!/usr/bin/env python3
"""
Volume Anomaly Detection Agent - Data Processing Module

This module handles all data processing specific to the Volume Anomaly Detection Agent,
including volume spike detection, anomaly classification, and significance analysis.
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

class VolumeAnomalyProcessor:
    """
    Specialized data processor for Volume Anomaly Detection Agent
    
    Focuses on identifying unusual volume spikes and classifying their significance
    """
    
    def __init__(self):
        self.min_data_points = 30  # Minimum data points for reliable percentile analysis
        # Focus on statistical outliers, not institutional thresholds
        self.spike_thresholds = {
            'low': 2.0,     # 2σ statistical outlier
            'medium': 3.0,  # 3σ statistical outlier
            'high': 4.0     # 4σ extreme statistical outlier
        }
        
        # Institutional detection is handled by separate agent
        # This agent focuses on: statistical anomalies, patterns, retail-driven spikes
        
    def calculate_volume_statistics(self, data: pd.DataFrame, window: int = 90) -> Dict[str, Any]:
        """
        Calculate comprehensive volume statistics for anomaly detection
        
        Args:
            data: DataFrame with OHLCV data
            window: Rolling window for statistics calculation
            
        Returns:
            Dict containing volume statistics
        """
        if len(data) < self.min_data_points:
            return {"error": "Insufficient data for volume statistics"}
        
        try:
            volume_series = data['volume']
            
            # Basic statistics
            volume_mean = volume_series.mean()
            volume_std = volume_series.std()
            volume_median = volume_series.median()
            
            # Percentile analysis
            percentiles = {}
            for p in [50, 75, 90, 95, 99]:
                percentiles[f'percentile_{p}'] = volume_series.quantile(p / 100)
            
            # Rolling statistics for trend analysis
            rolling_mean = volume_series.rolling(window=20).mean()
            rolling_std = volume_series.rolling(window=20).std()
            
            # Current volume context
            current_volume = volume_series.iloc[-1]
            recent_avg = rolling_mean.iloc[-10:].mean()  # Last 10 days average
            
            # Z-score analysis
            z_scores = (volume_series - volume_mean) / volume_std
            current_z_score = z_scores.iloc[-1]
            
            # Volume volatility (coefficient of variation)
            volume_cv = volume_std / volume_mean if volume_mean > 0 else 0
            
            return {
                'volume_mean': float(volume_mean),
                'volume_std': float(volume_std), 
                'volume_median': float(volume_median),
                'volume_cv': float(volume_cv),
                'current_volume': float(current_volume),
                'current_z_score': float(current_z_score),
                'recent_average': float(recent_avg) if not pd.isna(recent_avg) else float(volume_mean),
                'percentiles': percentiles,
                'volume_range': {
                    'min': float(volume_series.min()),
                    'max': float(volume_series.max()),
                    'range_ratio': float(volume_series.max() / volume_series.min()) if volume_series.min() > 0 else 0
                }
            }
            
        except Exception as e:
            return {"error": f"Volume statistics calculation failed: {str(e)}"}
    
    def detect_volume_anomalies(self, data: pd.DataFrame, lookback: int = 60) -> List[Dict[str, Any]]:
        """
        Detect volume anomalies using multiple detection methods
        
        Args:
            data: DataFrame with OHLCV data
            lookback: Number of recent days to analyze for anomalies
            
        Returns:
            List of detected anomalies with metadata
        """
        if len(data) < self.min_data_points:
            return []
        
        try:
            # Use last N days for analysis
            recent_data = data.tail(lookback).copy()
            volume_series = data['volume']
            
            # Calculate statistical context for anomaly detection
            volume_mean = volume_series.mean()
            volume_std = volume_series.std()
            
            # Calculate z-scores for statistical anomaly detection
            volume_z_scores = (volume_series - volume_mean) / volume_std
            
            anomalies = []
            
            for i in range(len(recent_data)):
                current_idx = recent_data.index[i]
                current_volume = recent_data['volume'].iloc[i]
                current_price = recent_data['close'].iloc[i]
                
                # Get statistical context at this point
                current_z_score = volume_z_scores.loc[current_idx] if current_idx in volume_z_scores.index else 0
                
                # Calculate volume ratio for context (but use z-score for detection)
                ratio_mean = current_volume / volume_mean if volume_mean > 0 else 1.0
                
                # Determine if this is a statistical anomaly (using z-scores)
                significance = None
                abs_z_score = abs(current_z_score)
                
                if abs_z_score >= self.spike_thresholds['high']:  # 4σ outlier
                    significance = 'high'
                elif abs_z_score >= self.spike_thresholds['medium']:  # 3σ outlier
                    significance = 'medium'
                elif abs_z_score >= self.spike_thresholds['low']:  # 2σ outlier
                    significance = 'low'
                
                if significance:
                    # Determine price context
                    price_context = self._determine_price_context(recent_data, i)
                    
                    # Focus on statistical anomaly properties only
                    anomaly_type = self._classify_anomaly_type(current_z_score, ratio_mean, price_context)
                    
                    # Determine non-institutional likely cause
                    likely_cause = self._determine_retail_cause(abs_z_score, price_context, ratio_mean)
                    
                    anomalies.append({
                        'date': current_idx.strftime('%Y-%m-%d'),
                        'volume_level': int(current_volume),
                        'volume_ratio': round(ratio_mean, 2),
                        'z_score': round(current_z_score, 2),
                        'significance': significance,
                        'anomaly_type': anomaly_type,
                        'price_context': price_context,
                        'likely_cause': likely_cause,
                        'price': round(current_price, 2)
                    })
            
            # Sort by significance and date
            significance_order = {'high': 3, 'medium': 2, 'low': 1}
            anomalies.sort(key=lambda x: (significance_order[x['significance']], x['date']), reverse=True)
            
            return anomalies[:10]  # Return top 10 most significant anomalies
            
        except Exception as e:
            return [{"error": f"Anomaly detection failed: {str(e)}"}]
    
    def _determine_price_context(self, data: pd.DataFrame, index: int) -> str:
        """Determine price context during volume spike"""
        try:
            if index == 0:
                return "insufficient_data"
            
            current_price = data['close'].iloc[index]
            prev_price = data['close'].iloc[index-1]
            price_change_pct = ((current_price - prev_price) / prev_price) * 100
            
            # Look at high/low for intraday context
            current_high = data['high'].iloc[index]
            current_low = data['low'].iloc[index]
            prev_close = prev_price
            
            # Determine breakout context
            if current_price > prev_close * 1.02:  # 2%+ gain
                if current_high > data['high'].iloc[max(0, index-5):index].max():
                    return "breakout_up"
                else:
                    return "price_surge"
            elif current_price < prev_close * 0.98:  # 2%+ loss
                if current_low < data['low'].iloc[max(0, index-5):index].min():
                    return "breakout_down"
                else:
                    return "price_decline"
            elif abs(price_change_pct) < 1.0:
                return "consolidation"
            else:
                return "normal_move"
                
        except Exception:
            return "unknown"
    
    def _classify_anomaly_type(self, z_score: float, volume_ratio: float, price_context: str) -> str:
        """Classify the type of statistical anomaly"""
        try:
            if abs(z_score) >= 4.0:
                return "extreme_outlier"
            elif abs(z_score) >= 3.0:
                return "significant_outlier"
            elif abs(z_score) >= 2.0:
                return "moderate_outlier"
            elif z_score > 0 and 'breakout' in price_context:
                return "breakout_spike"
            elif z_score > 0 and price_context == 'consolidation':
                return "quiet_accumulation"
            elif z_score < 0:
                return "volume_drought"
            else:
                return "irregular_pattern"
        except Exception:
            return "unknown"
    
    def _determine_retail_cause(self, abs_z_score: float, price_context: str, volume_ratio: float) -> str:
        """Determine likely non-institutional cause of volume anomaly"""
        try:
            # Focus on retail/market-driven causes, not institutional
            if abs_z_score >= 4.0:
                return "major_market_event"
            elif abs_z_score >= 3.0 and 'breakout' in price_context:
                return "technical_breakout"
            elif abs_z_score >= 3.0:
                return "news_reaction"
            elif 'breakout' in price_context:
                return "momentum_trading"
            elif price_context == 'consolidation':
                return "accumulation_interest"
            elif volume_ratio > 2.0:
                return "market_interest"
            else:
                return "trading_activity"
        except Exception:
            return "unknown"
    
    def _assess_sustainability(self, data: pd.DataFrame, index: int, volume_ratio: float) -> str:
        """Assess if volume spike is sustained over multiple periods"""
        try:
            # Look at next few days if available
            end_idx = min(index + 3, len(data))
            future_volumes = data['volume'].iloc[index:end_idx]
            
            if len(future_volumes) < 2:
                return "unclear"
            
            # Check if volume remains elevated
            current_vol = future_volumes.iloc[0]
            avg_future = future_volumes.iloc[1:].mean() if len(future_volumes) > 1 else current_vol
            
            # If future volume is still above 1.2x recent average, consider sustained
            if avg_future > current_vol * 0.6:  # At least 60% of spike volume
                return "sustained"
            elif volume_ratio > 3.0:  # Very high spikes are often temporary
                return "temporary"
            else:
                return "unclear"
                
        except Exception:
            return "unclear"
    
    
    def analyze_anomaly_patterns(self, anomalies: List[Dict], data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze patterns in detected anomalies
        
        Args:
            anomalies: List of detected anomalies
            data: Original OHLCV data
            
        Returns:
            Dict containing pattern analysis
        """
        if not anomalies:
            return {
                'anomaly_frequency': 'none',
                'anomaly_pattern': 'none',
                'dominant_causes': [],
                'temporal_clustering': 'none'
            }
        
        try:
            # Frequency analysis
            total_days = len(data)
            anomaly_count = len(anomalies)
            frequency_ratio = anomaly_count / total_days if total_days > 0 else 0
            
            if frequency_ratio > 0.1:  # More than 10% of days have anomalies
                frequency = 'high'
            elif frequency_ratio > 0.05:  # 5-10%
                frequency = 'medium'
            else:
                frequency = 'low'
            
            # Pattern analysis
            significance_counts = {}
            cause_counts = {}
            context_counts = {}
            
            for anomaly in anomalies:
                # Count significance levels
                sig = anomaly.get('significance', 'unknown')
                significance_counts[sig] = significance_counts.get(sig, 0) + 1
                
                # Count causes
                cause = anomaly.get('likely_cause', 'unknown')
                cause_counts[cause] = cause_counts.get(cause, 0) + 1
                
                # Count contexts
                context = anomaly.get('price_context', 'unknown')
                context_counts[context] = context_counts.get(context, 0) + 1
            
            # Determine dominant pattern
            max_sig = max(significance_counts.items(), key=lambda x: x[1]) if significance_counts else ('none', 0)
            if max_sig[1] >= len(anomalies) * 0.6:  # 60% or more are same significance
                pattern = f"dominated_by_{max_sig[0]}_significance"
            else:
                pattern = "mixed_significance"
            
            # Top causes
            sorted_causes = sorted(cause_counts.items(), key=lambda x: x[1], reverse=True)
            dominant_causes = [cause for cause, count in sorted_causes[:3]]
            
            # Temporal clustering analysis
            anomaly_dates = [pd.to_datetime(a['date']) for a in anomalies if 'date' in a]
            clustering = self._analyze_temporal_clustering(anomaly_dates)
            
            return {
                'anomaly_frequency': frequency,
                'anomaly_pattern': pattern,
                'frequency_ratio': round(frequency_ratio, 3),
                'significance_distribution': significance_counts,
                'dominant_causes': dominant_causes,
                'context_distribution': context_counts,
                'temporal_clustering': clustering,
                'total_anomalies': len(anomalies),
                'analysis_period_days': total_days
            }
            
        except Exception as e:
            return {
                'error': f"Pattern analysis failed: {str(e)}",
                'anomaly_frequency': 'unknown',
                'anomaly_pattern': 'unknown'
            }
    
    def _analyze_temporal_clustering(self, dates: List[datetime]) -> str:
        """Analyze if anomalies cluster in time"""
        if len(dates) < 2:
            return "insufficient_data"
        
        try:
            # Sort dates
            sorted_dates = sorted(dates)
            
            # Calculate gaps between anomalies
            gaps = [(sorted_dates[i+1] - sorted_dates[i]).days for i in range(len(sorted_dates)-1)]
            
            if not gaps:
                return "single_event"
            
            avg_gap = sum(gaps) / len(gaps)
            
            # Determine clustering
            if avg_gap <= 3:
                return "highly_clustered"
            elif avg_gap <= 7:
                return "weekly_pattern"
            elif avg_gap <= 30:
                return "monthly_pattern"
            else:
                return "scattered"
                
        except Exception:
            return "unknown"
    
    def assess_current_volume_status(self, data: pd.DataFrame, volume_stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess current volume status relative to historical patterns
        
        Args:
            data: DataFrame with OHLCV data
            volume_stats: Volume statistics from calculate_volume_statistics
            
        Returns:
            Dict containing current volume assessment
        """
        try:
            current_volume = volume_stats['current_volume']
            volume_mean = volume_stats['volume_mean']
            percentiles = volume_stats['percentiles']
            current_z_score = volume_stats['current_z_score']
            
            # Determine current status
            if current_volume >= percentiles['percentile_99']:
                status = "extremely_elevated"
                percentile = 99
            elif current_volume >= percentiles['percentile_95']:
                status = "highly_elevated"
                percentile = 95
            elif current_volume >= percentiles['percentile_90']:
                status = "elevated"
                percentile = 90
            elif current_volume >= percentiles['percentile_75']:
                status = "above_average"
                percentile = 75
            elif current_volume >= percentiles['percentile_50']:
                status = "normal"
                percentile = 50
            else:
                status = "below_average"
                percentile = 25  # Approximate
            
            # Recent trend analysis
            recent_volumes = data['volume'].tail(5)
            trend = "increasing" if recent_volumes.iloc[-1] > recent_volumes.iloc[0] else "decreasing"
            
            return {
                'current_status': status,
                'volume_percentile': percentile,
                'z_score': round(current_z_score, 2),
                'vs_mean_ratio': round(current_volume / volume_mean, 2),
                'recent_trend': trend,
                'status_description': self._get_status_description(status, percentile)
            }
            
        except Exception as e:
            return {'error': f"Current status assessment failed: {str(e)}"}
    
    def _get_status_description(self, status: str, percentile: int) -> str:
        """Get human-readable description of volume status"""
        descriptions = {
            "extremely_elevated": f"Current volume is in the top 1% of all observations (>{percentile}th percentile)",
            "highly_elevated": f"Current volume is in the top 5% of all observations (>{percentile}th percentile)", 
            "elevated": f"Current volume is above 90% of typical levels (>{percentile}th percentile)",
            "above_average": f"Current volume is above median levels (>{percentile}th percentile)",
            "normal": "Current volume is within normal range (around median)",
            "below_average": "Current volume is below typical levels"
        }
        return descriptions.get(status, "Volume status unclear")
    
    def process_volume_anomaly_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Main processing function for Volume Anomaly Detection Agent
        
        Args:
            data: DataFrame with OHLCV data (indexed by date)
            
        Returns:
            Dict containing all volume anomaly analysis data
        """
        if len(data) < self.min_data_points:
            return {
                "error": f"Insufficient data: need at least {self.min_data_points} data points, got {len(data)}",
                "data_length": len(data)
            }
        
        try:
            # Core analysis components
            volume_statistics = self.calculate_volume_statistics(data)
            if 'error' in volume_statistics:
                return volume_statistics
            
            detected_anomalies = self.detect_volume_anomalies(data)
            anomaly_patterns = self.analyze_anomaly_patterns(detected_anomalies, data)
            current_status = self.assess_current_volume_status(data, volume_statistics)
            
            # Quality assessment
            quality_score = self._assess_analysis_quality(detected_anomalies, volume_statistics)
            
            return {
                "analysis_timestamp": datetime.now().isoformat(),
                "data_period": f"{len(data)} days",
                "data_range": f"{data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}",
                "volume_statistics": volume_statistics,
                "significant_anomalies": detected_anomalies,
                "anomaly_patterns": anomaly_patterns,
                "current_volume_status": current_status,
                "quality_assessment": quality_score,
                "data_quality": "excellent" if len(data) > 90 else "good" if len(data) > 60 else "adequate"
            }
            
        except Exception as e:
            return {
                "error": f"Volume anomaly processing failed: {str(e)}",
                "data_length": len(data)
            }
    
    def _assess_analysis_quality(self, anomalies: List[Dict], volume_stats: Dict) -> Dict[str, Any]:
        """Assess the quality of anomaly detection analysis"""
        try:
            score = 0
            
            # Data quality (30 points)
            if 'error' not in volume_stats:
                score += 15
                cv = volume_stats.get('volume_cv', 0)
                if 0.3 < cv < 3.0:  # Reasonable volatility
                    score += 15
            
            # Anomaly detection quality (40 points)
            if anomalies and not any('error' in a for a in anomalies):
                score += 20
                
                # Quality based on anomaly characteristics
                high_sig_count = len([a for a in anomalies if a.get('significance') == 'high'])
                medium_sig_count = len([a for a in anomalies if a.get('significance') == 'medium'])
                
                if high_sig_count > 0:
                    score += 20
                elif medium_sig_count > 0:
                    score += 15
                else:
                    score += 10
            
            # Analysis completeness (30 points)
            complete_anomalies = [a for a in anomalies if all(k in a for k in 
                                ['date', 'significance', 'likely_cause', 'price_context'])]
            completeness_ratio = len(complete_anomalies) / max(1, len(anomalies))
            score += int(30 * completeness_ratio)
            
            return {
                'overall_score': min(100, score),
                'data_quality_score': 30 if 'error' not in volume_stats else 0,
                'detection_quality_score': min(40, score - 30) if score > 30 else 0,
                'completeness_score': int(30 * completeness_ratio) if anomalies else 0,
                'anomaly_count': len(anomalies),
                'high_significance_count': len([a for a in anomalies if a.get('significance') == 'high'])
            }
            
        except Exception:
            return {'overall_score': 0, 'error': 'Quality assessment failed'}


def test_volume_anomaly_processor():
    """Test function for Volume Anomaly Processor"""
    print("🔍 Testing Volume Anomaly Processor")
    print("=" * 50)
    
    # Create sample data with intentional volume spikes
    dates = pd.date_range(start='2024-06-01', end='2024-09-20', freq='D')
    np.random.seed(42)
    
    # Generate realistic price and volume data
    base_price = 2500
    base_volume = 1500000
    
    # Create price series
    price_changes = np.random.normal(0.001, 0.02, len(dates))
    prices = base_price * np.cumprod(1 + price_changes)
    
    # Create volume series with spikes
    volumes = np.random.lognormal(np.log(base_volume), 0.5, len(dates))
    
    # Add intentional volume spikes
    spike_dates = [20, 45, 80, 100]  # Days with volume spikes
    for spike_day in spike_dates:
        if spike_day < len(volumes):
            volumes[spike_day] *= np.random.uniform(3.0, 6.0)  # 3-6x volume spike
    
    sample_data = pd.DataFrame({
        'open': prices,
        'high': prices * np.random.uniform(1.001, 1.02, len(dates)),
        'low': prices * np.random.uniform(0.98, 0.999, len(dates)),
        'close': prices,
        'volume': volumes.astype(int)
    }, index=dates)
    
    print(f"✅ Created sample data: {len(sample_data)} days")
    print(f"   Price range: ₹{sample_data['close'].min():.2f} - ₹{sample_data['close'].max():.2f}")
    print(f"   Volume range: {sample_data['volume'].min():,} - {sample_data['volume'].max():,}")
    
    # Test processor
    processor = VolumeAnomalyProcessor()
    result = processor.process_volume_anomaly_data(sample_data)
    
    if 'error' in result:
        print(f"❌ Processing failed: {result['error']}")
        return False
    
    print(f"\n📊 Volume Anomaly Analysis Results:")
    print(f"   Data period: {result['data_period']}")
    print(f"   Data quality: {result['data_quality']}")
    
    # Volume statistics
    vol_stats = result['volume_statistics']
    print(f"   Volume mean: {vol_stats['volume_mean']:,.0f}")
    print(f"   Volume std: {vol_stats['volume_std']:,.0f}")
    print(f"   Current Z-score: {vol_stats['current_z_score']:.2f}")
    
    # Anomalies detected
    anomalies = result['significant_anomalies']
    print(f"   Anomalies detected: {len(anomalies)}")
    
    if anomalies:
        for i, anomaly in enumerate(anomalies[:3]):  # Show first 3
            if 'error' not in anomaly:
                print(f"     {i+1}. {anomaly['date']}: {anomaly['volume_ratio']:.1f}x volume ({anomaly['significance']})")
    
    # Pattern analysis
    patterns = result['anomaly_patterns']
    print(f"   Anomaly frequency: {patterns.get('anomaly_frequency', 'unknown')}")
    print(f"   Pattern type: {patterns.get('anomaly_pattern', 'unknown')}")
    
    # Current status
    current = result['current_volume_status']
    print(f"   Current volume status: {current.get('current_status', 'unknown')}")
    print(f"   Volume percentile: {current.get('volume_percentile', 0)}th")
    
    # Quality assessment
    quality = result['quality_assessment']
    print(f"   Overall quality score: {quality.get('overall_score', 0)}/100")
    
    print("\n✅ Volume Anomaly Processor test completed successfully!")
    return True

if __name__ == "__main__":
    test_volume_anomaly_processor()