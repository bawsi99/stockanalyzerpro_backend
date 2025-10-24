#!/usr/bin/env python3
"""
Institutional Activity Agent - Data Processing Module

Detects smart money patterns through volume profile analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

class InstitutionalActivityProcessor:
    def __init__(self):
        self.volume_profile_bins = 50  # Price level granularity
        self.large_block_threshold = 2.0  # 2x average volume
        self.institutional_threshold = 3.0  # 3x for institutional classification
    
    def process_institutional_activity_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Main processing function for institutional activity analysis"""
        try:
            # 1. Calculate volume profile
            volume_profile = self._calculate_volume_profile(data)
            
            # 2. Detect large block transactions
            large_blocks = self._detect_large_blocks(data)
            
            # 3. Analyze accumulation/distribution
            accum_dist = self._analyze_accumulation_distribution(data)
            
            # 4. Smart money timing analysis
            smart_money = self._analyze_smart_money_timing(data, large_blocks)
            
            # 5. Calculate predictive indicators
            predictive = self._calculate_predictive_indicators(data, accum_dist)
            
            # 6. Quality assessment
            quality = self._assess_analysis_quality(volume_profile, large_blocks)
            
            return {
                'volume_profile': volume_profile,
                'large_block_analysis': large_blocks,
                'accumulation_distribution': accum_dist,
                'smart_money_timing': smart_money,
                'predictive_indicators': predictive,
                'quality_assessment': quality,
                'institutional_activity_level': self._determine_activity_level(large_blocks),
                'primary_activity': self._determine_primary_activity(accum_dist)
            }
            
        except Exception as e:
            return {'error': f"Institutional activity analysis failed: {str(e)}"}
    
    def _calculate_volume_profile(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate volume distribution at different price levels"""
        try:
            # Get price range
            price_min = data['low'].min()
            price_max = data['high'].max()
            price_range = price_max - price_min
            
            # Create price bins
            price_bins = np.linspace(price_min, price_max, self.volume_profile_bins)
            bin_width = price_range / self.volume_profile_bins
            
            # Calculate volume at each price level
            volume_at_price = []
            
            for i in range(len(price_bins) - 1):
                bin_low = price_bins[i]
                bin_high = price_bins[i + 1]
                bin_center = (bin_low + bin_high) / 2
                
                # Find days where price range overlaps this bin
                overlapping_days = data[
                    (data['low'] <= bin_high) & (data['high'] >= bin_low)
                ]
                
                # Calculate volume proportion for this price level
                total_volume = 0
                for _, day in overlapping_days.iterrows():
                    # Simple approximation: distribute volume evenly across day's price range
                    day_range = day['high'] - day['low']
                    if day_range > 0:
                        overlap_low = max(bin_low, day['low'])
                        overlap_high = min(bin_high, day['high'])
                        overlap_range = max(0, overlap_high - overlap_low)
                        volume_proportion = overlap_range / day_range
                        total_volume += day['volume'] * volume_proportion
                    else:
                        # Single price point
                        if bin_low <= day['close'] <= bin_high:
                            total_volume += day['volume']
                
                volume_at_price.append({
                    'price_level': bin_center,
                    'volume': total_volume,
                    'price_range': [bin_low, bin_high]
                })
            
            # Find key levels
            volume_at_price.sort(key=lambda x: x['volume'], reverse=True)
            
            # Point of Control (POC) - highest volume price level
            poc = volume_at_price[0] if volume_at_price else {'price_level': 0, 'volume': 0}
            
            # Value Area (70% of volume)
            total_volume = sum(level['volume'] for level in volume_at_price)
            value_area_volume = total_volume * 0.7
            
            # Find value area high and low
            sorted_by_price = sorted(volume_at_price, key=lambda x: x['price_level'])
            cumulative_volume = 0
            value_area_levels = []
            
            # Start from POC and expand outward
            poc_index = next(i for i, level in enumerate(sorted_by_price) 
                           if level['price_level'] == poc['price_level'])
            
            value_area_levels.append(sorted_by_price[poc_index])
            cumulative_volume += sorted_by_price[poc_index]['volume']
            
            # Expand outward from POC
            low_index, high_index = poc_index - 1, poc_index + 1
            
            while cumulative_volume < value_area_volume and (low_index >= 0 or high_index < len(sorted_by_price)):
                low_vol = sorted_by_price[low_index]['volume'] if low_index >= 0 else 0
                high_vol = sorted_by_price[high_index]['volume'] if high_index < len(sorted_by_price) else 0
                
                if low_vol >= high_vol and low_index >= 0:
                    value_area_levels.append(sorted_by_price[low_index])
                    cumulative_volume += low_vol
                    low_index -= 1
                elif high_index < len(sorted_by_price):
                    value_area_levels.append(sorted_by_price[high_index])
                    cumulative_volume += high_vol
                    high_index += 1
                else:
                    break
            
            value_area_high = max(level['price_level'] for level in value_area_levels)
            value_area_low = min(level['price_level'] for level in value_area_levels)
            
            return {
                'volume_at_price': volume_at_price,
                'point_of_control': poc,
                'value_area_high': value_area_high,
                'value_area_low': value_area_low,
                'total_volume': total_volume,
                'price_range': [price_min, price_max]
            }
            
        except Exception as e:
            return {'error': f"Volume profile calculation failed: {str(e)}"}
    
    def _detect_large_blocks(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect large block transactions indicating institutional activity"""
        try:
            # Calculate volume statistics
            volume_mean = data['volume'].mean()
            volume_std = data['volume'].std()
            volume_median = data['volume'].median()
            
            # Define thresholds
            large_block_threshold = volume_mean * self.large_block_threshold
            institutional_threshold = volume_mean * self.institutional_threshold
            
            # Identify large blocks
            large_blocks = []
            institutional_blocks = []
            
            for date, row in data.iterrows():
                volume = row['volume']
                
                if volume > institutional_threshold:
                    block_info = {
                        'date': date.strftime('%Y-%m-%d'),
                        'volume': int(volume),
                        'volume_ratio': volume / volume_mean,
                        'price': row['close'],
                        'classification': 'institutional',
                        'size_category': self._classify_block_size(volume, volume_mean)
                    }
                    institutional_blocks.append(block_info)
                    large_blocks.append(block_info)
                    
                elif volume > large_block_threshold:
                    block_info = {
                        'date': date.strftime('%Y-%m-%d'),
                        'volume': int(volume),
                        'volume_ratio': volume / volume_mean,
                        'price': row['close'],
                        'classification': 'large_block',
                        'size_category': self._classify_block_size(volume, volume_mean)
                    }
                    large_blocks.append(block_info)
            
            # Analyze patterns
            activity_level = self._assess_institutional_activity_level(institutional_blocks, len(data))
            
            return {
                'large_blocks': large_blocks,
                'institutional_blocks': institutional_blocks,
                'activity_level': activity_level,
                'total_large_blocks': len(large_blocks),
                'institutional_block_count': len(institutional_blocks),
                'thresholds': {
                    'large_block': large_block_threshold,
                    'institutional': institutional_threshold
                },
                'volume_statistics': {
                    'mean': volume_mean,
                    'median': volume_median,
                    'std': volume_std
                }
            }
            
        except Exception as e:
            return {'error': f"Large block detection failed: {str(e)}"}
    
    def _analyze_accumulation_distribution(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze accumulation and distribution patterns"""
        try:
            # Calculate Accumulation/Distribution Line (A/D Line)
            ad_line = []
            cumulative_ad = 0
            
            for _, row in data.iterrows():
                # Money Flow Multiplier
                if row['high'] != row['low']:
                    mf_multiplier = ((row['close'] - row['low']) - (row['high'] - row['close'])) / (row['high'] - row['low'])
                else:
                    mf_multiplier = 0
                
                # Money Flow Volume
                mf_volume = mf_multiplier * row['volume']
                cumulative_ad += mf_volume
                ad_line.append(cumulative_ad)
            
            data = data.copy()
            data['ad_line'] = ad_line
            
            # Calculate A/D Line trend
            recent_ad = ad_line[-20:]  # Last 20 days
            if len(recent_ad) > 5:
                ad_trend = np.polyfit(range(len(recent_ad)), recent_ad, 1)[0]
            else:
                ad_trend = 0
            
            # Determine accumulation vs distribution
            if ad_trend > 0:
                primary_pattern = 'accumulation'
                pattern_strength = min(abs(ad_trend) / (data['volume'].mean() * 1000), 1.0)
            elif ad_trend < 0:
                primary_pattern = 'distribution'
                pattern_strength = min(abs(ad_trend) / (data['volume'].mean() * 1000), 1.0)
            else:
                primary_pattern = 'neutral'
                pattern_strength = 0
            
            # Calculate buying/selling pressure
            buying_pressure = len([x for x in recent_ad if x > 0]) / len(recent_ad) if recent_ad else 0.5
            selling_pressure = 1 - buying_pressure
            
            # Institutional flow estimation
            institutional_flow = self._estimate_institutional_flow(data)
            
            return {
                'ad_line': ad_line,
                'ad_trend': ad_trend,
                'primary_pattern': primary_pattern,
                'pattern_strength': pattern_strength,
                'buying_pressure': buying_pressure,
                'selling_pressure': selling_pressure,
                'institutional_flow': institutional_flow,
                'current_ad_level': ad_line[-1] if ad_line else 0
            }
            
        except Exception as e:
            return {'error': f"Accumulation/distribution analysis failed: {str(e)}"}
    
    def _analyze_smart_money_timing(self, data: pd.DataFrame, large_blocks: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze smart money timing patterns"""
        try:
            if 'error' in large_blocks:
                return {'error': 'Large block data unavailable'}
            
            institutional_blocks = large_blocks.get('institutional_blocks', [])
            
            if not institutional_blocks:
                return {
                    'timing_pattern': 'no_institutional_activity',
                    'entry_timing': 'unknown',
                    'activity_clusters': [],
                    'timing_quality': 'poor'
                }
            
            # Analyze timing relative to price movements
            timing_analysis = []
            
            for block in institutional_blocks:
                block_date = pd.to_datetime(block['date'])
                
                # Find price context around this date
                block_data = data[data.index == block_date]
                if not block_data.empty:
                    # Look at price movement before and after
                    idx = data.index.get_loc(block_date)
                    
                    # Price context
                    if idx > 5:
                        price_before = data.iloc[idx-5:idx]['close'].mean()
                    else:
                        price_before = block['price']
                    
                    if idx < len(data) - 5:
                        price_after = data.iloc[idx+1:idx+6]['close'].mean()
                    else:
                        price_after = block['price']
                    
                    # Determine timing quality
                    if price_after > block['price'] > price_before:
                        timing_type = 'accumulation_on_dip'
                    elif price_after > block['price']:
                        timing_type = 'early_accumulation'
                    elif block['price'] > price_before:
                        timing_type = 'breakout_accumulation'
                    else:
                        timing_type = 'distribution'
                    
                    timing_analysis.append({
                        'date': block['date'],
                        'timing_type': timing_type,
                        'volume_ratio': block['volume_ratio'],
                        'price_context': {
                            'before': price_before,
                            'during': block['price'],
                            'after': price_after
                        }
                    })
            
            # Determine overall timing pattern
            timing_types = [t['timing_type'] for t in timing_analysis]
            if not timing_types:
                primary_timing = 'unknown'
            else:
                primary_timing = max(set(timing_types), key=timing_types.count)
            
            # Activity clusters
            dates = [pd.to_datetime(block['date']) for block in institutional_blocks]
            clusters = self._find_activity_clusters(dates)
            
            return {
                'timing_pattern': primary_timing,
                'timing_analysis': timing_analysis,
                'activity_clusters': clusters,
                'timing_quality': self._assess_timing_quality(timing_analysis)
            }
            
        except Exception as e:
            return {'error': f"Smart money timing analysis failed: {str(e)}"}
    
    def _calculate_predictive_indicators(self, data: pd.DataFrame, accum_dist: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate predictive indicators for future price movement"""
        try:
            if 'error' in accum_dist:
                return {'error': 'Accumulation/distribution data unavailable'}
            
            # Volume-price trend correlation
            recent_data = data.tail(20)
            if len(recent_data) < 10:
                return {'error': 'Insufficient data for prediction'}
            
            price_trend = np.polyfit(range(len(recent_data)), recent_data['close'], 1)[0]
            volume_trend = np.polyfit(range(len(recent_data)), recent_data['volume'], 1)[0]
            
            # Normalize trends
            price_trend_norm = price_trend / recent_data['close'].mean()
            volume_trend_norm = volume_trend / recent_data['volume'].mean()
            
            # Correlation strength
            correlation = np.corrcoef(recent_data['close'], recent_data['volume'])[0, 1]
            
            # Predictive signals
            signals = []
            
            # Volume leading price
            if volume_trend_norm > 0.02 and price_trend_norm > 0:
                signals.append('volume_confirms_uptrend')
            elif volume_trend_norm < -0.02 and price_trend_norm < 0:
                signals.append('volume_confirms_downtrend')
            elif volume_trend_norm > 0.02 and price_trend_norm < 0:
                signals.append('volume_suggests_reversal_up')
            elif volume_trend_norm < -0.02 and price_trend_norm > 0:
                signals.append('volume_warns_reversal_down')
            
            # A/D Line signals
            ad_trend = accum_dist.get('ad_trend', 0)
            if ad_trend > 0 and accum_dist.get('pattern_strength', 0) > 0.5:
                signals.append('institutional_accumulation')
            elif ad_trend < 0 and accum_dist.get('pattern_strength', 0) > 0.5:
                signals.append('institutional_distribution')
            
            # Overall prediction
            bullish_signals = len([s for s in signals if 'up' in s or 'accumulation' in s or 'confirms_uptrend' in s])
            bearish_signals = len([s for s in signals if 'down' in s or 'distribution' in s or 'confirms_downtrend' in s])
            
            if bullish_signals > bearish_signals:
                prediction = 'bullish'
                confidence = min(bullish_signals / max(len(signals), 1), 1.0)
            elif bearish_signals > bullish_signals:
                prediction = 'bearish'
                confidence = min(bearish_signals / max(len(signals), 1), 1.0)
            else:
                prediction = 'neutral'
                confidence = 0.5
            
            return {
                'prediction': prediction,
                'confidence': confidence,
                'signals': signals,
                'correlations': {
                    'price_volume_correlation': correlation,
                    'price_trend': price_trend_norm,
                    'volume_trend': volume_trend_norm
                },
                'predictive_strength': self._calculate_predictive_strength(signals, correlation)
            }
            
        except Exception as e:
            return {'error': f"Predictive indicator calculation failed: {str(e)}"}
    
    # Helper methods
    def _classify_block_size(self, volume: float, mean_volume: float) -> str:
        """Classify block size relative to average"""
        ratio = volume / mean_volume
        if ratio > 5:
            return 'very_large'
        elif ratio > 3:
            return 'large'
        elif ratio > 2:
            return 'medium'
        else:
            return 'small'
    
    def _assess_institutional_activity_level(self, institutional_blocks: List[Dict], total_days: int) -> str:
        """Assess overall institutional activity level"""
        if not institutional_blocks:
            return 'very_low'
        
        frequency = len(institutional_blocks) / total_days
        
        if frequency > 0.2:
            return 'very_high'
        elif frequency > 0.1:
            return 'high'
        elif frequency > 0.05:
            return 'medium'
        elif frequency > 0.02:
            return 'low'
        else:
            return 'very_low'
    
    def _estimate_institutional_flow(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Estimate net institutional money flow"""
        recent_data = data.tail(10)
        
        total_flow = 0
        for _, row in recent_data.iterrows():
            # Simple flow estimation based on close relative to range
            range_position = (row['close'] - row['low']) / (row['high'] - row['low']) if row['high'] != row['low'] else 0.5
            flow = (range_position - 0.5) * row['volume']
            total_flow += flow
        
        avg_volume = recent_data['volume'].mean()
        flow_ratio = total_flow / (avg_volume * len(recent_data)) if avg_volume > 0 else 0
        
        return {
            'net_flow': total_flow,
            'flow_ratio': flow_ratio,
            'flow_direction': 'inflow' if total_flow > 0 else 'outflow'
        }
    
    def _find_activity_clusters(self, dates: List[pd.Timestamp]) -> List[Dict[str, Any]]:
        """Find clusters of institutional activity"""
        if len(dates) < 2:
            return []
        
        dates = sorted(dates)
        clusters = []
        current_cluster = [dates[0]]
        
        for i in range(1, len(dates)):
            if (dates[i] - dates[i-1]).days <= 5:  # Within 5 days
                current_cluster.append(dates[i])
            else:
                if len(current_cluster) > 1:
                    clusters.append({
                        'start_date': current_cluster[0].strftime('%Y-%m-%d'),
                        'end_date': current_cluster[-1].strftime('%Y-%m-%d'),
                        'activity_count': len(current_cluster),
                        'duration_days': (current_cluster[-1] - current_cluster[0]).days
                    })
                current_cluster = [dates[i]]
        
        # Add final cluster
        if len(current_cluster) > 1:
            clusters.append({
                'start_date': current_cluster[0].strftime('%Y-%m-%d'),
                'end_date': current_cluster[-1].strftime('%Y-%m-%d'),
                'activity_count': len(current_cluster),
                'duration_days': (current_cluster[-1] - current_cluster[0]).days
            })
        
        return clusters
    
    def _assess_timing_quality(self, timing_analysis: List[Dict]) -> str:
        """Assess quality of institutional timing"""
        if not timing_analysis:
            return 'poor'
        
        good_timing = len([t for t in timing_analysis if t['timing_type'] in 
                          ['accumulation_on_dip', 'early_accumulation']])
        total_timing = len(timing_analysis)
        
        quality_ratio = good_timing / total_timing
        
        if quality_ratio > 0.7:
            return 'excellent'
        elif quality_ratio > 0.5:
            return 'good'
        elif quality_ratio > 0.3:
            return 'fair'
        else:
            return 'poor'
    
    def _calculate_predictive_strength(self, signals: List[str], correlation: float) -> str:
        """Calculate overall predictive strength"""
        signal_strength = len(signals) * 0.2
        correlation_strength = abs(correlation) * 0.8
        
        total_strength = signal_strength + correlation_strength
        
        if total_strength > 0.8:
            return 'very_high'
        elif total_strength > 0.6:
            return 'high'
        elif total_strength > 0.4:
            return 'medium'
        elif total_strength > 0.2:
            return 'low'
        else:
            return 'very_low'
    
    def _assess_analysis_quality(self, volume_profile: Dict, large_blocks: Dict) -> Dict[str, Any]:
        """Assess overall analysis quality"""
        try:
            quality_score = 0
            factors = []
            
            # Volume profile quality
            if 'error' not in volume_profile:
                quality_score += 30
                factors.append('volume_profile_valid')
            
            # Large block detection quality
            if 'error' not in large_blocks:
                quality_score += 30
                block_count = large_blocks.get('total_large_blocks', 0)
                if block_count > 0:
                    quality_score += min(block_count * 2, 20)
                    factors.append('large_blocks_detected')
            
            # Data sufficiency
            total_volume = volume_profile.get('total_volume', 0)
            if total_volume > 0:
                quality_score += 20
                factors.append('sufficient_volume_data')
            
            return {
                'overall_score': min(quality_score, 100),
                'quality_factors': factors,
                'analysis_reliability': self._determine_reliability(quality_score)
            }
            
        except Exception as e:
            return {
                'overall_score': 0,
                'error': f"Quality assessment failed: {str(e)}"
            }
    
    def _determine_reliability(self, score: int) -> str:
        """Determine analysis reliability based on score"""
        if score >= 80:
            return 'high'
        elif score >= 60:
            return 'medium'
        elif score >= 40:
            return 'low'
        else:
            return 'very_low'
    
    def _determine_activity_level(self, large_blocks: Dict) -> str:
        """Determine overall institutional activity level"""
        if 'error' in large_blocks:
            return 'unknown'
        
        return large_blocks.get('activity_level', 'unknown')
    
    def _determine_primary_activity(self, accum_dist: Dict) -> str:
        """Determine primary institutional activity type"""
        if 'error' in accum_dist:
            return 'unknown'
        
        return accum_dist.get('primary_pattern', 'unknown')


def test_institutional_activity_processor():
    """Test function for Institutional Activity Processor"""
    print("🏛️ Testing Institutional Activity Processor")
    print("=" * 60)
    
    # Create test data with institutional activity patterns
    dates = pd.date_range(start='2024-07-01', end='2024-10-20', freq='D')
    np.random.seed(42)
    
    base_price = 2400
    base_volume = 1500000
    
    # Generate price data with some institutional accumulation periods
    price_changes = np.random.normal(0.001, 0.015, len(dates))
    
    # Add accumulation phases
    accumulation_periods = [(20, 30), (70, 80)]
    for start, end in accumulation_periods:
        if end < len(price_changes):
            price_changes[start:end] = np.random.normal(0.003, 0.008, end-start)  # Gentle uptrend
    
    prices = base_price * np.cumprod(1 + price_changes)
    
    # Generate volume with institutional blocks
    volumes = np.random.lognormal(np.log(base_volume), 0.4, len(dates))
    
    # Add institutional blocks during accumulation
    for start, end in accumulation_periods:
        for i in range(start, min(end, len(volumes))):
            if np.random.random() > 0.7:  # 30% chance of institutional block
                volumes[i] *= np.random.uniform(3.0, 6.0)
    
    # Add some random large blocks
    random_blocks = np.random.choice(len(dates), 5, replace=False)
    for idx in random_blocks:
        volumes[idx] *= np.random.uniform(2.5, 4.0)
    
    test_data = pd.DataFrame({
        'open': prices + np.random.normal(0, 2, len(dates)),
        'high': prices + np.abs(np.random.normal(6, 3, len(dates))),
        'low': prices - np.abs(np.random.normal(6, 3, len(dates))),
        'close': prices,
        'volume': volumes.astype(int)
    }, index=dates)
    
    # Ensure realistic OHLC relationships
    test_data['high'] = np.maximum(test_data[['open', 'close']].max(axis=1), test_data['high'])
    test_data['low'] = np.minimum(test_data[['open', 'close']].min(axis=1), test_data['low'])
    
    print(f"✅ Created test data: {len(test_data)} days")
    print(f"   Price range: ${test_data['close'].min():.2f} - ${test_data['close'].max():.2f}")
    print(f"   Volume range: {test_data['volume'].min():,} - {test_data['volume'].max():,}")
    
    # Process data
    processor = InstitutionalActivityProcessor()
    results = processor.process_institutional_activity_data(test_data)
    
    if 'error' in results:
        print(f"❌ Analysis failed: {results['error']}")
        return False
    
    print("✅ Analysis completed successfully")
    
    # Display key results
    large_blocks = results.get('large_block_analysis', {})
    accum_dist = results.get('accumulation_distribution', {})
    predictive = results.get('predictive_indicators', {})
    quality = results.get('quality_assessment', {})
    
    print(f"\n📊 Analysis Results:")
    print(f"   Activity Level: {results.get('institutional_activity_level', 'unknown')}")
    print(f"   Primary Activity: {results.get('primary_activity', 'unknown')}")
    print(f"   Large Blocks: {large_blocks.get('total_large_blocks', 0)}")
    print(f"   Institutional Blocks: {large_blocks.get('institutional_block_count', 0)}")
    print(f"   A/D Pattern: {accum_dist.get('primary_pattern', 'unknown')}")
    print(f"   Prediction: {predictive.get('prediction', 'unknown')}")
    print(f"   Quality Score: {quality.get('overall_score', 0)}/100")
    
    # Display volume profile summary
    volume_profile = results.get('volume_profile', {})
    if 'error' not in volume_profile:
        poc = volume_profile.get('point_of_control', {})
        print(f"\n📈 Volume Profile:")
        print(f"   Point of Control: ${poc.get('price_level', 0):.2f}")
        print(f"   Value Area High: ${volume_profile.get('value_area_high', 0):.2f}")
        print(f"   Value Area Low: ${volume_profile.get('value_area_low', 0):.2f}")
    
    return True

if __name__ == "__main__":
    test_institutional_activity_processor()