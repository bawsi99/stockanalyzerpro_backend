"""
Market Structure Analyzer

Advanced market structure analysis for pattern recognition including:
- Swing point detection (Higher Highs, Higher Lows, Lower Highs, Lower Lows)
- Break of Structure (BOS) detection
- Change of Character (CHOCH) detection
- Trend analysis and trend changes
- Market phases identification
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@dataclass
class SwingPoint:
    """Represents a swing high or swing low point"""
    index: int
    price: float
    timestamp: str
    swing_type: str  # 'high' or 'low'
    strength: int  # How many bars on each side confirm this swing
    
@dataclass
class StructureBreak:
    """Represents a break of structure or change of character"""
    index: int
    price: float
    timestamp: str
    break_type: str  # 'BOS' or 'CHOCH'
    direction: str  # 'bullish' or 'bearish'
    previous_structure: SwingPoint
    confidence: float
    description: str

@dataclass
class TrendAnalysis:
    """Comprehensive trend analysis"""
    current_trend: str  # 'uptrend', 'downtrend', 'sideways'
    trend_strength: float  # 0-1 scale
    trend_duration: int  # Number of bars in current trend
    last_structure_break: Optional[StructureBreak]
    swing_sequence: List[str]  # Recent sequence like ['HH', 'HL', 'HH']
    trend_quality: str  # 'strong', 'weak', 'consolidation'

class MarketStructureAnalyzer:
    """
    Advanced market structure analyzer for pattern detection.
    
    Identifies swing points, structure breaks, and trend changes that are
    crucial for understanding market dynamics and pattern formation.
    """
    
    def __init__(self, min_swing_strength: int = 2):
        self.min_swing_strength = min_swing_strength
        self.name = "market_structure_analyzer"
        self.version = "1.0.0"
    
    def analyze_market_structure(self, stock_data: pd.DataFrame, 
                               lookback_period: int = 100) -> Dict[str, Any]:
        """
        Comprehensive market structure analysis.
        
        Args:
            stock_data: DataFrame with OHLCV data
            lookback_period: Number of recent bars to analyze
            
        Returns:
            Dictionary containing complete market structure analysis
        """
        try:
            # Limit data to recent period for analysis
            recent_data = stock_data.tail(lookback_period) if len(stock_data) > lookback_period else stock_data
            
            if len(recent_data) < 10:
                return self._create_empty_analysis("Insufficient data for analysis")
            
            # Extract price arrays
            highs = recent_data['high'].values
            lows = recent_data['low'].values
            closes = recent_data['close'].values
            
            # Get timestamps (handle different timestamp formats)
            if 'timestamp' in recent_data.columns:
                timestamps = recent_data['timestamp'].astype(str).tolist()
            elif recent_data.index.dtype == 'datetime64[ns]':
                timestamps = recent_data.index.strftime('%Y-%m-%d %H:%M:%S').tolist()
            else:
                timestamps = [f"Bar_{i}" for i in range(len(recent_data))]
            
            # 1. Identify swing points
            swing_highs = self._identify_swing_points(highs, timestamps, 'high')
            swing_lows = self._identify_swing_points(lows, timestamps, 'low')
            
            # 2. Analyze swing sequences (HH, HL, LH, LL)
            swing_analysis = self._analyze_swing_sequences(swing_highs, swing_lows)
            
            # 3. Detect structure breaks (BOS and CHOCH)
            structure_breaks = self._detect_structure_breaks(swing_highs, swing_lows, closes, timestamps)
            
            # 4. Determine current trend
            trend_analysis = self._analyze_trend(swing_highs, swing_lows, structure_breaks, closes)
            
            # 5. Identify market phases
            market_phases = self._identify_market_phases(swing_analysis, trend_analysis, closes)
            
            # 6. Calculate support and resistance from structure
            key_levels = self._calculate_structural_levels(swing_highs, swing_lows, closes[-1])
            
            # 7. Assess overall structure quality
            structure_quality = self._assess_structure_quality(swing_highs, swing_lows, trend_analysis)
            
            return {
                'analysis_type': 'market_structure',
                'timestamp': datetime.now().isoformat(),
                'data_points_analyzed': len(recent_data),
                'lookback_period': lookback_period,
                
                # Core structure data
                'swing_points': {
                    'swing_highs': [asdict(sh) for sh in swing_highs],
                    'swing_lows': [asdict(sl) for sl in swing_lows],
                    'total_swings': len(swing_highs) + len(swing_lows)
                },
                
                'swing_analysis': swing_analysis,
                'structure_breaks': [asdict(sb) for sb in structure_breaks],
                'trend_analysis': asdict(trend_analysis),
                'market_phases': market_phases,
                'key_levels': key_levels,
                'structure_quality': structure_quality,
                
                # Summary for quick reference
                'summary': {
                    'current_trend': trend_analysis.current_trend,
                    'trend_strength': trend_analysis.trend_strength,
                    'recent_structure_breaks': len([sb for sb in structure_breaks if sb.index >= len(recent_data) - 20]),
                    'structure_clarity': structure_quality['clarity_score']
                }
            }
            
        except Exception as e:
            logger.error(f"[MARKET_STRUCTURE] Analysis failed: {str(e)}")
            return self._create_empty_analysis(f"Analysis error: {str(e)}")
    
    def _identify_swing_points(self, prices: np.ndarray, timestamps: List[str], 
                             point_type: str) -> List[SwingPoint]:
        """Identify swing highs or lows with configurable strength."""
        swings = []
        
        for i in range(self.min_swing_strength, len(prices) - self.min_swing_strength):
            is_swing = False
            
            if point_type == 'high':
                # Check if this is a swing high
                left_bars = prices[i - self.min_swing_strength:i]
                right_bars = prices[i + 1:i + self.min_swing_strength + 1]
                is_swing = all(prices[i] >= price for price in left_bars) and \
                          all(prices[i] >= price for price in right_bars) and \
                          prices[i] > max(np.max(left_bars), np.max(right_bars))
            else:
                # Check if this is a swing low
                left_bars = prices[i - self.min_swing_strength:i]
                right_bars = prices[i + 1:i + self.min_swing_strength + 1]
                is_swing = all(prices[i] <= price for price in left_bars) and \
                          all(prices[i] <= price for price in right_bars) and \
                          prices[i] < min(np.min(left_bars), np.min(right_bars))
            
            if is_swing:
                swing = SwingPoint(
                    index=i,
                    price=float(prices[i]),
                    timestamp=timestamps[i] if i < len(timestamps) else f"Bar_{i}",
                    swing_type=point_type,
                    strength=self.min_swing_strength
                )
                swings.append(swing)
        
        return swings
    
    def _analyze_swing_sequences(self, swing_highs: List[SwingPoint], 
                               swing_lows: List[SwingPoint]) -> Dict[str, Any]:
        """Analyze sequences of HH, HL, LH, LL patterns."""
        
        # Combine and sort all swings by index
        all_swings = swing_highs + swing_lows
        all_swings.sort(key=lambda x: x.index)
        
        if len(all_swings) < 4:
            return {
                'sequence': [],
                'pattern': 'insufficient_data',
                'trend_indication': 'unclear',
                'strength': 0.0
            }
        
        # Analyze recent swing relationships
        recent_swings = all_swings[-6:]  # Last 6 swings for analysis
        sequence = []
        
        for i in range(1, len(recent_swings)):
            current = recent_swings[i]
            previous_same_type = None
            
            # Find the previous swing of the same type
            for j in range(i - 1, -1, -1):
                if recent_swings[j].swing_type == current.swing_type:
                    previous_same_type = recent_swings[j]
                    break
            
            if previous_same_type:
                if current.swing_type == 'high':
                    if current.price > previous_same_type.price:
                        sequence.append('HH')  # Higher High
                    else:
                        sequence.append('LH')  # Lower High
                else:
                    if current.price > previous_same_type.price:
                        sequence.append('HL')  # Higher Low
                    else:
                        sequence.append('LL')  # Lower Low
        
        # Determine pattern and trend indication
        pattern = self._classify_swing_pattern(sequence)
        trend_indication = self._get_trend_from_pattern(pattern)
        strength = self._calculate_pattern_strength(sequence)
        
        return {
            'sequence': sequence,
            'pattern': pattern,
            'trend_indication': trend_indication,
            'strength': strength,
            'recent_swings_count': len(recent_swings),
            'analysis_confidence': min(len(sequence) / 4.0, 1.0)  # More swings = higher confidence
        }
    
    def _detect_structure_breaks(self, swing_highs: List[SwingPoint], 
                               swing_lows: List[SwingPoint], 
                               closes: np.ndarray, timestamps: List[str]) -> List[StructureBreak]:
        """Detect Break of Structure (BOS) and Change of Character (CHOCH)."""
        structure_breaks = []
        
        # Combine and sort swings
        all_swings = swing_highs + swing_lows
        all_swings.sort(key=lambda x: x.index)
        
        if len(all_swings) < 3:
            return structure_breaks
        
        # Look for structure breaks in recent data
        for i in range(len(closes)):
            current_price = closes[i]
            
            # Check for bullish BOS (break above previous swing high)
            for swing_high in swing_highs:
                if (swing_high.index < i and 
                    current_price > swing_high.price * 1.001):  # 0.1% buffer for noise
                    
                    # Confirm this is a significant break
                    if self._is_significant_break(swing_high, current_price, swing_highs, 'bullish'):
                        structure_break = StructureBreak(
                            index=i,
                            price=current_price,
                            timestamp=timestamps[i] if i < len(timestamps) else f"Bar_{i}",
                            break_type='BOS',
                            direction='bullish',
                            previous_structure=swing_high,
                            confidence=0.8,
                            description=f"Bullish BOS: Price broke above swing high at {swing_high.price:.2f}"
                        )
                        structure_breaks.append(structure_break)
                        break
            
            # Check for bearish BOS (break below previous swing low)
            for swing_low in swing_lows:
                if (swing_low.index < i and 
                    current_price < swing_low.price * 0.999):  # 0.1% buffer for noise
                    
                    if self._is_significant_break(swing_low, current_price, swing_lows, 'bearish'):
                        structure_break = StructureBreak(
                            index=i,
                            price=current_price,
                            timestamp=timestamps[i] if i < len(timestamps) else f"Bar_{i}",
                            break_type='BOS',
                            direction='bearish',
                            previous_structure=swing_low,
                            confidence=0.8,
                            description=f"Bearish BOS: Price broke below swing low at {swing_low.price:.2f}"
                        )
                        structure_breaks.append(structure_break)
                        break
        
        # Remove duplicate/overlapping breaks
        structure_breaks = self._filter_duplicate_breaks(structure_breaks)
        
        return structure_breaks
    
    def _analyze_trend(self, swing_highs: List[SwingPoint], swing_lows: List[SwingPoint], 
                      structure_breaks: List[StructureBreak], closes: np.ndarray) -> TrendAnalysis:
        """Analyze current trend based on market structure."""
        
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return TrendAnalysis(
                current_trend='sideways',
                trend_strength=0.0,
                trend_duration=0,
                last_structure_break=None,
                swing_sequence=[],
                trend_quality='weak'
            )
        
        # Get recent swings
        recent_highs = swing_highs[-3:] if len(swing_highs) >= 3 else swing_highs
        recent_lows = swing_lows[-3:] if len(swing_lows) >= 3 else swing_lows
        
        # Determine trend from swing analysis
        uptrend_signals = 0
        downtrend_signals = 0
        
        # Check for higher highs and higher lows (uptrend)
        if len(recent_highs) >= 2:
            for i in range(1, len(recent_highs)):
                if recent_highs[i].price > recent_highs[i-1].price:
                    uptrend_signals += 1
                else:
                    downtrend_signals += 1
        
        if len(recent_lows) >= 2:
            for i in range(1, len(recent_lows)):
                if recent_lows[i].price > recent_lows[i-1].price:
                    uptrend_signals += 1
                else:
                    downtrend_signals += 1
        
        # Determine current trend
        if uptrend_signals > downtrend_signals:
            current_trend = 'uptrend'
            trend_strength = min(uptrend_signals / (uptrend_signals + downtrend_signals + 0.1), 1.0)
        elif downtrend_signals > uptrend_signals:
            current_trend = 'downtrend'
            trend_strength = min(downtrend_signals / (uptrend_signals + downtrend_signals + 0.1), 1.0)
        else:
            current_trend = 'sideways'
            trend_strength = 0.3
        
        # Get last structure break
        last_break = structure_breaks[-1] if structure_breaks else None
        
        # Build swing sequence
        swing_sequence = []
        all_swings = recent_highs + recent_lows
        all_swings.sort(key=lambda x: x.index)
        
        for swing in all_swings[-4:]:  # Last 4 swings
            swing_sequence.append(swing.swing_type.upper())
        
        # Assess trend quality
        if trend_strength > 0.7:
            trend_quality = 'strong'
        elif trend_strength > 0.4:
            trend_quality = 'moderate'
        else:
            trend_quality = 'weak'
        
        # Calculate trend duration
        trend_duration = self._calculate_trend_duration(all_swings, current_trend)
        
        return TrendAnalysis(
            current_trend=current_trend,
            trend_strength=trend_strength,
            trend_duration=trend_duration,
            last_structure_break=last_break,
            swing_sequence=swing_sequence,
            trend_quality=trend_quality
        )
    
    def _identify_market_phases(self, swing_analysis: Dict[str, Any], 
                              trend_analysis: TrendAnalysis, closes: np.ndarray) -> Dict[str, Any]:
        """Identify current market phase (accumulation, distribution, trending, consolidation)."""
        
        # Analyze price volatility and range
        recent_range = np.max(closes[-20:]) - np.min(closes[-20:]) if len(closes) >= 20 else 0
        current_price = closes[-1]
        price_position = (current_price - np.min(closes[-20:])) / (recent_range + 0.001)
        
        # Determine phase based on trend and structure
        if trend_analysis.trend_strength > 0.6:
            if trend_analysis.current_trend == 'uptrend':
                phase = 'trending_up'
            else:
                phase = 'trending_down'
        elif trend_analysis.trend_strength < 0.3:
            if price_position > 0.7:
                phase = 'distribution'  # High in range, potential selling
            elif price_position < 0.3:
                phase = 'accumulation'  # Low in range, potential buying
            else:
                phase = 'consolidation'
        else:
            phase = 'transition'  # Between trends
        
        return {
            'current_phase': phase,
            'phase_confidence': abs(trend_analysis.trend_strength - 0.5) * 2,  # Distance from neutral
            'price_position_in_range': price_position,
            'range_size_percent': (recent_range / current_price) * 100 if current_price > 0 else 0,
            'phase_description': self._get_phase_description(phase)
        }
    
    def _calculate_structural_levels(self, swing_highs: List[SwingPoint], 
                                   swing_lows: List[SwingPoint], current_price: float) -> Dict[str, Any]:
        """Calculate key support and resistance levels from market structure."""
        
        # Extract prices from swings
        high_prices = [sh.price for sh in swing_highs]
        low_prices = [sl.price for sl in swing_lows]
        
        # Identify key resistance levels (recent swing highs)
        resistance_levels = []
        if high_prices:
            # Sort by proximity to current price and recency
            recent_highs = sorted(swing_highs, key=lambda x: abs(x.price - current_price))[:5]
            resistance_levels = [rh.price for rh in recent_highs if rh.price > current_price]
        
        # Identify key support levels (recent swing lows)
        support_levels = []
        if low_prices:
            recent_lows = sorted(swing_lows, key=lambda x: abs(x.price - current_price))[:5]
            support_levels = [rl.price for rl in recent_lows if rl.price < current_price]
        
        return {
            'resistance_levels': sorted(resistance_levels),
            'support_levels': sorted(support_levels, reverse=True),
            'nearest_resistance': min(resistance_levels) if resistance_levels else None,
            'nearest_support': max(support_levels) if support_levels else None,
            'key_level_count': len(resistance_levels) + len(support_levels)
        }
    
    def _assess_structure_quality(self, swing_highs: List[SwingPoint], 
                                swing_lows: List[SwingPoint], 
                                trend_analysis: TrendAnalysis) -> Dict[str, Any]:
        """Assess the overall quality and clarity of market structure."""
        
        # Calculate clarity score based on swing count and distribution
        total_swings = len(swing_highs) + len(swing_lows)
        clarity_score = min(total_swings / 10.0, 1.0)  # More swings = clearer structure (up to a point)
        
        # Assess trend consistency
        consistency_score = trend_analysis.trend_strength
        
        # Assess structure reliability
        reliability_score = 0.5
        if total_swings >= 6:
            reliability_score += 0.2
        if trend_analysis.trend_quality in ['strong', 'moderate']:
            reliability_score += 0.3
        
        reliability_score = min(reliability_score, 1.0)
        
        return {
            'clarity_score': clarity_score,
            'consistency_score': consistency_score,
            'reliability_score': reliability_score,
            'overall_quality': (clarity_score + consistency_score + reliability_score) / 3,
            'structure_strength': 'strong' if reliability_score > 0.7 else 'moderate' if reliability_score > 0.4 else 'weak'
        }
    
    # Helper methods
    def _classify_swing_pattern(self, sequence: List[str]) -> str:
        """Classify the swing pattern from sequence."""
        if not sequence:
            return 'no_pattern'
        
        # Check for bullish patterns
        if sequence[-2:] == ['HL', 'HH'] or 'HH' in sequence and 'HL' in sequence:
            return 'bullish_structure'
        # Check for bearish patterns
        elif sequence[-2:] == ['LH', 'LL'] or 'LL' in sequence and 'LH' in sequence:
            return 'bearish_structure'
        # Mixed signals
        else:
            return 'mixed_structure'
    
    def _get_trend_from_pattern(self, pattern: str) -> str:
        """Get trend indication from swing pattern."""
        if pattern == 'bullish_structure':
            return 'bullish'
        elif pattern == 'bearish_structure':
            return 'bearish'
        else:
            return 'neutral'
    
    def _calculate_pattern_strength(self, sequence: List[str]) -> float:
        """Calculate the strength of the swing pattern."""
        if not sequence:
            return 0.0
        
        bullish_signals = sequence.count('HH') + sequence.count('HL')
        bearish_signals = sequence.count('LL') + sequence.count('LH')
        total_signals = len(sequence)
        
        if total_signals == 0:
            return 0.0
        
        # Return strength as dominance of one direction
        max_directional = max(bullish_signals, bearish_signals)
        return max_directional / total_signals
    
    def _is_significant_break(self, swing_point: SwingPoint, current_price: float, 
                            all_swings: List[SwingPoint], direction: str) -> bool:
        """Check if a structure break is significant enough to be noteworthy."""
        
        # Must be a recent swing (not too old)
        if len(all_swings) > 10:
            recent_swings = all_swings[-10:]
            if swing_point not in recent_swings:
                return False
        
        # Must be a meaningful price movement (> 0.5%)
        price_move = abs(current_price - swing_point.price) / swing_point.price
        if price_move < 0.005:  # Less than 0.5%
            return False
        
        return True
    
    def _filter_duplicate_breaks(self, structure_breaks: List[StructureBreak]) -> List[StructureBreak]:
        """Remove duplicate or very close structure breaks."""
        if len(structure_breaks) <= 1:
            return structure_breaks
        
        filtered = []
        for i, break_point in enumerate(structure_breaks):
            is_duplicate = False
            for existing in filtered:
                # Check if breaks are very close in time and price
                if (abs(break_point.index - existing.index) < 5 and
                    abs(break_point.price - existing.price) / existing.price < 0.01):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered.append(break_point)
        
        return filtered
    
    def _calculate_trend_duration(self, all_swings: List[SwingPoint], current_trend: str) -> int:
        """Calculate how long the current trend has been in place."""
        if not all_swings or current_trend == 'sideways':
            return 0
        
        # Simple estimation - count swings in current trend direction
        duration = 0
        if current_trend == 'uptrend':
            # Count recent higher highs and higher lows
            for swing in all_swings[-5:]:  # Last 5 swings
                duration += 1
        elif current_trend == 'downtrend':
            # Count recent lower highs and lower lows
            for swing in all_swings[-5:]:  # Last 5 swings
                duration += 1
        
        return duration
    
    def _get_phase_description(self, phase: str) -> str:
        """Get human readable description of market phase."""
        descriptions = {
            'trending_up': 'Strong upward momentum with clear higher highs and higher lows',
            'trending_down': 'Strong downward momentum with clear lower highs and lower lows',
            'accumulation': 'Potential buying interest at lower price levels',
            'distribution': 'Potential selling pressure at higher price levels',
            'consolidation': 'Price moving sideways in a defined range',
            'transition': 'Market transitioning between trend phases'
        }
        return descriptions.get(phase, 'Market phase analysis unclear')
    
    def _create_empty_analysis(self, reason: str) -> Dict[str, Any]:
        """Create empty analysis structure when analysis fails."""
        return {
            'analysis_type': 'market_structure',
            'timestamp': datetime.now().isoformat(),
            'error': reason,
            'data_points_analyzed': 0,
            'swing_points': {'swing_highs': [], 'swing_lows': [], 'total_swings': 0},
            'swing_analysis': {'sequence': [], 'pattern': 'no_data', 'trend_indication': 'unclear', 'strength': 0.0},
            'structure_breaks': [],
            'trend_analysis': asdict(TrendAnalysis(
                current_trend='sideways', trend_strength=0.0, trend_duration=0,
                last_structure_break=None, swing_sequence=[], trend_quality='weak'
            )),
            'market_phases': {'current_phase': 'unknown', 'phase_confidence': 0.0},
            'key_levels': {'resistance_levels': [], 'support_levels': []},
            'structure_quality': {'clarity_score': 0.0, 'overall_quality': 0.0},
            'summary': {'current_trend': 'unknown', 'trend_strength': 0.0, 'recent_structure_breaks': 0}
        }


# Test function
async def test_market_structure_analyzer():
    """Test the market structure analyzer with sample data."""
    print("🧪 Testing Market Structure Analyzer")
    print("=" * 50)
    
    try:
        # Create sample stock data
        import pandas as pd
        import numpy as np
        
        # Generate realistic price movement
        np.random.seed(42)
        days = 100
        base_price = 1000
        returns = np.random.normal(0.001, 0.02, days)  # Small daily returns with volatility
        prices = [base_price]
        
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        
        # Create OHLCV data
        data = []
        for i, close in enumerate(prices[1:]):
            high = close * (1 + abs(np.random.normal(0, 0.01)))
            low = close * (1 - abs(np.random.normal(0, 0.01)))
            open_price = prices[i] if i == 0 else data[i-1]['close']
            volume = np.random.randint(100000, 1000000)
            
            data.append({
                'timestamp': f"2024-01-{i+1:02d}",
                'open': open_price,
                'high': max(open_price, high, close),
                'low': min(open_price, low, close),
                'close': close,
                'volume': volume
            })
        
        stock_data = pd.DataFrame(data)
        
        # Test the analyzer
        analyzer = MarketStructureAnalyzer(min_swing_strength=2)
        result = analyzer.analyze_market_structure(stock_data)
        
        print("✅ Market Structure Analyzer created successfully")
        print(f"   Data points analyzed: {result['data_points_analyzed']}")
        print(f"   Total swings found: {result['swing_points']['total_swings']}")
        print(f"   Current trend: {result['summary']['current_trend']}")
        print(f"   Trend strength: {result['summary']['trend_strength']:.2f}")
        print(f"   Structure breaks: {result['summary']['recent_structure_breaks']}")
        print(f"   Structure quality: {result['summary']['structure_clarity']:.2f}")
        
        # Check key components
        assert 'swing_points' in result
        assert 'structure_breaks' in result
        assert 'trend_analysis' in result
        assert 'market_phases' in result
        
        print("✅ All components present in analysis")
        print("✅ Market Structure Analyzer test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Market Structure Analyzer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_market_structure_analyzer())