#!/usr/bin/env python3
"""
Institutional Activity Agent - Integration Module

This module provides a unified interface for the Institutional Activity Agent,
combining data processing and visualization capabilities.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

from .processor import InstitutionalActivityProcessor
from .charts import InstitutionalActivityChartGenerator

class InstitutionalActivityAgent:
    """
    Unified Institutional Activity Agent
    
    Combines volume profile analysis, large block detection, and smart money timing
    to provide comprehensive institutional activity insights.
    """
    
    def __init__(self):
        self.processor = InstitutionalActivityProcessor()
        self.chart_generator = InstitutionalActivityChartGenerator()
        
        # Configuration
        self.config = {
            'min_data_points': 30,
            'analysis_lookback_days': 90,
            'chart_save_path': None,
            'enable_charts': True,
            'verbose': True
        }
    
    def analyze(self, data: pd.DataFrame, stock_symbol: str = "STOCK", 
                save_chart: bool = False, chart_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Complete institutional activity analysis
        
        Args:
            data: DataFrame with OHLCV data
            stock_symbol: Stock symbol for labeling
            save_chart: Whether to save generated chart
            chart_path: Path for saved chart
            
        Returns:
            Complete analysis results including charts
        """
        if self.config['verbose']:
            print(f"🏛️ Analyzing institutional activity for {stock_symbol}")
            print(f"   Data period: {len(data)} days")
            print(f"   Date range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
        
        # Process data
        analysis_results = self.processor.process_institutional_activity_data(data)
        
        if 'error' in analysis_results:
            return {
                'error': f"Analysis failed: {analysis_results['error']}",
                'symbol': stock_symbol,
                'analysis_results': None,
                'chart_bytes': None
            }
        
        # Generate chart if enabled
        chart_bytes = None
        if self.config['enable_charts']:
            if chart_path is None and save_chart:
                chart_path = f"{stock_symbol.lower()}_institutional_activity.png"
            
            chart_bytes = self.chart_generator.generate_institutional_activity_chart(
                data, analysis_results, stock_symbol, 
                save_path=chart_path if save_chart else None
            )
        
        # Prepare summary
        summary = self._create_analysis_summary(analysis_results, stock_symbol)
        
        if self.config['verbose']:
            self._print_summary(summary)
        
        return {
            'symbol': stock_symbol,
            'analysis_results': analysis_results,
            'summary': summary,
            'chart_bytes': chart_bytes,
            'recommendations': self._generate_recommendations(analysis_results)
        }
    
    def batch_analyze(self, data_dict: Dict[str, pd.DataFrame], 
                     save_charts: bool = False, 
                     output_dir: str = ".") -> Dict[str, Any]:
        """
        Analyze multiple stocks for institutional activity
        
        Args:
            data_dict: Dictionary of {symbol: DataFrame} with OHLCV data
            save_charts: Whether to save all charts
            output_dir: Directory for saved charts
            
        Returns:
            Dictionary of analysis results for each symbol
        """
        print(f"🏛️ Batch analyzing {len(data_dict)} stocks for institutional activity")
        
        results = {}
        successful = 0
        failed = 0
        
        for symbol, data in data_dict.items():
            try:
                chart_path = f"{output_dir}/{symbol.lower()}_institutional.png" if save_charts else None
                result = self.analyze(data, symbol, save_charts, chart_path)
                
                if 'error' not in result:
                    results[symbol] = result
                    successful += 1
                else:
                    results[symbol] = result
                    failed += 1
                    if self.config['verbose']:
                        print(f"❌ Analysis failed for {symbol}: {result['error']}")
                        
            except Exception as e:
                results[symbol] = {
                    'error': f"Critical error: {str(e)}",
                    'symbol': symbol
                }
                failed += 1
        
        # Create batch summary
        batch_summary = self._create_batch_summary(results)
        
        print(f"\n📊 Batch Analysis Complete:")
        print(f"   Successful: {successful}")
        print(f"   Failed: {failed}")
        print(f"   Total: {len(data_dict)}")
        
        return {
            'individual_results': results,
            'batch_summary': batch_summary,
            'statistics': {
                'total_analyzed': len(data_dict),
                'successful': successful,
                'failed': failed,
                'success_rate': successful / len(data_dict) if data_dict else 0
            }
        }
    
    def _create_analysis_summary(self, analysis_results: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """Create a concise analysis summary"""
        try:
            # Extract key metrics
            activity_level = analysis_results.get('institutional_activity_level', 'unknown')
            primary_activity = analysis_results.get('primary_activity', 'unknown')
            
            # Large blocks
            large_blocks = analysis_results.get('large_block_analysis', {})
            total_blocks = large_blocks.get('total_large_blocks', 0)
            institutional_blocks = large_blocks.get('institutional_block_count', 0)
            
            # Volume profile
            volume_profile = analysis_results.get('volume_profile', {})
            poc = volume_profile.get('point_of_control', {})
            
            # Predictive
            predictive = analysis_results.get('predictive_indicators', {})
            prediction = predictive.get('prediction', 'unknown')
            confidence = predictive.get('confidence', 0)
            
            # Quality
            quality = analysis_results.get('quality_assessment', {})
            quality_score = quality.get('overall_score', 0)
            
            return {
                'symbol': symbol,
                'overall_assessment': {
                    'activity_level': activity_level,
                    'primary_pattern': primary_activity,
                    'institutional_presence': 'high' if institutional_blocks >= 3 else 'medium' if institutional_blocks >= 1 else 'low'
                },
                'key_metrics': {
                    'total_large_blocks': total_blocks,
                    'institutional_blocks': institutional_blocks,
                    'point_of_control': poc.get('price_level', 0),
                    'prediction': prediction,
                    'confidence': confidence,
                    'quality_score': quality_score
                },
                'institutional_sentiment': self._determine_sentiment(analysis_results)
            }
            
        except Exception as e:
            return {
                'symbol': symbol,
                'error': f"Summary creation failed: {str(e)}"
            }
    
    def _determine_sentiment(self, analysis_results: Dict[str, Any]) -> str:
        """Determine overall institutional sentiment"""
        try:
            # Get key indicators
            primary_activity = analysis_results.get('primary_activity', 'unknown')
            predictive = analysis_results.get('predictive_indicators', {})
            prediction = predictive.get('prediction', 'neutral')
            
            # Large block analysis
            large_blocks = analysis_results.get('large_block_analysis', {})
            institutional_count = large_blocks.get('institutional_block_count', 0)
            
            # Smart money timing
            timing = analysis_results.get('smart_money_timing', {})
            timing_quality = timing.get('timing_quality', 'poor')
            
            # Determine sentiment
            if primary_activity == 'accumulation' and prediction == 'bullish' and institutional_count > 0:
                return 'bullish'
            elif primary_activity == 'distribution' and prediction == 'bearish':
                return 'bearish'
            elif institutional_count > 0 and timing_quality in ['good', 'excellent']:
                return 'cautiously_bullish'
            elif institutional_count == 0:
                return 'neutral'
            else:
                return 'mixed'
                
        except Exception:
            return 'unknown'
    
    def _print_summary(self, summary: Dict[str, Any]):
        """Print formatted analysis summary"""
        try:
            if 'error' in summary:
                print(f"❌ Summary error: {summary['error']}")
                return
            
            symbol = summary['symbol']
            overall = summary['overall_assessment']
            metrics = summary['key_metrics']
            sentiment = summary['institutional_sentiment']
            
            print(f"\n📋 INSTITUTIONAL ACTIVITY SUMMARY - {symbol}")
            print("-" * 60)
            print(f"Overall Activity Level: {overall['activity_level'].replace('_', ' ').title()}")
            print(f"Primary Pattern: {overall['primary_pattern'].replace('_', ' ').title()}")
            print(f"Institutional Presence: {overall['institutional_presence'].title()}")
            print(f"Institutional Sentiment: {sentiment.replace('_', ' ').title()}")
            
            print(f"\nKey Metrics:")
            print(f"  Large Blocks Detected: {metrics['total_large_blocks']}")
            print(f"  Institutional Blocks: {metrics['institutional_blocks']}")
            print(f"  Point of Control: ₹{metrics['point_of_control']:.2f}")
            print(f"  Price Prediction: {metrics['prediction'].title()} ({metrics['confidence']:.1%} confidence)")
            print(f"  Analysis Quality: {metrics['quality_score']}/100")
            
        except Exception as e:
            print(f"❌ Print summary error: {str(e)}")
    
    def _create_batch_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary of batch analysis results"""
        try:
            successful_results = {k: v for k, v in results.items() if 'error' not in v}
            
            if not successful_results:
                return {'error': 'No successful analyses to summarize'}
            
            # Aggregate statistics
            activity_levels = []
            predictions = []
            sentiments = []
            institutional_counts = []
            quality_scores = []
            
            for symbol, result in successful_results.items():
                summary = result.get('summary', {})
                if 'error' not in summary:
                    overall = summary.get('overall_assessment', {})
                    metrics = summary.get('key_metrics', {})
                    
                    activity_levels.append(overall.get('activity_level', 'unknown'))
                    predictions.append(metrics.get('prediction', 'unknown'))
                    sentiments.append(summary.get('institutional_sentiment', 'unknown'))
                    institutional_counts.append(metrics.get('institutional_blocks', 0))
                    quality_scores.append(metrics.get('quality_score', 0))
            
            # Count frequencies
            from collections import Counter
            
            return {
                'total_stocks': len(results),
                'successful_analyses': len(successful_results),
                'aggregated_metrics': {
                    'activity_levels': dict(Counter(activity_levels)),
                    'predictions': dict(Counter(predictions)),
                    'sentiments': dict(Counter(sentiments)),
                    'avg_institutional_blocks': np.mean(institutional_counts) if institutional_counts else 0,
                    'avg_quality_score': np.mean(quality_scores) if quality_scores else 0
                },
                'top_institutional_activity': self._get_top_institutional_stocks(successful_results)
            }
            
        except Exception as e:
            return {'error': f"Batch summary creation failed: {str(e)}"}
    
    def _get_top_institutional_stocks(self, results: Dict[str, Any], top_n: int = 5) -> List[Dict]:
        """Get top stocks by institutional activity"""
        try:
            stock_scores = []
            
            for symbol, result in results.items():
                summary = result.get('summary', {})
                if 'error' not in summary:
                    metrics = summary.get('key_metrics', {})
                    institutional_blocks = metrics.get('institutional_blocks', 0)
                    quality_score = metrics.get('quality_score', 0)
                    
                    # Create composite score
                    composite_score = institutional_blocks * 20 + quality_score * 0.5
                    
                    stock_scores.append({
                        'symbol': symbol,
                        'institutional_blocks': institutional_blocks,
                        'quality_score': quality_score,
                        'composite_score': composite_score,
                        'sentiment': summary.get('institutional_sentiment', 'unknown')
                    })
            
            # Sort by composite score and return top N
            return sorted(stock_scores, key=lambda x: x['composite_score'], reverse=True)[:top_n]
            
        except Exception:
            return []
    
    def _generate_recommendations(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        recommendations = []
        
        try:
            # Extract key data
            activity_level = analysis_results.get('institutional_activity_level', 'unknown')
            primary_activity = analysis_results.get('primary_activity', 'unknown')
            
            large_blocks = analysis_results.get('large_block_analysis', {})
            institutional_blocks = large_blocks.get('institutional_block_count', 0)
            
            predictive = analysis_results.get('predictive_indicators', {})
            prediction = predictive.get('prediction', 'unknown')
            confidence = predictive.get('confidence', 0)
            
            timing = analysis_results.get('smart_money_timing', {})
            timing_quality = timing.get('timing_quality', 'unknown')
            
            # Generate recommendations
            if institutional_blocks >= 3 and primary_activity == 'accumulation':
                recommendations.append("Strong institutional accumulation detected - consider position building on dips")
            
            if timing_quality in ['excellent', 'good'] and prediction == 'bullish':
                recommendations.append("Smart money timing is favorable - good entry opportunity")
            
            if primary_activity == 'distribution' and institutional_blocks >= 2:
                recommendations.append("Institutional distribution pattern - exercise caution, consider profit taking")
            
            if activity_level in ['very_high', 'high'] and confidence > 0.7:
                recommendations.append(f"High institutional activity with {confidence:.0%} confidence in {prediction} direction")
            
            if institutional_blocks == 0:
                recommendations.append("Low institutional interest - rely on technical analysis and market sentiment")
            
            # Volume profile recommendations
            volume_profile = analysis_results.get('volume_profile', {})
            if 'error' not in volume_profile:
                poc = volume_profile.get('point_of_control', {})
                if poc:
                    recommendations.append(f"Key support/resistance at ₹{poc.get('price_level', 0):.2f} (Point of Control)")
            
            if not recommendations:
                recommendations.append("Mixed signals - maintain cautious approach and monitor for clearer patterns")
            
            return recommendations
            
        except Exception as e:
            return [f"Error generating recommendations: {str(e)}"]
    
    def configure(self, **kwargs):
        """Update agent configuration"""
        for key, value in kwargs.items():
            if key in self.config:
                self.config[key] = value
                if self.config['verbose']:
                    print(f"✅ Updated {key}: {value}")
            else:
                if self.config['verbose']:
                    print(f"⚠️  Unknown configuration key: {key}")
    
    def get_analysis_explanation(self) -> str:
        """Get explanation of the analysis methodology"""
        return """
INSTITUTIONAL ACTIVITY AGENT - ANALYSIS METHODOLOGY

🏛️ CORE COMPONENTS:

1. Volume Profile Analysis
   - Calculates volume distribution at different price levels
   - Identifies Point of Control (POC) - highest volume price
   - Determines Value Area (70% of volume range)

2. Large Block Detection  
   - Identifies transactions 2x+ above average volume (large blocks)
   - Flags transactions 3x+ above average as institutional blocks
   - Analyzes frequency and timing patterns

3. Accumulation/Distribution Analysis
   - Uses Accumulation/Distribution Line (A/D Line) 
   - Money Flow Multiplier based on close position in range
   - Determines buying vs selling pressure

4. Smart Money Timing Analysis
   - Evaluates institutional entry timing relative to price moves
   - Classifies timing as: accumulation on dips, early accumulation, 
     breakout accumulation, or distribution
   - Identifies activity clusters in time

5. Predictive Indicators
   - Correlates volume trends with price movements
   - Generates signals for potential price direction
   - Calculates confidence levels based on signal strength

📊 INTERPRETATION GUIDE:

Activity Levels:
- Very High: >20% of days have institutional blocks
- High: 10-20% institutional block frequency
- Medium: 5-10% frequency
- Low: 2-5% frequency  
- Very Low: <2% frequency

Timing Quality:
- Excellent: >70% of blocks show good timing
- Good: 50-70% good timing
- Fair: 30-50% good timing
- Poor: <30% good timing

Sentiment Indicators:
- Bullish: Accumulation + favorable timing + bullish prediction
- Bearish: Distribution pattern + bearish signals
- Cautiously Bullish: Some positive signals with caveats
- Mixed: Conflicting signals
- Neutral: Limited institutional activity

🎯 USE CASES:

1. Entry Timing: Look for accumulation on dips with good timing quality
2. Risk Assessment: Distribution patterns suggest caution
3. Support/Resistance: Use POC and Value Area levels
4. Trend Confirmation: Volume should confirm price movements
5. Position Sizing: Higher institutional activity may justify larger positions

⚠️  LIMITATIONS:

- Based on daily volume data (intraday data would be more precise)
- Cannot distinguish between different types of institutions
- Market conditions and news events can override patterns
- Should be combined with other technical and fundamental analysis
"""

def demo_institutional_activity_analysis():
    """Demonstration of the Institutional Activity Agent"""
    print("🎯 INSTITUTIONAL ACTIVITY AGENT DEMONSTRATION")
    print("=" * 80)
    
    # Create agent
    agent = InstitutionalActivityAgent()
    
    # Show methodology
    print(agent.get_analysis_explanation())
    
    # Create demo data
    print("\n🔬 Creating demonstration data with institutional patterns...")
    dates = pd.date_range(start='2024-06-01', end='2024-10-20', freq='D')
    np.random.seed(42)
    
    # Demo stock with clear institutional accumulation
    base_price = 2500
    price_changes = np.random.normal(0.002, 0.015, len(dates))
    
    # Add accumulation period
    accumulation_start, accumulation_end = 30, 50
    price_changes[accumulation_start:accumulation_end] = np.random.normal(0.005, 0.008, accumulation_end-accumulation_start)
    
    prices = base_price * np.cumprod(1 + price_changes)
    volumes = np.random.lognormal(np.log(1800000), 0.4, len(dates))
    
    # Add institutional blocks during accumulation
    for i in range(accumulation_start, accumulation_end):
        if np.random.random() > 0.6:  # 40% chance
            volumes[i] *= np.random.uniform(3.5, 6.0)
    
    demo_data = pd.DataFrame({
        'open': prices + np.random.normal(0, 3, len(dates)),
        'high': prices + np.abs(np.random.normal(8, 4, len(dates))),
        'low': prices - np.abs(np.random.normal(8, 4, len(dates))),
        'close': prices,
        'volume': volumes.astype(int)
    }, index=dates)
    
    # Ensure realistic OHLC relationships
    demo_data['high'] = np.maximum(demo_data[['open', 'close']].max(axis=1), demo_data['high'])
    demo_data['low'] = np.minimum(demo_data[['open', 'close']].min(axis=1), demo_data['low'])
    
    print(f"✅ Demo data created: {len(demo_data)} days")
    
    # Run analysis
    print("\n🏛️ Running institutional activity analysis...")
    result = agent.analyze(demo_data, "DEMO_STOCK", save_chart=True, chart_path="demo_institutional_activity.png")
    
    if 'error' not in result:
        print("\n✅ Analysis completed successfully!")
        
        # Show recommendations
        recommendations = result.get('recommendations', [])
        print(f"\n💡 RECOMMENDATIONS:")
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
        
        # Show chart info
        if result.get('chart_bytes'):
            print(f"\n📊 Chart generated and saved as 'demo_institutional_activity.png'")
        
    else:
        print(f"❌ Demo analysis failed: {result['error']}")

# Alias for backwards compatibility
InstitutionalActivityIntegration = InstitutionalActivityAgent

if __name__ == "__main__":
    demo_institutional_activity_analysis()