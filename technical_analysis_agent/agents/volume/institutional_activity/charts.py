#!/usr/bin/env python3
"""
Institutional Activity Agent - Chart Generation Module

Creates volume profile and institutional activity visualizations
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from typing import Dict, List, Any, Optional
import io
import warnings
warnings.filterwarnings('ignore')

class InstitutionalActivityChartGenerator:
    """Chart generator for Institutional Activity Agent"""
    
    def __init__(self):
        self.chart_style = {
            'figure_size': (16, 12),
            'dpi': 300,
            'colors': {
                'volume_profile': '#2196f3',
                'poc_line': '#ff5722',
                'value_area': '#4caf50',
                'institutional_block': '#9c27b0',
                'large_block': '#ff9800',
                'accumulation': '#4caf50',
                'distribution': '#f44336',
                'background': '#fafafa',
                'grid': '#e0e0e0'
            }
        }
    
    def generate_institutional_activity_chart(self, data: pd.DataFrame,
                                            analysis_data: Dict[str, Any],
                                            stock_symbol: str = "STOCK",
                                            save_path: Optional[str] = None) -> Optional[bytes]:
        """Generate comprehensive institutional activity chart"""
        try:
            plt.style.use('default')
            fig = plt.figure(figsize=self.chart_style['figure_size'],
                           dpi=self.chart_style['dpi'],
                           facecolor='white')
            
            # Create 5-panel layout
            gs = fig.add_gridspec(5, 2, height_ratios=[2.5, 2, 1.5, 1, 0.5], 
                                hspace=0.35, wspace=0.15)
            
            # Panel layout
            ax_volume_profile = fig.add_subplot(gs[0, :])
            ax_blocks = fig.add_subplot(gs[1, 0])
            ax_accumulation = fig.add_subplot(gs[1, 1])
            ax_timing = fig.add_subplot(gs[2, :])
            ax_summary = fig.add_subplot(gs[3, :])
            ax_legend = fig.add_subplot(gs[4, :])
            
            # Generate panels
            self._plot_volume_profile(ax_volume_profile, data, analysis_data)
            self._plot_large_blocks(ax_blocks, data, analysis_data)
            self._plot_accumulation_distribution(ax_accumulation, data, analysis_data)
            self._plot_institutional_timing(ax_timing, data, analysis_data)
            self._plot_analysis_summary(ax_summary, analysis_data)
            self._add_chart_legend(ax_legend, analysis_data)
            
            # Main title
            activity_level = analysis_data.get('institutional_activity_level', 'unknown')
            primary_activity = analysis_data.get('primary_activity', 'unknown')
            
            fig.suptitle(f'{stock_symbol} - Institutional Activity Analysis\n'
                        f'Activity Level: {activity_level.replace("_", " ").title()} | '
                        f'Pattern: {primary_activity.replace("_", " ").title()}',
                        fontsize=16, fontweight='bold', y=0.98)
            
            # Convert to bytes
            buf = io.BytesIO()
            fig.canvas.draw()
            fig.savefig(buf, format='png', dpi=self.chart_style['dpi'],
                       bbox_inches='tight', facecolor='white', edgecolor='none')
            buf.seek(0)
            chart_bytes = buf.getvalue()
            buf.close()
            
            if save_path:
                with open(save_path, 'wb') as f:
                    f.write(chart_bytes)
                print(f"💾 Institutional Activity chart saved: {save_path}")
            
            plt.close(fig)
            return chart_bytes
            
        except Exception as e:
            print(f"❌ Chart generation failed: {e}")
            if 'fig' in locals():
                plt.close(fig)
            return None
    
    def _plot_volume_profile(self, ax, data: pd.DataFrame, analysis_data: Dict[str, Any]):
        """Plot volume profile with price levels"""
        try:
            # Try to get volume profile from institutional_analysis first, then fallback to direct volume_profile
            institutional_analysis = analysis_data.get('institutional_analysis', {})
            if isinstance(institutional_analysis, dict) and 'volume_profile' in institutional_analysis:
                volume_profile = institutional_analysis['volume_profile']
            else:
                volume_profile = analysis_data.get('volume_profile', {})
            
            if 'error' in volume_profile:
                ax.text(0.5, 0.5, 'Volume profile unavailable',
                       transform=ax.transAxes, ha='center', va='center')
                return
            
            volume_at_price = volume_profile.get('volume_at_price', [])
            if not volume_at_price:
                ax.text(0.5, 0.5, 'No volume profile data',
                       transform=ax.transAxes, ha='center', va='center')
                return
            
            # Plot price candlesticks
            for i, (date, row) in enumerate(data.iterrows()):
                color = 'green' if row['close'] > row['open'] else 'red'
                # High-Low line
                ax.plot([i, i], [row['low'], row['high']], color='black', linewidth=1)
                # Body
                body_height = abs(row['close'] - row['open'])
                body_bottom = min(row['open'], row['close'])
                rect = Rectangle((i-0.3, body_bottom), 0.6, body_height,
                               facecolor=color, alpha=0.7, edgecolor='black')
                ax.add_patch(rect)
            
            # Create volume profile on right side
            ax2 = ax.twinx()
            
            # Plot volume bars horizontally
            prices = [level['price_level'] for level in volume_at_price]
            volumes = [level['volume'] for level in volume_at_price]
            
            max_volume = max(volumes) if volumes else 1
            normalized_volumes = [v / max_volume * len(data) * 0.3 for v in volumes]
            
            for price, norm_vol in zip(prices, normalized_volumes):
                ax2.barh(price, norm_vol, height=2, alpha=0.6,
                        color=self.chart_style['colors']['volume_profile'])
            
            # Add key levels
            poc = volume_profile.get('point_of_control', {})
            if poc:
                poc_price = poc.get('price_level', 0)
                ax.axhline(y=poc_price, color=self.chart_style['colors']['poc_line'],
                          linewidth=3, alpha=0.8, label='Point of Control')
            
            va_high = volume_profile.get('value_area_high', 0)
            va_low = volume_profile.get('value_area_low', 0)
            
            if va_high and va_low:
                ax.axhline(y=va_high, color=self.chart_style['colors']['value_area'],
                          linestyle='--', alpha=0.7, label='Value Area High')
                ax.axhline(y=va_low, color=self.chart_style['colors']['value_area'],
                          linestyle='--', alpha=0.7, label='Value Area Low')
                
                # Shade value area
                ax.axhspan(va_low, va_high, alpha=0.1,
                          color=self.chart_style['colors']['value_area'])
            
            ax.set_title('Volume Profile with Price Action', fontsize=14, pad=20)
            ax.set_ylabel('Price', fontsize=12)
            ax.legend(loc='upper left', fontsize=9)
            ax.grid(True, alpha=0.3)
            
            # Format x-axis
            ax.set_xticks(range(0, len(data), max(1, len(data)//10)))
            ax.set_xticklabels([data.index[i].strftime('%m/%d') 
                               for i in range(0, len(data), max(1, len(data)//10))],
                              rotation=45)
            
            ax2.set_xlabel('Volume Profile', fontsize=10)
            ax2.set_xlim(0, max(normalized_volumes) if normalized_volumes else 1)
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Volume profile error: {str(e)}',
                   transform=ax.transAxes, ha='center', va='center')
    
    def _plot_large_blocks(self, ax, data: pd.DataFrame, analysis_data: Dict[str, Any]):
        """Plot large block detection analysis"""
        try:
            # Try to get large block data from institutional_analysis first
            institutional_analysis = analysis_data.get('institutional_analysis', {})
            if isinstance(institutional_analysis, dict) and 'large_block_analysis' in institutional_analysis:
                large_block_data = institutional_analysis['large_block_analysis']
            else:
                large_block_data = analysis_data.get('large_block_analysis', {})
            
            if 'error' in large_block_data:
                ax.text(0.5, 0.5, 'Large block data unavailable',
                       transform=ax.transAxes, ha='center', va='center')
                return
            
            # Plot volume bars
            volume_data = data['volume']
            dates = range(len(data))
            
            # Color code volume bars
            colors = []
            large_blocks = large_block_data.get('large_blocks', [])
            institutional_blocks = large_block_data.get('institutional_blocks', [])
            
            # Create lookup for block dates
            institutional_dates = {block['date'] for block in institutional_blocks}
            large_block_dates = {block['date'] for block in large_blocks if block['classification'] != 'institutional'}
            
            for date, volume in zip(data.index, volume_data):
                date_str = date.strftime('%Y-%m-%d')
                if date_str in institutional_dates:
                    colors.append(self.chart_style['colors']['institutional_block'])
                elif date_str in large_block_dates:
                    colors.append(self.chart_style['colors']['large_block'])
                else:
                    colors.append('lightblue')
            
            ax.bar(dates, volume_data, color=colors, alpha=0.7)
            
            # Add threshold lines
            thresholds = large_block_data.get('thresholds', {})
            if thresholds:
                ax.axhline(y=thresholds.get('large_block', 0), color='orange', 
                          linestyle='--', alpha=0.8, label='Large Block Threshold')
                ax.axhline(y=thresholds.get('institutional', 0), color='purple', 
                          linestyle='--', alpha=0.8, label='Institutional Threshold')
            
            ax.set_title('Large Block Detection', fontsize=12)
            ax.set_ylabel('Volume', fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            
            # Format x-axis
            ax.set_xticks(range(0, len(data), max(1, len(data)//5)))
            ax.set_xticklabels([data.index[i].strftime('%m/%d') 
                               for i in range(0, len(data), max(1, len(data)//5))],
                              rotation=45, fontsize=8)
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Large block plot error: {str(e)}',
                   transform=ax.transAxes, ha='center', va='center')
    
    def _plot_accumulation_distribution(self, ax, data: pd.DataFrame, analysis_data: Dict[str, Any]):
        """Plot accumulation/distribution analysis"""
        try:
            # Try to get A/D data from institutional_analysis first, then fallback to direct accumulation_distribution
            institutional_analysis = analysis_data.get('institutional_analysis', {})
            if isinstance(institutional_analysis, dict) and 'accumulation_distribution' in institutional_analysis:
                ad_data = institutional_analysis['accumulation_distribution']
            else:
                ad_data = analysis_data.get('accumulation_distribution', {})
            
            if 'error' in ad_data:
                ax.text(0.5, 0.5, 'A/D data unavailable',
                       transform=ax.transAxes, ha='center', va='center')
                return
            
            ad_line = ad_data.get('ad_line', [])
            if not ad_line:
                ax.text(0.5, 0.5, 'No A/D line data',
                       transform=ax.transAxes, ha='center', va='center')
                return
            
            # Plot A/D line
            dates = range(len(ad_line))
            primary_pattern = ad_data.get('primary_pattern', 'neutral')
            
            color = self.chart_style['colors']['accumulation'] if primary_pattern == 'accumulation' else \
                   self.chart_style['colors']['distribution'] if primary_pattern == 'distribution' else 'gray'
            
            ax.plot(dates, ad_line, color=color, linewidth=2, 
                   label=f'A/D Line ({primary_pattern.title()})')
            
            # Add trend line
            if len(ad_line) > 10:
                z = np.polyfit(dates, ad_line, 1)
                trend_line = np.poly1d(z)
                ax.plot(dates, trend_line(dates), '--', color='red', alpha=0.7, label='Trend')
            
            # Add zero line
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            ax.set_title('Accumulation/Distribution Analysis', fontsize=12)
            ax.set_ylabel('A/D Line', fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            
            # Format x-axis
            ax.set_xticks(range(0, len(data), max(1, len(data)//5)))
            ax.set_xticklabels([data.index[i].strftime('%m/%d') 
                               for i in range(0, len(data), max(1, len(data)//5))],
                              rotation=45, fontsize=8)
            
        except Exception as e:
            ax.text(0.5, 0.5, f'A/D plot error: {str(e)}',
                   transform=ax.transAxes, ha='center', va='center')
    
    def _plot_institutional_timing(self, ax, data: pd.DataFrame, analysis_data: Dict[str, Any]):
        """Plot institutional timing analysis"""
        try:
            # Try to get timing data from institutional_analysis first
            institutional_analysis = analysis_data.get('institutional_analysis', {})
            if isinstance(institutional_analysis, dict) and 'smart_money_timing' in institutional_analysis:
                timing_data = institutional_analysis['smart_money_timing']
            else:
                timing_data = analysis_data.get('smart_money_timing', {})
            
            if 'error' in timing_data:
                ax.text(0.5, 0.5, 'Timing data unavailable',
                       transform=ax.transAxes, ha='center', va='center')
                return
            
            timing_analysis = timing_data.get('timing_analysis', [])
            if not timing_analysis:
                ax.text(0.5, 0.5, 'No institutional timing data',
                       transform=ax.transAxes, ha='center', va='center')
                return
            
            # Plot price line
            dates = range(len(data))
            ax.plot(dates, data['close'], color='blue', linewidth=1, alpha=0.7, label='Price')
            
            # Mark institutional activity with different colors
            timing_colors = {
                'accumulation_on_dip': 'green',
                'early_accumulation': 'lightgreen',
                'breakout_accumulation': 'orange',
                'distribution': 'red'
            }
            
            for timing in timing_analysis:
                timing_date = pd.to_datetime(timing['date'])
                if timing_date in data.index:
                    idx = data.index.get_loc(timing_date)
                    timing_type = timing['timing_type']
                    color = timing_colors.get(timing_type, 'gray')
                    
                    ax.scatter(idx, data.loc[timing_date, 'close'], 
                             color=color, s=100, alpha=0.8, 
                             label=timing_type.replace('_', ' ').title())
            
            # Remove duplicate labels
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), fontsize=8)
            
            ax.set_title('Smart Money Timing Analysis', fontsize=12)
            ax.set_ylabel('Price', fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Format x-axis
            ax.set_xticks(range(0, len(data), max(1, len(data)//10)))
            ax.set_xticklabels([data.index[i].strftime('%m/%d') 
                               for i in range(0, len(data), max(1, len(data)//10))],
                              rotation=45, fontsize=8)
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Timing plot error: {str(e)}',
                   transform=ax.transAxes, ha='center', va='center')
    
    def _plot_analysis_summary(self, ax, analysis_data: Dict[str, Any]):
        """Plot analysis summary metrics"""
        try:
            # Remove axes for clean text display
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 10)
            ax.axis('off')
            
            # Get key metrics
            activity_level = analysis_data.get('institutional_activity_level', 'unknown')
            primary_activity = analysis_data.get('primary_activity', 'unknown')
            
            large_blocks = analysis_data.get('large_block_analysis', {})
            total_blocks = large_blocks.get('total_large_blocks', 0)
            institutional_blocks = large_blocks.get('institutional_block_count', 0)
            
            predictive = analysis_data.get('predictive_indicators', {})
            prediction = predictive.get('prediction', 'unknown')
            confidence = predictive.get('confidence', 0) * 100
            
            quality = analysis_data.get('quality_assessment', {})
            quality_score = quality.get('overall_score', 0)
            
            # Display summary text
            summary_text = f"""INSTITUTIONAL ACTIVITY SUMMARY
            
Activity Level: {activity_level.replace('_', ' ').title()}
Primary Pattern: {primary_activity.replace('_', ' ').title()}
Large Blocks Detected: {total_blocks}
Institutional Blocks: {institutional_blocks}

PREDICTIVE OUTLOOK
Prediction: {prediction.title()}
Confidence: {confidence:.1f}%

ANALYSIS QUALITY
Quality Score: {quality_score}/100"""
            
            ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=11,
                   verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.3))
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Summary error: {str(e)}',
                   transform=ax.transAxes, ha='center', va='center')
    
    def _add_chart_legend(self, ax, analysis_data: Dict[str, Any]):
        """Add comprehensive chart legend"""
        try:
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 10)
            ax.axis('off')
            
            legend_text = """LEGEND & KEY INSIGHTS
            
Volume Profile: Blue bars show volume distribution at price levels
Point of Control (POC): Red line - highest volume price level
Value Area: Green lines - 70% of total volume range
Institutional Blocks: Purple markers - 3x+ average volume
Large Blocks: Orange markers - 2x+ average volume
A/D Line: Green (accumulation) / Red (distribution)
            
Smart Money Timing:
• Green: Accumulation on dips (excellent timing)
• Light Green: Early accumulation (good timing)  
• Orange: Breakout accumulation (fair timing)
• Red: Distribution (poor timing for buyers)"""
            
            ax.text(0.05, 0.95, legend_text, transform=ax.transAxes, fontsize=9,
                   verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Legend error: {str(e)}',
                   transform=ax.transAxes, ha='center', va='center')


def test_institutional_activity_charts():
    """Test function for Institutional Activity Chart Generator"""
    print("📊 Testing Institutional Activity Chart Generator")
    print("=" * 60)
    
    # Import the processor for test data
    from institutional_activity_processor import InstitutionalActivityProcessor
    
    # Create test data
    dates = pd.date_range(start='2024-07-01', end='2024-10-20', freq='D')
    np.random.seed(42)
    
    base_price = 2400
    base_volume = 1500000
    
    # Generate test data with patterns
    price_changes = np.random.normal(0.001, 0.015, len(dates))
    accumulation_periods = [(20, 30), (70, 80)]
    for start, end in accumulation_periods:
        if end < len(price_changes):
            price_changes[start:end] = np.random.normal(0.003, 0.008, end-start)
    
    prices = base_price * np.cumprod(1 + price_changes)
    volumes = np.random.lognormal(np.log(base_volume), 0.4, len(dates))
    
    # Add institutional blocks
    for start, end in accumulation_periods:
        for i in range(start, min(end, len(volumes))):
            if np.random.random() > 0.7:
                volumes[i] *= np.random.uniform(3.0, 6.0)
    
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
    
    # Process data
    processor = InstitutionalActivityProcessor()
    analysis_results = processor.process_institutional_activity_data(test_data)
    
    if 'error' in analysis_results:
        print(f"❌ Analysis failed: {analysis_results['error']}")
        return False
    
    print("✅ Analysis completed successfully")
    
    # Generate chart
    chart_generator = InstitutionalActivityChartGenerator()
    chart_bytes = chart_generator.generate_institutional_activity_chart(
        test_data, analysis_results, "TEST_STOCK", save_path="test_institutional_activity.png"
    )
    
    if chart_bytes:
        print("✅ Chart generated successfully!")
        print(f"   Chart size: {len(chart_bytes):,} bytes")
        return True
    else:
        print("❌ Chart generation failed")
        return False

# Alias for backwards compatibility
InstitutionalActivityCharts = InstitutionalActivityChartGenerator

if __name__ == "__main__":
    test_institutional_activity_charts()