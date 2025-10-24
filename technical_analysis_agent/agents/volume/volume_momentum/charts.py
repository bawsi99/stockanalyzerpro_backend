#!/usr/bin/env python3
"""
Volume Trend Momentum Chart Generator

Creates comprehensive visualizations for Volume Trend Momentum analysis
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from typing import Dict, List, Any, Optional, Tuple
import io
import warnings
warnings.filterwarnings('ignore')

# Try to import seaborn, use basic styling if not available
try:
    import seaborn as sns
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    HAS_SEABORN = True
except ImportError:
    plt.style.use('ggplot')
    HAS_SEABORN = False

class VolumeTrendMomentumChartGenerator:
    """
    Chart generator for Volume Trend Momentum analysis
    
    Creates multi-panel charts showing:
    1. Price with volume overlay and trend lines
    2. Volume momentum indicators (ROC, oscillator)
    3. Momentum cycle analysis with phases
    4. Future implications dashboard
    """
    
    def __init__(self):
        self.figure_size = (16, 12)
        self.dpi = 100
        
    def generate_volume_momentum_chart(self, 
                                     data: pd.DataFrame, 
                                     analysis_results: Dict[str, Any],
                                     symbol: str = "STOCK",
                                     save_path: Optional[str] = None) -> Optional[bytes]:
        """
        Generate comprehensive volume momentum chart
        
        Args:
            data: OHLCV DataFrame
            analysis_results: Results from VolumeTrendMomentumProcessor
            symbol: Stock symbol for title
            save_path: Optional path to save chart
            
        Returns:
            Chart as bytes if successful, None otherwise
        """
        try:
            if 'error' in analysis_results:
                print(f"❌ Cannot generate chart: {analysis_results['error']}")
                return None
            
            # Create figure with subplots
            fig = plt.figure(figsize=self.figure_size, dpi=self.dpi)
            
            # Define layout: 4 rows, 2 columns
            gs = fig.add_gridspec(4, 2, height_ratios=[2, 1, 1, 1], width_ratios=[3, 1],
                                hspace=0.3, wspace=0.3)
            
            # 1. Main price and volume chart
            ax1 = fig.add_subplot(gs[0, 0])
            self._plot_price_volume_trends(ax1, data, analysis_results)
            
            # 2. Summary dashboard
            ax_dash = fig.add_subplot(gs[0, 1])
            self._plot_summary_dashboard(ax_dash, analysis_results, symbol)
            
            # 3. Volume momentum indicators
            ax2 = fig.add_subplot(gs[1, :])
            self._plot_momentum_indicators(ax2, data, analysis_results)
            
            # 4. Momentum cycles and phases
            ax3 = fig.add_subplot(gs[2, :])
            self._plot_momentum_cycles(ax3, data, analysis_results)
            
            # 5. Future implications
            ax4 = fig.add_subplot(gs[3, :])
            self._plot_future_implications(ax4, analysis_results)
            
            # Overall title
            fig.suptitle(f'Volume Trend Momentum Analysis - {symbol}', 
                        fontsize=16, fontweight='bold', y=0.95)
            
            # Save or return bytes
            fig.canvas.draw()
            if save_path:
                fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight', 
                           facecolor='white', edgecolor='none')
                print(f"✅ Volume momentum chart saved to: {save_path}")
            
            # Return as bytes
            buffer = io.BytesIO()
            fig.savefig(buffer, format='png', dpi=self.dpi, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            buffer.seek(0)
            chart_bytes = buffer.read()
            buffer.close()
            plt.close(fig)
            
            return chart_bytes
            
        except Exception as e:
            print(f"❌ Chart generation failed: {str(e)}")
            plt.close('all')
            return None
    
    def _plot_price_volume_trends(self, ax: plt.Axes, data: pd.DataFrame, 
                                results: Dict[str, Any]):
        """Plot price with volume and trend lines"""
        try:
            # Price line
            ax.plot(data.index, data['close'], color='#2E86AB', linewidth=2, label='Price')
            
            # Volume trend lines
            volume_trends = results.get('volume_trend_analysis', {})
            
            # Plot trend lines for different timeframes
            colors = {'short_term': '#F24236', 'medium_term': '#FF8C42', 'long_term': '#43AA8B'}
            
            for timeframe, color in colors.items():
                trend_data = volume_trends.get(timeframe, {})
                if trend_data and 'direction' in trend_data:
                    period = trend_data.get('period_days', 20)
                    if len(data) >= period:
                        # Get trend line data
                        recent_data = data.tail(period)
                        x_vals = np.arange(len(recent_data))
                        volumes = recent_data['volume'].values
                        
                        # Linear regression
                        coeffs = np.polyfit(x_vals, volumes, 1)
                        trend_line = np.poly1d(coeffs)(x_vals)
                        
                        # Scale to price range for visualization
                        price_range = data['close'].max() - data['close'].min()
                        volume_range = trend_line.max() - trend_line.min()
                        if volume_range > 0:
                            scaled_trend = ((trend_line - trend_line.min()) / volume_range) * price_range * 0.1
                            scaled_trend += data['close'].min()
                            
                            ax.plot(recent_data.index, scaled_trend, 
                                   color=color, linestyle='--', alpha=0.7,
                                   label=f'{timeframe.replace("_", " ").title()} Trend')
            
            # Volume momentum phase annotations
            cycle_analysis = results.get('cycle_analysis', {})
            current_phase = cycle_analysis.get('current_phase', 'unknown')
            
            if current_phase != 'unknown':
                # Add phase indicator
                phase_colors = {
                    'building': '#90EE90', 'peak': '#FFD700', 
                    'declining': '#FFA07A', 'trough': '#87CEEB',
                    'transitioning': '#DDA0DD'
                }
                phase_color = phase_colors.get(current_phase, '#DDDDDD')
                
                # Add background rectangle for current phase
                y_min, y_max = ax.get_ylim()
                recent_period = min(20, len(data) // 4)
                ax.axvspan(data.index[-recent_period], data.index[-1], 
                          alpha=0.2, color=phase_color, 
                          label=f'Current Phase: {current_phase.title()}')
            
            ax.set_ylabel('Price ($)', fontweight='bold')
            ax.set_title('Price Action with Volume Trend Analysis', fontweight='bold')
            ax.legend(loc='upper left', fontsize=9)
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Price/Volume Chart Error: {str(e)}', 
                   ha='center', va='center', transform=ax.transAxes)
    
    def _plot_summary_dashboard(self, ax: plt.Axes, results: Dict[str, Any], symbol: str):
        """Plot summary dashboard with key metrics"""
        try:
            ax.axis('off')
            
            # Extract key metrics
            trend_direction = results.get('volume_trend_direction', 'unknown')
            trend_strength = results.get('trend_strength', 'unknown')
            momentum_phase = results.get('momentum_phase', 'unknown')
            
            future_implications = results.get('future_implications', {})
            continuation_prob = future_implications.get('trend_continuation_probability', 0)
            exhaustion_warning = future_implications.get('momentum_exhaustion_warning', False)
            
            sustainability = results.get('sustainability_assessment', {})
            sustainability_score = sustainability.get('sustainability_score', 0)
            sustainability_level = sustainability.get('overall_sustainability', 'unknown')
            
            quality = results.get('quality_assessment', {})
            quality_score = quality.get('overall_score', 0)
            
            # Create dashboard text
            dashboard_text = f"""
{symbol} MOMENTUM DASHBOARD
{'='*25}

📊 CURRENT STATUS
Volume Trend: {trend_direction.title()}
Trend Strength: {trend_strength.title()}
Momentum Phase: {momentum_phase.title()}

🔮 FUTURE OUTLOOK
Continuation Probability: {continuation_prob:.1%}
Momentum Exhaustion: {'⚠️ YES' if exhaustion_warning else '✅ NO'}

⏳ SUSTAINABILITY
Overall Rating: {sustainability_level.title()}
Sustainability Score: {sustainability_score:.0f}/100

🎯 ANALYSIS QUALITY
Reliability Score: {quality_score}/100
Analysis Status: ✅ Complete
            """
            
            # Color coding based on metrics
            if continuation_prob > 0.6 and sustainability_score > 60:
                box_color = '#90EE90'  # Light green
            elif continuation_prob < 0.4 or sustainability_score < 40:
                box_color = '#FFA07A'  # Light salmon
            else:
                box_color = '#F0F0F0'  # Light gray
            
            # Add background box
            bbox_props = dict(boxstyle="round,pad=0.3", facecolor=box_color, alpha=0.7)
            ax.text(0.05, 0.95, dashboard_text, transform=ax.transAxes, 
                   verticalalignment='top', fontfamily='monospace',
                   fontsize=9, bbox=bbox_props)
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Dashboard Error: {str(e)}', 
                   ha='center', va='center', transform=ax.transAxes)
    
    def _plot_momentum_indicators(self, ax: plt.Axes, data: pd.DataFrame, 
                                results: Dict[str, Any]):
        """Plot volume momentum indicators"""
        try:
            momentum_analysis = results.get('momentum_analysis', {})
            if 'error' in momentum_analysis:
                ax.text(0.5, 0.5, 'Momentum indicators unavailable', 
                       ha='center', va='center', transform=ax.transAxes)
                return
            
            # Get ROC indicators
            roc_indicators = momentum_analysis.get('rate_of_change_indicators', {})
            
            # Plot different ROC periods
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
            for i, (period, indicator) in enumerate(roc_indicators.items()):
                roc_values = indicator.get('values', [])
                if roc_values:
                    # Create date range for ROC values
                    period_num = int(period.replace('roc_', '').replace('d', ''))
                    roc_dates = data.index[period_num:][:len(roc_values)]
                    
                    ax.plot(roc_dates, roc_values, 
                           color=colors[i % len(colors)], linewidth=2,
                           label=f'ROC {period_num}d', alpha=0.8)
            
            # Add momentum oscillator
            momentum_osc = momentum_analysis.get('momentum_oscillator', {})
            momentum_values = momentum_osc.get('values', [])
            if momentum_values:
                osc_dates = data.index[14:][:len(momentum_values)]  # 14-day momentum
                ax.plot(osc_dates, momentum_values, 
                       color='#8B4513', linewidth=2, 
                       label='Momentum Oscillator', alpha=0.9)
                
                # Add zero line
                ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                
                # Add moving average
                momentum_ma = momentum_osc.get('moving_average', 0)
                ax.axhline(y=momentum_ma, color='red', linestyle='--', alpha=0.7,
                          label=f'MA: {momentum_ma:.0f}')
            
            ax.set_ylabel('Volume Momentum (%)', fontweight='bold')
            ax.set_title('Volume Momentum Indicators', fontweight='bold')
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Momentum Indicators Error: {str(e)}', 
                   ha='center', va='center', transform=ax.transAxes)
    
    def _plot_momentum_cycles(self, ax: plt.Axes, data: pd.DataFrame, 
                            results: Dict[str, Any]):
        """Plot momentum cycles and phases"""
        try:
            cycle_analysis = results.get('cycle_analysis', {})
            momentum_analysis = results.get('momentum_analysis', {})
            
            if 'error' in cycle_analysis or 'error' in momentum_analysis:
                ax.text(0.5, 0.5, 'Cycle analysis unavailable', 
                       ha='center', va='center', transform=ax.transAxes)
                return
            
            # Get base momentum data (ROC 10-day)
            roc_10d = momentum_analysis.get('rate_of_change_indicators', {}).get('roc_10d', {})
            momentum_values = roc_10d.get('values', [])
            
            if momentum_values:
                # Create date range
                momentum_dates = data.index[10:][:len(momentum_values)]
                
                # Plot momentum line
                ax.plot(momentum_dates, momentum_values, 
                       color='#2E86AB', linewidth=2, label='Volume ROC 10d')
                
                # Mark peaks and troughs
                peaks_troughs = cycle_analysis.get('momentum_peaks_troughs', {})
                peaks = peaks_troughs.get('peaks', [])
                troughs = peaks_troughs.get('troughs', [])
                
                # Plot peaks
                for peak in peaks:
                    peak_idx = peak.get('index', 0)
                    if peak_idx < len(momentum_dates):
                        ax.scatter(momentum_dates[peak_idx], peak.get('value', 0), 
                                 color='red', s=100, marker='^', 
                                 label='Peaks' if peak == peaks[0] else "", zorder=5)
                
                # Plot troughs
                for trough in troughs:
                    trough_idx = trough.get('index', 0)
                    if trough_idx < len(momentum_dates):
                        ax.scatter(momentum_dates[trough_idx], trough.get('value', 0), 
                                 color='green', s=100, marker='v', 
                                 label='Troughs' if trough == troughs[0] else "", zorder=5)
                
                # Highlight current phase
                current_phase = cycle_analysis.get('current_phase', 'unknown')
                phase_colors = {
                    'building': '#90EE90', 'peak': '#FFD700', 
                    'declining': '#FFA07A', 'trough': '#87CEEB',
                    'transitioning': '#DDA0DD'
                }
                
                if current_phase in phase_colors:
                    # Highlight recent period
                    recent_period = min(20, len(momentum_dates) // 4)
                    ax.axvspan(momentum_dates[-recent_period], momentum_dates[-1], 
                              alpha=0.3, color=phase_colors[current_phase],
                              label=f'Current: {current_phase.title()}')
            
            # Add zero line
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            ax.set_ylabel('Momentum Cycles', fontweight='bold')
            ax.set_title('Volume Momentum Cycles & Phase Analysis', fontweight='bold')
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Cycle Analysis Error: {str(e)}', 
                   ha='center', va='center', transform=ax.transAxes)
    
    def _plot_future_implications(self, ax: plt.Axes, results: Dict[str, Any]):
        """Plot future implications as horizontal bar chart"""
        try:
            future_implications = results.get('future_implications', {})
            sustainability = results.get('sustainability_assessment', {})
            
            if 'error' in future_implications:
                ax.text(0.5, 0.5, 'Future implications unavailable', 
                       ha='center', va='center', transform=ax.transAxes)
                return
            
            # Prepare metrics for bar chart
            metrics = {
                'Trend Continuation': future_implications.get('trend_continuation_probability', 0) * 100,
                'Sustainability Score': sustainability.get('sustainability_score', 0),
                'Quality Score': results.get('quality_assessment', {}).get('overall_score', 0)
            }
            
            # Create horizontal bar chart
            bars = ax.barh(list(metrics.keys()), list(metrics.values()),
                          color=['#4CAF50', '#2196F3', '#FF9800'], alpha=0.8)
            
            # Add value labels on bars
            for i, (bar, value) in enumerate(zip(bars, metrics.values())):
                ax.text(value + 2, bar.get_y() + bar.get_height()/2, 
                       f'{value:.1f}%' if 'Score' in list(metrics.keys())[i] else f'{value:.1f}%',
                       ha='left', va='center', fontweight='bold', fontsize=10)
            
            # Add warning indicators
            exhaustion_warning = future_implications.get('momentum_exhaustion_warning', False)
            if exhaustion_warning:
                ax.text(0.98, 0.95, '⚠️ MOMENTUM EXHAUSTION WARNING', 
                       transform=ax.transAxes, ha='right', va='top',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.8),
                       fontweight='bold', color='red')
            
            # Acceleration signal
            acc_signal = future_implications.get('volume_acceleration_signal', 'neutral')
            if acc_signal != 'neutral':
                signal_text = acc_signal.replace('_', ' ').title()
                color = 'green' if 'acceleration' in acc_signal else 'red'
                ax.text(0.02, 0.95, f'🚀 {signal_text}', 
                       transform=ax.transAxes, ha='left', va='top',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.3),
                       fontweight='bold')
            
            ax.set_xlim(0, 110)
            ax.set_xlabel('Percentage / Score', fontweight='bold')
            ax.set_title('Future Implications & Key Metrics', fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Future Implications Error: {str(e)}', 
                   ha='center', va='center', transform=ax.transAxes)


def test_volume_momentum_chart_generator():
    """Test function for chart generator"""
    print("🎨 Testing Volume Trend Momentum Chart Generator")
    print("=" * 60)
    
    # Import the processor for test data
    from volume_trend_momentum_processor import VolumeTrendMomentumProcessor
    
    # Create test data
    dates = pd.date_range(start='2024-07-01', end='2024-10-20', freq='D')
    np.random.seed(42)
    
    base_price = 2400
    base_volume = 1500000
    
    # Generate trending price data
    price_changes = np.random.normal(0.003, 0.015, len(dates))
    prices = base_price * np.cumprod(1 + price_changes)
    
    # Generate volume with momentum cycles
    volumes = []
    for i, date in enumerate(dates):
        trend_factor = 1 + (i / len(dates)) * 0.2
        cycle_factor = 1 + 0.3 * np.sin(2 * np.pi * i / 20)
        random_factor = np.random.lognormal(0, 0.3)
        volume = base_volume * trend_factor * cycle_factor * random_factor
        volumes.append(int(volume))
    
    test_data = pd.DataFrame({
        'open': prices + np.random.normal(0, 2, len(dates)),
        'high': prices + np.abs(np.random.normal(8, 4, len(dates))),
        'low': prices - np.abs(np.random.normal(8, 4, len(dates))),
        'close': prices,
        'volume': volumes
    }, index=dates)
    
    # Ensure realistic OHLC relationships
    test_data['high'] = np.maximum(test_data[['open', 'close']].max(axis=1), test_data['high'])
    test_data['low'] = np.minimum(test_data[['open', 'close']].min(axis=1), test_data['low'])
    
    print(f"✅ Created test data: {len(test_data)} days")
    
    # Process data
    processor = VolumeTrendMomentumProcessor()
    results = processor.process_volume_trend_momentum_data(test_data)
    
    if 'error' in results:
        print(f"❌ Analysis failed: {results['error']}")
        return False
    
    print("✅ Analysis completed, generating chart...")
    
    # Generate chart
    chart_generator = VolumeTrendMomentumChartGenerator()
    chart_bytes = chart_generator.generate_volume_momentum_chart(
        test_data, results, "TEST_STOCK", 
        save_path="volume_momentum_test_chart.png"
    )
    
    if chart_bytes:
        print(f"✅ Chart generated successfully: {len(chart_bytes)} bytes")
        print("📁 Chart saved as: volume_momentum_test_chart.png")
        return True
    else:
        print("❌ Chart generation failed")
        return False


# Alias for backwards compatibility
VolumeTrendMomentumCharts = VolumeTrendMomentumChartGenerator

if __name__ == "__main__":
    test_volume_momentum_chart_generator()