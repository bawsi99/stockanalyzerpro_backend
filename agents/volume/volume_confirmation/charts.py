#!/usr/bin/env python3
"""
Volume Confirmation Agent - Chart Generation Module

This module creates specialized visualizations for the Volume Confirmation Agent,
focusing on price-volume relationship charts with confirmation indicators.
"""

import pandas as pd
import numpy as np
# Set matplotlib backend for headless operation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from typing import Dict, List, Any, Optional
import io
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class VolumeConfirmationChartGenerator:
    """
    Specialized chart generator for Volume Confirmation Agent
    
    Creates focused price-volume relationship visualizations
    """
    
    def __init__(self):
        self.chart_style = {
            'figure_size': (16, 12),
            'dpi': 300,
            'colors': {
                'price_up': '#26a69a',
                'price_down': '#ef5350', 
                'volume': '#42a5f5',
                'volume_ma': '#ff7043',
                'confirmation': '#4caf50',
                'divergence': '#f44336',
                'background': '#fafafa',
                'grid': '#e0e0e0',
                'text': '#424242'
            }
        }
    
    def generate_volume_confirmation_chart(self, data: pd.DataFrame, 
                                         analysis_data: Dict[str, Any],
                                         stock_symbol: str = "STOCK",
                                         save_path: Optional[str] = None) -> Optional[bytes]:
        """
        Generate volume confirmation chart with analysis overlay
        
        Args:
            data: DataFrame with OHLCV data
            analysis_data: Analysis results from VolumeConfirmationProcessor
            stock_symbol: Stock symbol for chart title
            save_path: Optional path to save chart file
            
        Returns:
            Chart as bytes for LLM analysis, or None if generation fails
        """
        try:
            plt.style.use('default')
            fig = plt.figure(figsize=self.chart_style['figure_size'], 
                           dpi=self.chart_style['dpi'], 
                           facecolor='white')
            
            # Create subplot layout
            gs = fig.add_gridspec(4, 2, height_ratios=[3, 2, 1.5, 0.5], hspace=0.3, wspace=0.15)
            
            # Main price-volume chart
            ax_price = fig.add_subplot(gs[0, :])
            ax_volume = fig.add_subplot(gs[1, :], sharex=ax_price)
            ax_correlation = fig.add_subplot(gs[2, 0])
            ax_summary = fig.add_subplot(gs[2, 1])
            ax_legend = fig.add_subplot(gs[3, :])
            
            # Plot price action with volume confirmation indicators
            self._plot_price_with_confirmations(ax_price, data, analysis_data)
            
            # Plot volume with moving averages
            self._plot_volume_analysis(ax_volume, data, analysis_data)
            
            # Plot correlation analysis
            self._plot_correlation_analysis(ax_correlation, data, analysis_data)
            
            # Plot summary metrics
            self._plot_summary_metrics(ax_summary, analysis_data)
            
            # Add legend and labels
            self._add_chart_legend(ax_legend, analysis_data)
            
            # Set main title
            overall_assessment = analysis_data.get('overall_assessment', {})
            confirmation_status = overall_assessment.get('confirmation_status', 'unknown')
            confidence = overall_assessment.get('confidence_score', 0)
            
            fig.suptitle(f'{stock_symbol} - Volume Confirmation Analysis\n'
                        f'Status: {confirmation_status.replace("_", " ").title()} '
                        f'(Confidence: {confidence}%)', 
                        fontsize=16, fontweight='bold', y=0.98)
            
            # Convert to bytes
            buf = io.BytesIO()
            fig.canvas.draw()
            fig.savefig(buf, format='png', dpi=self.chart_style['dpi'], 
                       bbox_inches='tight', facecolor='white', edgecolor='none')
            buf.seek(0)
            chart_bytes = buf.getvalue()
            buf.close()
            
            # Save to file if path provided
            if save_path:
                with open(save_path, 'wb') as f:
                    f.write(chart_bytes)
                print(f"💾 Volume Confirmation chart saved: {save_path}")
            
            plt.close(fig)
            return chart_bytes
            
        except Exception as e:
            print(f"❌ Chart generation failed: {e}")
            if 'fig' in locals():
                plt.close(fig)
            return None
    
    def _plot_price_with_confirmations(self, ax, data: pd.DataFrame, analysis_data: Dict[str, Any]):
        """Plot price action with volume confirmation indicators"""
        try:
            # Plot candlestick-style price chart
            for i in range(len(data)):
                row = data.iloc[i]
                date = row.name
                
                # Determine color based on price direction
                color = self.chart_style['colors']['price_up'] if row['close'] >= row['open'] else self.chart_style['colors']['price_down']
                
                # Plot price bar
                ax.plot([date, date], [row['low'], row['high']], color=color, linewidth=1.5, alpha=0.8)
                ax.plot([date, date], [row['open'], row['close']], color=color, linewidth=4, alpha=0.9)
            
            # Add volume moving average
            volume_ma_20 = data['volume'].rolling(window=20).mean()
            ax2 = ax.twinx()
            ax2.fill_between(data.index, volume_ma_20, alpha=0.2, 
                           color=self.chart_style['colors']['volume_ma'], label='Volume MA(20)')
            ax2.set_ylabel('Volume', fontsize=10, color=self.chart_style['colors']['text'])
            
            # Highlight confirmation/divergence events
            recent_movements = analysis_data.get('recent_movements', [])
            for movement in recent_movements:
                try:
                    movement_date = pd.to_datetime(movement['date'])
                    if movement_date in data.index:
                        price = data.loc[movement_date, 'close']
                        
                        if movement['volume_response'] == 'confirming':
                            ax.scatter(movement_date, price, 
                                     color=self.chart_style['colors']['confirmation'], 
                                     s=100, marker='^', zorder=5)
                        elif movement['volume_response'] == 'diverging':
                            ax.scatter(movement_date, price, 
                                     color=self.chart_style['colors']['divergence'], 
                                     s=100, marker='v', zorder=5)
                except Exception:
                    continue
            
            ax.set_ylabel('Price (₹)', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, color=self.chart_style['colors']['grid'])
            ax.set_title('Price Action with Volume Confirmation Signals', fontsize=14, pad=20)
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Price chart error: {str(e)}', 
                   transform=ax.transAxes, ha='center', va='center')
    
    def _plot_volume_analysis(self, ax, data: pd.DataFrame, analysis_data: Dict[str, Any]):
        """Plot volume analysis with moving averages"""
        try:
            # Plot volume bars
            volume_colors = []
            volume_ma_20 = data['volume'].rolling(window=20).mean()
            
            for i in range(len(data)):
                volume_ratio = data['volume'].iloc[i] / volume_ma_20.iloc[i] if volume_ma_20.iloc[i] > 0 else 1.0
                
                if volume_ratio > 1.5:
                    color = self.chart_style['colors']['confirmation']  # High volume
                elif volume_ratio < 0.7:
                    color = self.chart_style['colors']['divergence']  # Low volume
                else:
                    color = self.chart_style['colors']['volume']  # Normal volume
                
                volume_colors.append(color)
            
            # Plot volume bars with color coding
            ax.bar(data.index, data['volume'], color=volume_colors, alpha=0.7, width=0.8)
            
            # Plot volume moving averages
            ax.plot(data.index, volume_ma_20, 
                   color=self.chart_style['colors']['volume_ma'], 
                   linewidth=2, label='Volume MA(20)')
            
            # Add volume ratio indicators
            volume_averages = analysis_data.get('volume_averages', {})
            current_volume = volume_averages.get('current_volume', 0)
            volume_20d_avg = volume_averages.get('volume_20d_avg', 0)
            
            if volume_20d_avg > 0:
                ratio = current_volume / volume_20d_avg
                ax.axhline(y=volume_20d_avg, color='orange', linestyle='--', alpha=0.7, 
                          label=f'Current Ratio: {ratio:.2f}x')
            
            ax.set_ylabel('Volume', fontsize=12, fontweight='bold')
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(True, alpha=0.3, color=self.chart_style['colors']['grid'])
            ax.set_title('Volume Analysis with Confirmation Indicators', fontsize=12)
            
            # Format x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
            ax.xaxis.set_major_locator(mdates.WeekdayLocator())
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Volume chart error: {str(e)}', 
                   transform=ax.transAxes, ha='center', va='center')
    
    def _plot_correlation_analysis(self, ax, data: pd.DataFrame, analysis_data: Dict[str, Any]):
        """Plot price-volume correlation analysis"""
        try:
            correlation_data = analysis_data.get('price_volume_correlation', {})
            
            if 'error' in correlation_data:
                ax.text(0.5, 0.5, 'Correlation analysis unavailable', 
                       transform=ax.transAxes, ha='center', va='center')
                return
            
            # Calculate rolling correlation for visualization
            data_copy = data.copy()
            data_copy['price_change'] = data_copy['close'].pct_change()
            data_copy['volume_change'] = data_copy['volume'].pct_change()
            
            rolling_corr = data_copy['price_change'].rolling(window=20).corr(data_copy['volume_change'])
            
            # Plot correlation over time
            ax.plot(data.index, rolling_corr, color='blue', linewidth=2, label='Rolling Correlation')
            ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
            ax.axhline(y=0.3, color='green', linestyle='--', alpha=0.7, label='Strong (+)')
            ax.axhline(y=-0.3, color='red', linestyle='--', alpha=0.7, label='Strong (-)')
            
            # Highlight current correlation
            current_corr = correlation_data.get('correlation_coefficient', 0)
            ax.axhline(y=current_corr, color='orange', linewidth=3, alpha=0.8, 
                      label=f'Current: {current_corr:.3f}')
            
            ax.set_ylabel('Correlation', fontsize=10)
            ax.set_title('Price-Volume Correlation', fontsize=11)
            ax.legend(fontsize=8, loc='upper right')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-1, 1)
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Correlation error: {str(e)}', 
                   transform=ax.transAxes, ha='center', va='center')
    
    def _plot_summary_metrics(self, ax, analysis_data: Dict[str, Any]):
        """Plot summary metrics and assessment"""
        try:
            ax.axis('off')
            
            # Get key metrics
            overall_assessment = analysis_data.get('overall_assessment', {})
            correlation_data = analysis_data.get('price_volume_correlation', {})
            trend_support = analysis_data.get('trend_support', {})
            
            # Create summary text
            summary_lines = []
            
            # Overall assessment
            status = overall_assessment.get('confirmation_status', 'unknown')
            strength = overall_assessment.get('confirmation_strength', 'unknown')
            confidence = overall_assessment.get('confidence_score', 0)
            
            summary_lines.append(f"📊 VOLUME CONFIRMATION SUMMARY")
            summary_lines.append(f"")
            summary_lines.append(f"Overall Status: {status.replace('_', ' ').title()}")
            summary_lines.append(f"Confirmation Strength: {strength.title()}")
            summary_lines.append(f"Confidence Score: {confidence}%")
            summary_lines.append(f"")
            
            # Correlation metrics
            if 'error' not in correlation_data:
                corr_coef = correlation_data.get('correlation_coefficient', 0)
                corr_strength = correlation_data.get('correlation_strength', 'unknown')
                summary_lines.append(f"Price-Volume Correlation: {corr_coef:.3f} ({corr_strength})")
            
            # Trend support
            if 'error' not in trend_support:
                current_trend = trend_support.get('current_trend', 'unknown')
                uptrend_support = trend_support.get('uptrend_volume_support', 'unknown')
                summary_lines.append(f"Current Trend: {current_trend.title()}")
                summary_lines.append(f"Volume Support: {uptrend_support.title()}")
            
            # Recent confirmations
            recent_movements = analysis_data.get('recent_movements', [])
            confirming_count = len([m for m in recent_movements if m.get('volume_response') == 'confirming'])
            diverging_count = len([m for m in recent_movements if m.get('volume_response') == 'diverging'])
            
            summary_lines.append(f"")
            summary_lines.append(f"Recent Signals:")
            summary_lines.append(f"  Confirming: {confirming_count}")
            summary_lines.append(f"  Diverging: {diverging_count}")
            
            # Display summary text
            y_start = 0.95
            for i, line in enumerate(summary_lines):
                font_weight = 'bold' if line.startswith('📊') or line.startswith('Overall') else 'normal'
                font_size = 12 if line.startswith('📊') else 10
                
                ax.text(0.05, y_start - i * 0.08, line, 
                       transform=ax.transAxes, fontsize=font_size, 
                       fontweight=font_weight, verticalalignment='top')
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Summary error: {str(e)}', 
                   transform=ax.transAxes, ha='center', va='center')
    
    def _add_chart_legend(self, ax, analysis_data: Dict[str, Any]):
        """Add comprehensive legend for chart symbols"""
        try:
            ax.axis('off')
            
            legend_items = [
                ('Price Bars:', 'Green=Up, Red=Down'),
                ('Volume Bars:', 'Green=High Vol, Red=Low Vol, Blue=Normal'),
                ('Signals:', '▲=Volume Confirms, ▼=Volume Diverges'),
                ('Lines:', 'Orange=Volume MA, Blue=Correlation'),
            ]
            
            # Create legend text
            legend_text = "Chart Legend: "
            for item, description in legend_items:
                legend_text += f"{item} {description}  |  "
            
            legend_text = legend_text.rstrip("  |  ")
            
            ax.text(0.5, 0.5, legend_text, transform=ax.transAxes, 
                   ha='center', va='center', fontsize=9, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.7))
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Legend error: {str(e)}', 
                   transform=ax.transAxes, ha='center', va='center')


def test_volume_confirmation_charts():
    """Test function for Volume Confirmation Chart Generator"""
    print("🎨 Testing Volume Confirmation Chart Generator")
    print("=" * 60)
    
    # Import and use the processor for test data
    from volume_confirmation_processor import VolumeConfirmationProcessor
    
    # Create sample data
    dates = pd.date_range(start='2024-08-01', end='2024-09-20', freq='D')
    np.random.seed(42)
    
    base_price = 2450
    price_trend = np.cumsum(np.random.normal(0.8, 12, len(dates)))
    prices = base_price + price_trend
    
    sample_data = pd.DataFrame({
        'open': prices + np.random.normal(0, 4, len(dates)),
        'high': prices + np.abs(np.random.normal(8, 6, len(dates))),
        'low': prices - np.abs(np.random.normal(8, 6, len(dates))),
        'close': prices,
        'volume': np.abs(np.random.lognormal(14.3, 0.7, len(dates)))
    }, index=dates)
    
    # Ensure realistic OHLC relationships
    sample_data['high'] = np.maximum(sample_data[['open', 'close']].max(axis=1), sample_data['high'])
    sample_data['low'] = np.minimum(sample_data[['open', 'close']].min(axis=1), sample_data['low'])
    
    print(f"✅ Created sample data: {len(sample_data)} days")
    
    # Process data
    processor = VolumeConfirmationProcessor()
    analysis_data = processor.process_volume_confirmation_data(sample_data)
    
    if 'error' in analysis_data:
        print(f"❌ Data processing failed: {analysis_data['error']}")
        return False
    
    print("✅ Data processing completed")
    
    # Generate chart
    chart_generator = VolumeConfirmationChartGenerator()
    chart_bytes = chart_generator.generate_volume_confirmation_chart(
        sample_data, analysis_data, "TEST_STOCK", "test_volume_confirmation_chart.png"
    )
    
    if chart_bytes:
        print(f"✅ Chart generated successfully: {len(chart_bytes)} bytes")
        print("💾 Chart saved as: test_volume_confirmation_chart.png")
        
        # Display key analysis results
        overall = analysis_data.get('overall_assessment', {})
        print(f"\n📊 Analysis Results:")
        print(f"   Status: {overall.get('confirmation_status', 'unknown')}")
        print(f"   Strength: {overall.get('confirmation_strength', 'unknown')}")
        print(f"   Confidence: {overall.get('confidence_score', 0)}%")
        
        return True
    else:
        print("❌ Chart generation failed")
        return False

# Alias for backwards compatibility
VolumeConfirmationCharts = VolumeConfirmationChartGenerator

if __name__ == "__main__":
    test_volume_confirmation_charts()