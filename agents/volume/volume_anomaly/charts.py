#!/usr/bin/env python3
"""
Volume Anomaly Detection Agent - Chart Generation Module

This module creates specialized visualizations for the Volume Anomaly Detection Agent,
focusing on volume spike detection and anomaly classification charts.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from typing import Dict, List, Any, Optional
import io
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib backend for headless operation
import matplotlib
matplotlib.use('Agg')

class VolumeAnomalyChartGenerator:
    """
    Specialized chart generator for Volume Anomaly Detection Agent
    
    Creates focused volume spike detection and anomaly classification visualizations
    """
    
    def __init__(self):
        self.chart_style = {
            'figure_size': (16, 12),
            'dpi': 300,
            'colors': {
                'normal_volume': '#42a5f5',
                'low_anomaly': '#ffb74d',
                'medium_anomaly': '#ff8a65',
                'high_anomaly': '#e53935',
                'volume_ma': '#37474f',
                'percentile_95': '#f44336',
                'percentile_90': '#ff9800',
                'percentile_75': '#ffc107',
                'background': '#fafafa',
                'grid': '#e0e0e0',
                'text': '#424242'
            }
        }
    
    def generate_volume_anomaly_chart(self, data: pd.DataFrame, 
                                    analysis_data: Dict[str, Any],
                                    stock_symbol: str = "STOCK",
                                    save_path: Optional[str] = None) -> Optional[bytes]:
        """
        Generate volume anomaly detection chart with analysis overlay
        
        Args:
            data: DataFrame with OHLCV data
            analysis_data: Analysis results from VolumeAnomalyProcessor
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
            
            # Create subplot layout: 4 main sections
            gs = fig.add_gridspec(5, 2, height_ratios=[2.5, 2, 1, 1, 0.5], hspace=0.35, wspace=0.15)
            
            # Main volume histogram with anomalies
            ax_volume = fig.add_subplot(gs[0, :])
            # Volume percentile analysis
            ax_percentiles = fig.add_subplot(gs[1, 0])
            # Volume trend analysis
            ax_trends = fig.add_subplot(gs[1, 1])
            # Anomaly timeline
            ax_timeline = fig.add_subplot(gs[2, :])
            # Summary metrics
            ax_summary = fig.add_subplot(gs[3, :])
            # Legend
            ax_legend = fig.add_subplot(gs[4, :])
            
            # Plot main volume analysis with anomalies
            self._plot_volume_with_anomalies(ax_volume, data, analysis_data)
            
            # Plot volume percentile analysis
            self._plot_volume_percentiles(ax_percentiles, data, analysis_data)
            
            # Plot volume trends and patterns
            self._plot_volume_trends(ax_trends, data, analysis_data)
            
            # Plot anomaly timeline
            self._plot_anomaly_timeline(ax_timeline, data, analysis_data)
            
            # Plot summary metrics
            self._plot_anomaly_summary(ax_summary, analysis_data)
            
            # Add legend
            self._add_anomaly_legend(ax_legend, analysis_data)
            
            # Set main title
            current_status = analysis_data.get('current_volume_status', {})
            anomaly_count = len(analysis_data.get('significant_anomalies', []))
            
            fig.suptitle(f'{stock_symbol} - Volume Anomaly Detection Analysis\\n'
                        f'Status: {current_status.get("current_status", "unknown").replace("_", " ").title()} | '
                        f'Anomalies Detected: {anomaly_count}', 
                        fontsize=16, fontweight='bold', y=0.98)
            
            # Convert to bytes
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=self.chart_style['dpi'], 
                       bbox_inches='tight', facecolor='white', edgecolor='none')
            buf.seek(0)
            chart_bytes = buf.getvalue()
            buf.close()
            
            # Save to file if path provided
            if save_path:
                with open(save_path, 'wb') as f:
                    f.write(chart_bytes)
                print(f"💾 Volume Anomaly chart saved: {save_path}")
            
            plt.close(fig)
            return chart_bytes
            
        except Exception as e:
            print(f"❌ Chart generation failed: {e}")
            if 'fig' in locals():
                plt.close(fig)
            return None
    
    def _plot_volume_with_anomalies(self, ax, data: pd.DataFrame, analysis_data: Dict[str, Any]):
        """Plot volume bars with anomaly highlighting"""
        try:
            # Get volume statistics and anomalies
            volume_stats = analysis_data.get('volume_statistics', {})
            anomalies = analysis_data.get('significant_anomalies', [])
            
            if 'error' in volume_stats:
                ax.text(0.5, 0.5, 'Volume statistics unavailable', 
                       transform=ax.transAxes, ha='center', va='center')
                return
            
            # Create anomaly lookup for coloring
            anomaly_dates = {}
            for anomaly in anomalies:
                if 'error' not in anomaly and 'date' in anomaly:
                    anomaly_dates[anomaly['date']] = anomaly['significance']
            
            # Color volume bars based on anomaly significance
            volume_colors = []
            for date in data.index:
                date_str = date.strftime('%Y-%m-%d')
                if date_str in anomaly_dates:
                    significance = anomaly_dates[date_str]
                    if significance == 'high':
                        color = self.chart_style['colors']['high_anomaly']
                    elif significance == 'medium':
                        color = self.chart_style['colors']['medium_anomaly']
                    else:
                        color = self.chart_style['colors']['low_anomaly']
                else:
                    color = self.chart_style['colors']['normal_volume']
                volume_colors.append(color)
            
            # Plot volume bars
            ax.bar(data.index, data['volume'], color=volume_colors, alpha=0.8, width=0.8)
            
            # Add volume moving average
            volume_ma_20 = data['volume'].rolling(window=20).mean()
            ax.plot(data.index, volume_ma_20, 
                   color=self.chart_style['colors']['volume_ma'], 
                   linewidth=2, alpha=0.8, label='Volume MA(20)')
            
            # Add percentile lines for reference
            percentiles = volume_stats.get('percentiles', {})
            if percentiles:
                ax.axhline(y=percentiles.get('percentile_95', 0), 
                          color=self.chart_style['colors']['percentile_95'], 
                          linestyle='--', alpha=0.7, linewidth=1, label='95th Percentile')
                ax.axhline(y=percentiles.get('percentile_75', 0), 
                          color=self.chart_style['colors']['percentile_75'], 
                          linestyle='--', alpha=0.7, linewidth=1, label='75th Percentile')
            
            ax.set_ylabel('Volume', fontsize=12, fontweight='bold')
            ax.set_title('Volume Analysis with Anomaly Detection', fontsize=14, pad=20)
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(True, alpha=0.3, color=self.chart_style['colors']['grid'])
            
            # Format x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
            ax.xaxis.set_major_locator(mdates.WeekdayLocator())
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Volume chart error: {str(e)}', 
                   transform=ax.transAxes, ha='center', va='center')
    
    def _plot_volume_percentiles(self, ax, data: pd.DataFrame, analysis_data: Dict[str, Any]):
        """Plot volume percentile distribution analysis"""
        try:
            volume_stats = analysis_data.get('volume_statistics', {})
            
            if 'error' in volume_stats:
                ax.text(0.5, 0.5, 'Statistics unavailable', 
                       transform=ax.transAxes, ha='center', va='center')
                return
            
            # Get percentile data
            percentiles = volume_stats.get('percentiles', {})
            current_volume = volume_stats.get('current_volume', 0)
            volume_mean = volume_stats.get('volume_mean', 0)
            
            # Create percentile bar chart
            percentile_values = []
            percentile_labels = []
            percentile_colors = []
            
            for p in [50, 75, 90, 95, 99]:
                if f'percentile_{p}' in percentiles:
                    percentile_values.append(percentiles[f'percentile_{p}'])
                    percentile_labels.append(f'{p}th')
                    
                    # Color based on significance
                    if p >= 99:
                        percentile_colors.append(self.chart_style['colors']['high_anomaly'])
                    elif p >= 95:
                        percentile_colors.append(self.chart_style['colors']['medium_anomaly'])
                    elif p >= 90:
                        percentile_colors.append(self.chart_style['colors']['low_anomaly'])
                    else:
                        percentile_colors.append(self.chart_style['colors']['normal_volume'])
            
            # Plot percentile bars
            bars = ax.bar(percentile_labels, percentile_values, color=percentile_colors, alpha=0.7)
            
            # Add current volume line
            ax.axhline(y=current_volume, color='red', linewidth=3, alpha=0.8, 
                      label=f'Current: {current_volume:,.0f}')
            
            # Add mean line
            ax.axhline(y=volume_mean, color='blue', linewidth=2, alpha=0.6,
                      linestyle=':', label=f'Mean: {volume_mean:,.0f}')
            
            ax.set_ylabel('Volume', fontsize=10)
            ax.set_title('Volume Percentile Analysis', fontsize=11)
            ax.legend(fontsize=8, loc='upper left')
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, percentile_values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + max(percentile_values)*0.01,
                       f'{value:,.0f}', ha='center', va='bottom', fontsize=8)
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Percentile error: {str(e)}', 
                   transform=ax.transAxes, ha='center', va='center')
    
    def _plot_volume_trends(self, ax, data: pd.DataFrame, analysis_data: Dict[str, Any]):
        """Plot volume trend analysis"""
        try:
            anomaly_patterns = analysis_data.get('anomaly_patterns', {})
            current_status = analysis_data.get('current_volume_status', {})
            
            # Create trend visualization
            recent_volume = data['volume'].tail(30)  # Last 30 days
            dates = recent_volume.index
            
            # Plot recent volume trend
            ax.plot(dates, recent_volume, color='blue', linewidth=2, marker='o', 
                   markersize=3, alpha=0.7, label='Recent Volume')
            
            # Add trend line
            if len(recent_volume) > 5:
                x_numeric = np.arange(len(recent_volume))
                coeffs = np.polyfit(x_numeric, recent_volume, 1)
                trend_line = np.poly1d(coeffs)(x_numeric)
                
                trend_color = 'green' if coeffs[0] > 0 else 'red'
                ax.plot(dates, trend_line, color=trend_color, linewidth=2, 
                       linestyle='--', alpha=0.8, label=f'Trend ({"↗" if coeffs[0] > 0 else "↘"})')
            
            # Add moving average
            ma_7 = recent_volume.rolling(window=7).mean()
            ax.plot(dates, ma_7, color='orange', linewidth=1.5, alpha=0.8, label='7-day MA')
            
            ax.set_ylabel('Volume', fontsize=10)
            ax.set_title('Recent Volume Trend (30 days)', fontsize=11)
            ax.legend(fontsize=8, loc='upper right')
            ax.grid(True, alpha=0.3)
            
            # Format x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
            ax.xaxis.set_major_locator(mdates.WeekdayLocator())
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, fontsize=8)
            
            # Add trend info text
            trend_info = current_status.get('recent_trend', 'unknown')
            z_score = current_status.get('z_score', 0)
            ax.text(0.05, 0.95, f'Trend: {trend_info}\\nZ-score: {z_score:.2f}', 
                   transform=ax.transAxes, va='top', fontsize=9,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.7))
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Trend error: {str(e)}', 
                   transform=ax.transAxes, ha='center', va='center')
    
    def _plot_anomaly_timeline(self, ax, data: pd.DataFrame, analysis_data: Dict[str, Any]):
        """Plot timeline of anomalies"""
        try:
            anomalies = analysis_data.get('significant_anomalies', [])
            
            if not anomalies or any('error' in a for a in anomalies):
                ax.text(0.5, 0.5, 'No significant anomalies detected', 
                       transform=ax.transAxes, ha='center', va='center')
                ax.set_title('Anomaly Timeline', fontsize=11)
                return
            
            # Plot timeline
            dates = []
            ratios = []
            colors = []
            sizes = []
            
            for anomaly in anomalies:
                if 'error' not in anomaly:
                    dates.append(pd.to_datetime(anomaly['date']))
                    ratios.append(anomaly['volume_ratio'])
                    
                    # Color and size based on significance
                    significance = anomaly['significance']
                    if significance == 'high':
                        colors.append(self.chart_style['colors']['high_anomaly'])
                        sizes.append(100)
                    elif significance == 'medium':
                        colors.append(self.chart_style['colors']['medium_anomaly'])
                        sizes.append(80)
                    else:
                        colors.append(self.chart_style['colors']['low_anomaly'])
                        sizes.append(60)
            
            # Create scatter plot
            scatter = ax.scatter(dates, ratios, c=colors, s=sizes, alpha=0.8, edgecolors='black', linewidth=1)
            
            # Add statistical threshold reference lines
            ax.axhline(y=4.0, color='red', linestyle='--', alpha=0.5, label='High Threshold (4σ)')
            ax.axhline(y=3.0, color='orange', linestyle='--', alpha=0.5, label='Medium Threshold (3σ)')
            ax.axhline(y=2.0, color='yellow', linestyle='--', alpha=0.5, label='Low Threshold (2σ)')
            
            # Annotate top anomalies
            sorted_anomalies = sorted(anomalies, key=lambda x: x.get('volume_ratio', 0), reverse=True)
            for i, anomaly in enumerate(sorted_anomalies[:3]):  # Top 3
                if 'error' not in anomaly:
                    date = pd.to_datetime(anomaly['date'])
                    ratio = anomaly['volume_ratio']
                    ax.annotate(f"{ratio:.1f}x", (date, ratio), 
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=8, ha='left')
            
            ax.set_ylabel('Volume Ratio (x)', fontsize=10)
            ax.set_title('Volume Anomaly Timeline', fontsize=11)
            ax.legend(fontsize=8, loc='upper right')
            ax.grid(True, alpha=0.3)
            
            # Format x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
            ax.xaxis.set_major_locator(mdates.WeekdayLocator())
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, fontsize=8)
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Timeline error: {str(e)}', 
                   transform=ax.transAxes, ha='center', va='center')
    
    def _plot_anomaly_summary(self, ax, analysis_data: Dict[str, Any]):
        """Plot summary metrics and statistics"""
        try:
            ax.axis('off')
            
            # Get key data
            anomaly_patterns = analysis_data.get('anomaly_patterns', {})
            current_status = analysis_data.get('current_volume_status', {})
            volume_stats = analysis_data.get('volume_statistics', {})
            quality_assessment = analysis_data.get('quality_assessment', {})
            anomalies = analysis_data.get('significant_anomalies', [])
            
            # Create summary sections
            summary_sections = []
            
            # Anomaly Overview
            if anomalies and not any('error' in a for a in anomalies):
                high_count = len([a for a in anomalies if a.get('significance') == 'high'])
                medium_count = len([a for a in anomalies if a.get('significance') == 'medium'])
                low_count = len([a for a in anomalies if a.get('significance') == 'low'])
                
                summary_sections.append(f"📊 ANOMALY DETECTION SUMMARY")
                summary_sections.append(f"Total Anomalies: {len(anomalies)}")
                summary_sections.append(f"High Significance: {high_count} | Medium: {medium_count} | Low: {low_count}")
                summary_sections.append(f"Frequency: {anomaly_patterns.get('anomaly_frequency', 'unknown').title()}")
                summary_sections.append(f"Pattern: {anomaly_patterns.get('anomaly_pattern', 'unknown').replace('_', ' ').title()}")
                
                # Top causes
                dominant_causes = anomaly_patterns.get('dominant_causes', [])
                if dominant_causes:
                    causes_str = ', '.join(dominant_causes[:3]).replace('_', ' ').title()
                    summary_sections.append(f"Top Causes: {causes_str}")
                
                summary_sections.append("")
            
            # Current Status
            if 'error' not in current_status:
                summary_sections.append(f"📈 CURRENT VOLUME STATUS")
                summary_sections.append(f"Status: {current_status.get('current_status', 'unknown').replace('_', ' ').title()}")
                summary_sections.append(f"Percentile: {current_status.get('volume_percentile', 0)}th")
                summary_sections.append(f"Z-Score: {current_status.get('z_score', 0):.2f}")
                summary_sections.append(f"vs Mean: {current_status.get('vs_mean_ratio', 1.0):.2f}x")
                summary_sections.append("")
            
            # Statistics Overview
            if 'error' not in volume_stats:
                cv = volume_stats.get('volume_cv', 0)
                summary_sections.append(f"📋 VOLUME STATISTICS")
                summary_sections.append(f"Mean Volume: {volume_stats.get('volume_mean', 0):,.0f}")
                summary_sections.append(f"Volatility (CV): {cv:.2f}")
                summary_sections.append(f"Range Ratio: {volume_stats.get('volume_range', {}).get('range_ratio', 0):.1f}x")
                summary_sections.append("")
            
            # Quality Assessment
            if quality_assessment and 'error' not in quality_assessment:
                summary_sections.append(f"✅ ANALYSIS QUALITY")
                summary_sections.append(f"Overall Score: {quality_assessment.get('overall_score', 0)}/100")
                summary_sections.append(f"Detection Quality: {quality_assessment.get('detection_quality_score', 0)}/40")
                summary_sections.append(f"High Significance Count: {quality_assessment.get('high_significance_count', 0)}")
            
            # Display summary sections
            y_start = 0.95
            line_height = 0.1
            
            for i, section in enumerate(summary_sections):
                font_weight = 'bold' if section.startswith('📊') or section.startswith('📈') or section.startswith('📋') or section.startswith('✅') else 'normal'
                font_size = 11 if section.startswith(('📊', '📈', '📋', '✅')) else 9
                
                ax.text(0.05, y_start - i * line_height, section, 
                       transform=ax.transAxes, fontsize=font_size, 
                       fontweight=font_weight, verticalalignment='top')
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Summary error: {str(e)}', 
                   transform=ax.transAxes, ha='center', va='center')
    
    def _add_anomaly_legend(self, ax, analysis_data: Dict[str, Any]):
        """Add comprehensive legend for anomaly detection"""
        try:
            ax.axis('off')
            
            # Create legend items focused on statistical analysis
            legend_items = [
                ('Volume Bars:', 'Red=High Anomaly (4σ+), Orange=Medium (3σ+), Yellow=Low (2σ+), Blue=Normal'),
                ('Percentile Lines:', '95th (Red), 75th (Yellow) percentiles for statistical reference'),
                ('Timeline Points:', 'Size indicates statistical significance level (z-score based)'),
                ('Analysis Quality:', 'Statistical outlier detection with z-score thresholds'),
                ('Analysis Focus:', 'Retail/market-driven anomalies (institutional analysis handled separately)')
            ]
            
            # Create legend text
            legend_text = "Chart Legend: "
            for item, description in legend_items:
                legend_text += f"{item} {description}  |  "
            
            legend_text = legend_text.rstrip("  |  ")
            
            # Display legend
            ax.text(0.5, 0.5, legend_text, transform=ax.transAxes, 
                   ha='center', va='center', fontsize=9, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.7))
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Legend error: {str(e)}', 
                   transform=ax.transAxes, ha='center', va='center')


def test_volume_anomaly_charts():
    """Test function for Volume Anomaly Chart Generator"""
    print("🎨 Testing Volume Anomaly Chart Generator")
    print("=" * 60)
    
    # Import and use the processor for test data
    from volume_anomaly_processor import VolumeAnomalyProcessor
    
    # Create sample data with volume spikes
    dates = pd.date_range(start='2024-07-01', end='2024-09-20', freq='D')
    np.random.seed(42)
    
    base_price = 2400
    base_volume = 1800000
    
    # Generate price data
    price_changes = np.random.normal(0.001, 0.018, len(dates))
    prices = base_price * np.cumprod(1 + price_changes)
    
    # Generate volume data with intentional spikes
    volumes = np.random.lognormal(np.log(base_volume), 0.6, len(dates))
    
    # Add volume spikes at specific dates
    spike_indices = [15, 35, 55, 70]  # Days with spikes
    for idx in spike_indices:
        if idx < len(volumes):
            volumes[idx] *= np.random.uniform(2.8, 5.5)
    
    sample_data = pd.DataFrame({
        'open': prices + np.random.normal(0, 3, len(dates)),
        'high': prices + np.abs(np.random.normal(8, 5, len(dates))),
        'low': prices - np.abs(np.random.normal(8, 5, len(dates))),
        'close': prices,
        'volume': volumes.astype(int)
    }, index=dates)
    
    # Ensure realistic OHLC relationships
    sample_data['high'] = np.maximum(sample_data[['open', 'close']].max(axis=1), sample_data['high'])
    sample_data['low'] = np.minimum(sample_data[['open', 'close']].min(axis=1), sample_data['low'])
    
    print(f"✅ Created sample data: {len(sample_data)} days")
    print(f"   Volume range: {sample_data['volume'].min():,} - {sample_data['volume'].max():,}")
    
    # Process data
    processor = VolumeAnomalyProcessor()
    analysis_data = processor.process_volume_anomaly_data(sample_data)
    
    if 'error' in analysis_data:
        print(f"❌ Data processing failed: {analysis_data['error']}")
        return False
    
    print("✅ Data processing completed")
    print(f"   Anomalies detected: {len(analysis_data.get('significant_anomalies', []))}")
    
    # Generate chart
    chart_generator = VolumeAnomalyChartGenerator()
    chart_bytes = chart_generator.generate_volume_anomaly_chart(
        sample_data, analysis_data, "TEST_STOCK", "test_volume_anomaly_chart.png"
    )
    
    if chart_bytes:
        print(f"✅ Chart generated successfully: {len(chart_bytes)} bytes")
        print("💾 Chart saved as: test_volume_anomaly_chart.png")
        
        # Display analysis summary
        patterns = analysis_data.get('anomaly_patterns', {})
        current_status = analysis_data.get('current_volume_status', {})
        
        print(f"\\n📊 Analysis Results:")
        print(f"   Anomaly frequency: {patterns.get('anomaly_frequency', 'unknown')}")
        print(f"   Current status: {current_status.get('current_status', 'unknown')}")
        print(f"   Volume percentile: {current_status.get('volume_percentile', 0)}th")
        
        return True
    else:
        print("❌ Chart generation failed")
        return False

# Alias for backwards compatibility
VolumeAnomalyCharts = VolumeAnomalyChartGenerator

if __name__ == "__main__":
    test_volume_anomaly_charts()
