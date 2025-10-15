#!/usr/bin/env python3
"""
Enhanced Market Structure Chart Generator

This module extends the basic chart generator with advanced visual elements:
- Fibonacci retracements between major swing points
- Trend channels connecting swing highs and lows
- Structure break lines through BOS/CHOCH points
- Price labels on important swing points
- Time-based phase annotations

Enhanced Visual Features:
1. Fibonacci Retracements (0.236, 0.382, 0.5, 0.618, 0.786)
2. Trend Channels with parallel lines
3. BOS/CHOCH break lines with direction indicators
4. Enhanced swing point labeling
5. Market phase highlighting
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional
import os
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedMarketStructureCharts:
    """
    Enhanced chart generator with advanced market structure visualizations.
    """
    
    def __init__(self, output_dir: str = "enhanced_charts"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Enhanced color scheme
        self.colors = {
            'price': '#1f77b4',
            'price_fill': '#1f77b4',
            'swing_high': '#ff4444', 
            'swing_low': '#44ff44',
            'bos_bullish': '#00aa00',
            'bos_bearish': '#aa0000',
            'choch_bullish': '#66cc66',
            'choch_bearish': '#cc6666',
            'support': '#44ff44',
            'resistance': '#ff4444',
            'trend_up': '#00aa00',
            'trend_down': '#aa0000',
            'neutral': '#888888',
            'volume_up': '#2ca02c',
            'volume_down': '#d62728',
            # New colors for enhanced features
            'fibonacci': '#ffa500',
            'trend_channel': '#9370db',
            'break_line': '#ff6347',
            'phase_highlight': '#ffd700',
            'price_label': '#000000'
        }
        
        # Fibonacci levels
        self.fib_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
        
        plt.style.use('seaborn-v0_8-darkgrid')
    
    def create_enhanced_chart(self, 
                            stock_data: pd.DataFrame, 
                            analysis_data: Dict[str, Any], 
                            symbol: str, 
                            scenario: str) -> str:
        """
        Create enhanced market structure chart with advanced visual elements.
        """
        try:
            logger.info(f"Generating enhanced chart for {symbol} - {scenario}")
            
            # Create figure with enhanced layout
            fig = plt.figure(figsize=(24, 18))
            gs = fig.add_gridspec(4, 3, height_ratios=[4, 1, 1, 1], width_ratios=[4, 1, 1], 
                                hspace=0.3, wspace=0.2)
            
            # Main price chart (larger)
            ax_main = fig.add_subplot(gs[0, :2])
            
            # Fibonacci analysis panel
            ax_fib = fig.add_subplot(gs[0, 2])
            
            # Volume chart
            ax_volume = fig.add_subplot(gs[1, :2], sharex=ax_main)
            
            # Analysis metrics
            ax_metrics1 = fig.add_subplot(gs[1, 2])
            ax_metrics2 = fig.add_subplot(gs[2, 0])
            ax_metrics3 = fig.add_subplot(gs[2, 1])
            
            # Enhanced summary
            ax_summary = fig.add_subplot(gs[3, :])
            
            # Set main title
            fig.suptitle(f'Enhanced Market Structure Analysis - {symbol} ({scenario})', 
                        fontsize=20, fontweight='bold', y=0.98)
            
            # Plot all enhanced components
            self._plot_enhanced_price_structure(ax_main, stock_data, analysis_data)
            self._plot_fibonacci_analysis(ax_fib, analysis_data)
            self._plot_volume_analysis(ax_volume, stock_data, analysis_data)
            self._plot_swing_metrics(ax_metrics1, analysis_data)
            self._plot_structure_metrics(ax_metrics2, analysis_data)
            self._plot_fibonacci_metrics(ax_metrics3, analysis_data)
            self._plot_enhanced_summary(ax_summary, analysis_data, scenario)
            
            # Format and save
            self._format_enhanced_axes(ax_main, ax_volume, stock_data)
            
            # Save chart
            filename = f"{symbol}_{scenario}_enhanced_structure.png"
            filepath = self.output_dir / filename
            
            plt.savefig(filepath, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close(fig)
            
            logger.info(f"Enhanced chart saved: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Enhanced chart generation failed: {e}")
            plt.close('all')
            return None
    
    def _plot_enhanced_price_structure(self, ax, stock_data: pd.DataFrame, analysis_data: Dict):
        """Plot enhanced price structure with all advanced visual elements"""
        
        dates = pd.to_datetime(stock_data.index)
        
        # Plot price action
        ax.plot(dates, stock_data['close'], color=self.colors['price'], 
               linewidth=3, label='Close Price', zorder=5)
        
        # Fill between high and low
        ax.fill_between(dates, stock_data['low'], stock_data['high'], 
                       alpha=0.1, color=self.colors['price_fill'], zorder=1)
        
        # 1. Enhanced swing points with price labels
        self._plot_enhanced_swing_points(ax, analysis_data)
        
        # 2. Fibonacci retracements
        self._plot_fibonacci_retracements(ax, analysis_data)
        
        # 3. Trend channels
        self._plot_trend_channels(ax, analysis_data)
        
        # 4. Structure break lines
        self._plot_structure_break_lines(ax, analysis_data)
        
        # 5. Market phase highlighting
        self._plot_market_phases(ax, stock_data, analysis_data)
        
        # 6. Enhanced support/resistance levels
        self._plot_enhanced_levels(ax, analysis_data)
        
        # Add current price indicator
        current_price = stock_data['close'].iloc[-1]
        ax.axhline(y=current_price, color='black', linestyle=':', 
                  alpha=0.8, linewidth=3, label=f'Current: ${current_price:.2f}')
        
        ax.set_title('Enhanced Price Action & Market Structure', fontsize=16, fontweight='bold')
        ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
        ax.grid(True, alpha=0.3)
    
    def _plot_enhanced_swing_points(self, ax, analysis_data: Dict):
        """Enhanced swing point visualization with price labels and connections"""
        
        swing_points = analysis_data.get('swing_points', {})
        swing_highs = swing_points.get('swing_highs', [])
        swing_lows = swing_points.get('swing_lows', [])
        
        # Plot swing highs with enhanced styling
        high_dates, high_prices, high_strengths = [], [], []
        for swing in swing_highs:
            try:
                high_dates.append(pd.to_datetime(swing['date']))
                high_prices.append(swing['price'])
                high_strengths.append(swing['strength'])
            except:
                continue
        
        if high_dates:
            # Enhanced sizing based on strength
            sizes = [150 if s == 'strong' else 100 if s == 'medium' else 60 
                    for s in high_strengths]
            
            scatter = ax.scatter(high_dates, high_prices, c=self.colors['swing_high'], 
                               s=sizes, marker='^', alpha=0.9, edgecolors='black', 
                               linewidth=2, label='Swing Highs', zorder=7)
            
            # Add price labels for strong swings
            for date, price, strength in zip(high_dates, high_prices, high_strengths):
                if strength in ['strong', 'medium']:
                    ax.annotate(f'${price:.2f}', (date, price), 
                              xytext=(5, 15), textcoords='offset points',
                              fontsize=9, fontweight='bold', 
                              color=self.colors['price_label'],
                              bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                       alpha=0.8, edgecolor=self.colors['swing_high']))
            
            # Connect swing highs with enhanced trend line
            if len(high_dates) > 1:
                ax.plot(high_dates, high_prices, color=self.colors['swing_high'], 
                       linestyle='--', alpha=0.6, linewidth=2.5, zorder=3)
        
        # Plot swing lows with enhanced styling
        low_dates, low_prices, low_strengths = [], [], []
        for swing in swing_lows:
            try:
                low_dates.append(pd.to_datetime(swing['date']))
                low_prices.append(swing['price'])
                low_strengths.append(swing['strength'])
            except:
                continue
        
        if low_dates:
            sizes = [150 if s == 'strong' else 100 if s == 'medium' else 60 
                    for s in low_strengths]
            
            ax.scatter(low_dates, low_prices, c=self.colors['swing_low'], 
                      s=sizes, marker='v', alpha=0.9, edgecolors='black', 
                      linewidth=2, label='Swing Lows', zorder=7)
            
            # Add price labels for strong swings
            for date, price, strength in zip(low_dates, low_prices, low_strengths):
                if strength in ['strong', 'medium']:
                    ax.annotate(f'${price:.2f}', (date, price), 
                              xytext=(5, -20), textcoords='offset points',
                              fontsize=9, fontweight='bold',
                              color=self.colors['price_label'],
                              bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                       alpha=0.8, edgecolor=self.colors['swing_low']))
            
            # Connect swing lows with enhanced trend line
            if len(low_dates) > 1:
                ax.plot(low_dates, low_prices, color=self.colors['swing_low'], 
                       linestyle='--', alpha=0.6, linewidth=2.5, zorder=3)
    
    def _plot_fibonacci_retracements(self, ax, analysis_data: Dict):
        """Plot Fibonacci retracements between major swing points"""
        
        swing_points = analysis_data.get('swing_points', {})
        swing_highs = swing_points.get('swing_highs', [])
        swing_lows = swing_points.get('swing_lows', [])
        
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return
        
        # Find the most significant swing high and low for Fibonacci calculation
        # Use the strongest swings or most recent major move
        strong_highs = [s for s in swing_highs if s.get('strength') == 'strong']
        strong_lows = [s for s in swing_lows if s.get('strength') == 'strong']
        
        if not strong_highs or not strong_lows:
            # Fallback to medium strength
            strong_highs = [s for s in swing_highs if s.get('strength') in ['strong', 'medium']]
            strong_lows = [s for s in swing_lows if s.get('strength') in ['strong', 'medium']]
        
        if not strong_highs or not strong_lows:
            return
        
        # Get the most recent significant high and low
        recent_high = max(strong_highs, key=lambda x: pd.to_datetime(x['date']))
        recent_low = max(strong_lows, key=lambda x: pd.to_datetime(x['date']))
        
        high_price = recent_high['price']
        low_price = recent_low['price']
        price_range = high_price - low_price
        
        xlim = ax.get_xlim()
        
        # Plot Fibonacci retracement levels
        for level in self.fib_levels:
            if high_price > low_price:  # Uptrend retracement
                fib_price = high_price - (price_range * level)
                label = f'Fib {level:.1%} (${fib_price:.2f})'
            else:  # Downtrend retracement  
                fib_price = low_price + (price_range * level)
                label = f'Fib {level:.1%} (${fib_price:.2f})'
            
            # Draw Fibonacci level
            ax.axhline(y=fib_price, color=self.colors['fibonacci'], 
                      linestyle=':', alpha=0.7, linewidth=1.5, zorder=2)
            
            # Add Fibonacci level label
            ax.text(xlim[1], fib_price, f'  {level:.1%}', 
                   verticalalignment='center', fontsize=8,
                   color=self.colors['fibonacci'], fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                            alpha=0.7, edgecolor=self.colors['fibonacci']))
        
        # Add Fibonacci range annotation
        mid_date = pd.to_datetime(recent_high['date']) if recent_high['date'] > recent_low['date'] else pd.to_datetime(recent_low['date'])
        ax.annotate(f'Fib Range: ${low_price:.2f} - ${high_price:.2f}', 
                   (mid_date, (high_price + low_price) / 2),
                   xytext=(20, 0), textcoords='offset points',
                   fontsize=10, fontweight='bold', 
                   color=self.colors['fibonacci'],
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                            alpha=0.9, edgecolor=self.colors['fibonacci']),
                   zorder=6)
    
    def _plot_trend_channels(self, ax, analysis_data: Dict):
        """Plot trend channels connecting swing highs and lows"""
        
        swing_points = analysis_data.get('swing_points', {})
        swing_highs = swing_points.get('swing_highs', [])
        swing_lows = swing_points.get('swing_lows', [])
        
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return
        
        # Convert dates and sort by date
        high_points = [(pd.to_datetime(s['date']), s['price']) for s in swing_highs]
        low_points = [(pd.to_datetime(s['date']), s['price']) for s in swing_lows]
        
        high_points.sort(key=lambda x: x[0])
        low_points.sort(key=lambda x: x[0])
        
        if len(high_points) >= 2:
            # Calculate trend line for highs
            high_dates = [p[0] for p in high_points]
            high_prices = [p[1] for p in high_points]
            
            # Use linear regression for trend line
            x_high = np.array([d.timestamp() for d in high_dates])
            y_high = np.array(high_prices)
            
            if len(x_high) >= 2:
                slope_high, intercept_high = np.polyfit(x_high, y_high, 1)
                
                # Extend trend line across chart
                xlim = ax.get_xlim()
                x_extended = np.linspace(xlim[0], xlim[1], 100)
                x_timestamps = [pd.to_datetime(x, unit='D', origin='1970-01-01').timestamp() for x in x_extended]
                y_trend_high = slope_high * np.array(x_timestamps) + intercept_high
                
                # Plot upper trend line
                ax.plot([pd.to_datetime(x, unit='D', origin='1970-01-01') for x in x_extended], 
                       y_trend_high, color=self.colors['trend_channel'], 
                       linestyle='-', alpha=0.7, linewidth=2, 
                       label='Upper Channel', zorder=3)
        
        if len(low_points) >= 2:
            # Calculate trend line for lows
            low_dates = [p[0] for p in low_points]
            low_prices = [p[1] for p in low_points]
            
            x_low = np.array([d.timestamp() for d in low_dates])
            y_low = np.array(low_prices)
            
            if len(x_low) >= 2:
                slope_low, intercept_low = np.polyfit(x_low, y_low, 1)
                
                # Extend trend line across chart
                xlim = ax.get_xlim()
                x_extended = np.linspace(xlim[0], xlim[1], 100)
                x_timestamps = [pd.to_datetime(x, unit='D', origin='1970-01-01').timestamp() for x in x_extended]
                y_trend_low = slope_low * np.array(x_timestamps) + intercept_low
                
                # Plot lower trend line
                ax.plot([pd.to_datetime(x, unit='D', origin='1970-01-01') for x in x_extended], 
                       y_trend_low, color=self.colors['trend_channel'], 
                       linestyle='-', alpha=0.7, linewidth=2,
                       label='Lower Channel', zorder=3)
                
                # Fill channel area if both trend lines exist
                if 'slope_high' in locals():
                    ax.fill_between([pd.to_datetime(x, unit='D', origin='1970-01-01') for x in x_extended], 
                                   y_trend_low, y_trend_high, 
                                   color=self.colors['trend_channel'], alpha=0.1, zorder=1)
    
    def _plot_structure_break_lines(self, ax, analysis_data: Dict):
        """Plot structure break lines through BOS/CHOCH points"""
        
        bos_choch = analysis_data.get('bos_choch_analysis', {})
        bos_events = bos_choch.get('bos_events', [])
        choch_events = bos_choch.get('choch_events', [])
        
        # Plot BOS break lines
        for i, bos in enumerate(bos_events):
            try:
                date = pd.to_datetime(bos['date'])
                price = bos['break_price']
                bos_type = bos['type']
                strength = bos.get('strength', 'medium')
                
                is_bullish = 'bullish' in bos_type
                color = self.colors['bos_bullish'] if is_bullish else self.colors['bos_bearish']
                arrow = '↑' if is_bullish else '↓'
                
                # Draw break line across the chart
                xlim = ax.get_xlim()
                line_style = '-' if strength == 'strong' else '--'
                line_width = 3 if strength == 'strong' else 2
                
                ax.axhline(y=price, color=color, linestyle=line_style, 
                          alpha=0.6, linewidth=line_width, zorder=4)
                
                # Enhanced BOS annotation with break line indicator
                y_offset = 20 + (i % 3) * 15 if is_bullish else -35 - (i % 3) * 15
                
                ax.annotate(f'BOS {arrow}\n${price:.2f}', (date, price), 
                          xytext=(15, y_offset), textcoords='offset points',
                          fontsize=11, fontweight='bold', color=color,
                          bbox=dict(boxstyle='round,pad=0.4', facecolor=color, 
                                   alpha=0.2, edgecolor=color, linewidth=2),
                          arrowprops=dict(arrowstyle='->', color=color, 
                                        alpha=0.8, linewidth=2),
                          zorder=8)
                
            except Exception as e:
                logger.debug(f"Failed to plot BOS break line: {e}")
        
        # Plot CHOCH break lines
        for i, choch in enumerate(choch_events):
            try:
                date = pd.to_datetime(choch['date'])
                choch_type = choch['type']
                
                # For CHOCH, estimate price from chart context
                ylim = ax.get_ylim()
                price = ylim[1] - (ylim[1] - ylim[0]) * (0.1 + i * 0.08)
                
                is_bullish = 'bullish' in choch_type
                color = self.colors['choch_bullish'] if is_bullish else self.colors['choch_bearish']
                symbol = '⟲' if is_bullish else '⟳'
                
                # Draw vertical line for CHOCH event
                ax.axvline(x=date, color=color, linestyle=':', alpha=0.6, 
                          linewidth=2, zorder=4)
                
                ax.annotate(f'CHoCH {symbol}', (date, price), 
                          xytext=(0, -25), textcoords='offset points',
                          fontsize=10, fontweight='bold', color=color,
                          bbox=dict(boxstyle='round,pad=0.4', facecolor=color, 
                                   alpha=0.2, edgecolor=color),
                          zorder=8)
                
            except Exception as e:
                logger.debug(f"Failed to plot CHOCH break line: {e}")
    
    def _plot_market_phases(self, ax, stock_data: pd.DataFrame, analysis_data: Dict):
        """Plot market phase highlighting"""
        
        trend_analysis = analysis_data.get('trend_analysis', {})
        trend_direction = trend_analysis.get('trend_direction', 'sideways')
        
        # Get date range for phase highlighting
        dates = pd.to_datetime(stock_data.index)
        
        # Simple phase detection based on trend direction
        if trend_direction == 'uptrend':
            phase_color = self.colors['trend_up']
            phase_label = 'Uptrend Phase'
        elif trend_direction == 'downtrend':
            phase_color = self.colors['trend_down']
            phase_label = 'Downtrend Phase'
        else:
            phase_color = self.colors['phase_highlight']
            phase_label = 'Consolidation Phase'
        
        # Add subtle background highlighting for current phase
        ylim = ax.get_ylim()
        
        # Highlight the last portion of the chart (recent phase)
        phase_start = dates[-len(dates)//3]  # Last 1/3 of data
        phase_end = dates[-1]
        
        ax.axvspan(phase_start, phase_end, alpha=0.05, color=phase_color, zorder=0)
        
        # Add phase label
        ax.text(phase_start + (phase_end - phase_start) / 2, 
               ylim[1] - (ylim[1] - ylim[0]) * 0.05,
               phase_label, ha='center', va='top', fontsize=10, 
               fontweight='bold', color=phase_color,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                        alpha=0.8, edgecolor=phase_color))
    
    def _plot_enhanced_levels(self, ax, analysis_data: Dict):
        """Plot enhanced support/resistance levels"""
        
        key_levels = analysis_data.get('key_levels', {})
        support_levels = key_levels.get('support_levels', [])
        resistance_levels = key_levels.get('resistance_levels', [])
        
        xlim = ax.get_xlim()
        
        # Plot enhanced support levels
        for i, support in enumerate(support_levels[-3:]):  # Show only recent 3 levels
            try:
                level = support['level']
                strength = support['strength']
                
                line_width = {'strong': 4, 'medium': 3, 'weak': 2}.get(strength, 2)
                line_style = {'strong': '-', 'medium': '--', 'weak': ':'}.get(strength, '--')
                alpha = {'strong': 0.9, 'medium': 0.7, 'weak': 0.5}.get(strength, 0.7)
                
                # Enhanced level line with gradient effect
                ax.axhline(y=level, color=self.colors['support'], 
                          linestyle=line_style, alpha=alpha, linewidth=line_width,
                          zorder=2)
                
                # Enhanced level label with strength indicator
                strength_indicator = '●' if strength == 'strong' else '◐' if strength == 'medium' else '○'
                ax.text(xlim[1], level, f'  S: ${level:.2f} {strength_indicator}', 
                       verticalalignment='center', fontsize=10,
                       color=self.colors['support'], fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                alpha=0.9, edgecolor=self.colors['support']))
                
            except Exception as e:
                logger.debug(f"Failed to plot enhanced support: {e}")
        
        # Plot enhanced resistance levels  
        for i, resistance in enumerate(resistance_levels[-3:]):
            try:
                level = resistance['level']
                strength = resistance['strength']
                
                line_width = {'strong': 4, 'medium': 3, 'weak': 2}.get(strength, 2)
                line_style = {'strong': '-', 'medium': '--', 'weak': ':'}.get(strength, '--')
                alpha = {'strong': 0.9, 'medium': 0.7, 'weak': 0.5}.get(strength, 0.7)
                
                ax.axhline(y=level, color=self.colors['resistance'], 
                          linestyle=line_style, alpha=alpha, linewidth=line_width,
                          zorder=2)
                
                # Enhanced level label
                strength_indicator = '●' if strength == 'strong' else '◐' if strength == 'medium' else '○'
                ax.text(xlim[1], level, f'  R: ${level:.2f} {strength_indicator}', 
                       verticalalignment='center', fontsize=10,
                       color=self.colors['resistance'], fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                alpha=0.9, edgecolor=self.colors['resistance']))
                
            except Exception as e:
                logger.debug(f"Failed to plot enhanced resistance: {e}")
    
    def _plot_fibonacci_analysis(self, ax, analysis_data: Dict):
        """Plot dedicated Fibonacci analysis panel"""
        
        ax.axis('off')
        
        # Fibonacci analysis summary
        fib_text = """
FIBONACCI ANALYSIS

Key Retracement Levels:
• 23.6% - Shallow pullback
• 38.2% - Common retracement  
• 50.0% - Psychological level
• 61.8% - Golden ratio
• 78.6% - Deep retracement

Extension Levels:
• 127.2% - Minimum target
• 161.8% - Primary target
• 261.8% - Extended target

Current Analysis:
Based on major swing
points identified in
market structure.

Confluence zones show
high probability areas
for support/resistance.
        """.strip()
        
        ax.text(0.05, 0.95, fib_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcyan', 
                        alpha=0.8, edgecolor=self.colors['fibonacci']))
    
    def _plot_volume_analysis(self, ax, stock_data: pd.DataFrame, analysis_data: Dict):
        """Enhanced volume analysis"""
        
        if 'volume' not in stock_data.columns:
            ax.text(0.5, 0.5, 'Volume Data Not Available', 
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=14, alpha=0.6)
            return
        
        dates = pd.to_datetime(stock_data.index)
        volumes = stock_data['volume']
        closes = stock_data['close']
        
        # Enhanced volume coloring
        colors = []
        for i in range(len(volumes)):
            if i == 0:
                colors.append(self.colors['volume_up'])
            else:
                if closes.iloc[i] >= closes.iloc[i-1]:
                    # Higher volume on up days is more significant
                    if volumes.iloc[i] > volumes.iloc[i-1] * 1.5:
                        colors.append('#228B22')  # Strong green for high volume up days
                    else:
                        colors.append(self.colors['volume_up'])
                else:
                    # Higher volume on down days shows selling pressure
                    if volumes.iloc[i] > volumes.iloc[i-1] * 1.5:
                        colors.append('#DC143C')  # Strong red for high volume down days
                    else:
                        colors.append(self.colors['volume_down'])
        
        ax.bar(dates, volumes, color=colors, alpha=0.7, width=0.8)
        
        # Enhanced moving averages
        if len(volumes) > 20:
            vol_ma20 = volumes.rolling(window=20).mean()
            ax.plot(dates, vol_ma20, color='blue', linewidth=2, 
                   alpha=0.8, label='Vol MA(20)')
        
        if len(volumes) > 50:
            vol_ma50 = volumes.rolling(window=50).mean()
            ax.plot(dates, vol_ma50, color='orange', linewidth=2, 
                   alpha=0.8, label='Vol MA(50)')
        
        # Volume spike detection
        if len(volumes) > 20:
            vol_threshold = vol_ma20 * 2  # 2x average volume
            spike_dates = dates[volumes > vol_threshold]
            spike_volumes = volumes[volumes > vol_threshold]
            
            if len(spike_dates) > 0:
                ax.scatter(spike_dates, spike_volumes, color='red', 
                          s=50, alpha=0.8, marker='*', 
                          label='Volume Spikes', zorder=5)
        
        ax.set_title('Enhanced Volume Analysis', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.ticklabel_format(style='plain', axis='y')
    
    def _plot_swing_metrics(self, ax, analysis_data: Dict):
        """Enhanced swing point metrics"""
        
        swing_points = analysis_data.get('swing_points', {})
        swing_highs = swing_points.get('swing_highs', [])
        swing_lows = swing_points.get('swing_lows', [])
        
        strength_counts = {'strong': 0, 'medium': 0, 'weak': 0}
        
        for swing in swing_highs + swing_lows:
            strength = swing.get('strength', 'unknown')
            if strength in strength_counts:
                strength_counts[strength] += 1
        
        strengths = list(strength_counts.keys())
        counts = list(strength_counts.values())
        colors = ['#228B22', '#FFA500', '#DC143C']  # Green, Orange, Red
        
        bars = ax.bar(strengths, counts, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Enhanced value labels
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{count}', ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        ax.set_title('Swing Strength Distribution', fontsize=12, fontweight='bold')
        ax.set_ylabel('Count', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    def _plot_structure_metrics(self, ax, analysis_data: Dict):
        """Enhanced structural break metrics"""
        
        bos_choch = analysis_data.get('bos_choch_analysis', {})
        bos_count = len(bos_choch.get('bos_events', []))
        choch_count = len(bos_choch.get('choch_events', []))
        
        bullish_bos = sum(1 for bos in bos_choch.get('bos_events', []) 
                         if 'bullish' in bos.get('type', ''))
        bearish_bos = bos_count - bullish_bos
        
        categories = ['Bull\nBOS', 'Bear\nBOS', 'CHoCH']
        counts = [bullish_bos, bearish_bos, choch_count]
        colors = ['#228B22', '#DC143C', '#4169E1']  # Green, Red, Blue
        
        bars = ax.bar(categories, counts, color=colors, alpha=0.8, 
                     edgecolor='black', linewidth=1.5)
        
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                   f'{count}', ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        ax.set_title('Structure Break Analysis', fontsize=12, fontweight='bold')
        ax.set_ylabel('Count', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    def _plot_fibonacci_metrics(self, ax, analysis_data: Dict):
        """Plot Fibonacci-related metrics"""
        
        ax.axis('off')
        
        # Calculate some Fibonacci-related metrics
        swing_points = analysis_data.get('swing_points', {})
        total_swings = swing_points.get('total_swings', 0)
        swing_density = swing_points.get('swing_density', 0)
        
        structure_quality = analysis_data.get('structure_quality', {})
        quality_score = structure_quality.get('quality_score', 0)
        
        fib_metrics_text = f"""
FIBONACCI METRICS

Swing Analysis:
• Total Swings: {total_swings}
• Swing Density: {swing_density:.3f}
• Structure Quality: {quality_score}/100

Retracement Zones:
• 38.2%-50%: High probability
• 61.8%: Golden ratio zone  
• 78.6%: Last chance area

Extension Targets:
• 127%-162%: Primary zone
• 200%-262%: Extended targets
        """.strip()
        
        ax.text(0.05, 0.95, fib_metrics_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round,pad=0.4', facecolor='wheat', 
                        alpha=0.8, edgecolor=self.colors['fibonacci']))
    
    def _plot_enhanced_summary(self, ax, analysis_data: Dict, scenario: str):
        """Enhanced analysis summary with more details"""
        
        ax.axis('off')
        
        # Extract enhanced metrics
        structure_quality = analysis_data.get('structure_quality', {})
        quality_score = structure_quality.get('quality_score', 0)
        quality_rating = structure_quality.get('quality_rating', 'unknown')
        
        trend_analysis = analysis_data.get('trend_analysis', {})
        trend_direction = trend_analysis.get('trend_direction', 'unknown')
        trend_strength = trend_analysis.get('trend_strength', 'unknown')
        
        swing_points = analysis_data.get('swing_points', {})
        total_swings = swing_points.get('total_swings', 0)
        swing_density = swing_points.get('swing_density', 0)
        
        bos_choch = analysis_data.get('bos_choch_analysis', {})
        structural_bias = bos_choch.get('structural_bias', 'neutral')
        bos_count = len(bos_choch.get('bos_events', []))
        choch_count = len(bos_choch.get('choch_events', []))
        
        current_state = analysis_data.get('current_state', {})
        structure_state = current_state.get('structure_state', 'unknown')
        
        # Enhanced summary with visual elements info
        summary_text = f"""
ENHANCED MARKET STRUCTURE ANALYSIS SUMMARY - {scenario.upper()}

📊 STRUCTURE QUALITY                    🎯 VISUAL ENHANCEMENTS                   📈 TREND ANALYSIS
   Quality Score: {quality_score}/100 ({quality_rating.title()})        ✓ Fibonacci Retracements (5 levels)         Direction: {trend_direction.title()}
   Confidence: {analysis_data.get('confidence_score', 0):.2f}                      ✓ Trend Channels with fills                 Strength: {trend_strength.title()}
                                        ✓ Structure Break Lines                     Current State: {structure_state.replace('_', ' ').title()}

🔄 SWING STRUCTURE                      ⚡ STRUCTURAL BREAKS                     🎨 ENHANCED FEATURES
   Total Swing Points: {total_swings}                BOS Events: {bos_count}                            ✓ Price Labels on Key Swings
   Swing Density: {swing_density:.3f}                   CHoCH Events: {choch_count}                        ✓ Market Phase Highlighting  
   Market Bias: {structural_bias.title()}                 Structural Bias: {structural_bias.title()}           ✓ Enhanced Level Indicators

📅 ANALYSIS PERIOD: Full Dataset | 🔍 DATA QUALITY: {analysis_data.get('data_quality', {}).get('overall_quality_score', 0)}/100 | 🎯 VISUAL CONFIDENCE: Enhanced
        """.strip()
        
        ax.text(0.02, 0.95, summary_text, transform=ax.transAxes,
               fontsize=11, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round,pad=0.7', facecolor='lightsteelblue', 
                        alpha=0.9, edgecolor='navy', linewidth=2))
    
    def _format_enhanced_axes(self, ax_main, ax_volume, stock_data):
        """Enhanced axis formatting"""
        
        dates = pd.to_datetime(stock_data.index)
        
        # Enhanced date formatting
        ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax_main.xaxis.set_major_locator(mdates.WeekdayLocator(interval=3))
        ax_volume.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax_volume.xaxis.set_major_locator(mdates.WeekdayLocator(interval=3))
        
        # Rotate and style date labels
        plt.setp(ax_main.xaxis.get_majorticklabels(), rotation=45, fontsize=9)
        plt.setp(ax_volume.xaxis.get_majorticklabels(), rotation=45, fontsize=9)
        
        # Enhanced y-axis formatting
        ax_main.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:.2f}'))
        ax_volume.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.0f}K'))
        
        # Enhanced grid
        ax_main.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax_volume.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        # Style improvements
        ax_main.spines['top'].set_visible(False)
        ax_main.spines['right'].set_visible(False)
        ax_volume.spines['top'].set_visible(False)
        ax_volume.spines['right'].set_visible(False)


# Test function to run enhanced chart generation
def test_enhanced_charts():
    """Test enhanced chart generation"""
    
    logger.info("Testing Enhanced Market Structure Charts...")
    
    # Import mock data from the original test
    from test_chart_generation import create_mock_data_scenarios
    
    # Create enhanced chart generator
    enhanced_generator = EnhancedMarketStructureCharts(output_dir="enhanced_charts_v2")
    
    # Generate scenarios
    scenarios = create_mock_data_scenarios()
    
    generated_charts = []
    
    for stock_data, analysis_data, symbol, scenario in scenarios:
        logger.info(f"\n{'='*60}")
        logger.info(f"Generating enhanced chart for {symbol} - {scenario}")
        logger.info(f"{'='*60}")
        
        # Generate enhanced chart
        chart_path = enhanced_generator.create_enhanced_chart(
            stock_data=stock_data,
            analysis_data=analysis_data,
            symbol=symbol,
            scenario=scenario
        )
        
        if chart_path:
            generated_charts.append(chart_path)
            file_size = os.path.getsize(chart_path)
            logger.info(f"✅ Enhanced chart generated: {chart_path}")
            logger.info(f"📊 File size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
        else:
            logger.error(f"❌ Failed to generate enhanced chart for {symbol}")
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("ENHANCED CHART GENERATION SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Scenarios tested: {len(scenarios)}")
    logger.info(f"Charts generated: {len(generated_charts)}")
    
    if generated_charts:
        logger.info(f"\n📊 Enhanced charts with new features:")
        for chart in generated_charts:
            logger.info(f"  ✨ {chart}")
        
        logger.info(f"\n🎯 New Visual Features Added:")
        logger.info(f"  🔹 Fibonacci Retracements (5 levels)")
        logger.info(f"  🔹 Trend Channels with fill areas") 
        logger.info(f"  🔹 Structure Break Lines through BOS/CHOCH")
        logger.info(f"  🔹 Price Labels on Key Swing Points")
        logger.info(f"  🔹 Market Phase Background Highlighting")
        logger.info(f"  🔹 Enhanced Support/Resistance Indicators")
        logger.info(f"  🔹 Dedicated Fibonacci Analysis Panel")
        logger.info(f"  🔹 Volume Spike Detection")
        logger.info(f"  🔹 Enhanced Summary with Visual Confirmation")
    
    logger.info(f"\n✨ Enhanced chart generation test completed!")


if __name__ == "__main__":
    test_enhanced_charts()