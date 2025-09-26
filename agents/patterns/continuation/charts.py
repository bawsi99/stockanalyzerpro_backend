"""
Continuation Patterns Charts Generator

This module creates specialized charts for continuation pattern analysis,
including triangles, flags, pennants, channels, and support/resistance levels.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as mlines
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import io
import logging

logger = logging.getLogger(__name__)

class ContinuationPatternsCharts:
    """
    Chart generator specialized for continuation pattern visualization
    
    Creates charts that highlight:
    - Triangle patterns (ascending, descending, symmetrical)
    - Flag and pennant patterns
    - Channel patterns
    - Support and resistance levels
    - Breakout levels and targets
    """
    
    def __init__(self):
        self.name = "continuation_patterns_charts"
        self.chart_style = {
            'figsize': (16, 10),
            'price_color': '#2E86C1',
            'volume_color': '#7FB3D3',
            'bullish_color': '#27AE60',
            'bearish_color': '#E74C3C',
            'triangle_color': '#9B59B6',
            'flag_color': '#F39C12',
            'channel_color': '#1ABC9C',
            'support_color': '#16A085',
            'resistance_color': '#DC7633',
            'breakout_color': '#E67E22'
        }
    
    async def create_chart(self, stock_data: pd.DataFrame, indicators: Dict[str, Any] = None) -> bytes:
        """
        Create comprehensive continuation patterns chart
        
        Args:
            stock_data: OHLCV price data
            indicators: Technical indicators dictionary
            
        Returns:
            Chart image as bytes
        """
        try:
            fig, axes = plt.subplots(2, 1, figsize=self.chart_style['figsize'], 
                                   gridspec_kw={'height_ratios': [3, 1]})
            
            # Main price chart with continuation patterns
            self._plot_price_with_patterns(axes[0], stock_data, indicators)
            
            # Volume chart with pattern confirmation
            self._plot_volume_analysis(axes[1], stock_data)
            
            # Style all axes
            for i, ax in enumerate(axes):
                self._style_axis(ax, i == 0)
            
            plt.tight_layout()
            
            # Convert to bytes
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            img_buffer.seek(0)
            chart_bytes = img_buffer.read()
            plt.close(fig)
            
            logger.info(f"[CONTINUATION_CHARTS] Chart created successfully")
            return chart_bytes
            
        except Exception as e:
            logger.error(f"[CONTINUATION_CHARTS] Chart creation failed: {str(e)}")
            # Return empty chart on failure
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, f'Chart creation failed: {str(e)}', ha='center', va='center')
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            img_buffer.seek(0)
            chart_bytes = img_buffer.read()
            plt.close(fig)
            return chart_bytes
    
    def _plot_price_with_patterns(self, ax, stock_data: pd.DataFrame, indicators: Dict[str, Any]):
        """Plot price data with continuation pattern identification"""
        prices = stock_data['close'].values
        highs = stock_data['high'].values
        lows = stock_data['low'].values
        x_axis = range(len(prices))
        
        # Plot price line and high/low envelope
        ax.plot(x_axis, prices, color=self.chart_style['price_color'], linewidth=2, label='Close Price')
        ax.fill_between(x_axis, lows, highs, alpha=0.1, color=self.chart_style['price_color'])
        
        # Identify and highlight continuation patterns
        self._highlight_triangle_patterns(ax, stock_data)
        self._highlight_flag_pennant_patterns(ax, stock_data)
        self._highlight_channel_patterns(ax, stock_data)
        
        # Add support/resistance levels
        self._add_support_resistance_levels(ax, stock_data)
        
        # Add breakout levels and targets
        self._add_breakout_analysis(ax, stock_data)
        
        # Add moving averages if available
        if indicators:
            if 'sma_20' in indicators:
                ax.plot(x_axis, indicators['sma_20'], color='orange', linewidth=1, alpha=0.7, label='SMA 20')
            if 'sma_50' in indicators:
                ax.plot(x_axis, indicators['sma_50'], color='red', linewidth=1, alpha=0.7, label='SMA 50')
        
        ax.set_title('Price Action with Continuation Patterns', fontsize=14, fontweight='bold')
        ax.set_ylabel('Price', fontsize=12)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
    
    def _highlight_triangle_patterns(self, ax, stock_data: pd.DataFrame):
        """Identify and highlight triangle patterns"""
        highs = stock_data['high'].values
        lows = stock_data['low'].values
        
        if len(highs) < 20:
            return
        
        # Look for triangle patterns
        lookback = min(30, len(highs) - 5)
        
        for i in range(lookback, len(highs) - 5):
            recent_highs = highs[i-lookback:i]
            recent_lows = lows[i-lookback:i]
            
            # Calculate trend lines
            upper_trend = self._calculate_trend_line(recent_highs)
            lower_trend = self._calculate_trend_line(recent_lows)
            
            if upper_trend and lower_trend:
                upper_slope = upper_trend['slope']
                lower_slope = lower_trend['slope']
                
                triangle_type = None
                color = self.chart_style['triangle_color']
                
                if upper_slope < -0.01 and abs(lower_slope) < 0.01:
                    triangle_type = "Descending Triangle"
                    color = self.chart_style['bearish_color']
                elif lower_slope > 0.01 and abs(upper_slope) < 0.01:
                    triangle_type = "Ascending Triangle"
                    color = self.chart_style['bullish_color']
                elif upper_slope < -0.005 and lower_slope > 0.005:
                    triangle_type = "Symmetrical Triangle"
                
                if triangle_type:
                    # Draw triangle lines
                    start_x = i - lookback
                    end_x = i
                    
                    # Upper trend line
                    upper_start = upper_trend['intercept'] + upper_slope * 0
                    upper_end = upper_trend['intercept'] + upper_slope * (end_x - start_x)
                    ax.plot([start_x, end_x], [upper_start, upper_end], 
                           color=color, linewidth=2, linestyle='--', alpha=0.8)
                    
                    # Lower trend line
                    lower_start = lower_trend['intercept'] + lower_slope * 0
                    lower_end = lower_trend['intercept'] + lower_slope * (end_x - start_x)
                    ax.plot([start_x, end_x], [lower_start, lower_end], 
                           color=color, linewidth=2, linestyle='--', alpha=0.8)
                    
                    # Add pattern label
                    mid_x = start_x + (end_x - start_x) * 0.7
                    mid_y = (upper_end + lower_end) / 2
                    ax.annotate(triangle_type, xy=(mid_x, mid_y), 
                              xytext=(mid_x + 2, mid_y + (upper_end - lower_end) * 0.1),
                              ha='left', fontsize=9, color=color, fontweight='bold',
                              bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    def _highlight_flag_pennant_patterns(self, ax, stock_data: pd.DataFrame):
        """Identify and highlight flag and pennant patterns"""
        prices = stock_data['close'].values
        highs = stock_data['high'].values
        lows = stock_data['low'].values
        
        if len(prices) < 15:
            return
        
        # Look for flag/pennant patterns after strong moves
        for i in range(10, len(prices) - 5):
            flagpole_start = max(0, i - 10)
            flagpole_move = (prices[i] - prices[flagpole_start]) / prices[flagpole_start]
            
            if abs(flagpole_move) > 0.05:  # At least 5% move
                # Check for consolidation
                consolidation_highs = highs[i:i+5]
                consolidation_lows = lows[i:i+5]
                consolidation_range = np.max(consolidation_highs) - np.min(consolidation_lows)
                avg_price = np.mean(prices[i:i+5])
                
                if consolidation_range / avg_price < 0.05:  # Tight consolidation
                    pattern_type = "Bull Flag" if flagpole_move > 0 else "Bear Flag"
                    color = self.chart_style['bullish_color'] if flagpole_move > 0 else self.chart_style['bearish_color']
                    
                    # Draw flagpole
                    ax.plot([flagpole_start, i], [prices[flagpole_start], prices[i]], 
                           color=color, linewidth=3, alpha=0.8, label='Flagpole' if i == 10 else "")
                    
                    # Draw consolidation box
                    consolidation_high = np.max(consolidation_highs)
                    consolidation_low = np.min(consolidation_lows)
                    
                    rect = patches.Rectangle((i, consolidation_low), 5, 
                                           consolidation_high - consolidation_low,
                                           linewidth=2, edgecolor=color, facecolor='none', 
                                           linestyle='-', alpha=0.7)
                    ax.add_patch(rect)
                    
                    # Add pattern label
                    ax.annotate(pattern_type, xy=(i + 2.5, consolidation_high), 
                              xytext=(i + 2.5, consolidation_high + avg_price * 0.02),
                              ha='center', fontsize=9, color=color, fontweight='bold')
    
    def _highlight_channel_patterns(self, ax, stock_data: pd.DataFrame):
        """Identify and highlight channel patterns"""
        highs = stock_data['high'].values
        lows = stock_data['low'].values
        
        if len(highs) < 25:
            return
        
        # Look for parallel channel patterns
        lookback = min(30, len(highs) - 5)
        
        for i in range(lookback, len(highs) - 5):
            recent_highs = highs[i-lookback:i]
            recent_lows = lows[i-lookback:i]
            
            # Calculate trend lines for channel
            upper_trend = self._calculate_trend_line(recent_highs)
            lower_trend = self._calculate_trend_line(recent_lows)
            
            if upper_trend and lower_trend:
                # Check if lines are roughly parallel
                slope_diff = abs(upper_trend['slope'] - lower_trend['slope'])
                
                if slope_diff < 0.02:  # Roughly parallel
                    avg_slope = (upper_trend['slope'] + lower_trend['slope']) / 2
                    
                    if avg_slope > 0.01:
                        channel_type = "Ascending Channel"
                        color = self.chart_style['bullish_color']
                    elif avg_slope < -0.01:
                        channel_type = "Descending Channel"
                        color = self.chart_style['bearish_color']
                    else:
                        channel_type = "Horizontal Channel"
                        color = self.chart_style['channel_color']
                    
                    # Draw channel lines
                    start_x = i - lookback
                    end_x = i
                    
                    # Upper channel line
                    upper_start = upper_trend['intercept']
                    upper_end = upper_trend['intercept'] + upper_trend['slope'] * (end_x - start_x)
                    ax.plot([start_x, end_x], [upper_start, upper_end], 
                           color=color, linewidth=2, alpha=0.7)
                    
                    # Lower channel line
                    lower_start = lower_trend['intercept']
                    lower_end = lower_trend['intercept'] + lower_trend['slope'] * (end_x - start_x)
                    ax.plot([start_x, end_x], [lower_start, lower_end], 
                           color=color, linewidth=2, alpha=0.7)
                    
                    # Fill channel area
                    x_fill = [start_x, end_x, end_x, start_x]
                    y_fill = [upper_start, upper_end, lower_end, lower_start]
                    ax.fill(x_fill, y_fill, color=color, alpha=0.1)
                    
                    # Add channel label
                    mid_x = start_x + (end_x - start_x) * 0.8
                    mid_y = (upper_end + lower_end) / 2
                    ax.annotate(channel_type, xy=(mid_x, mid_y), 
                              xytext=(mid_x, mid_y),
                              ha='center', fontsize=9, color=color, fontweight='bold',
                              bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    def _add_support_resistance_levels(self, ax, stock_data: pd.DataFrame):
        """Add key support and resistance levels"""
        highs = stock_data['high'].values
        lows = stock_data['low'].values
        
        # Find significant levels using pivot points
        window = min(10, len(highs) // 5)
        
        # Resistance levels (local maxima)
        resistance_levels = []
        for i in range(window, len(highs) - window):
            if highs[i] == np.max(highs[i-window:i+window+1]):
                resistance_levels.append(highs[i])
        
        # Support levels (local minima)
        support_levels = []
        for i in range(window, len(lows) - window):
            if lows[i] == np.min(lows[i-window:i+window+1]):
                support_levels.append(lows[i])
        
        # Draw resistance levels
        for level in resistance_levels[-3:]:  # Show top 3
            ax.axhline(y=level, color=self.chart_style['resistance_color'],
                      linestyle='--', alpha=0.6, linewidth=1.5)
            ax.text(len(highs) * 0.02, level, f'R: {level:.2f}', 
                   color=self.chart_style['resistance_color'], fontsize=8, fontweight='bold')
        
        # Draw support levels
        for level in support_levels[-3:]:  # Show top 3
            ax.axhline(y=level, color=self.chart_style['support_color'],
                      linestyle='--', alpha=0.6, linewidth=1.5)
            ax.text(len(lows) * 0.02, level, f'S: {level:.2f}', 
                   color=self.chart_style['support_color'], fontsize=8, fontweight='bold')
    
    def _add_breakout_analysis(self, ax, stock_data: pd.DataFrame):
        """Add breakout levels and target analysis"""
        current_price = stock_data['close'].iloc[-1]
        highs = stock_data['high'].values
        lows = stock_data['low'].values
        
        # Find recent high and low for potential breakout levels
        recent_high = np.max(highs[-20:])
        recent_low = np.min(lows[-20:])
        
        # Calculate potential breakout targets
        range_size = recent_high - recent_low
        
        # Upside breakout target
        upside_target = recent_high + range_size
        ax.axhline(y=upside_target, color=self.chart_style['breakout_color'],
                  linestyle=':', alpha=0.8, linewidth=2)
        ax.text(len(highs) * 0.95, upside_target, f'Target: {upside_target:.2f}', 
               color=self.chart_style['breakout_color'], fontsize=8, fontweight='bold', ha='right')
        
        # Downside breakout target
        downside_target = recent_low - range_size
        ax.axhline(y=downside_target, color=self.chart_style['breakout_color'],
                  linestyle=':', alpha=0.8, linewidth=2)
        ax.text(len(lows) * 0.95, downside_target, f'Target: {downside_target:.2f}', 
               color=self.chart_style['breakout_color'], fontsize=8, fontweight='bold', ha='right')
        
        # Mark current price
        ax.axhline(y=current_price, color='black', linestyle='-', alpha=0.8, linewidth=2)
        ax.text(len(highs) * 0.98, current_price, f'Current: {current_price:.2f}', 
               color='black', fontsize=9, fontweight='bold', ha='right')
    
    def _calculate_trend_line(self, prices: np.ndarray) -> Optional[Dict[str, float]]:
        """Calculate trend line for prices"""
        if len(prices) < 5:
            return None
        
        x = np.arange(len(prices))
        
        # Use linear regression to find trend line
        slope, intercept = np.polyfit(x, prices, 1)
        
        # Calculate R-squared for trend line quality
        y_pred = slope * x + intercept
        ss_res = np.sum((prices - y_pred) ** 2)
        ss_tot = np.sum((prices - np.mean(prices)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        if r_squared > 0.3:  # Reasonable trend line fit
            return {
                'slope': slope,
                'intercept': intercept,
                'r_squared': r_squared
            }
        
        return None
    
    def _plot_volume_analysis(self, ax, stock_data: pd.DataFrame):
        """Plot volume with pattern confirmation analysis"""
        if 'volume' not in stock_data.columns:
            ax.text(0.5, 0.5, 'Volume data not available', ha='center', va='center')
            ax.set_xlim(0, len(stock_data))
            return
        
        volume = stock_data['volume'].values
        x_axis = range(len(volume))
        
        # Plot volume bars with color coding
        colors = []
        for i in range(len(stock_data)):
            if i > 0:
                if stock_data['close'].iloc[i] > stock_data['close'].iloc[i-1]:
                    colors.append(self.chart_style['bullish_color'])
                else:
                    colors.append(self.chart_style['bearish_color'])
            else:
                colors.append(self.chart_style['volume_color'])
        
        bars = ax.bar(x_axis, volume, color=colors, alpha=0.7)
        
        # Add volume moving averages
        volume_ma20 = pd.Series(volume).rolling(window=20).mean()
        ax.plot(x_axis, volume_ma20, color='orange', linewidth=2, label='Volume MA(20)')
        
        # Highlight above-average volume periods
        avg_volume = np.mean(volume)
        high_volume_threshold = avg_volume * 1.5
        
        for i, vol in enumerate(volume):
            if vol > high_volume_threshold:
                bars[i].set_edgecolor('red')
                bars[i].set_linewidth(2)
        
        # Add breakout volume analysis
        recent_volume = np.mean(volume[-10:])
        if recent_volume > avg_volume * 1.2:
            ax.axhline(y=recent_volume, color='red', linestyle='--', alpha=0.7, linewidth=2)
            ax.text(len(volume) * 0.02, recent_volume, 'Breakout Volume?', 
                   color='red', fontsize=9, fontweight='bold')
        
        ax.set_title('Volume Analysis with Pattern Confirmation', fontsize=12, fontweight='bold')
        ax.set_ylabel('Volume', fontsize=10)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    def _style_axis(self, ax, show_title=True):
        """Apply consistent styling to chart axes"""
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#CCCCCC')
        ax.spines['bottom'].set_color('#CCCCCC')
        
        # Grid styling
        ax.grid(True, linestyle='-', alpha=0.2)
        ax.set_axisbelow(True)
        
        # Tick styling
        ax.tick_params(colors='#666666', which='both')
        
        if not show_title:
            ax.set_title('')