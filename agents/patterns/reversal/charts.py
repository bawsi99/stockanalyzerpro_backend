"""
Reversal Patterns Charts Generator

This module creates specialized charts for reversal pattern analysis,
including divergence visualization and pattern identification charts.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import io
import logging

logger = logging.getLogger(__name__)

class ReversalPatternsCharts:
    """
    Chart generator specialized for reversal pattern visualization
    
    Creates charts that highlight:
    - Price divergences with technical indicators
    - Double top/bottom patterns
    - Head and shoulders patterns
    - Support/resistance levels
    """
    
    def __init__(self):
        self.name = "reversal_patterns_charts"
        self.chart_style = {
            'figsize': (16, 12),
            'price_color': '#2E86C1',
            'volume_color': '#7FB3D3',
            'bullish_color': '#27AE60',
            'bearish_color': '#E74C3C',
            'divergence_color': '#F39C12',
            'pattern_color': '#8E44AD',
            'support_color': '#16A085',
            'resistance_color': '#DC7633'
        }
    
    async def create_chart(self, stock_data: pd.DataFrame, indicators: Dict[str, Any] = None) -> bytes:
        """
        Create comprehensive reversal patterns chart
        
        Args:
            stock_data: OHLCV price data
            indicators: Technical indicators dictionary
            
        Returns:
            Chart image as bytes
        """
        try:
            fig, axes = plt.subplots(4, 1, figsize=self.chart_style['figsize'], 
                                   gridspec_kw={'height_ratios': [3, 1, 1, 1]})
            
            # Main price chart with patterns
            self._plot_price_with_patterns(axes[0], stock_data, indicators)
            
            # Volume chart
            self._plot_volume_analysis(axes[1], stock_data)
            
            # RSI with divergence analysis
            if indicators and 'rsi' in indicators:
                self._plot_rsi_divergences(axes[2], stock_data, indicators['rsi'])
            else:
                axes[2].text(0.5, 0.5, 'RSI data not available', ha='center', va='center')
                axes[2].set_xlim(0, len(stock_data))
                axes[2].set_ylim(0, 100)
            
            # MACD with divergence analysis
            if indicators and all(k in indicators for k in ['macd', 'macd_signal', 'macd_histogram']):
                self._plot_macd_divergences(axes[3], stock_data, indicators)
            else:
                axes[3].text(0.5, 0.5, 'MACD data not available', ha='center', va='center')
                axes[3].set_xlim(0, len(stock_data))
            
            # Style all axes
            for i, ax in enumerate(axes):
                self._style_axis(ax, i == 0)  # Only show title on first axis
            
            plt.tight_layout()
            
            # Convert to bytes
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            img_buffer.seek(0)
            chart_bytes = img_buffer.read()
            plt.close(fig)
            
            logger.info(f"[REVERSAL_CHARTS] Chart created successfully")
            return chart_bytes
            
        except Exception as e:
            logger.error(f"[REVERSAL_CHARTS] Chart creation failed: {str(e)}")
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
        """Plot price data with reversal pattern identification"""
        prices = stock_data['close'].values
        highs = stock_data['high'].values
        lows = stock_data['low'].values
        x_axis = range(len(prices))
        
        # Plot price line
        ax.plot(x_axis, prices, color=self.chart_style['price_color'], linewidth=2, label='Close Price')
        
        # Plot high/low envelope
        ax.fill_between(x_axis, lows, highs, alpha=0.1, color=self.chart_style['price_color'])
        
        # Identify and highlight reversal patterns
        self._highlight_double_tops_bottoms(ax, stock_data)
        self._highlight_head_shoulders(ax, stock_data)
        
        # Add support/resistance levels
        self._add_support_resistance_levels(ax, stock_data)
        
        # Add moving averages if available
        if indicators:
            if 'sma_20' in indicators:
                ax.plot(x_axis, indicators['sma_20'], color='orange', linewidth=1, alpha=0.7, label='SMA 20')
            if 'sma_50' in indicators:
                ax.plot(x_axis, indicators['sma_50'], color='red', linewidth=1, alpha=0.7, label='SMA 50')
        
        ax.set_title('Price Action with Reversal Patterns', fontsize=14, fontweight='bold')
        ax.set_ylabel('Price', fontsize=12)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
    
    def _highlight_double_tops_bottoms(self, ax, stock_data: pd.DataFrame):
        """Identify and highlight double top/bottom patterns"""
        highs = stock_data['high'].values
        lows = stock_data['low'].values
        
        # Look for double tops
        for i in range(10, len(highs) - 10):
            current_high = highs[i]
            
            # Look for another high at similar level
            for j in range(i + 5, min(i + 15, len(highs))):
                if abs(highs[j] - current_high) / current_high < 0.02:  # Within 2%
                    # Check if there's a valley between the peaks
                    valley_low = np.min(lows[i:j+1])
                    if (current_high - valley_low) / current_high > 0.03:  # At least 3% decline
                        # Highlight double top pattern
                        ax.add_patch(patches.Rectangle((i-2, valley_low), j-i+4, current_high-valley_low,
                                                     linewidth=2, edgecolor=self.chart_style['bearish_color'],
                                                     facecolor='none', linestyle='--', alpha=0.7))
                        ax.annotate('Double Top', xy=(i+(j-i)/2, current_high), 
                                  xytext=(i+(j-i)/2, current_high + current_high*0.02),
                                  ha='center', fontsize=10, color=self.chart_style['bearish_color'],
                                  fontweight='bold')
                        break
        
        # Look for double bottoms
        for i in range(10, len(lows) - 10):
            current_low = lows[i]
            
            # Look for another low at similar level
            for j in range(i + 5, min(i + 15, len(lows))):
                if abs(lows[j] - current_low) / current_low < 0.02:  # Within 2%
                    # Check if there's a peak between the valleys
                    peak_high = np.max(highs[i:j+1])
                    if (peak_high - current_low) / current_low > 0.03:  # At least 3% rally
                        # Highlight double bottom pattern
                        ax.add_patch(patches.Rectangle((i-2, current_low), j-i+4, peak_high-current_low,
                                                     linewidth=2, edgecolor=self.chart_style['bullish_color'],
                                                     facecolor='none', linestyle='--', alpha=0.7))
                        ax.annotate('Double Bottom', xy=(i+(j-i)/2, current_low), 
                                  xytext=(i+(j-i)/2, current_low - current_low*0.02),
                                  ha='center', fontsize=10, color=self.chart_style['bullish_color'],
                                  fontweight='bold')
                        break
    
    def _highlight_head_shoulders(self, ax, stock_data: pd.DataFrame):
        """Identify and highlight head and shoulders patterns"""
        highs = stock_data['high'].values
        lows = stock_data['low'].values
        
        # Look for potential head and shoulders patterns
        for i in range(5, len(highs) - 10):
            left_shoulder = highs[i-5:i]
            head = highs[i:i+3]
            right_shoulder = highs[i+3:i+8]
            
            if len(left_shoulder) > 0 and len(head) > 0 and len(right_shoulder) > 0:
                left_peak = np.max(left_shoulder)
                head_peak = np.max(head)
                right_peak = np.max(right_shoulder)
                
                # Check if head is higher than shoulders
                if (head_peak > left_peak * 1.02 and head_peak > right_peak * 1.02 and
                    abs(left_peak - right_peak) / left_peak < 0.05):
                    
                    # Find neckline (support level)
                    neckline = np.min(lows[i-5:i+8])
                    
                    # Highlight head and shoulders pattern
                    # Draw neckline
                    ax.axhline(y=neckline, xmin=(i-5)/len(highs), xmax=(i+8)/len(highs),
                             color=self.chart_style['bearish_color'], linestyle='-', linewidth=2, alpha=0.8)
                    
                    # Mark the three peaks
                    left_peak_idx = i-5 + np.argmax(left_shoulder)
                    head_peak_idx = i + np.argmax(head)
                    right_peak_idx = i+3 + np.argmax(right_shoulder)
                    
                    ax.scatter([left_peak_idx, head_peak_idx, right_peak_idx],
                             [left_peak, head_peak, right_peak],
                             color=self.chart_style['bearish_color'], s=100, zorder=5)
                    
                    ax.annotate('H&S', xy=(head_peak_idx, head_peak), 
                              xytext=(head_peak_idx, head_peak + head_peak*0.02),
                              ha='center', fontsize=10, color=self.chart_style['bearish_color'],
                              fontweight='bold')
    
    def _add_support_resistance_levels(self, ax, stock_data: pd.DataFrame):
        """Add support and resistance levels"""
        highs = stock_data['high'].values
        lows = stock_data['low'].values
        
        # Find recent significant highs and lows
        window = min(20, len(highs) // 4)
        
        # Resistance levels (significant highs)
        for i in range(window, len(highs) - window):
            if highs[i] == np.max(highs[i-window:i+window+1]):
                # This is a local maximum
                resistance_level = highs[i]
                ax.axhline(y=resistance_level, color=self.chart_style['resistance_color'],
                         linestyle='--', alpha=0.6, linewidth=1)
        
        # Support levels (significant lows)
        for i in range(window, len(lows) - window):
            if lows[i] == np.min(lows[i-window:i+window+1]):
                # This is a local minimum
                support_level = lows[i]
                ax.axhline(y=support_level, color=self.chart_style['support_color'],
                         linestyle='--', alpha=0.6, linewidth=1)
    
    def _plot_volume_analysis(self, ax, stock_data: pd.DataFrame):
        """Plot volume with analysis for pattern confirmation"""
        if 'volume' not in stock_data.columns:
            ax.text(0.5, 0.5, 'Volume data not available', ha='center', va='center')
            ax.set_xlim(0, len(stock_data))
            return
        
        volume = stock_data['volume'].values
        x_axis = range(len(volume))
        
        # Plot volume bars
        colors = []
        for i in range(len(stock_data)):
            if i > 0:
                if stock_data['close'].iloc[i] > stock_data['close'].iloc[i-1]:
                    colors.append(self.chart_style['bullish_color'])
                else:
                    colors.append(self.chart_style['bearish_color'])
            else:
                colors.append(self.chart_style['volume_color'])
        
        ax.bar(x_axis, volume, color=colors, alpha=0.7)
        
        # Add volume moving average
        volume_ma = pd.Series(volume).rolling(window=20).mean()
        ax.plot(x_axis, volume_ma, color='orange', linewidth=2, label='Volume MA(20)')
        
        ax.set_title('Volume Analysis', fontsize=12, fontweight='bold')
        ax.set_ylabel('Volume', fontsize=10)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    def _plot_rsi_divergences(self, ax, stock_data: pd.DataFrame, rsi: np.ndarray):
        """Plot RSI with divergence highlighting"""
        x_axis = range(len(rsi))
        
        # Plot RSI
        ax.plot(x_axis, rsi, color=self.chart_style['divergence_color'], linewidth=2, label='RSI')
        
        # Add overbought/oversold levels
        ax.axhline(y=70, color='red', linestyle='--', alpha=0.5, linewidth=1)
        ax.axhline(y=30, color='green', linestyle='--', alpha=0.5, linewidth=1)
        ax.axhline(y=50, color='gray', linestyle='-', alpha=0.3, linewidth=1)
        
        # Highlight divergences
        self._highlight_rsi_divergences(ax, stock_data, rsi)
        
        ax.set_title('RSI Divergence Analysis', fontsize=12, fontweight='bold')
        ax.set_ylabel('RSI', fontsize=10)
        ax.set_ylim(0, 100)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    def _highlight_rsi_divergences(self, ax, stock_data: pd.DataFrame, rsi: np.ndarray):
        """Highlight bullish and bearish divergences"""
        prices = stock_data['close'].values
        
        # Look for divergences in recent periods
        lookback = min(20, len(prices) - 1)
        
        for i in range(lookback, len(prices)):
            if i >= 10:
                price_window = prices[i-10:i+1]
                rsi_window = rsi[i-10:i+1]
                
                if len(price_window) >= 3 and len(rsi_window) >= 3:
                    price_min_idx = np.argmin(price_window)
                    rsi_min_idx = np.argmin(rsi_window)
                    price_max_idx = np.argmax(price_window)
                    rsi_max_idx = np.argmax(rsi_window)
                    
                    # Check for bullish divergence (price lower low, RSI higher low)
                    if (price_min_idx == len(price_window) - 1 and rsi_min_idx < len(rsi_window) - 1):
                        if rsi_window[-1] > rsi_window[rsi_min_idx]:
                            # Draw divergence line
                            ax.plot([i-10+rsi_min_idx, i], [rsi_window[rsi_min_idx], rsi_window[-1]],
                                   color=self.chart_style['bullish_color'], linewidth=2, alpha=0.8)
                            ax.annotate('Bull Div', xy=(i, rsi[i]), xytext=(i, rsi[i] + 5),
                                      ha='center', fontsize=8, color=self.chart_style['bullish_color'])
                    
                    # Check for bearish divergence (price higher high, RSI lower high)
                    if (price_max_idx == len(price_window) - 1 and rsi_max_idx < len(rsi_window) - 1):
                        if rsi_window[-1] < rsi_window[rsi_max_idx]:
                            # Draw divergence line
                            ax.plot([i-10+rsi_max_idx, i], [rsi_window[rsi_max_idx], rsi_window[-1]],
                                   color=self.chart_style['bearish_color'], linewidth=2, alpha=0.8)
                            ax.annotate('Bear Div', xy=(i, rsi[i]), xytext=(i, rsi[i] + 5),
                                      ha='center', fontsize=8, color=self.chart_style['bearish_color'])
    
    def _plot_macd_divergences(self, ax, stock_data: pd.DataFrame, indicators: Dict[str, Any]):
        """Plot MACD with divergence analysis"""
        macd = indicators['macd']
        macd_signal = indicators['macd_signal']
        macd_histogram = indicators['macd_histogram']
        x_axis = range(len(macd))
        
        # Plot MACD line and signal line
        ax.plot(x_axis, macd, color=self.chart_style['divergence_color'], linewidth=2, label='MACD')
        ax.plot(x_axis, macd_signal, color='red', linewidth=1, label='Signal')
        
        # Plot MACD histogram
        colors = ['green' if h > 0 else 'red' for h in macd_histogram]
        ax.bar(x_axis, macd_histogram, color=colors, alpha=0.3, label='Histogram')
        
        # Add zero line
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
        
        # Highlight MACD divergences
        self._highlight_macd_divergences(ax, stock_data, macd)
        
        ax.set_title('MACD Divergence Analysis', fontsize=12, fontweight='bold')
        ax.set_ylabel('MACD', fontsize=10)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    def _highlight_macd_divergences(self, ax, stock_data: pd.DataFrame, macd: np.ndarray):
        """Highlight MACD divergences"""
        prices = stock_data['close'].values
        
        # Similar divergence detection as RSI but using MACD
        lookback = min(20, len(prices) - 1)
        
        for i in range(lookback, len(prices)):
            if i >= 10:
                price_window = prices[i-10:i+1]
                macd_window = macd[i-10:i+1]
                
                if len(price_window) >= 3 and len(macd_window) >= 3:
                    price_min_idx = np.argmin(price_window)
                    macd_min_idx = np.argmin(macd_window)
                    
                    # Check for bullish divergence
                    if (price_min_idx == len(price_window) - 1 and macd_min_idx < len(macd_window) - 1):
                        if macd_window[-1] > macd_window[macd_min_idx]:
                            # Draw divergence line
                            ax.plot([i-10+macd_min_idx, i], [macd_window[macd_min_idx], macd_window[-1]],
                                   color=self.chart_style['bullish_color'], linewidth=2, alpha=0.8)
    
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