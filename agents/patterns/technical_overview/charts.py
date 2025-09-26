"""
Technical Overview Charts Generator

This module creates comprehensive technical analysis charts showing all major indicators,
support/resistance levels, trend analysis, and risk assessment visualizations.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import io
import logging
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates

logger = logging.getLogger(__name__)

class TechnicalOverviewCharts:
    """
    Chart generator for comprehensive technical overview analysis
    
    Creates multi-panel charts showing:
    - Price action with moving averages and Bollinger Bands
    - Volume analysis with trend confirmation
    - MACD momentum indicator with signals
    - RSI with overbought/oversold levels
    - Stochastic oscillator for momentum confirmation
    - ADX for trend strength assessment
    - Support/resistance levels overlay
    - Risk zones and key price levels
    """
    
    def __init__(self):
        self.name = "technical_overview_charts"
        self.chart_style = {
            'figsize': (18, 14),
            'price_color': '#2E86C1',
            'volume_color': '#7FB3D3',
            'bullish_color': '#27AE60',
            'bearish_color': '#E74C3C',
            'ma_color': '#F39C12',
            'signal_color': '#9B59B6',
            'support_color': '#16A085',
            'resistance_color': '#DC7633',
            'risk_color': '#E67E22',
            'neutral_color': '#95A5A6'
        }
    
    async def create_chart(self, stock_data: pd.DataFrame, indicators: Dict[str, Any] = None) -> bytes:
        """
        Create comprehensive technical overview chart
        
        Args:
            stock_data: OHLCV price data with datetime index
            indicators: Technical indicators dictionary
            
        Returns:
            Chart image as bytes
        """
        try:
            # Create figure with 6 subplots in a grid layout
            fig = plt.figure(figsize=self.chart_style['figsize'])
            gs = gridspec.GridSpec(6, 2, height_ratios=[3, 1, 1, 1, 1, 1], width_ratios=[3, 1], 
                                 hspace=0.3, wspace=0.2)
            
            # Main price chart (takes up most space)
            ax_price = fig.add_subplot(gs[0, :])
            
            # Technical indicator panels
            ax_volume = fig.add_subplot(gs[1, :], sharex=ax_price)
            ax_macd = fig.add_subplot(gs[2, :], sharex=ax_price)
            ax_rsi = fig.add_subplot(gs[3, :], sharex=ax_price)
            ax_stoch = fig.add_subplot(gs[4, :], sharex=ax_price)
            ax_adx = fig.add_subplot(gs[5, :], sharex=ax_price)
            
            # Plot each component
            self._plot_price_analysis(ax_price, stock_data, indicators)
            self._plot_volume_analysis(ax_volume, stock_data)
            self._plot_macd_analysis(ax_macd, stock_data, indicators)
            self._plot_rsi_analysis(ax_rsi, stock_data, indicators)
            self._plot_stochastic_analysis(ax_stoch, stock_data, indicators)
            self._plot_adx_analysis(ax_adx, stock_data, indicators)
            
            # Style all axes
            axes = [ax_price, ax_volume, ax_macd, ax_rsi, ax_stoch, ax_adx]
            for i, ax in enumerate(axes):
                self._style_axis(ax, i == 0, i == len(axes) - 1)
            
            # Add overall title
            fig.suptitle('Comprehensive Technical Overview Analysis', fontsize=16, fontweight='bold', y=0.98)
            
            # Convert to bytes
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            img_buffer.seek(0)
            chart_bytes = img_buffer.read()
            plt.close(fig)
            
            logger.info(f"[TECHNICAL_OVERVIEW_CHARTS] Chart created successfully")
            return chart_bytes
            
        except Exception as e:
            logger.error(f"[TECHNICAL_OVERVIEW_CHARTS] Chart creation failed: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Return error chart
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.text(0.5, 0.5, f'Technical Overview Chart Error:\n{str(e)}', 
                   ha='center', va='center', fontsize=14, 
                   bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            img_buffer.seek(0)
            chart_bytes = img_buffer.read()
            plt.close(fig)
            return chart_bytes
    
    def _plot_price_analysis(self, ax, stock_data: pd.DataFrame, indicators: Dict[str, Any]):
        """Plot main price chart with moving averages, Bollinger Bands, and support/resistance"""
        
        # Ensure we have a proper index for plotting
        x_axis = range(len(stock_data))
        prices = stock_data['close'].values
        highs = stock_data['high'].values
        lows = stock_data['low'].values
        
        # Plot price line and high/low envelope
        ax.plot(x_axis, prices, color=self.chart_style['price_color'], linewidth=2.5, label='Close Price')
        ax.fill_between(x_axis, lows, highs, alpha=0.1, color=self.chart_style['price_color'], label='Price Range')
        
        # Plot moving averages if available
        if indicators:
            if 'sma_20' in indicators and indicators['sma_20'] is not None:
                sma_20 = indicators['sma_20']
                if isinstance(sma_20, (list, np.ndarray)) and len(sma_20) == len(x_axis):
                    ax.plot(x_axis, sma_20, color='orange', linewidth=2, alpha=0.8, label='SMA 20')
            
            if 'sma_50' in indicators and indicators['sma_50'] is not None:
                sma_50 = indicators['sma_50']
                if isinstance(sma_50, (list, np.ndarray)) and len(sma_50) == len(x_axis):
                    ax.plot(x_axis, sma_50, color='red', linewidth=2, alpha=0.8, label='SMA 50')
            
            if 'ema_12' in indicators and indicators['ema_12'] is not None:
                ema_12 = indicators['ema_12']
                if isinstance(ema_12, (list, np.ndarray)) and len(ema_12) == len(x_axis):
                    ax.plot(x_axis, ema_12, color='purple', linewidth=1.5, alpha=0.7, label='EMA 12')
            
            # Plot Bollinger Bands if available
            if all(k in indicators for k in ['bb_upper', 'bb_middle', 'bb_lower']):
                bb_upper = indicators['bb_upper']
                bb_middle = indicators['bb_middle']
                bb_lower = indicators['bb_lower']
                
                if all(isinstance(bb, (list, np.ndarray)) and len(bb) == len(x_axis) 
                       for bb in [bb_upper, bb_middle, bb_lower]):
                    ax.plot(x_axis, bb_upper, color='gray', linewidth=1, alpha=0.6, linestyle='--')
                    ax.plot(x_axis, bb_middle, color='gray', linewidth=1.5, alpha=0.7, label='BB Middle')
                    ax.plot(x_axis, bb_lower, color='gray', linewidth=1, alpha=0.6, linestyle='--')
                    ax.fill_between(x_axis, bb_upper, bb_lower, alpha=0.05, color='gray', label='Bollinger Bands')
        
        # Add support and resistance levels
        self._add_support_resistance_levels(ax, stock_data)
        
        # Add trend analysis
        self._add_trend_analysis(ax, stock_data)
        
        # Add risk zones
        self._add_risk_zones(ax, stock_data)
        
        ax.set_title('Price Action with Technical Indicators', fontsize=14, fontweight='bold')
        ax.set_ylabel('Price', fontsize=12)
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    def _plot_volume_analysis(self, ax, stock_data: pd.DataFrame):
        """Plot volume analysis with trend confirmation"""
        
        if 'volume' not in stock_data.columns:
            ax.text(0.5, 0.5, 'Volume data not available', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            ax.set_ylim(0, 1)
            return
        
        x_axis = range(len(stock_data))
        volume = stock_data['volume'].values
        
        # Color volume bars based on price movement
        colors = []
        for i in range(len(stock_data)):
            if i > 0:
                if stock_data['close'].iloc[i] > stock_data['close'].iloc[i-1]:
                    colors.append(self.chart_style['bullish_color'])
                else:
                    colors.append(self.chart_style['bearish_color'])
            else:
                colors.append(self.chart_style['volume_color'])
        
        # Plot volume bars
        bars = ax.bar(x_axis, volume, color=colors, alpha=0.7, width=0.8)
        
        # Add volume moving average
        if len(volume) >= 20:
            volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=1).mean()
            ax.plot(x_axis, volume_ma20, color='orange', linewidth=2, label='Volume MA(20)')
        
        # Highlight unusual volume
        avg_volume = np.mean(volume)
        high_volume_threshold = avg_volume * 1.5
        
        for i, (bar, vol) in enumerate(zip(bars, volume)):
            if vol > high_volume_threshold:
                bar.set_edgecolor('red')
                bar.set_linewidth(2)
                # Add annotation for significant volume spikes
                if vol > avg_volume * 2:
                    ax.annotate(f'{vol/1e6:.1f}M', xy=(i, vol), xytext=(i, vol * 1.1),
                              ha='center', fontsize=8, color='red', fontweight='bold')
        
        # Add volume trend analysis
        recent_volume = np.mean(volume[-10:]) if len(volume) >= 10 else avg_volume
        if recent_volume > avg_volume * 1.2:
            ax.axhline(y=recent_volume, color='red', linestyle='--', alpha=0.7, linewidth=2)
            ax.text(len(volume) * 0.02, recent_volume * 1.1, 'High Volume Trend', 
                   color='red', fontsize=9, fontweight='bold')
        
        ax.set_title('Volume Analysis with Trend Confirmation', fontsize=12, fontweight='bold')
        ax.set_ylabel('Volume', fontsize=10)
        if len(volume) >= 20:
            ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    def _plot_macd_analysis(self, ax, stock_data: pd.DataFrame, indicators: Dict[str, Any]):
        """Plot MACD analysis with signals and divergences"""
        
        if not indicators or not all(k in indicators for k in ['macd', 'macd_signal', 'macd_histogram']):
            ax.text(0.5, 0.5, 'MACD data not available', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            ax.set_ylim(-1, 1)
            return
        
        x_axis = range(len(stock_data))
        macd = indicators['macd']
        macd_signal = indicators['macd_signal']
        macd_histogram = indicators['macd_histogram']
        
        # Ensure arrays are proper length
        if not all(isinstance(arr, (list, np.ndarray)) and len(arr) == len(x_axis) 
                  for arr in [macd, macd_signal, macd_histogram]):
            ax.text(0.5, 0.5, 'MACD data length mismatch', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            return
        
        # Plot MACD line and signal line
        ax.plot(x_axis, macd, color=self.chart_style['signal_color'], linewidth=2, label='MACD')
        ax.plot(x_axis, macd_signal, color='red', linewidth=1.5, label='Signal')
        
        # Plot MACD histogram
        colors = [self.chart_style['bullish_color'] if h > 0 else self.chart_style['bearish_color'] 
                 for h in macd_histogram]
        ax.bar(x_axis, macd_histogram, color=colors, alpha=0.6, width=0.8, label='Histogram')
        
        # Add zero line
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
        
        # Identify and mark MACD signals
        self._mark_macd_signals(ax, x_axis, macd, macd_signal)
        
        # Add divergence analysis
        self._highlight_macd_divergences(ax, stock_data, x_axis, macd)
        
        ax.set_title('MACD Momentum Analysis', fontsize=12, fontweight='bold')
        ax.set_ylabel('MACD', fontsize=10)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    def _plot_rsi_analysis(self, ax, stock_data: pd.DataFrame, indicators: Dict[str, Any]):
        """Plot RSI analysis with overbought/oversold levels and divergences"""
        
        if not indicators or 'rsi' not in indicators or indicators['rsi'] is None:
            ax.text(0.5, 0.5, 'RSI data not available', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            ax.set_ylim(0, 100)
            return
        
        x_axis = range(len(stock_data))
        rsi = indicators['rsi']
        
        if not isinstance(rsi, (list, np.ndarray)) or len(rsi) != len(x_axis):
            ax.text(0.5, 0.5, 'RSI data length mismatch', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            ax.set_ylim(0, 100)
            return
        
        # Plot RSI line
        ax.plot(x_axis, rsi, color=self.chart_style['signal_color'], linewidth=2, label='RSI')
        
        # Add overbought/oversold levels
        ax.axhline(y=70, color=self.chart_style['bearish_color'], linestyle='--', alpha=0.7, linewidth=1.5, label='Overbought (70)')
        ax.axhline(y=30, color=self.chart_style['bullish_color'], linestyle='--', alpha=0.7, linewidth=1.5, label='Oversold (30)')
        ax.axhline(y=50, color='gray', linestyle='-', alpha=0.4, linewidth=1)
        
        # Fill overbought/oversold zones
        ax.fill_between(x_axis, 70, 100, alpha=0.1, color=self.chart_style['bearish_color'])
        ax.fill_between(x_axis, 0, 30, alpha=0.1, color=self.chart_style['bullish_color'])
        
        # Highlight extreme levels
        for i, rsi_val in enumerate(rsi):
            if rsi_val > 80:
                ax.scatter(i, rsi_val, color='red', s=30, zorder=5)
            elif rsi_val < 20:
                ax.scatter(i, rsi_val, color='green', s=30, zorder=5)
        
        # Add RSI divergence analysis
        self._highlight_rsi_divergences(ax, stock_data, x_axis, rsi)
        
        ax.set_title('RSI Momentum Analysis', fontsize=12, fontweight='bold')
        ax.set_ylabel('RSI', fontsize=10)
        ax.set_ylim(0, 100)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    def _plot_stochastic_analysis(self, ax, stock_data: pd.DataFrame, indicators: Dict[str, Any]):
        """Plot Stochastic oscillator analysis"""
        
        # Generate simple stochastic if not available
        if not indicators or 'stoch_k' not in indicators:
            # Calculate basic stochastic
            high_14 = stock_data['high'].rolling(window=14, min_periods=1).max()
            low_14 = stock_data['low'].rolling(window=14, min_periods=1).min()
            stoch_k = ((stock_data['close'] - low_14) / (high_14 - low_14) * 100).fillna(50)
            stoch_d = stoch_k.rolling(window=3, min_periods=1).mean()
        else:
            stoch_k = indicators['stoch_k']
            stoch_d = indicators.get('stoch_d', pd.Series(stoch_k).rolling(window=3).mean())
        
        x_axis = range(len(stock_data))
        
        # Plot Stochastic lines
        ax.plot(x_axis, stoch_k, color=self.chart_style['signal_color'], linewidth=2, label='%K')
        ax.plot(x_axis, stoch_d, color='red', linewidth=1.5, label='%D')
        
        # Add overbought/oversold levels
        ax.axhline(y=80, color=self.chart_style['bearish_color'], linestyle='--', alpha=0.7, linewidth=1, label='Overbought')
        ax.axhline(y=20, color=self.chart_style['bullish_color'], linestyle='--', alpha=0.7, linewidth=1, label='Oversold')
        ax.axhline(y=50, color='gray', linestyle='-', alpha=0.4, linewidth=1)
        
        # Fill zones
        ax.fill_between(x_axis, 80, 100, alpha=0.1, color=self.chart_style['bearish_color'])
        ax.fill_between(x_axis, 0, 20, alpha=0.1, color=self.chart_style['bullish_color'])
        
        # Mark crossovers
        self._mark_stochastic_signals(ax, x_axis, stoch_k, stoch_d)
        
        ax.set_title('Stochastic Oscillator', fontsize=12, fontweight='bold')
        ax.set_ylabel('Stochastic', fontsize=10)
        ax.set_ylim(0, 100)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    def _plot_adx_analysis(self, ax, stock_data: pd.DataFrame, indicators: Dict[str, Any]):
        """Plot ADX trend strength analysis"""
        
        # Generate simple ADX approximation if not available
        if not indicators or 'adx' not in indicators:
            # Simple trend strength approximation using price volatility and direction
            returns = stock_data['close'].pct_change().fillna(0)
            volatility = returns.rolling(window=14, min_periods=1).std()
            trend_strength = (volatility * 100).clip(0, 100).fillna(25)
        else:
            trend_strength = indicators['adx']
        
        x_axis = range(len(stock_data))
        
        # Plot ADX line
        ax.plot(x_axis, trend_strength, color=self.chart_style['signal_color'], linewidth=2.5, label='Trend Strength')
        
        # Add trend strength zones
        ax.axhline(y=25, color='red', linestyle='--', alpha=0.7, linewidth=1, label='Strong Trend (25+)')
        ax.axhline(y=50, color='darkred', linestyle='--', alpha=0.7, linewidth=1, label='Very Strong (50+)')
        
        # Fill zones
        ax.fill_between(x_axis, 25, 100, alpha=0.1, color=self.chart_style['bullish_color'])
        ax.fill_between(x_axis, 0, 25, alpha=0.1, color=self.chart_style['neutral_color'])
        
        # Add directional movement if available
        if indicators and 'di_plus' in indicators and 'di_minus' in indicators:
            di_plus = indicators['di_plus']
            di_minus = indicators['di_minus']
            ax.plot(x_axis, di_plus, color='green', linewidth=1.5, alpha=0.7, label='DI+')
            ax.plot(x_axis, di_minus, color='red', linewidth=1.5, alpha=0.7, label='DI-')
        
        ax.set_title('ADX Trend Strength Analysis', fontsize=12, fontweight='bold')
        ax.set_ylabel('ADX', fontsize=10)
        ax.set_ylim(0, 100)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    def _add_support_resistance_levels(self, ax, stock_data: pd.DataFrame):
        """Add support and resistance levels to price chart"""
        
        highs = stock_data['high'].values
        lows = stock_data['low'].values
        
        # Find significant levels using pivot points
        window = min(10, len(highs) // 4)
        if window < 2:
            return
        
        # Find resistance levels (local maxima)
        resistance_levels = []
        for i in range(window, len(highs) - window):
            if highs[i] == np.max(highs[i-window:i+window+1]):
                resistance_levels.append(highs[i])
        
        # Find support levels (local minima)
        support_levels = []
        for i in range(window, len(lows) - window):
            if lows[i] == np.min(lows[i-window:i+window+1]):
                support_levels.append(lows[i])
        
        # Draw most significant levels
        resistance_levels = sorted(list(set(resistance_levels)), reverse=True)[:3]
        support_levels = sorted(list(set(support_levels)))[:3]
        
        for level in resistance_levels:
            ax.axhline(y=level, color=self.chart_style['resistance_color'],
                      linestyle='--', alpha=0.7, linewidth=2)
            ax.text(len(highs) * 0.02, level, f'R: {level:.2f}', 
                   color=self.chart_style['resistance_color'], fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        
        for level in support_levels:
            ax.axhline(y=level, color=self.chart_style['support_color'],
                      linestyle='--', alpha=0.7, linewidth=2)
            ax.text(len(lows) * 0.02, level, f'S: {level:.2f}', 
                   color=self.chart_style['support_color'], fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    
    def _add_trend_analysis(self, ax, stock_data: pd.DataFrame):
        """Add trend analysis to price chart"""
        
        prices = stock_data['close'].values
        x_axis = range(len(prices))
        
        # Calculate trend lines for recent periods
        if len(prices) >= 20:
            # Short-term trend (last 20 periods)
            recent_x = np.array(x_axis[-20:])
            recent_prices = prices[-20:]
            
            try:
                # Linear regression for trend line
                slope, intercept = np.polyfit(recent_x - recent_x[0], recent_prices, 1)
                trend_line = slope * (recent_x - recent_x[0]) + intercept
                
                # Determine trend direction and color
                if slope > 0:
                    trend_color = self.chart_style['bullish_color']
                    trend_label = 'Uptrend'
                else:
                    trend_color = self.chart_style['bearish_color']
                    trend_label = 'Downtrend'
                
                # Draw trend line
                ax.plot(recent_x, trend_line, color=trend_color, linewidth=2.5, 
                       alpha=0.8, linestyle='-', label=f'{trend_label} Line')
                
                # Add trend strength annotation
                r_squared = self._calculate_r_squared(recent_prices, trend_line)
                strength = 'Strong' if r_squared > 0.7 else 'Moderate' if r_squared > 0.4 else 'Weak'
                
                ax.text(recent_x[-1], trend_line[-1], f'{strength} {trend_label}', 
                       color=trend_color, fontsize=9, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
                       
            except Exception as e:
                logger.debug(f"Trend line calculation failed: {e}")
    
    def _add_risk_zones(self, ax, stock_data: pd.DataFrame):
        """Add risk assessment zones to price chart"""
        
        current_price = stock_data['close'].iloc[-1]
        recent_high = stock_data['high'].tail(20).max()
        recent_low = stock_data['low'].tail(20).min()
        
        # Calculate volatility-based risk zones
        returns = stock_data['close'].pct_change().dropna()
        volatility = returns.std()
        
        # Risk zones based on volatility
        upper_risk = current_price * (1 + volatility * 2)
        lower_risk = current_price * (1 - volatility * 2)
        
        # Add risk zone shading
        ax.axhspan(upper_risk, recent_high * 1.1, alpha=0.1, color=self.chart_style['risk_color'], 
                  label='High Risk Zone')
        ax.axhspan(recent_low * 0.9, lower_risk, alpha=0.1, color=self.chart_style['risk_color'])
        
        # Mark current price
        ax.axhline(y=current_price, color='black', linestyle='-', alpha=0.8, linewidth=2.5)
        ax.text(len(stock_data) * 0.98, current_price, f'Current: {current_price:.2f}', 
               color='black', fontsize=10, fontweight='bold', ha='right',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))
    
    def _mark_macd_signals(self, ax, x_axis, macd, macd_signal):
        """Mark MACD buy/sell signals"""
        
        for i in range(1, len(macd)):
            # Bullish crossover (MACD crosses above signal)
            if macd[i-1] <= macd_signal[i-1] and macd[i] > macd_signal[i]:
                ax.scatter(x_axis[i], macd[i], color=self.chart_style['bullish_color'], 
                          s=50, marker='^', zorder=5, label='Buy Signal' if i == 1 else "")
            
            # Bearish crossover (MACD crosses below signal)
            elif macd[i-1] >= macd_signal[i-1] and macd[i] < macd_signal[i]:
                ax.scatter(x_axis[i], macd[i], color=self.chart_style['bearish_color'], 
                          s=50, marker='v', zorder=5, label='Sell Signal' if i == 1 else "")
    
    def _mark_stochastic_signals(self, ax, x_axis, stoch_k, stoch_d):
        """Mark Stochastic crossover signals"""
        
        for i in range(1, len(stoch_k)):
            # Bullish crossover in oversold zone
            if (stoch_k[i-1] <= stoch_d[i-1] and stoch_k[i] > stoch_d[i] and 
                stoch_k[i] < 20):
                ax.scatter(x_axis[i], stoch_k[i], color=self.chart_style['bullish_color'], 
                          s=40, marker='^', zorder=5)
            
            # Bearish crossover in overbought zone
            elif (stoch_k[i-1] >= stoch_d[i-1] and stoch_k[i] < stoch_d[i] and 
                  stoch_k[i] > 80):
                ax.scatter(x_axis[i], stoch_k[i], color=self.chart_style['bearish_color'], 
                          s=40, marker='v', zorder=5)
    
    def _highlight_rsi_divergences(self, ax, stock_data, x_axis, rsi):
        """Highlight RSI divergences with price"""
        
        prices = stock_data['close'].values
        
        # Look for divergences in recent periods
        lookback = min(20, len(prices) - 2)
        
        for i in range(lookback, len(prices) - 1):
            if i >= 10:
                price_window = prices[i-10:i+1]
                rsi_window = rsi[i-10:i+1]
                
                if len(price_window) >= 3 and len(rsi_window) >= 3:
                    price_min_idx = np.argmin(price_window)
                    rsi_min_idx = np.argmin(rsi_window)
                    
                    # Bullish divergence (price lower low, RSI higher low)
                    if (price_min_idx == len(price_window) - 1 and 
                        rsi_min_idx < len(rsi_window) - 1 and
                        rsi_window[-1] > rsi_window[rsi_min_idx]):
                        
                        ax.plot([i-10+rsi_min_idx, i], 
                               [rsi_window[rsi_min_idx], rsi_window[-1]],
                               color=self.chart_style['bullish_color'], 
                               linewidth=2, alpha=0.8, linestyle=':')
                        ax.text(i, rsi[i] + 5, 'Bull Div', ha='center', fontsize=8, 
                               color=self.chart_style['bullish_color'], fontweight='bold')
    
    def _highlight_macd_divergences(self, ax, stock_data, x_axis, macd):
        """Highlight MACD divergences with price"""
        
        prices = stock_data['close'].values
        
        # Similar to RSI divergence but for MACD
        lookback = min(20, len(prices) - 2)
        
        for i in range(lookback, len(prices) - 1):
            if i >= 10:
                price_window = prices[i-10:i+1]
                macd_window = macd[i-10:i+1]
                
                if len(price_window) >= 3 and len(macd_window) >= 3:
                    price_max_idx = np.argmax(price_window)
                    macd_max_idx = np.argmax(macd_window)
                    
                    # Bearish divergence (price higher high, MACD lower high)
                    if (price_max_idx == len(price_window) - 1 and 
                        macd_max_idx < len(macd_window) - 1 and
                        macd_window[-1] < macd_window[macd_max_idx]):
                        
                        ax.plot([i-10+macd_max_idx, i], 
                               [macd_window[macd_max_idx], macd_window[-1]],
                               color=self.chart_style['bearish_color'], 
                               linewidth=2, alpha=0.8, linestyle=':')
    
    def _calculate_r_squared(self, actual, predicted):
        """Calculate R-squared for trend line quality"""
        try:
            ss_res = np.sum((actual - predicted) ** 2)
            ss_tot = np.sum((actual - np.mean(actual)) ** 2)
            return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        except:
            return 0
    
    def _style_axis(self, ax, is_top=False, is_bottom=False):
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
        ax.tick_params(axis='x', labelsize=9)
        ax.tick_params(axis='y', labelsize=9)
        
        # Only show x-axis labels on bottom chart
        if not is_bottom:
            ax.set_xticklabels([])
        
        # Reduce tick density for cleaner look
        ax.locator_params(axis='y', nbins=6)
        if is_bottom:
            ax.locator_params(axis='x', nbins=8)