#!/usr/bin/env python3
"""
Market Structure Charts Generator

This module generates comprehensive charts for market structure analysis including:
- Swing point visualization
- BOS/CHOCH annotations
- Support/resistance level highlights
- Trend structure visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Dict, Any, Optional, List
import io
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class MarketStructureCharts:
    """
    Chart generator for market structure analysis visualization.
    
    Generates comprehensive charts showing:
    - Price action with swing points highlighted
    - BOS/CHOCH events marked
    - Support/resistance levels
    - Trend structure visualization
    """
    
    def __init__(self):
        self.name = "market_structure_charts"
        self.chart_style = 'seaborn-v0_8-darkgrid'
        
        # Color scheme
        self.colors = {
            'price': '#1f77b4',
            'swing_high': '#ff4444',
            'swing_low': '#44ff44', 
            'bos_bullish': '#00ff00',
            'bos_bearish': '#ff0000',
            'support': '#44ff44',
            'resistance': '#ff4444',
            'trend_up': '#00aa00',
            'trend_down': '#aa0000',
            'neutral': '#888888'
        }
    
    def generate_market_structure_chart(self, 
                                      stock_data: pd.DataFrame, 
                                      analysis_data: Dict[str, Any],
                                      symbol: str) -> Optional[bytes]:
        """
        Generate comprehensive market structure analysis chart.
        
        Args:
            stock_data: DataFrame with OHLCV data
            analysis_data: Market structure analysis results
            symbol: Stock symbol for chart title
            
        Returns:
            Chart image as bytes, or None if generation fails
        """
        try:
            logger.info(f"[MARKET_STRUCTURE_CHARTS] Generating chart for {symbol}")
            
            if stock_data is None or stock_data.empty:
                logger.error("No stock data provided for chart generation")
                return None
                
            if 'error' in analysis_data:
                logger.error(f"Analysis contains error: {analysis_data['error']}")
                return None
            
            # Set up the plot style
            try:
                plt.style.use(self.chart_style)
            except:
                plt.style.use('default')
            
            # Create figure with subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), height_ratios=[3, 1])
            fig.suptitle(f'Market Structure Analysis - {symbol}', fontsize=16, fontweight='bold')
            
            # Prepare data
            dates = pd.to_datetime(stock_data.index)
            prices = stock_data['close']
            highs = stock_data['high'] 
            lows = stock_data['low']
            volumes = stock_data['volume'] if 'volume' in stock_data.columns else None
            
            # Main price chart
            self._plot_price_action(ax1, dates, prices, highs, lows, analysis_data)
            self._plot_swing_points(ax1, analysis_data)
            self._plot_bos_choch_events(ax1, analysis_data)
            self._plot_key_levels(ax1, analysis_data)
            self._add_trend_annotations(ax1, analysis_data)
            
            # Volume subplot
            if volumes is not None:
                self._plot_volume(ax2, dates, volumes)
            else:
                ax2.text(0.5, 0.5, 'Volume data not available', 
                        transform=ax2.transAxes, ha='center', va='center',
                        fontsize=12, alpha=0.6)
            
            # Add analysis summary
            self._add_analysis_summary(fig, analysis_data)
            
            # Format chart
            self._format_chart(ax1, ax2, dates)
            
            # Save to bytes
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            buffer.seek(0)
            chart_bytes = buffer.getvalue()
            buffer.close()
            plt.close(fig)
            
            logger.info(f"[MARKET_STRUCTURE_CHARTS] Chart generated successfully for {symbol}")
            return chart_bytes
            
        except Exception as e:
            logger.error(f"[MARKET_STRUCTURE_CHARTS] Chart generation failed: {e}")
            plt.close('all')  # Clean up any open figures
            return None
    
    def _plot_price_action(self, ax, dates, prices, highs, lows, analysis_data):
        """Plot main price action"""
        try:
            # Plot price line
            ax.plot(dates, prices, color=self.colors['price'], linewidth=2, label='Close Price')
            
            # Fill between high and low for better visualization
            ax.fill_between(dates, lows, highs, alpha=0.1, color=self.colors['price'])
            
        except Exception as e:
            logger.error(f"Price action plotting failed: {e}")
    
    def _plot_swing_points(self, ax, analysis_data):
        """Plot swing points on the chart"""
        try:
            swing_points = analysis_data.get('swing_points', {})
            swing_highs = swing_points.get('swing_highs', [])
            swing_lows = swing_points.get('swing_lows', [])
            
            # Plot swing highs
            for swing_high in swing_highs:
                try:
                    date = pd.to_datetime(swing_high['date'])
                    price = swing_high['price']
                    strength = swing_high['strength']
                    
                    # Different markers for different strengths
                    marker_size = {'strong': 100, 'medium': 70, 'weak': 40}.get(strength, 50)
                    
                    ax.scatter(date, price, color=self.colors['swing_high'], 
                             s=marker_size, marker='^', alpha=0.8, 
                             edgecolors='black', linewidth=1, 
                             label='Swing High' if swing_high == swing_highs[0] else "")
                    
                    # Add text annotation for strong swings
                    if strength == 'strong':
                        ax.annotate('SH', (date, price), xytext=(5, 5), 
                                  textcoords='offset points', fontsize=8, 
                                  fontweight='bold', color=self.colors['swing_high'])
                        
                except Exception as e:
                    logger.debug(f"Failed to plot swing high: {e}")
                    continue
            
            # Plot swing lows
            for swing_low in swing_lows:
                try:
                    date = pd.to_datetime(swing_low['date'])
                    price = swing_low['price']
                    strength = swing_low['strength']
                    
                    # Different markers for different strengths
                    marker_size = {'strong': 100, 'medium': 70, 'weak': 40}.get(strength, 50)
                    
                    ax.scatter(date, price, color=self.colors['swing_low'], 
                             s=marker_size, marker='v', alpha=0.8,
                             edgecolors='black', linewidth=1,
                             label='Swing Low' if swing_low == swing_lows[0] else "")
                    
                    # Add text annotation for strong swings
                    if strength == 'strong':
                        ax.annotate('SL', (date, price), xytext=(5, -10), 
                                  textcoords='offset points', fontsize=8, 
                                  fontweight='bold', color=self.colors['swing_low'])
                        
                except Exception as e:
                    logger.debug(f"Failed to plot swing low: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Swing point plotting failed: {e}")
    
    def _plot_bos_choch_events(self, ax, analysis_data):
        """Plot BOS and CHOCH events"""
        try:
            bos_choch = analysis_data.get('bos_choch_analysis', {})
            bos_events = bos_choch.get('bos_events', [])
            choch_events = bos_choch.get('choch_events', [])
            
            # Plot BOS events
            for bos in bos_events:
                try:
                    date = pd.to_datetime(bos['date'])
                    price = bos['break_price']
                    bos_type = bos['type']
                    
                    color = self.colors['bos_bullish'] if 'bullish' in bos_type else self.colors['bos_bearish']
                    marker = '↑' if 'bullish' in bos_type else '↓'
                    
                    ax.annotate(f'BOS {marker}', (date, price), 
                              xytext=(10, 10 if 'bullish' in bos_type else -15),
                              textcoords='offset points', fontsize=10, 
                              fontweight='bold', color=color,
                              bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.3))
                              
                except Exception as e:
                    logger.debug(f"Failed to plot BOS event: {e}")
                    continue
            
            # Plot CHOCH events
            for choch in choch_events:
                try:
                    date = pd.to_datetime(choch['date'])
                    # For CHOCH, we need to estimate price from the chart
                    # This is simplified - in a real implementation, you'd store the price
                    choch_type = choch['type']
                    
                    color = self.colors['bos_bullish'] if 'bullish' in choch_type else self.colors['bos_bearish']
                    marker = '⟲' if 'bullish' in choch_type else '⟳'
                    
                    # Place annotation at a reasonable position
                    ylim = ax.get_ylim()
                    y_pos = ylim[1] - (ylim[1] - ylim[0]) * 0.1
                    
                    ax.annotate(f'CHoCH {marker}', (date, y_pos), 
                              xytext=(0, -20), textcoords='offset points',
                              fontsize=9, fontweight='bold', color=color,
                              bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.3))
                              
                except Exception as e:
                    logger.debug(f"Failed to plot CHOCH event: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"BOS/CHOCH plotting failed: {e}")
    
    def _plot_key_levels(self, ax, analysis_data):
        """Plot support and resistance levels"""
        try:
            key_levels = analysis_data.get('key_levels', {})
            support_levels = key_levels.get('support_levels', [])
            resistance_levels = key_levels.get('resistance_levels', [])
            
            xlim = ax.get_xlim()
            
            # Plot support levels
            for support in support_levels:
                try:
                    level = support['level']
                    strength = support['strength']
                    
                    line_style = {'strong': '-', 'medium': '--', 'weak': ':'}.get(strength, '--')
                    alpha = {'strong': 0.8, 'medium': 0.6, 'weak': 0.4}.get(strength, 0.6)
                    
                    ax.axhline(y=level, color=self.colors['support'], 
                             linestyle=line_style, alpha=alpha, linewidth=2)
                    
                    # Add level label
                    ax.text(xlim[1], level, f'  S: {level:.2f}', 
                           verticalalignment='center', fontsize=9,
                           color=self.colors['support'], fontweight='bold')
                           
                except Exception as e:
                    logger.debug(f"Failed to plot support level: {e}")
                    continue
            
            # Plot resistance levels
            for resistance in resistance_levels:
                try:
                    level = resistance['level']
                    strength = resistance['strength']
                    
                    line_style = {'strong': '-', 'medium': '--', 'weak': ':'}.get(strength, '--')
                    alpha = {'strong': 0.8, 'medium': 0.6, 'weak': 0.4}.get(strength, 0.6)
                    
                    ax.axhline(y=level, color=self.colors['resistance'], 
                             linestyle=line_style, alpha=alpha, linewidth=2)
                    
                    # Add level label
                    ax.text(xlim[1], level, f'  R: {level:.2f}', 
                           verticalalignment='center', fontsize=9,
                           color=self.colors['resistance'], fontweight='bold')
                           
                except Exception as e:
                    logger.debug(f"Failed to plot resistance level: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Key levels plotting failed: {e}")
    
    def _add_trend_annotations(self, ax, analysis_data):
        """Add trend direction and quality annotations"""
        try:
            trend_analysis = analysis_data.get('trend_analysis', {})
            trend_direction = trend_analysis.get('trend_direction', 'unknown')
            trend_strength = trend_analysis.get('trend_strength', 'unknown')
            trend_quality = trend_analysis.get('trend_quality', 'unknown')
            
            # Color based on trend
            if trend_direction == 'uptrend':
                trend_color = self.colors['trend_up']
                trend_symbol = '↗'
            elif trend_direction == 'downtrend':
                trend_color = self.colors['trend_down']
                trend_symbol = '↙'
            else:
                trend_color = self.colors['neutral']
                trend_symbol = '→'
            
            # Add trend annotation
            trend_text = f'Trend: {trend_direction.title()} {trend_symbol}\nStrength: {trend_strength.title()}\nQuality: {trend_quality.title()}'
            
            ax.text(0.02, 0.98, trend_text, transform=ax.transAxes, 
                   verticalalignment='top', fontsize=10, fontweight='bold',
                   color=trend_color, bbox=dict(boxstyle='round,pad=0.5', 
                   facecolor='white', alpha=0.8, edgecolor=trend_color))
                   
        except Exception as e:
            logger.error(f"Trend annotation failed: {e}")
    
    def _plot_volume(self, ax, dates, volumes):
        """Plot volume bars"""
        try:
            colors = ['red' if i > 0 and volumes.iloc[i] < volumes.iloc[i-1] else 'green' 
                     for i in range(len(volumes))]
            colors[0] = 'green'  # First bar default
            
            ax.bar(dates, volumes, color=colors, alpha=0.6, width=1)
            ax.set_ylabel('Volume', fontsize=10)
            ax.ticklabel_format(style='plain', axis='y')
            
            # Add volume moving average
            if len(volumes) > 20:
                volume_ma = volumes.rolling(window=20).mean()
                ax.plot(dates, volume_ma, color='blue', linewidth=1, alpha=0.7, label='Volume MA(20)')
                ax.legend(loc='upper right', fontsize=8)
                
        except Exception as e:
            logger.error(f"Volume plotting failed: {e}")
    
    def _add_analysis_summary(self, fig, analysis_data):
        """Add analysis summary text box"""
        try:
            # Extract key metrics
            structure_quality = analysis_data.get('structure_quality', {})
            quality_rating = structure_quality.get('quality_rating', 'unknown')
            quality_score = structure_quality.get('quality_score', 0)
            
            current_state = analysis_data.get('current_state', {})
            structure_state = current_state.get('structure_state', 'unknown')
            price_position = current_state.get('price_position_description', 'unknown')
            
            bos_choch = analysis_data.get('bos_choch_analysis', {})
            structural_bias = bos_choch.get('structural_bias', 'unknown')
            
            swing_points = analysis_data.get('swing_points', {})
            total_swings = swing_points.get('total_swings', 0)
            
            # Create summary text
            summary_text = f"""MARKET STRUCTURE SUMMARY:
Structure Quality: {quality_rating.title()} ({quality_score}/100)
Current State: {structure_state.replace('_', ' ').title()}
Price Position: {price_position.replace('_', ' ').title()}
Structural Bias: {structural_bias.title()}
Total Swing Points: {total_swings}
Analysis Confidence: {analysis_data.get('confidence_score', 0):.2f}"""
            
            # Add text box
            fig.text(0.02, 0.02, summary_text, fontsize=9, 
                    verticalalignment='bottom', fontfamily='monospace',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
                    
        except Exception as e:
            logger.error(f"Analysis summary failed: {e}")
    
    def _format_chart(self, ax1, ax2, dates):
        """Format chart axes and appearance"""
        try:
            # Format main chart
            ax1.set_ylabel('Price', fontsize=12, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc='upper left', fontsize=10)
            
            # Format dates
            if len(dates) > 50:
                ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
                ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
            else:
                ax1.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(dates)//10)))
                ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
            
            ax1.tick_params(axis='x', rotation=45)
            
            # Format volume chart
            ax2.set_ylabel('Volume', fontsize=10)
            ax2.grid(True, alpha=0.3)
            ax2.xaxis.set_major_locator(ax1.xaxis.get_major_locator())
            ax2.xaxis.set_major_formatter(ax1.xaxis.get_major_formatter())
            ax2.tick_params(axis='x', rotation=45)
            
            # Align x-axes
            ax1.set_xlim(dates.min(), dates.max())
            ax2.set_xlim(dates.min(), dates.max())
            
            # Remove x-axis labels from top chart
            ax1.set_xlabel('')
            ax1.tick_params(axis='x', labelbottom=False)
            
        except Exception as e:
            logger.error(f"Chart formatting failed: {e}")

    def generate_quick_structure_chart(self, stock_data: pd.DataFrame, symbol: str) -> Optional[bytes]:
        """
        Generate a simplified market structure chart without full analysis data.
        
        Args:
            stock_data: DataFrame with OHLCV data
            symbol: Stock symbol
            
        Returns:
            Chart image as bytes, or None if generation fails
        """
        try:
            logger.info(f"[MARKET_STRUCTURE_CHARTS] Generating quick chart for {symbol}")
            
            if stock_data is None or stock_data.empty:
                return None
            
            # Set up plot
            plt.style.use('default')
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            fig.suptitle(f'Market Structure - {symbol}', fontsize=14, fontweight='bold')
            
            # Plot price
            dates = pd.to_datetime(stock_data.index)
            prices = stock_data['close']
            highs = stock_data['high']
            lows = stock_data['low']
            
            ax.plot(dates, prices, color=self.colors['price'], linewidth=2)
            ax.fill_between(dates, lows, highs, alpha=0.1, color=self.colors['price'])
            
            # Basic formatting
            ax.set_ylabel('Price', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
            
            # Save to bytes
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            chart_bytes = buffer.getvalue()
            buffer.close()
            plt.close(fig)
            
            return chart_bytes
            
        except Exception as e:
            logger.error(f"Quick chart generation failed: {e}")
            plt.close('all')
            return None