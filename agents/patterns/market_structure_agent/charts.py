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
            
            # Add current price indicator
            self._add_current_price_indicator(ax1, analysis_data, dates)
            
            # Add analysis summary
            self._add_analysis_summary(fig, analysis_data)
            
            # Add legend with key visual encodings
            try:
                from matplotlib.lines import Line2D
                legend_elems = [
                    Line2D([], [], color=self.colors['price'], lw=2, label='Close Price'),
                    Line2D([], [], marker='^', color='w', markerfacecolor=self.colors['swing_high'],
                           markeredgecolor='black', linestyle='None', label='Swing High (▲)'),
                    Line2D([], [], marker='v', color='w', markerfacecolor=self.colors['swing_low'],
                           markeredgecolor='black', linestyle='None', label='Swing Low (▼)'),
                    Line2D([], [], color=self.colors['support'], lw=2, linestyle='--', label='Support'),
                    Line2D([], [], color=self.colors['resistance'], lw=2, linestyle='--', label='Resistance'),
                    Line2D([], [], color=self.colors['bos_bullish'], lw=0, marker='^', label='BOS ↑ (bullish)'),
                    Line2D([], [], color=self.colors['bos_bearish'], lw=0, marker='v', label='BOS ↓ (bearish)'),
                    Line2D([], [], color='orange', lw=2, linestyle=':', label='Current Price')
                ]
                ax1.legend(handles=legend_elems, loc='upper left', fontsize=9, frameon=True)
            except Exception as _:
                pass

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
        """Plot swing points with improved visual hierarchy based on strength"""
        try:
            swing_points = analysis_data.get('swing_points', {})
            swing_highs = swing_points.get('swing_highs', [])
            swing_lows = swing_points.get('swing_lows', [])
            
            # Plot swing highs with enhanced styling
            for swing_high in swing_highs:
                try:
                    date = pd.to_datetime(swing_high['date'])
                    price = swing_high['price']
                    strength = swing_high['strength']
                    
                    # Enhanced visual differentiation by strength
                    size_map = {'strong': 120, 'medium': 80, 'weak': 50}
                    alpha_map = {'strong': 1.0, 'medium': 0.8, 'weak': 0.6}
                    edge_width = {'strong': 2, 'medium': 1.5, 'weak': 1}.get(strength, 1)
                    
                    marker_size = size_map.get(strength, 60)
                    alpha = alpha_map.get(strength, 0.7)
                    
                    ax.scatter(date, price, color=self.colors['swing_high'], 
                             s=marker_size, marker='^', alpha=alpha, 
                             edgecolors='darkred', linewidth=edge_width, 
                             label='Swing High' if swing_high == swing_highs[0] else "", zorder=5)
                    
                    # Only add SH label for strong swings to reduce clutter
                    if strength == 'strong':
                        ax.annotate('SH', (date, price), xytext=(3, 8), 
                                  textcoords='offset points', fontsize=9, 
                                  fontweight='bold', color='darkred',
                                  bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='darkred'))
                        
                except Exception as e:
                    logger.debug(f"Failed to plot swing high: {e}")
                    continue
            
            # Plot swing lows with enhanced styling
            for swing_low in swing_lows:
                try:
                    date = pd.to_datetime(swing_low['date'])
                    price = swing_low['price']
                    strength = swing_low['strength']
                    
                    # Enhanced visual differentiation by strength
                    size_map = {'strong': 120, 'medium': 80, 'weak': 50}
                    alpha_map = {'strong': 1.0, 'medium': 0.8, 'weak': 0.6}
                    edge_width = {'strong': 2, 'medium': 1.5, 'weak': 1}.get(strength, 1)
                    
                    marker_size = size_map.get(strength, 60)
                    alpha = alpha_map.get(strength, 0.7)
                    
                    ax.scatter(date, price, color=self.colors['swing_low'], 
                             s=marker_size, marker='v', alpha=alpha,
                             edgecolors='darkgreen', linewidth=edge_width,
                             label='Swing Low' if swing_low == swing_lows[0] else "", zorder=5)
                    
                    # Only add SL label for strong swings to reduce clutter
                    if strength == 'strong':
                        ax.annotate('SL', (date, price), xytext=(3, -12), 
                                  textcoords='offset points', fontsize=9, 
                                  fontweight='bold', color='darkgreen',
                                  bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='darkgreen'))
                        
                except Exception as e:
                    logger.debug(f"Failed to plot swing low: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Swing point plotting failed: {e}")
    
    def _plot_bos_choch_events(self, ax, analysis_data):
        """Plot BOS and CHOCH events with smart positioning to prevent overlaps"""
        try:
            bos_choch = analysis_data.get('bos_choch_analysis', {})
            bos_events = bos_choch.get('bos_events', [])
            choch_events = bos_choch.get('choch_events', [])
            
            # Track annotation positions to prevent overlaps
            used_positions = []  # Store (date, y_position) tuples
            
            # Plot BOS events with improved positioning
            for i, bos in enumerate(bos_events):
                try:
                    date = pd.to_datetime(bos['date'])
                    price = bos['break_price']
                    bos_type = bos['type']
                    strength = bos.get('strength', 'medium')
                    
                    color = self.colors['bos_bullish'] if 'bullish' in bos_type else self.colors['bos_bearish']
                    marker = '↑' if 'bullish' in bos_type else '↓'
                    
                    # Implement alternating positions for consecutive BOS events
                    base_offset = 15 if 'bullish' in bos_type else -25
                    # Alternate above/below for better spacing in congested areas
                    if i > 0:
                        prev_date = pd.to_datetime(bos_events[i-1]['date'])
                        days_apart = abs((date - prev_date).days)
                        if days_apart <= 30:  # Events within 30 days - use alternating positions
                            base_offset = base_offset * (-1 if i % 2 == 1 else 1)
                    
                    y_offset = self._find_annotation_offset(date, price, used_positions, base_offset)
                    
                    # Add price to the label and use different styles for strength
                    font_size = {'strong': 10, 'medium': 9, 'weak': 8}.get(strength, 9)
                    alpha = {'strong': 0.8, 'medium': 0.6, 'weak': 0.4}.get(strength, 0.6)
                    
                    annotation = ax.annotate(f'BOS {marker} {price:.1f}', (date, price), 
                              xytext=(5, y_offset), textcoords='offset points', 
                              fontsize=font_size, fontweight='bold', color=color,
                              bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=alpha),
                              ha='left', va='center')
                    
                    # Record this position
                    used_positions.append((date, price + y_offset/100))
                              
                except Exception as e:
                    logger.debug(f"Failed to plot BOS event: {e}")
                    continue
            
            # Plot CHOCH events (if any)
            for choch in choch_events:
                try:
                    date = pd.to_datetime(choch['date'])
                    choch_type = choch['type']
                    
                    color = self.colors['bos_bullish'] if 'bullish' in choch_type else self.colors['bos_bearish']
                    marker = '⟲' if 'bullish' in choch_type else '⟳'
                    
                    # Position CHoCH events at top of chart
                    ylim = ax.get_ylim()
                    y_pos = ylim[1] - (ylim[1] - ylim[0]) * 0.08
                    
                    y_offset = self._find_annotation_offset(date, y_pos, used_positions, -15)
                    
                    ax.annotate(f'CHoCH {marker}', (date, y_pos), 
                              xytext=(5, y_offset), textcoords='offset points',
                              fontsize=9, fontweight='bold', color=color,
                              bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.6),
                              ha='left', va='center')
                    
                    used_positions.append((date, y_pos))
                              
                except Exception as e:
                    logger.debug(f"Failed to plot CHOCH event: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"BOS/CHOCH plotting failed: {e}")
    
    def _find_annotation_offset(self, target_date, target_price, used_positions, base_offset, min_separation_days=7):
        """Find a Y offset for annotation that avoids overlapping with existing annotations"""
        import datetime as dt
        
        for used_date, used_price in used_positions:
            # Check if dates are close (within min_separation_days)
            date_diff = abs((target_date - used_date).days)
            
            if date_diff <= min_separation_days:
                # Dates are close, check if prices would cause overlap
                price_diff = abs(target_price - used_price)
                
                # If overlap detected, adjust the offset
                if price_diff < abs(base_offset) * 0.01:  # Convert offset to price units roughly
                    return base_offset + (20 if base_offset > 0 else -20)  # Stack them
        
        return base_offset
    
    def _plot_key_levels(self, ax, analysis_data):
        """Plot filtered support and resistance levels to reduce clutter"""
        try:
            key_levels = analysis_data.get('key_levels', {})
            current_price = key_levels.get('current_price', 0)
            
            if current_price == 0:
                return
            
            # Filter levels by proximity and strength
            support_levels = self._filter_key_levels(
                key_levels.get('support_levels', []), current_price, max_levels=5
            )
            resistance_levels = self._filter_key_levels(
                key_levels.get('resistance_levels', []), current_price, max_levels=5
            )
            
            # Track label positions to prevent overlaps
            label_positions = []
            min_label_spacing = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02  # 2% of chart height
            
            # Plot support levels
            for support in support_levels:
                try:
                    level = support['level']
                    strength = support['strength']
                    
                    line_style = {'strong': '-', 'medium': '--', 'weak': ':'}.get(strength, '--')
                    alpha = {'strong': 0.9, 'medium': 0.7, 'weak': 0.5}.get(strength, 0.6)
                    linewidth = {'strong': 2.5, 'medium': 2, 'weak': 1.5}.get(strength, 2)
                    
                    ax.axhline(y=level, color=self.colors['support'], 
                             linestyle=line_style, alpha=alpha, linewidth=linewidth)
                    
                    # Smart label positioning to avoid overlaps
                    label_y = self._find_label_position(level, label_positions, min_label_spacing)
                    label_positions.append(label_y)
                    
                    # Add level label with strength indicator
                    strength_symbol = {'strong': '●', 'medium': '◐', 'weak': '○'}.get(strength, '◐')
                    ax.text(1.005, label_y, f'S {strength_symbol} {level:.1f}', 
                           transform=ax.get_yaxis_transform(),
                           verticalalignment='center', fontsize=8, clip_on=False,
                           color=self.colors['support'], fontweight='bold', ha='left')
                           
                except Exception as e:
                    logger.debug(f"Failed to plot support level: {e}")
                    continue
            
            # Plot resistance levels
            for resistance in resistance_levels:
                try:
                    level = resistance['level']
                    strength = resistance['strength']
                    
                    line_style = {'strong': '-', 'medium': '--', 'weak': ':'}.get(strength, '--')
                    alpha = {'strong': 0.9, 'medium': 0.7, 'weak': 0.5}.get(strength, 0.6)
                    linewidth = {'strong': 2.5, 'medium': 2, 'weak': 1.5}.get(strength, 2)
                    
                    ax.axhline(y=level, color=self.colors['resistance'], 
                             linestyle=line_style, alpha=alpha, linewidth=linewidth)
                    
                    # Smart label positioning
                    label_y = self._find_label_position(level, label_positions, min_label_spacing)
                    label_positions.append(label_y)
                    
                    # Add level label with strength indicator
                    strength_symbol = {'strong': '●', 'medium': '◐', 'weak': '○'}.get(strength, '◐')
                    ax.text(1.005, label_y, f'R {strength_symbol} {level:.1f}', 
                           transform=ax.get_yaxis_transform(),
                           verticalalignment='center', fontsize=8, clip_on=False,
                           color=self.colors['resistance'], fontweight='bold', ha='left')
                           
                except Exception as e:
                    logger.debug(f"Failed to plot resistance level: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Key levels plotting failed: {e}")
    
    def _filter_key_levels(self, levels, current_price, max_levels=5, price_range_pct=0.15):
        """Filter levels by proximity to current price and strength"""
        if not levels or current_price == 0:
            return []
            
        # Filter by price proximity (within ±15% of current price)
        nearby_levels = [
            level for level in levels 
            if abs(level['level'] - current_price) / current_price < price_range_pct
        ]
        
        # Sort by strength (strong > medium > weak) then by proximity to current price
        strength_priority = {'strong': 3, 'medium': 2, 'weak': 1}
        
        sorted_levels = sorted(nearby_levels, key=lambda x: (
            strength_priority.get(x['strength'], 0),  # Strength priority
            -abs(x['level'] - current_price)  # Proximity (negative for closer = higher priority)
        ), reverse=True)
        
        return sorted_levels[:max_levels]
    
    def _find_label_position(self, target_y, existing_positions, min_spacing):
        """Find a position for a label that doesn't overlap with existing labels"""
        for pos in existing_positions:
            if abs(target_y - pos) < min_spacing:
                # Find an offset position
                offset = min_spacing * 1.2
                # Try above first
                candidate = target_y + offset
                if not any(abs(candidate - pos) < min_spacing for pos in existing_positions):
                    return candidate
                # Try below
                candidate = target_y - offset
                if not any(abs(candidate - pos) < min_spacing for pos in existing_positions):
                    return candidate
        return target_y
    
    def _add_current_price_indicator(self, ax, analysis_data, dates):
        """Add current price indicator line and label"""
        try:
            key_levels = analysis_data.get('key_levels', {})
            current_price = key_levels.get('current_price')
            
            if not current_price:
                # Try getting from current_state as fallback
                current_state = analysis_data.get('current_state', {})
                current_price = current_state.get('current_price')
            
            if current_price and len(dates) > 0:
                # Add horizontal dotted line for current price
                ax.axhline(y=current_price, color='orange', linestyle=':', 
                          linewidth=2.5, alpha=0.9, zorder=4)
                
                # Add current price label at the right edge
                ax.text(dates.iloc[-1] if hasattr(dates, 'iloc') else dates[-1], 
                       current_price, f'  Current: {current_price:.2f}',
                       verticalalignment='center', horizontalalignment='left',
                       fontsize=10, fontweight='bold', color='darkorange',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                alpha=0.9, edgecolor='orange', linewidth=1.5))
                
        except Exception as e:
            logger.debug(f"Failed to add current price indicator: {e}")
    
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
        """Add analysis summary text box with improved positioning"""
        try:
            # Extract key metrics
            structure_quality = analysis_data.get('structure_quality', {})
            quality_rating = structure_quality.get('quality_rating', 'unknown')
            quality_score = structure_quality.get('quality_score', 0)
            
            current_state = analysis_data.get('current_state', {})
            structure_state = current_state.get('structure_state', 'unknown')
            current_price = current_state.get('current_price', 0)
            
            bos_choch = analysis_data.get('bos_choch_analysis', {})
            structural_bias = bos_choch.get('structural_bias', 'unknown')
            recent_break = bos_choch.get('recent_structural_break', {})
            
            key_levels = analysis_data.get('key_levels', {})
            nearest_support = key_levels.get('nearest_support', {}).get('level', 'N/A')
            nearest_resistance = key_levels.get('nearest_resistance', {}).get('level', 'N/A')
            
            swing_points = analysis_data.get('swing_points', {})
            total_swings = swing_points.get('total_swings', 0)
            
            # Create more concise summary text
            recent_break_type = recent_break.get('type', 'none').replace('_', ' ').title()
            
            # Format confidence score as percentage
            confidence_raw = analysis_data.get('confidence_score', 0)
            confidence_pct = f"{confidence_raw*100:.0f}%" if confidence_raw <= 1 else f"{confidence_raw:.0f}%"
            
            summary_text = f"""STRUCTURE: {quality_rating.title()} ({quality_score}/100) | {structural_bias.title()} Bias
CURRENT: {current_price:.1f} | Support: {nearest_support} | Resistance: {nearest_resistance}
RECENT: {recent_break_type} | STATE: {structure_state.replace('_', ' ').title()}
SWINGS: {total_swings} total | CONFIDENCE: {confidence_pct}"""
            
            # Position at bottom-left with improved font size and styling
            fig.text(0.02, 0.02, summary_text, fontsize=9,
                    verticalalignment='bottom', fontfamily='monospace', fontweight='normal',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.95, edgecolor='gray', linewidth=1))
                    
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