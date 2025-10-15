#!/usr/bin/env python3
"""
Market Structure Chart Generation Test

This module tests chart generation for different market structure scenarios:
- Uptrend with clear BOS events
- Downtrend with structural breaks
- Sideways/consolidation with ranging structure
- Complex multi-phase market structure

Tests chart generation, saving, and mock LLM integration setup.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple
import os
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketStructureChartGenerator:
    """
    Enhanced chart generator for market structure analysis with comprehensive visualizations.
    """
    
    def __init__(self, output_dir: str = "test_charts"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Enhanced color scheme for market structure
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
            'consolidation': '#ffd700'
        }
        
        plt.style.use('seaborn-v0_8-darkgrid')
    
    def create_comprehensive_chart(self, 
                                 stock_data: pd.DataFrame, 
                                 analysis_data: Dict[str, Any], 
                                 symbol: str, 
                                 scenario: str) -> str:
        """
        Create comprehensive market structure chart and save as image.
        
        Args:
            stock_data: OHLCV DataFrame
            analysis_data: Market structure analysis results
            symbol: Stock symbol
            scenario: Scenario name for filename
            
        Returns:
            Path to saved chart image
        """
        try:
            logger.info(f"Generating chart for {symbol} - {scenario}")
            
            # Create figure with enhanced layout
            fig = plt.figure(figsize=(20, 16))
            gs = fig.add_gridspec(4, 2, height_ratios=[3, 1, 1, 1], width_ratios=[3, 1], 
                                hspace=0.3, wspace=0.2)
            
            # Main price chart
            ax_main = fig.add_subplot(gs[0, :])
            
            # Volume chart
            ax_volume = fig.add_subplot(gs[1, :], sharex=ax_main)
            
            # Analysis metrics
            ax_metrics1 = fig.add_subplot(gs[2, 0])
            ax_metrics2 = fig.add_subplot(gs[2, 1])
            
            # Summary stats
            ax_summary = fig.add_subplot(gs[3, :])
            
            # Set main title
            fig.suptitle(f'Market Structure Analysis - {symbol} ({scenario})', 
                        fontsize=18, fontweight='bold', y=0.98)
            
            # Plot all components
            self._plot_price_structure(ax_main, stock_data, analysis_data)
            self._plot_volume_analysis(ax_volume, stock_data, analysis_data)
            self._plot_swing_metrics(ax_metrics1, analysis_data)
            self._plot_structure_metrics(ax_metrics2, analysis_data)
            self._plot_analysis_summary(ax_summary, analysis_data, scenario)
            
            # Format and save
            self._format_chart_axes(ax_main, ax_volume, stock_data)
            
            # Save chart
            filename = f"{symbol}_{scenario}_market_structure.png"
            filepath = self.output_dir / filename
            
            plt.savefig(filepath, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close(fig)
            
            logger.info(f"Chart saved: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Chart generation failed: {e}")
            plt.close('all')
            return None
    
    def _plot_price_structure(self, ax, stock_data: pd.DataFrame, analysis_data: Dict):
        """Plot comprehensive price structure with all market structure elements"""
        
        dates = pd.to_datetime(stock_data.index)
        
        # Plot price action
        ax.plot(dates, stock_data['close'], color=self.colors['price'], 
               linewidth=2.5, label='Close Price', zorder=3)
        
        # Fill between high and low
        ax.fill_between(dates, stock_data['low'], stock_data['high'], 
                       alpha=0.15, color=self.colors['price_fill'], zorder=1)
        
        # Plot swing points
        self._plot_swing_points_enhanced(ax, analysis_data)
        
        # Plot BOS/CHOCH events  
        self._plot_structural_breaks(ax, analysis_data)
        
        # Plot support/resistance levels
        self._plot_key_levels_enhanced(ax, analysis_data)
        
        # Plot trend structure
        self._plot_trend_structure(ax, analysis_data)
        
        # Add current price indicator
        current_price = stock_data['close'].iloc[-1]
        ax.axhline(y=current_price, color='black', linestyle=':', 
                  alpha=0.7, linewidth=2, label=f'Current: {current_price:.2f}')
        
        ax.set_title('Price Action & Market Structure', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    def _plot_swing_points_enhanced(self, ax, analysis_data: Dict):
        """Enhanced swing point visualization with connections"""
        
        swing_points = analysis_data.get('swing_points', {})
        swing_highs = swing_points.get('swing_highs', [])
        swing_lows = swing_points.get('swing_lows', [])
        
        # Plot swing highs
        high_dates, high_prices, high_strengths = [], [], []
        for swing in swing_highs:
            try:
                high_dates.append(pd.to_datetime(swing['date']))
                high_prices.append(swing['price'])
                high_strengths.append(swing['strength'])
            except:
                continue
        
        if high_dates:
            # Different sizes for different strengths
            sizes = [120 if s == 'strong' else 80 if s == 'medium' else 50 
                    for s in high_strengths]
            
            ax.scatter(high_dates, high_prices, c=self.colors['swing_high'], 
                      s=sizes, marker='^', alpha=0.8, edgecolors='black', 
                      linewidth=1.5, label='Swing Highs', zorder=5)
            
            # Connect swing highs with trend line
            if len(high_dates) > 1:
                ax.plot(high_dates, high_prices, color=self.colors['swing_high'], 
                       linestyle='--', alpha=0.5, linewidth=2, zorder=2)
        
        # Plot swing lows
        low_dates, low_prices, low_strengths = [], [], []
        for swing in swing_lows:
            try:
                low_dates.append(pd.to_datetime(swing['date']))
                low_prices.append(swing['price'])
                low_strengths.append(swing['strength'])
            except:
                continue
        
        if low_dates:
            sizes = [120 if s == 'strong' else 80 if s == 'medium' else 50 
                    for s in low_strengths]
            
            ax.scatter(low_dates, low_prices, c=self.colors['swing_low'], 
                      s=sizes, marker='v', alpha=0.8, edgecolors='black', 
                      linewidth=1.5, label='Swing Lows', zorder=5)
            
            # Connect swing lows with trend line
            if len(low_dates) > 1:
                ax.plot(low_dates, low_prices, color=self.colors['swing_low'], 
                       linestyle='--', alpha=0.5, linewidth=2, zorder=2)
    
    def _plot_structural_breaks(self, ax, analysis_data: Dict):
        """Plot BOS and CHOCH events with enhanced visualization"""
        
        bos_choch = analysis_data.get('bos_choch_analysis', {})
        bos_events = bos_choch.get('bos_events', [])
        choch_events = bos_choch.get('choch_events', [])
        
        # Plot BOS events
        for i, bos in enumerate(bos_events):
            try:
                date = pd.to_datetime(bos['date'])
                price = bos['break_price']
                bos_type = bos['type']
                strength = bos.get('strength', 'medium')
                
                is_bullish = 'bullish' in bos_type
                color = self.colors['bos_bullish'] if is_bullish else self.colors['bos_bearish']
                arrow = '↑' if is_bullish else '↓'
                
                # Different annotation positions to avoid overlap
                y_offset = 15 + (i % 3) * 10 if is_bullish else -25 - (i % 3) * 10
                
                ax.annotate(f'BOS {arrow}', (date, price), 
                          xytext=(10, y_offset), textcoords='offset points',
                          fontsize=11, fontweight='bold', color=color,
                          bbox=dict(boxstyle='round,pad=0.4', facecolor=color, 
                                   alpha=0.3, edgecolor=color),
                          arrowprops=dict(arrowstyle='->', color=color, alpha=0.7),
                          zorder=6)
                
            except Exception as e:
                logger.debug(f"Failed to plot BOS: {e}")
        
        # Plot CHOCH events
        for i, choch in enumerate(choch_events):
            try:
                date = pd.to_datetime(choch['date'])
                # Estimate price position for CHOCH
                ylim = ax.get_ylim()
                price = ylim[1] - (ylim[1] - ylim[0]) * (0.1 + i * 0.05)
                
                choch_type = choch['type']
                is_bullish = 'bullish' in choch_type
                color = self.colors['choch_bullish'] if is_bullish else self.colors['choch_bearish']
                symbol = '⟲' if is_bullish else '⟳'
                
                ax.annotate(f'CHoCH {symbol}', (date, price), 
                          xytext=(0, -15), textcoords='offset points',
                          fontsize=10, fontweight='bold', color=color,
                          bbox=dict(boxstyle='round,pad=0.4', facecolor=color, 
                                   alpha=0.3, edgecolor=color),
                          zorder=6)
                
            except Exception as e:
                logger.debug(f"Failed to plot CHOCH: {e}")
    
    def _plot_key_levels_enhanced(self, ax, analysis_data: Dict):
        """Enhanced support/resistance level visualization"""
        
        key_levels = analysis_data.get('key_levels', {})
        support_levels = key_levels.get('support_levels', [])
        resistance_levels = key_levels.get('resistance_levels', [])
        
        xlim = ax.get_xlim()
        
        # Plot support levels
        for support in support_levels[-5:]:  # Show only recent 5 levels
            try:
                level = support['level']
                strength = support['strength']
                
                line_width = {'strong': 3, 'medium': 2, 'weak': 1}.get(strength, 2)
                line_style = {'strong': '-', 'medium': '--', 'weak': ':'}.get(strength, '--')
                alpha = {'strong': 0.9, 'medium': 0.7, 'weak': 0.5}.get(strength, 0.7)
                
                ax.axhline(y=level, color=self.colors['support'], 
                          linestyle=line_style, alpha=alpha, linewidth=line_width,
                          zorder=2)
                
                # Add level label
                ax.text(xlim[1], level, f'  S: {level:.2f} ({strength})', 
                       verticalalignment='center', fontsize=9,
                       color=self.colors['support'], fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
                
            except Exception as e:
                logger.debug(f"Failed to plot support: {e}")
        
        # Plot resistance levels  
        for resistance in resistance_levels[-5:]:  # Show only recent 5 levels
            try:
                level = resistance['level']
                strength = resistance['strength']
                
                line_width = {'strong': 3, 'medium': 2, 'weak': 1}.get(strength, 2)
                line_style = {'strong': '-', 'medium': '--', 'weak': ':'}.get(strength, '--')
                alpha = {'strong': 0.9, 'medium': 0.7, 'weak': 0.5}.get(strength, 0.7)
                
                ax.axhline(y=level, color=self.colors['resistance'], 
                          linestyle=line_style, alpha=alpha, linewidth=line_width,
                          zorder=2)
                
                # Add level label
                ax.text(xlim[1], level, f'  R: {level:.2f} ({strength})', 
                       verticalalignment='center', fontsize=9,
                       color=self.colors['resistance'], fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
                
            except Exception as e:
                logger.debug(f"Failed to plot resistance: {e}")
    
    def _plot_trend_structure(self, ax, analysis_data: Dict):
        """Plot trend structure and market regime indicators"""
        
        trend_analysis = analysis_data.get('trend_analysis', {})
        trend_direction = trend_analysis.get('trend_direction', 'sideways')
        trend_strength = trend_analysis.get('trend_strength', 'weak')
        
        # Color and symbol based on trend
        if trend_direction == 'uptrend':
            color = self.colors['trend_up']
            symbol = '↗'
        elif trend_direction == 'downtrend':
            color = self.colors['trend_down'] 
            symbol = '↙'
        else:
            color = self.colors['neutral']
            symbol = '→'
        
        # Add trend indicator box
        trend_text = f'{symbol} {trend_direction.upper()}\n{trend_strength.upper()}'
        
        ax.text(0.02, 0.98, trend_text, transform=ax.transAxes,
               verticalalignment='top', horizontalalignment='left',
               fontsize=12, fontweight='bold', color=color,
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                        alpha=0.9, edgecolor=color, linewidth=2))
    
    def _plot_volume_analysis(self, ax, stock_data: pd.DataFrame, analysis_data: Dict):
        """Plot volume with trend analysis"""
        
        if 'volume' not in stock_data.columns:
            ax.text(0.5, 0.5, 'Volume Data Not Available', 
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=14, alpha=0.6)
            return
        
        dates = pd.to_datetime(stock_data.index)
        volumes = stock_data['volume']
        closes = stock_data['close']
        
        # Color volume bars based on price direction
        colors = []
        for i in range(len(volumes)):
            if i == 0:
                colors.append(self.colors['volume_up'])
            else:
                color = self.colors['volume_up'] if closes.iloc[i] >= closes.iloc[i-1] else self.colors['volume_down']
                colors.append(color)
        
        ax.bar(dates, volumes, color=colors, alpha=0.7, width=0.8)
        
        # Add volume moving averages
        if len(volumes) > 20:
            vol_ma20 = volumes.rolling(window=20).mean()
            ax.plot(dates, vol_ma20, color='blue', linewidth=2, 
                   alpha=0.8, label='Vol MA(20)')
        
        if len(volumes) > 50:
            vol_ma50 = volumes.rolling(window=50).mean()
            ax.plot(dates, vol_ma50, color='orange', linewidth=2, 
                   alpha=0.8, label='Vol MA(50)')
        
        ax.set_title('Volume Analysis', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.ticklabel_format(style='plain', axis='y')
    
    def _plot_swing_metrics(self, ax, analysis_data: Dict):
        """Plot swing point metrics as bar chart"""
        
        swing_points = analysis_data.get('swing_points', {})
        
        # Extract swing strength distribution
        swing_highs = swing_points.get('swing_highs', [])
        swing_lows = swing_points.get('swing_lows', [])
        
        strength_counts = {'strong': 0, 'medium': 0, 'weak': 0}
        
        for swing in swing_highs + swing_lows:
            strength = swing.get('strength', 'unknown')
            if strength in strength_counts:
                strength_counts[strength] += 1
        
        strengths = list(strength_counts.keys())
        counts = list(strength_counts.values())
        colors = ['#ff4444', '#ffaa44', '#44ff44']  # Red, Orange, Green
        
        bars = ax.bar(strengths, counts, color=colors, alpha=0.7, edgecolor='black')
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{count}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_title('Swing Point Strength Distribution', fontsize=11, fontweight='bold')
        ax.set_ylabel('Count')
    
    def _plot_structure_metrics(self, ax, analysis_data: Dict):
        """Plot structural break metrics"""
        
        bos_choch = analysis_data.get('bos_choch_analysis', {})
        bos_count = len(bos_choch.get('bos_events', []))
        choch_count = len(bos_choch.get('choch_events', []))
        
        # Count bullish vs bearish BOS
        bullish_bos = sum(1 for bos in bos_choch.get('bos_events', []) 
                         if 'bullish' in bos.get('type', ''))
        bearish_bos = bos_count - bullish_bos
        
        categories = ['Bullish\nBOS', 'Bearish\nBOS', 'CHoCH\nEvents']
        counts = [bullish_bos, bearish_bos, choch_count]
        colors = ['#44aa44', '#aa4444', '#4444aa']
        
        bars = ax.bar(categories, counts, color=colors, alpha=0.7, edgecolor='black')
        
        # Add value labels
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{count}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_title('Structural Break Analysis', fontsize=11, fontweight='bold')
        ax.set_ylabel('Count')
    
    def _plot_analysis_summary(self, ax, analysis_data: Dict, scenario: str):
        """Plot comprehensive analysis summary"""
        
        ax.axis('off')  # Turn off axes for text summary
        
        # Extract key metrics
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
        
        current_state = analysis_data.get('current_state', {})
        structure_state = current_state.get('structure_state', 'unknown')
        
        # Create formatted summary
        summary_text = f"""
MARKET STRUCTURE ANALYSIS SUMMARY - {scenario.upper()}

📊 STRUCTURE QUALITY
   Quality Score: {quality_score}/100 ({quality_rating.title()})
   
📈 TREND ANALYSIS  
   Direction: {trend_direction.title()}
   Strength: {trend_strength.title()}
   Current State: {structure_state.replace('_', ' ').title()}
   
🔄 SWING STRUCTURE
   Total Swing Points: {total_swings}
   Swing Density: {swing_density:.3f}
   
⚡ STRUCTURAL BIAS
   Market Bias: {structural_bias.title()}
   
📅 ANALYSIS PERIOD
   Data Quality Score: {analysis_data.get('data_quality', {}).get('overall_quality_score', 0)}/100
   Confidence Level: {analysis_data.get('confidence_score', 0):.2f}
        """.strip()
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
               fontsize=11, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round,pad=0.7', facecolor='lightblue', 
                        alpha=0.8, edgecolor='navy'))
    
    def _format_chart_axes(self, ax_main, ax_volume, stock_data):
        """Format chart axes for better readability"""
        
        # Format x-axis dates
        dates = pd.to_datetime(stock_data.index)
        
        # Set date formatting
        ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax_main.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        ax_volume.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax_volume.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        
        # Rotate date labels
        plt.setp(ax_main.xaxis.get_majorticklabels(), rotation=45)
        plt.setp(ax_volume.xaxis.get_majorticklabels(), rotation=45)
        
        # Set y-axis formatting
        ax_main.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:.2f}'))
        ax_volume.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
        
        # Add grid
        ax_main.grid(True, alpha=0.3)
        ax_volume.grid(True, alpha=0.3)


def create_mock_data_scenarios() -> List[Tuple[pd.DataFrame, Dict, str, str]]:
    """
    Create mock data for different market structure scenarios.
    
    Returns:
        List of (stock_data, analysis_data, symbol, scenario) tuples
    """
    scenarios = []
    
    # Scenario 1: Strong Uptrend with BOS
    stock_data_1, analysis_1 = create_uptrend_scenario()
    scenarios.append((stock_data_1, analysis_1, "TECH", "strong_uptrend"))
    
    # Scenario 2: Clear Downtrend with Structural Breaks
    stock_data_2, analysis_2 = create_downtrend_scenario()
    scenarios.append((stock_data_2, analysis_2, "BANK", "clear_downtrend"))
    
    # Scenario 3: Sideways Consolidation
    stock_data_3, analysis_3 = create_sideways_scenario()
    scenarios.append((stock_data_3, analysis_3, "UTIL", "sideways_consolidation"))
    
    # Scenario 4: Complex Multi-Phase Structure
    stock_data_4, analysis_4 = create_complex_scenario()
    scenarios.append((stock_data_4, analysis_4, "GROWTH", "complex_structure"))
    
    return scenarios


def create_uptrend_scenario() -> Tuple[pd.DataFrame, Dict]:
    """Create mock data for strong uptrend scenario"""
    
    # Generate dates
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    
    # Generate uptrending price data with realistic noise
    np.random.seed(42)
    base_trend = np.linspace(100, 180, len(dates))
    noise = np.random.normal(0, 2, len(dates))
    volatility = np.random.normal(0, 5, len(dates))
    
    close_prices = base_trend + noise
    high_prices = close_prices + np.abs(volatility) + np.random.uniform(0.5, 3, len(dates))
    low_prices = close_prices - np.abs(volatility) - np.random.uniform(0.5, 3, len(dates))
    open_prices = close_prices + np.random.uniform(-2, 2, len(dates))
    volumes = np.random.randint(1000000, 5000000, len(dates))
    
    stock_data = pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes
    }, index=dates)
    
    # Create corresponding analysis data
    analysis_data = {
        'swing_points': {
            'swing_highs': [
                {'date': '2024-02-15', 'price': 115.5, 'strength': 'medium', 'index': 45},
                {'date': '2024-04-10', 'price': 135.2, 'strength': 'strong', 'index': 100},
                {'date': '2024-06-20', 'price': 155.8, 'strength': 'strong', 'index': 171},
                {'date': '2024-08-25', 'price': 168.9, 'strength': 'medium', 'index': 237},
                {'date': '2024-11-10', 'price': 177.3, 'strength': 'weak', 'index': 314},
            ],
            'swing_lows': [
                {'date': '2024-01-20', 'price': 102.3, 'strength': 'strong', 'index': 20},
                {'date': '2024-03-05', 'price': 118.7, 'strength': 'medium', 'index': 64},
                {'date': '2024-05-15', 'price': 142.1, 'strength': 'strong', 'index': 135},
                {'date': '2024-07-30', 'price': 158.4, 'strength': 'medium', 'index': 211},
                {'date': '2024-10-05', 'price': 165.2, 'strength': 'weak', 'index': 278},
            ],
            'total_swings': 10,
            'swing_density': 0.027
        },
        'bos_choch_analysis': {
            'bos_events': [
                {'type': 'bullish_bos', 'date': '2024-04-10', 'break_price': 135.2, 'strength': 'strong'},
                {'type': 'bullish_bos', 'date': '2024-06-20', 'break_price': 155.8, 'strength': 'strong'},
                {'type': 'bullish_bos', 'date': '2024-08-25', 'break_price': 168.9, 'strength': 'medium'},
            ],
            'choch_events': [],
            'structural_bias': 'bullish'
        },
        'trend_analysis': {
            'trend_direction': 'uptrend',
            'trend_strength': 'strong',
            'trend_quality': 'excellent'
        },
        'key_levels': {
            'support_levels': [
                {'level': 102.3, 'strength': 'strong', 'date': '2024-01-20'},
                {'level': 118.7, 'strength': 'medium', 'date': '2024-03-05'},
                {'level': 142.1, 'strength': 'strong', 'date': '2024-05-15'},
                {'level': 158.4, 'strength': 'medium', 'date': '2024-07-30'},
            ],
            'resistance_levels': [
                {'level': 115.5, 'strength': 'weak', 'date': '2024-02-15'},
                {'level': 135.2, 'strength': 'weak', 'date': '2024-04-10'},
                {'level': 155.8, 'strength': 'weak', 'date': '2024-06-20'},
                {'level': 177.3, 'strength': 'medium', 'date': '2024-11-10'},
            ]
        },
        'structure_quality': {
            'quality_score': 95,
            'quality_rating': 'excellent'
        },
        'current_state': {
            'structure_state': 'trending_strongly',
            'current_price': close_prices[-1]
        },
        'confidence_score': 0.92,
        'data_quality': {
            'overall_quality_score': 100
        }
    }
    
    return stock_data, analysis_data


def create_downtrend_scenario() -> Tuple[pd.DataFrame, Dict]:
    """Create mock data for clear downtrend scenario"""
    
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    
    np.random.seed(123)
    base_trend = np.linspace(200, 120, len(dates))  # Downward trend
    noise = np.random.normal(0, 3, len(dates))
    volatility = np.random.normal(0, 6, len(dates))
    
    close_prices = base_trend + noise
    high_prices = close_prices + np.abs(volatility) + np.random.uniform(0.5, 4, len(dates))
    low_prices = close_prices - np.abs(volatility) - np.random.uniform(0.5, 4, len(dates))
    open_prices = close_prices + np.random.uniform(-3, 3, len(dates))
    volumes = np.random.randint(2000000, 8000000, len(dates))
    
    stock_data = pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes
    }, index=dates)
    
    analysis_data = {
        'swing_points': {
            'swing_highs': [
                {'date': '2024-01-25', 'price': 198.7, 'strength': 'strong', 'index': 25},
                {'date': '2024-03-20', 'price': 185.4, 'strength': 'medium', 'index': 79},
                {'date': '2024-05-30', 'price': 165.2, 'strength': 'strong', 'index': 150},
                {'date': '2024-08-15', 'price': 145.8, 'strength': 'medium', 'index': 227},
                {'date': '2024-10-20', 'price': 132.1, 'strength': 'weak', 'index': 293},
            ],
            'swing_lows': [
                {'date': '2024-02-28', 'price': 175.3, 'strength': 'medium', 'index': 59},
                {'date': '2024-04-25', 'price': 158.9, 'strength': 'strong', 'index': 115},
                {'date': '2024-07-10', 'price': 138.7, 'strength': 'strong', 'index': 191},
                {'date': '2024-09-18', 'price': 125.4, 'strength': 'medium', 'index': 261},
                {'date': '2024-11-30', 'price': 118.2, 'strength': 'strong', 'index': 334},
            ],
            'total_swings': 10,
            'swing_density': 0.027
        },
        'bos_choch_analysis': {
            'bos_events': [
                {'type': 'bearish_bos', 'date': '2024-02-28', 'break_price': 175.3, 'strength': 'strong'},
                {'type': 'bearish_bos', 'date': '2024-04-25', 'break_price': 158.9, 'strength': 'strong'},
                {'type': 'bearish_bos', 'date': '2024-07-10', 'break_price': 138.7, 'strength': 'strong'},
                {'type': 'bearish_bos', 'date': '2024-09-18', 'break_price': 125.4, 'strength': 'medium'},
            ],
            'choch_events': [
                {'type': 'bearish_choch', 'date': '2024-06-15'}
            ],
            'structural_bias': 'bearish'
        },
        'trend_analysis': {
            'trend_direction': 'downtrend',
            'trend_strength': 'strong',
            'trend_quality': 'good'
        },
        'key_levels': {
            'support_levels': [
                {'level': 175.3, 'strength': 'weak', 'date': '2024-02-28'},
                {'level': 158.9, 'strength': 'weak', 'date': '2024-04-25'},
                {'level': 138.7, 'strength': 'medium', 'date': '2024-07-10'},
                {'level': 118.2, 'strength': 'strong', 'date': '2024-11-30'},
            ],
            'resistance_levels': [
                {'level': 198.7, 'strength': 'strong', 'date': '2024-01-25'},
                {'level': 185.4, 'strength': 'medium', 'date': '2024-03-20'},
                {'level': 165.2, 'strength': 'strong', 'date': '2024-05-30'},
                {'level': 145.8, 'strength': 'medium', 'date': '2024-08-15'},
            ]
        },
        'structure_quality': {
            'quality_score': 88,
            'quality_rating': 'good'
        },
        'current_state': {
            'structure_state': 'trending_down_strongly',
            'current_price': close_prices[-1]
        },
        'confidence_score': 0.89,
        'data_quality': {
            'overall_quality_score': 95
        }
    }
    
    return stock_data, analysis_data


def create_sideways_scenario() -> Tuple[pd.DataFrame, Dict]:
    """Create mock data for sideways consolidation scenario"""
    
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    
    np.random.seed(456)
    # Create sideways movement between 145-155
    base_price = 150
    range_movement = np.sin(np.linspace(0, 8*np.pi, len(dates))) * 5  # Oscillation within range
    noise = np.random.normal(0, 2, len(dates))
    
    close_prices = base_price + range_movement + noise
    high_prices = close_prices + np.random.uniform(1, 4, len(dates))
    low_prices = close_prices - np.random.uniform(1, 4, len(dates))
    open_prices = close_prices + np.random.uniform(-2, 2, len(dates))
    volumes = np.random.randint(800000, 3000000, len(dates))
    
    stock_data = pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes
    }, index=dates)
    
    analysis_data = {
        'swing_points': {
            'swing_highs': [
                {'date': '2024-01-30', 'price': 154.8, 'strength': 'medium', 'index': 30},
                {'date': '2024-03-15', 'price': 155.2, 'strength': 'weak', 'index': 74},
                {'date': '2024-05-20', 'price': 155.7, 'strength': 'medium', 'index': 140},
                {'date': '2024-07-25', 'price': 154.9, 'strength': 'weak', 'index': 206},
                {'date': '2024-09-30', 'price': 155.1, 'strength': 'weak', 'index': 273},
                {'date': '2024-11-25', 'price': 154.6, 'strength': 'medium', 'index': 329},
            ],
            'swing_lows': [
                {'date': '2024-02-14', 'price': 145.3, 'strength': 'medium', 'index': 45},
                {'date': '2024-04-10', 'price': 144.8, 'strength': 'strong', 'index': 100},
                {'date': '2024-06-05', 'price': 145.1, 'strength': 'weak', 'index': 156},
                {'date': '2024-08-20', 'price': 145.2, 'strength': 'medium', 'index': 232},
                {'date': '2024-10-15', 'price': 144.9, 'strength': 'weak', 'index': 288},
                {'date': '2024-12-10', 'price': 145.4, 'strength': 'medium', 'index': 344},
            ],
            'total_swings': 12,
            'swing_density': 0.033
        },
        'bos_choch_analysis': {
            'bos_events': [],  # No significant BOS in sideways movement
            'choch_events': [
                {'type': 'bullish_choch', 'date': '2024-04-15'},
                {'type': 'bearish_choch', 'date': '2024-08-05'},
            ],
            'structural_bias': 'neutral'
        },
        'trend_analysis': {
            'trend_direction': 'sideways',
            'trend_strength': 'weak',
            'trend_quality': 'fair'
        },
        'key_levels': {
            'support_levels': [
                {'level': 144.8, 'strength': 'strong', 'date': '2024-04-10'},
                {'level': 145.1, 'strength': 'medium', 'date': '2024-06-05'},
                {'level': 145.2, 'strength': 'medium', 'date': '2024-08-20'},
            ],
            'resistance_levels': [
                {'level': 155.7, 'strength': 'strong', 'date': '2024-05-20'},
                {'level': 154.9, 'strength': 'medium', 'date': '2024-07-25'},
                {'level': 155.1, 'strength': 'medium', 'date': '2024-09-30'},
            ]
        },
        'structure_quality': {
            'quality_score': 75,
            'quality_rating': 'fair'
        },
        'current_state': {
            'structure_state': 'consolidating',
            'current_price': close_prices[-1]
        },
        'confidence_score': 0.78,
        'data_quality': {
            'overall_quality_score': 90
        }
    }
    
    return stock_data, analysis_data


def create_complex_scenario() -> Tuple[pd.DataFrame, Dict]:
    """Create mock data for complex multi-phase structure"""
    
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    
    np.random.seed(789)
    
    # Complex scenario: Uptrend -> Consolidation -> Downtrend -> Recovery
    phase1 = np.linspace(120, 160, len(dates)//4)  # Uptrend
    phase2 = np.ones(len(dates)//4) * 160 + np.random.normal(0, 3, len(dates)//4)  # Consolidation
    phase3 = np.linspace(160, 130, len(dates)//4)  # Downtrend  
    phase4 = np.linspace(130, 155, len(dates) - 3*(len(dates)//4))  # Recovery
    
    base_trend = np.concatenate([phase1, phase2, phase3, phase4])
    noise = np.random.normal(0, 2.5, len(dates))
    volatility = np.random.normal(0, 4, len(dates))
    
    close_prices = base_trend + noise
    high_prices = close_prices + np.abs(volatility) + np.random.uniform(0.5, 3.5, len(dates))
    low_prices = close_prices - np.abs(volatility) - np.random.uniform(0.5, 3.5, len(dates))
    open_prices = close_prices + np.random.uniform(-2.5, 2.5, len(dates))
    volumes = np.random.randint(1500000, 6000000, len(dates))
    
    stock_data = pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes
    }, index=dates)
    
    analysis_data = {
        'swing_points': {
            'swing_highs': [
                {'date': '2024-02-20', 'price': 145.2, 'strength': 'strong', 'index': 51},
                {'date': '2024-03-25', 'price': 158.7, 'strength': 'strong', 'index': 84},
                {'date': '2024-05-15', 'price': 165.3, 'strength': 'medium', 'index': 135},
                {'date': '2024-06-30', 'price': 163.8, 'strength': 'weak', 'index': 181},
                {'date': '2024-08-10', 'price': 155.9, 'strength': 'medium', 'index': 222},
                {'date': '2024-11-20', 'price': 154.1, 'strength': 'weak', 'index': 324},
            ],
            'swing_lows': [
                {'date': '2024-01-15', 'price': 122.4, 'strength': 'strong', 'index': 15},
                {'date': '2024-03-10', 'price': 142.8, 'strength': 'medium', 'index': 69},
                {'date': '2024-04-20', 'price': 152.1, 'strength': 'weak', 'index': 110},
                {'date': '2024-07-25', 'price': 158.2, 'strength': 'weak', 'index': 206},
                {'date': '2024-09-15', 'price': 132.7, 'strength': 'strong', 'index': 258},
                {'date': '2024-10-30', 'price': 138.9, 'strength': 'medium', 'index': 303},
            ],
            'total_swings': 12,
            'swing_density': 0.033
        },
        'bos_choch_analysis': {
            'bos_events': [
                {'type': 'bullish_bos', 'date': '2024-03-25', 'break_price': 158.7, 'strength': 'strong'},
                {'type': 'bearish_bos', 'date': '2024-07-25', 'break_price': 158.2, 'strength': 'medium'},
                {'type': 'bearish_bos', 'date': '2024-09-15', 'break_price': 132.7, 'strength': 'strong'},
                {'type': 'bullish_bos', 'date': '2024-11-20', 'break_price': 154.1, 'strength': 'medium'},
            ],
            'choch_events': [
                {'type': 'bearish_choch', 'date': '2024-06-15'},
                {'type': 'bullish_choch', 'date': '2024-10-05'}
            ],
            'structural_bias': 'neutral'
        },
        'trend_analysis': {
            'trend_direction': 'uptrend',  # Currently in recovery phase
            'trend_strength': 'medium',
            'trend_quality': 'fair'
        },
        'key_levels': {
            'support_levels': [
                {'level': 132.7, 'strength': 'strong', 'date': '2024-09-15'},
                {'level': 138.9, 'strength': 'medium', 'date': '2024-10-30'},
                {'level': 142.8, 'strength': 'weak', 'date': '2024-03-10'},
            ],
            'resistance_levels': [
                {'level': 165.3, 'strength': 'strong', 'date': '2024-05-15'},
                {'level': 158.7, 'strength': 'medium', 'date': '2024-03-25'},
                {'level': 155.9, 'strength': 'medium', 'date': '2024-08-10'},
            ]
        },
        'structure_quality': {
            'quality_score': 82,
            'quality_rating': 'good'
        },
        'current_state': {
            'structure_state': 'transitional_recovery',
            'current_price': close_prices[-1]
        },
        'confidence_score': 0.85,
        'data_quality': {
            'overall_quality_score': 92
        }
    }
    
    return stock_data, analysis_data


def main():
    """Run chart generation tests for all scenarios"""
    
    logger.info("Starting Market Structure Chart Generation Tests...")
    
    # Create chart generator
    chart_generator = MarketStructureChartGenerator(output_dir="test_market_structure_charts")
    
    # Generate all scenarios
    scenarios = create_mock_data_scenarios()
    
    generated_charts = []
    
    for stock_data, analysis_data, symbol, scenario in scenarios:
        logger.info(f"\n{'='*60}")
        logger.info(f"Generating chart for {symbol} - {scenario}")
        logger.info(f"{'='*60}")
        
        # Generate chart
        chart_path = chart_generator.create_comprehensive_chart(
            stock_data=stock_data,
            analysis_data=analysis_data,
            symbol=symbol,
            scenario=scenario
        )
        
        if chart_path:
            generated_charts.append(chart_path)
            logger.info(f"✅ Successfully generated: {chart_path}")
            
            # Log chart information for LLM prompt integration
            chart_info = {
                'chart_path': chart_path,
                'chart_size_bytes': os.path.getsize(chart_path),
                'scenario': scenario,
                'symbol': symbol,
                'chart_description': f"Comprehensive market structure chart showing price action, swing points, BOS/CHOCH events, and trend analysis for {scenario} scenario",
                'visual_elements': [
                    'Price action with high/low fills',
                    'Swing points with strength-based sizing',
                    'BOS/CHOCH event annotations',
                    'Support/resistance levels',
                    'Volume analysis',
                    'Trend structure indicators',
                    'Analysis summary metrics'
                ]
            }
            
            logger.info(f"Chart Info for LLM Integration:")
            logger.info(f"  - Path: {chart_info['chart_path']}")
            logger.info(f"  - Size: {chart_info['chart_size_bytes']:,} bytes")
            logger.info(f"  - Elements: {', '.join(chart_info['visual_elements'])}")
            
        else:
            logger.error(f"❌ Failed to generate chart for {symbol} - {scenario}")
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("CHART GENERATION SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total scenarios: {len(scenarios)}")
    logger.info(f"Successfully generated: {len(generated_charts)}")
    logger.info(f"Failed: {len(scenarios) - len(generated_charts)}")
    
    if generated_charts:
        logger.info(f"\nGenerated charts:")
        for chart_path in generated_charts:
            logger.info(f"  📊 {chart_path}")
        
        logger.info(f"\n🎯 Next steps:")
        logger.info(f"  1. Review generated charts visually")
        logger.info(f"  2. Integrate with LLM multimodal analysis")
        logger.info(f"  3. Test prompt generation with chart metadata")
        logger.info(f"  4. Validate LLM responses with visual context")
    
    logger.info(f"\n✨ Chart generation test completed!")


if __name__ == "__main__":
    main()