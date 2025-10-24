#!/usr/bin/env python3
"""
Pattern Chart Generator for Cross-Validation Agent

Creates static matplotlib-based charts suitable for multimodal LLM analysis.
Focuses on pattern visualization with clean, simple charts rather than complex interactive dashboards.
"""

import pandas as pd
import numpy as np
# Force non-interactive backend for headless chart generation
import matplotlib
matplotlib.use('Agg')  # Use Anti-Grain Geometry backend (no GUI)
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle, FancyBboxPatch
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class PatternChartGenerator:
    """
    Self-contained chart generator for pattern visualization.
    
    Creates static PNG charts suitable for LLM analysis featuring:
    - Clean candlestick price charts with moving averages
    - Volume subplot with volume moving average
    - RSI subplot (always included for comprehensive analysis)
    - Basic pattern overlays (trend lines, levels, divergences)
    - High-resolution PNG output optimized for AI analysis
    """
    
    def __init__(self):
        self.name = "pattern_chart_generator"
        
        # Define chart colors - self-contained
        self.colors = {
            'bull': '#00C851', 'bear': '#ff4444',
            'sma20': '#2196F3', 'sma50': '#FF9800', 
            'support': '#4CAF50', 'resistance': '#F44336',
            'current_price': '#9E9E9E',
            'volume': '#666666', 'rsi': '#9467bd',
            'overbought': '#ff0000', 'oversold': '#00ff00'
        }
        
        logger.info("✅ Pattern chart generator initialized")
    
    def generate_pattern_chart(
        self,
        stock_data: pd.DataFrame,
        detected_patterns: List[Dict[str, Any]],
        symbol: str = "STOCK",
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a clean pattern visualization chart suitable for LLM analysis.
        
        Args:
            stock_data: DataFrame with OHLCV data
            detected_patterns: List of detected patterns to overlay
            symbol: Stock symbol for chart title
            save_path: Optional path to save the chart image
            
        Returns:
            Dictionary with chart info and saved image path
        """
        try:
            logger.info(f"[PATTERN_CHART] Generating chart for {symbol} with {len(detected_patterns)} patterns")
            logger.info(f"[PATTERN_CHART] Stock data shape: {stock_data.shape if stock_data is not None else 'None'}")
            logger.info(f"[PATTERN_CHART] Stock data columns: {list(stock_data.columns) if stock_data is not None else 'None'}")
            logger.info(f"[PATTERN_CHART] Save path provided: {save_path}")
            
            if stock_data is None or stock_data.empty:
                return self._build_error_result("No stock data provided")
            
            # Prepare data
            logger.info(f"[PATTERN_CHART] Preparing data...")
            df = stock_data.copy()
            logger.info(f"[PATTERN_CHART] Data copied, computing SMAs...")
            df['sma20'] = df['close'].rolling(20).mean()
            df['sma50'] = df['close'].rolling(50).mean()
            logger.info(f"[PATTERN_CHART] SMAs computed successfully")
            
            # Always include RSI for comprehensive analysis (valuable for pattern context)
            has_rsi_pane = True
            has_divergence_patterns = any('divergence' in p.get('pattern_name', '').lower() for p in detected_patterns)
            
            # Calculate RSI for all charts
            rsi_data = self._calculate_rsi(df['close'])
            
            # Always use 3-pane layout (Price, Volume, RSI)
            logger.info(f"[PATTERN_CHART] Creating 3-pane matplotlib figure...")
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12), 
                                               gridspec_kw={'height_ratios': [3, 1, 1]})
            axes = [ax1, ax2, ax3]
            logger.info(f"[PATTERN_CHART] Figure created successfully")
            
            # Plot price pane
            logger.info(f"[PATTERN_CHART] Plotting price pane...")
            self._plot_price_pane(ax1, df, symbol, detected_patterns)
            logger.info(f"[PATTERN_CHART] Price pane completed")
            
            # Plot volume pane  
            logger.info(f"[PATTERN_CHART] Plotting volume pane...")
            self._plot_volume_pane(axes[1], df)
            logger.info(f"[PATTERN_CHART] Volume pane completed")
            
            # Plot RSI pane (always included)
            if rsi_data is not None and len(axes) > 2:
                logger.info(f"[PATTERN_CHART] Plotting RSI pane...")
                self._plot_rsi_pane(axes[2], df, rsi_data, detected_patterns)
                logger.info(f"[PATTERN_CHART] RSI pane completed")
            
            # Add pattern overlays
            if detected_patterns:
                try:
                    self._add_pattern_overlays(axes, df, detected_patterns, rsi_data)
                except Exception as e:
                    logger.warning(f"Failed to add pattern overlays: {e}")
            
            # Styling and layout
            logger.info(f"[PATTERN_CHART] Applying layout adjustments...")
            plt.tight_layout()
            plt.subplots_adjust(hspace=0.1)
            logger.info(f"[PATTERN_CHART] Layout completed")
            
            # Save chart
            saved_path = None
            if save_path:
                logger.info(f"[PATTERN_CHART] Attempting to save chart...")
                saved_path = self._save_chart(fig, symbol, save_path)
                logger.info(f"[PATTERN_CHART] Save attempt completed, path: {saved_path}")
            else:
                logger.info(f"[PATTERN_CHART] No save path provided, skipping save")
            
            logger.info(f"[PATTERN_CHART] Closing matplotlib figure...")
            plt.close(fig)  # Free memory
            logger.info(f"[PATTERN_CHART] Figure closed successfully")
            
            return {
                'success': True,
                'chart_type': 'pattern_visualization',
                'symbol': symbol,
                'patterns_count': len(detected_patterns),
                'has_rsi_pane': has_rsi_pane,
                'has_divergence_patterns': has_divergence_patterns,
                'saved_image_path': saved_path,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"[PATTERN_CHART] Chart generation failed: {e}")
            return self._build_error_result(str(e))
    
    def _plot_price_pane(self, ax, df: pd.DataFrame, symbol: str, patterns: List[Dict]):
        """Plot candlestick price chart with moving averages"""
        
        # Candlesticks
        for i, (idx, row) in enumerate(df.iterrows()):
            color = self.colors['bull'] if row['close'] >= row['open'] else self.colors['bear']
            
            # High-low line
            ax.plot([i, i], [row['low'], row['high']], color='black', linewidth=0.5)
            
            # Body rectangle
            body_height = abs(row['close'] - row['open'])
            body_bottom = min(row['open'], row['close'])
            rect = Rectangle((i-0.3, body_bottom), 0.6, body_height, 
                           facecolor=color, edgecolor='black', linewidth=0.5)
            ax.add_patch(rect)
        
        # Moving averages
        ax.plot(range(len(df)), df['sma20'], color=self.colors['sma20'], 
               linewidth=1, label='SMA20', alpha=0.8)
        ax.plot(range(len(df)), df['sma50'], color=self.colors['sma50'], 
               linewidth=1, label='SMA50', alpha=0.8)
        
        # Current price line
        current_price = df['close'].iloc[-1]
        ax.axhline(y=current_price, color=self.colors['current_price'], 
                  linestyle='--', alpha=0.7, label=f'Current: {current_price:.2f}')
        
        # Styling
        ax.set_title(f'{symbol} - Price Action & Patterns', fontsize=14, fontweight='bold')
        ax.set_ylabel('Price', fontsize=12)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Format x-axis dates
        if len(df) > 0:
            date_labels = [df.index[i] if i < len(df) else df.index[-1] 
                          for i in range(0, len(df), max(1, len(df)//10))]
            ax.set_xticks(range(0, len(df), max(1, len(df)//10)))
            ax.set_xticklabels([d.strftime('%m-%d') for d in date_labels], rotation=45)
    
    def _plot_volume_pane(self, ax, df: pd.DataFrame):
        """Plot volume bars with moving average"""
        
        # Volume bars
        colors = [self.colors['bull'] if close >= open_ else self.colors['bear'] 
                 for close, open_ in zip(df['close'], df['open'])]
        
        ax.bar(range(len(df)), df['volume'], color=colors, alpha=0.7, width=0.8)
        
        # Volume moving average
        vma20 = df['volume'].rolling(20).mean()
        ax.plot(range(len(df)), vma20, color=self.colors['volume'], linewidth=1, 
               linestyle='--', label='VMA20', alpha=0.8)
        
        # Styling
        ax.set_ylabel('Volume', fontsize=12)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Format volume numbers
        ax.yaxis.set_major_formatter(plt.FuncFormatter(self._format_volume))
    
    def _plot_rsi_pane(self, ax, df: pd.DataFrame, rsi_data: np.ndarray, patterns: List[Dict]):
        """Plot RSI with overbought/oversold levels"""
        
        # RSI line
        ax.plot(range(len(rsi_data)), rsi_data, color=self.colors['rsi'], linewidth=2, label='RSI(14)')
        
        # Reference lines
        ax.axhline(y=70, color=self.colors['overbought'], linestyle='--', alpha=0.7, label='Overbought')
        ax.axhline(y=30, color=self.colors['oversold'], linestyle='--', alpha=0.7, label='Oversold') 
        ax.axhline(y=50, color='gray', linestyle='-', alpha=0.5)
        
        # Fill overbought/oversold areas
        ax.fill_between(range(len(rsi_data)), 70, 100, alpha=0.1, color=self.colors['overbought'])
        ax.fill_between(range(len(rsi_data)), 0, 30, alpha=0.1, color=self.colors['oversold'])
        
        # Styling
        ax.set_ylabel('RSI', fontsize=12)
        ax.set_ylim(0, 100)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        if len(df) > 0:
            date_labels = [df.index[i] if i < len(df) else df.index[-1] 
                          for i in range(0, len(df), max(1, len(df)//10))]
            ax.set_xticks(range(0, len(df), max(1, len(df)//10)))
            ax.set_xticklabels([d.strftime('%m-%d') for d in date_labels], rotation=45)
    
    def _add_pattern_overlays(self, axes, df: pd.DataFrame, patterns: List[Dict], rsi_data=None):
        """Add pattern overlays - self-contained implementation"""
        
        for pattern in patterns:
            try:
                pattern_type = pattern.get('pattern_type', '').lower()
                pattern_name = pattern.get('pattern_name', '').lower()
                
                # Handle geometric patterns (lines, channels, etc.)
                if 'line' in pattern_type or 'trend' in pattern_type:
                    self._draw_trend_line(axes[0], pattern, df)
                elif 'channel' in pattern_type:
                    self._draw_channel(axes[0], pattern, df)
                elif 'support' in pattern_name or 'resistance' in pattern_name:
                    self._draw_level(axes[0], pattern, df)
                elif 'triple' in pattern_name:
                    self._draw_triple_pattern(axes[0], pattern, df)
                elif 'divergence' in pattern_name and rsi_data is not None and len(axes) > 2:
                    self._draw_divergence(axes[0], axes[2], pattern, df, rsi_data)
                else:
                    # Generic pattern annotation for unrecognized patterns
                    self._add_generic_pattern_annotation(axes[0], pattern, df)
                
            except Exception as e:
                logger.warning(f"Failed to draw pattern {pattern.get('pattern_name', 'unknown')}: {e}")
    
    def _draw_trend_line(self, ax, pattern: Dict, df: pd.DataFrame):
        """Draw trend line on price chart"""
        try:
            start_idx = pattern.get('start_index', 0)
            end_idx = pattern.get('end_index', len(df)-1)
            start_price = pattern.get('start_price', df['close'].iloc[start_idx])
            end_price = pattern.get('end_price', df['close'].iloc[end_idx])
            
            color = self.colors['support'] if pattern.get('trend_direction') == 'up' else self.colors['resistance']
            
            ax.plot([start_idx, end_idx], [start_price, end_price], 
                   color=color, linewidth=2, linestyle='-', alpha=0.8,
                   label=pattern.get('pattern_name', 'Trend Line'))
        except Exception as e:
            logger.warning(f"Failed to draw trend line: {e}")
    
    def _draw_level(self, ax, pattern: Dict, df: pd.DataFrame):
        """Draw support/resistance level"""
        try:
            level = pattern.get('level', pattern.get('price', df['close'].iloc[-1]))
            is_support = 'support' in pattern.get('pattern_name', '').lower()
            
            color = self.colors['support'] if is_support else self.colors['resistance']
            linestyle = '--' if is_support else '-'
            
            ax.axhline(y=level, color=color, linestyle=linestyle, 
                      alpha=0.7, linewidth=1.5,
                      label=pattern.get('pattern_name', 'Level'))
        except Exception as e:
            logger.warning(f"Failed to draw level: {e}")
    
    def _draw_triple_pattern(self, ax, pattern: Dict, df: pd.DataFrame):
        """Draw triple top/bottom pattern"""
        try:
            peaks = pattern.get('peaks', [])
            if len(peaks) >= 3:
                # Draw lines connecting the peaks
                peak_indices = [p.get('index', 0) for p in peaks[:3]]
                peak_prices = [p.get('price', df['close'].iloc[idx]) for p, idx in zip(peaks[:3], peak_indices)]
                
                is_top = 'top' in pattern.get('pattern_name', '').lower()
                color = self.colors['resistance'] if is_top else self.colors['support']
                
                ax.plot(peak_indices, peak_prices, 'o-', color=color, 
                       markersize=6, linewidth=2, alpha=0.8,
                       label=pattern.get('pattern_name', 'Triple Pattern'))
        except Exception as e:
            logger.warning(f"Failed to draw triple pattern: {e}")
    
    def _draw_divergence(self, price_ax, rsi_ax, pattern: Dict, df: pd.DataFrame, rsi_data: np.ndarray):
        """Draw divergence lines on both price and RSI panes"""
        try:
            start_idx = pattern.get('start_index', 0)
            end_idx = pattern.get('end_index', len(df)-1)
            
            is_bullish = 'bullish' in pattern.get('pattern_name', '').lower()
            color = self.colors['bull'] if is_bullish else self.colors['bear']
            
            # Price divergence line
            start_price = df['close'].iloc[start_idx]
            end_price = df['close'].iloc[end_idx]
            price_ax.plot([start_idx, end_idx], [start_price, end_price], 
                         color=color, linewidth=1.5, linestyle=':', alpha=0.8,
                         label=f'Price {pattern.get("pattern_name", "Divergence")}')
            
            # RSI divergence line
            start_rsi = rsi_data[start_idx]
            end_rsi = rsi_data[end_idx]
            rsi_ax.plot([start_idx, end_idx], [start_rsi, end_rsi], 
                       color=color, linewidth=1.5, linestyle=':', alpha=0.8,
                       label=f'RSI {pattern.get("pattern_name", "Divergence")}')
        except Exception as e:
            logger.warning(f"Failed to draw divergence: {e}")
    
    def _draw_channel(self, ax, pattern: Dict, df: pd.DataFrame):
        """Draw channel pattern"""
        try:
            upper_line = pattern.get('upper_line', {})
            lower_line = pattern.get('lower_line', {})
            
            if upper_line and lower_line:
                # Upper channel line
                ax.plot([upper_line.get('start_index', 0), upper_line.get('end_index', len(df)-1)],
                       [upper_line.get('start_price', 0), upper_line.get('end_price', 0)],
                       color=self.colors['resistance'], linewidth=1.5, alpha=0.7)
                
                # Lower channel line  
                ax.plot([lower_line.get('start_index', 0), lower_line.get('end_index', len(df)-1)],
                       [lower_line.get('start_price', 0), lower_line.get('end_price', 0)],
                       color=self.colors['support'], linewidth=1.5, alpha=0.7,
                       label=pattern.get('pattern_name', 'Channel'))
        except Exception as e:
            logger.warning(f"Failed to draw channel: {e}")
    
    def _add_generic_pattern_annotation(self, ax, pattern: Dict, df: pd.DataFrame):
        """Add generic text annotation for unrecognized patterns"""
        try:
            pattern_name = pattern.get('pattern_name', 'Pattern')
            # Add text annotation at the end of the chart
            ax.text(len(df) - 1, df['close'].iloc[-1], pattern_name, 
                   fontsize=10, ha='right', va='bottom', 
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        except Exception as e:
            logger.warning(f"Failed to add generic annotation: {e}")
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> np.ndarray:
        """Calculate RSI indicator"""
        try:
            delta = prices.diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=period).mean()
            avg_loss = loss.rolling(window=period).mean()
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi.fillna(50).values
        except Exception:
            return np.full(len(prices), 50)  # Fallback to neutral RSI
    
    def _save_chart(self, fig, symbol: str, save_path: str) -> Optional[str]:
        """Save chart as PNG image"""
        try:
            # Create output directory
            save_path_obj = Path(save_path)
            output_dir = save_path_obj.parent / "charts"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save as high-resolution PNG for LLM analysis
            png_path = output_dir / f"{symbol}_pattern_chart.png"
            fig.savefig(png_path, dpi=150, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            
            logger.info(f"✅ Saved pattern chart: {png_path}")
            return str(png_path)
            
        except Exception as e:
            logger.error(f"❌ Failed to save chart: {e}")
            return None
    
    def _format_volume(self, x, p):
        """Format volume numbers for display"""
        if x >= 1e9:
            return f'{x/1e9:.1f}B'
        elif x >= 1e6:
            return f'{x/1e6:.1f}M'
        elif x >= 1e3:
            return f'{x/1e3:.1f}K'
        else:
            return f'{int(x)}'
    
    def generate_pattern_chart_bytes(
        self,
        stock_data: pd.DataFrame,
        detected_patterns: List[Dict[str, Any]],
        symbol: str = "STOCK"
    ) -> Optional[bytes]:
        """
        Generate a pattern visualization chart and return as bytes (like market structure agent).
        
        Args:
            stock_data: DataFrame with OHLCV data
            detected_patterns: List of detected patterns to overlay
            symbol: Stock symbol for chart title
            
        Returns:
            Chart image as bytes or None if failed
        """
        try:
            logger.info(f"[PATTERN_CHART] Generating chart bytes for {symbol} with {len(detected_patterns)} patterns")
            
            if stock_data is None or stock_data.empty:
                logger.error(f"[PATTERN_CHART] No stock data provided")
                return None
            
            # Prepare data
            logger.info(f"[PATTERN_CHART] Preparing data...")
            df = stock_data.copy()
            df['sma20'] = df['close'].rolling(20).mean()
            df['sma50'] = df['close'].rolling(50).mean()
            logger.info(f"[PATTERN_CHART] SMAs computed successfully")
            
            # Calculate RSI
            rsi_data = self._calculate_rsi(df['close'])
            
            # Create 3-pane layout (Price, Volume, RSI)
            logger.info(f"[PATTERN_CHART] Creating 3-pane matplotlib figure...")
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12), 
                                               gridspec_kw={'height_ratios': [3, 1, 1]})
            axes = [ax1, ax2, ax3]
            logger.info(f"[PATTERN_CHART] Figure created successfully")
            
            # Plot price pane
            logger.info(f"[PATTERN_CHART] Plotting price pane...")
            self._plot_price_pane(ax1, df, symbol, detected_patterns)
            logger.info(f"[PATTERN_CHART] Price pane completed")
            
            # Plot volume pane  
            logger.info(f"[PATTERN_CHART] Plotting volume pane...")
            self._plot_volume_pane(axes[1], df)
            logger.info(f"[PATTERN_CHART] Volume pane completed")
            
            # Plot RSI pane
            if rsi_data is not None and len(axes) > 2:
                logger.info(f"[PATTERN_CHART] Plotting RSI pane...")
                self._plot_rsi_pane(axes[2], df, rsi_data, detected_patterns)
                logger.info(f"[PATTERN_CHART] RSI pane completed")
            
            # Add pattern overlays
            if detected_patterns:
                try:
                    self._add_pattern_overlays(axes, df, detected_patterns, rsi_data)
                except Exception as e:
                    logger.warning(f"Failed to add pattern overlays: {e}")
            
            # Styling and layout
            logger.info(f"[PATTERN_CHART] Applying layout adjustments...")
            plt.tight_layout()
            plt.subplots_adjust(hspace=0.1)
            logger.info(f"[PATTERN_CHART] Layout completed")
            
            # Convert to bytes
            logger.info(f"[PATTERN_CHART] Converting chart to bytes...")
            import io
            img_buffer = io.BytesIO()
            fig.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            img_buffer.seek(0)
            chart_bytes = img_buffer.getvalue()
            img_buffer.close()
            
            logger.info(f"[PATTERN_CHART] Closing matplotlib figure...")
            plt.close(fig)  # Free memory
            logger.info(f"[PATTERN_CHART] Chart generated successfully: {len(chart_bytes)} bytes")
            
            return chart_bytes
            
        except Exception as e:
            logger.error(f"[PATTERN_CHART] Chart bytes generation failed: {e}")
            return None
    
    def _build_error_result(self, error_message: str) -> Dict[str, Any]:
        """Build error result"""
        return {
            'success': False,
            'error': error_message,
            'chart_type': 'pattern_visualization',
            'saved_image_path': None,
            'timestamp': datetime.now().isoformat()
        }
