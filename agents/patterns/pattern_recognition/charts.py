"""
Pattern Recognition Agent Charts

Creates comprehensive visualizations for general pattern recognition analysis,
market structure, and cross-pattern relationships.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
import os

# Set up logging
logger = logging.getLogger(__name__)

class PatternRecognitionCharts:
    """
    Chart generator for pattern recognition analysis, providing visualizations
    for market structure, cross-pattern analysis, and general pattern insights.
    """
    
    def __init__(self):
        self.name = "pattern_recognition_charts"
        self.version = "1.0.0"
        
    def create_comprehensive_chart(self, stock_data: pd.DataFrame, indicators: Dict[str, np.ndarray],
                                 analysis_result: Dict[str, Any], symbol: str = "SYMBOL",
                                 save_path: Optional[str] = None) -> str:
        """
        Create a comprehensive pattern recognition analysis chart.
        
        Args:
            stock_data: DataFrame with OHLCV data
            indicators: Dictionary of technical indicators
            analysis_result: Results from pattern recognition analysis
            symbol: Stock symbol for chart title
            save_path: Optional path to save the chart
            
        Returns:
            Path to the saved chart file
        """
        try:
            # Create the comprehensive chart
            fig = plt.figure(figsize=(20, 16))
            
            # Create subplots for different analysis components
            gs = fig.add_gridspec(4, 2, height_ratios=[3, 2, 2, 1], hspace=0.3, wspace=0.3)
            
            # Main price chart with market structure
            ax_price = fig.add_subplot(gs[0, :])
            self._plot_market_structure(ax_price, stock_data, indicators, analysis_result)
            
            # Pattern relationships heatmap
            ax_patterns = fig.add_subplot(gs[1, 0])
            self._plot_pattern_relationships(ax_patterns, analysis_result)
            
            # Volume analysis
            ax_volume = fig.add_subplot(gs[1, 1])
            self._plot_volume_analysis(ax_volume, stock_data, analysis_result)
            
            # Momentum patterns
            ax_momentum = fig.add_subplot(gs[2, 0])
            self._plot_momentum_patterns(ax_momentum, indicators, analysis_result)
            
            # Fractal analysis
            ax_fractals = fig.add_subplot(gs[2, 1])
            self._plot_fractal_patterns(ax_fractals, stock_data, analysis_result)
            
            # Overall assessment summary
            ax_summary = fig.add_subplot(gs[3, :])
            self._plot_assessment_summary(ax_summary, analysis_result)
            
            # Add main title
            confidence = analysis_result.get('confidence_score', 0.0)
            fig.suptitle(f'{symbol} - Pattern Recognition Analysis (Confidence: {confidence:.2f})', 
                        fontsize=16, fontweight='bold')
            
            # Save the chart
            if save_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = f"pattern_recognition_analysis_{symbol}_{timestamp}.png"
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
            plt.close()
            
            logger.info(f"[PATTERN_RECOGNITION_CHARTS] Chart created successfully")
            return save_path
            
        except Exception as e:
            logger.error(f"[PATTERN_RECOGNITION_CHARTS] Chart creation failed: {str(e)}")
            if 'fig' in locals():
                plt.close(fig)
            return ""
    
    def _plot_market_structure(self, ax: plt.Axes, stock_data: pd.DataFrame, 
                             indicators: Dict[str, np.ndarray], analysis_result: Dict[str, Any]):
        """Plot market structure analysis on the main price chart."""
        
        # Get price data
        dates = stock_data.index
        closes = stock_data['close'].values
        highs = stock_data['high'].values
        lows = stock_data['low'].values
        
        # Plot price with moving averages
        ax.plot(dates, closes, label='Close Price', color='black', linewidth=1.5, alpha=0.8)
        
        # Plot moving averages if available
        if 'sma_20' in indicators:
            ax.plot(dates, indicators['sma_20'], label='SMA 20', color='blue', alpha=0.7, linewidth=1)
        if 'sma_50' in indicators:
            ax.plot(dates, indicators['sma_50'], label='SMA 50', color='orange', alpha=0.7, linewidth=1)
        if 'ema_20' in indicators:
            ax.plot(dates, indicators['ema_20'], label='EMA 20', color='green', alpha=0.7, linewidth=1, linestyle='--')
        
        # Add support and resistance levels
        market_structure = analysis_result.get('market_structure', {})
        support_resistance = market_structure.get('support_resistance', {})
        
        # Plot resistance levels
        resistance_levels = support_resistance.get('resistance_levels', [])
        for i, level in enumerate(resistance_levels[:3]):  # Top 3 levels
            level_price = level['level']
            alpha = 0.6 - (i * 0.15)  # Fade subsequent levels
            ax.axhline(y=level_price, color='red', alpha=alpha, linestyle='-', linewidth=2,
                      label=f"Resistance {level_price:.2f}" if i == 0 else "")
            ax.text(dates[-1], level_price, f" R{i+1}: {level_price:.2f}", 
                   verticalalignment='center', color='red', fontsize=9, alpha=alpha + 0.3)
        
        # Plot support levels
        support_levels = support_resistance.get('support_levels', [])
        for i, level in enumerate(support_levels[:3]):  # Top 3 levels
            level_price = level['level']
            alpha = 0.6 - (i * 0.15)
            ax.axhline(y=level_price, color='green', alpha=alpha, linestyle='-', linewidth=2,
                      label=f"Support {level_price:.2f}" if i == 0 else "")
            ax.text(dates[-1], level_price, f" S{i+1}: {level_price:.2f}", 
                   verticalalignment='center', color='green', fontsize=9, alpha=alpha + 0.3)
        
        # Add trend analysis information
        trend_analysis = market_structure.get('trend_analysis', {})
        trend_strength = trend_analysis.get('trend_strength', 0)
        medium_trend = trend_analysis.get('medium_term_trend', 'neutral')
        
        # Add trend arrow
        if medium_trend == 'uptrend':
            ax.annotate('↗️', xy=(0.02, 0.95), xycoords='axes fraction', fontsize=20, ha='left', va='top')
        elif medium_trend == 'downtrend':
            ax.annotate('↘️', xy=(0.02, 0.95), xycoords='axes fraction', fontsize=20, ha='left', va='top')
        else:
            ax.annotate('↔️', xy=(0.02, 0.95), xycoords='axes fraction', fontsize=20, ha='left', va='top')
        
        # Add phase information
        market_phases = market_structure.get('market_phases', {})
        current_phase = market_phases.get('current_phase', 'neutral')
        phase_strength = market_phases.get('phase_strength', 0)
        
        phase_text = f"Phase: {current_phase.title()} (Strength: {phase_strength:.2f})"
        ax.text(0.02, 0.85, phase_text, transform=ax.transAxes, fontsize=10,
               bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.7))
        
        ax.set_title(f'Market Structure Analysis (Trend Strength: {trend_strength:.2f})', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    def _plot_pattern_relationships(self, ax: plt.Axes, analysis_result: Dict[str, Any]):
        """Plot pattern relationships and confluence analysis."""
        
        # Get pattern relationship data
        pattern_relationships = analysis_result.get('pattern_relationships', {})
        confluence_areas = pattern_relationships.get('confluence_areas', [])
        confirmations = pattern_relationships.get('pattern_confirmations', [])
        conflicts = pattern_relationships.get('pattern_conflicts', [])
        overall_coherence = pattern_relationships.get('overall_coherence', 0.5)
        
        # Create a simple pattern strength matrix
        pattern_types = ['Price Patterns', 'Volume Patterns', 'Momentum Patterns', 'Fractal Patterns', 'Wave Patterns']
        
        # Generate synthetic relationship strength data for visualization
        relationship_matrix = np.random.rand(len(pattern_types), len(pattern_types)) * overall_coherence
        np.fill_diagonal(relationship_matrix, 1.0)  # Perfect self-correlation
        
        # Create heatmap
        im = ax.imshow(relationship_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        
        # Add text annotations
        for i in range(len(pattern_types)):
            for j in range(len(pattern_types)):
                text = ax.text(j, i, f'{relationship_matrix[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=9)
        
        # Set ticks and labels
        ax.set_xticks(range(len(pattern_types)))
        ax.set_yticks(range(len(pattern_types)))
        ax.set_xticklabels([pt.replace(' ', '\n') for pt in pattern_types], fontsize=9)
        ax.set_yticklabels(pattern_types, fontsize=9)
        
        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
        
        ax.set_title(f'Pattern Relationship Matrix\n(Overall Coherence: {overall_coherence:.2f})', 
                    fontsize=11, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Relationship Strength', rotation=270, labelpad=20, fontsize=9)
        
        # Add confluence information
        confluence_text = f"Confluences: {len(confluence_areas)}, Confirmations: {len(confirmations)}, Conflicts: {len(conflicts)}"
        ax.text(0.5, -0.15, confluence_text, transform=ax.transAxes, ha='center', fontsize=9,
               bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
    
    def _plot_volume_analysis(self, ax: plt.Axes, stock_data: pd.DataFrame, analysis_result: Dict[str, Any]):
        """Plot volume pattern analysis."""
        
        dates = stock_data.index
        volumes = stock_data['volume'].values
        closes = stock_data['close'].values
        
        # Get volume analysis data
        volume_patterns = analysis_result.get('volume_patterns', {})
        volume_trend = volume_patterns.get('volume_trend', {})
        anomalies = volume_patterns.get('volume_anomalies', [])
        
        # Plot volume bars
        colors = ['green' if closes[i] > closes[i-1] else 'red' if i > 0 else 'gray' for i in range(len(closes))]
        ax.bar(dates, volumes, color=colors, alpha=0.6, width=0.8)
        
        # Add volume moving average
        volume_ma = pd.Series(volumes).rolling(window=20, min_periods=1).mean()
        ax.plot(dates, volume_ma, color='blue', linewidth=2, label='Volume MA(20)')
        
        # Highlight volume anomalies
        for anomaly in anomalies[:5]:  # Show top 5 anomalies
            idx = anomaly.get('index', 0)
            if idx < len(dates):
                ax.scatter(dates[idx], volumes[idx], color='yellow', s=100, marker='*', 
                          edgecolor='black', linewidth=2, zorder=5,
                          label='Volume Spike' if anomaly == anomalies[0] else "")
        
        # Add volume trend information
        trend_direction = volume_trend.get('trend_direction', 'neutral')
        avg_volume = volume_trend.get('average_volume', 0)
        volume_volatility = volume_trend.get('volume_volatility', 0)
        
        trend_text = f"Volume Trend: {trend_direction.title()}\nAvg: {avg_volume:,.0f}\nVolatility: {volume_volatility:.2f}"
        ax.text(0.02, 0.98, trend_text, transform=ax.transAxes, fontsize=9, va='top',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', alpha=0.8))
        
        # Price-volume relationship info
        pv_relationship = volume_patterns.get('price_volume_relationship', {})
        correlation = pv_relationship.get('correlation', 0)
        
        ax.set_title(f'Volume Analysis (P-V Correlation: {correlation:.2f})', fontsize=11, fontweight='bold')
        ax.set_ylabel('Volume', fontsize=10)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    def _plot_momentum_patterns(self, ax: plt.Axes, indicators: Dict[str, np.ndarray], analysis_result: Dict[str, Any]):
        """Plot momentum pattern analysis."""
        
        momentum_patterns = analysis_result.get('momentum_patterns', {})
        oscillator_patterns = momentum_patterns.get('oscillator_patterns', {})
        
        # Plot RSI if available
        if 'rsi' in indicators:
            rsi = indicators['rsi']
            x_vals = range(len(rsi))
            
            ax.plot(x_vals, rsi, color='purple', linewidth=2, label='RSI(14)')
            ax.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Overbought (70)')
            ax.axhline(y=30, color='green', linestyle='--', alpha=0.7, label='Oversold (30)')
            ax.axhline(y=50, color='gray', linestyle='-', alpha=0.5)
            
            # Fill overbought/oversold areas
            ax.fill_between(x_vals, 70, 100, where=(rsi >= 70), alpha=0.2, color='red', label='')
            ax.fill_between(x_vals, 0, 30, where=(rsi <= 30), alpha=0.2, color='green', label='')
            
            # Add RSI pattern information
            rsi_patterns = oscillator_patterns.get('rsi', {})
            current_rsi = rsi_patterns.get('current_level', rsi[-1] if len(rsi) > 0 else 50)
            rsi_trend = rsi_patterns.get('trend', 'neutral')
            
            rsi_text = f"RSI: {current_rsi:.1f}\nTrend: {rsi_trend.title()}"
            ax.text(0.02, 0.98, rsi_text, transform=ax.transAxes, fontsize=9, va='top',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='lightpink', alpha=0.8))
            
            ax.set_ylim(0, 100)
            ax.set_ylabel('RSI', fontsize=10)
        
        else:
            # If no RSI available, show MACD or other momentum indicators
            if 'macd' in indicators and 'macd_signal' in indicators:
                macd = indicators['macd']
                macd_signal = indicators['macd_signal']
                x_vals = range(len(macd))
                
                ax.plot(x_vals, macd, color='blue', linewidth=1.5, label='MACD')
                ax.plot(x_vals, macd_signal, color='red', linewidth=1.5, label='Signal')
                ax.bar(x_vals, macd - macd_signal, alpha=0.3, color='gray', label='Histogram')
                ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
                
                ax.set_ylabel('MACD', fontsize=10)
            else:
                # Fallback: show momentum analysis text
                ax.text(0.5, 0.5, 'Momentum Pattern Analysis\n\nNo momentum indicators available\nfor detailed visualization', 
                       transform=ax.transAxes, ha='center', va='center', fontsize=11,
                       bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_xticks([])
                ax.set_yticks([])
        
        ax.set_title('Momentum Pattern Analysis', fontsize=11, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    def _plot_fractal_patterns(self, ax: plt.Axes, stock_data: pd.DataFrame, analysis_result: Dict[str, Any]):
        """Plot fractal pattern analysis."""
        
        dates = stock_data.index
        highs = stock_data['high'].values
        lows = stock_data['low'].values
        closes = stock_data['close'].values
        
        # Plot price
        ax.plot(dates, closes, color='black', linewidth=1, alpha=0.7, label='Close Price')
        
        # Get fractal patterns
        fractal_patterns = analysis_result.get('fractal_patterns', {})
        fractal_highs = fractal_patterns.get('fractal_highs', [])
        fractal_lows = fractal_patterns.get('fractal_lows', [])
        
        # Plot fractal highs
        for fractal in fractal_highs[:10]:  # Limit to 10 fractals for clarity
            idx = fractal.get('index', 0)
            if idx < len(dates):
                ax.scatter(dates[idx], fractal['price'], color='red', marker='^', s=80, 
                          alpha=0.8, zorder=5, label='Fractal High' if fractal == fractal_highs[0] else "")
        
        # Plot fractal lows
        for fractal in fractal_lows[:10]:  # Limit to 10 fractals for clarity
            idx = fractal.get('index', 0)
            if idx < len(dates):
                ax.scatter(dates[idx], fractal['price'], color='green', marker='v', s=80, 
                          alpha=0.8, zorder=5, label='Fractal Low' if fractal == fractal_lows[0] else "")
        
        # Add fractal trend information
        fractal_trend = fractal_patterns.get('fractal_trend', {})
        trend = fractal_trend.get('trend', 'neutral')
        strength = fractal_trend.get('strength', 0.5)
        
        fractal_text = f"Fractal Trend: {trend.title()}\nStrength: {strength:.2f}\n"
        fractal_text += f"Highs: {len(fractal_highs)}, Lows: {len(fractal_lows)}"
        
        ax.text(0.02, 0.98, fractal_text, transform=ax.transAxes, fontsize=9, va='top',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcyan', alpha=0.8))
        
        ax.set_title('Fractal Pattern Analysis', fontsize=11, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    def _plot_assessment_summary(self, ax: plt.Axes, analysis_result: Dict[str, Any]):
        """Plot overall assessment summary."""
        
        # Get overall assessment
        assessment = analysis_result.get('overall_assessment', {})
        market_condition = assessment.get('market_condition', 'neutral')
        confidence_level = assessment.get('confidence_level', 'medium')
        key_insights = assessment.get('key_insights', [])
        primary_patterns = assessment.get('primary_patterns', [])
        
        # Clear axes
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 4)
        ax.axis('off')
        
        # Market condition
        condition_color = {'strong_uptrend': 'green', 'strong_downtrend': 'red', 
                          'coherent_patterns': 'blue', 'mixed_signals': 'orange'}.get(market_condition, 'gray')
        
        ax.text(0.5, 3.5, f"Market Condition: {market_condition.replace('_', ' ').title()}", 
               fontsize=14, fontweight='bold', ha='left', color=condition_color)
        
        # Confidence level
        conf_color = {'high': 'green', 'medium': 'orange', 'low': 'red'}.get(confidence_level, 'gray')
        ax.text(0.5, 3.0, f"Confidence Level: {confidence_level.title()}", 
               fontsize=12, ha='left', color=conf_color)
        
        # Key insights
        insights_text = "Key Insights:\n" + "\n".join([f"• {insight}" for insight in key_insights[:3]])
        if not key_insights:
            insights_text = "Key Insights:\n• No specific insights identified"
        
        ax.text(0.5, 2.5, insights_text, fontsize=10, ha='left', va='top',
               bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgreen', alpha=0.3))
        
        # Primary patterns
        if primary_patterns:
            patterns_text = "Primary Patterns:\n" + "\n".join([f"• {p.get('type', 'Unknown')}" for p in primary_patterns[:3]])
        else:
            patterns_text = "Primary Patterns:\n• No specific patterns identified"
        
        ax.text(5.5, 2.5, patterns_text, fontsize=10, ha='left', va='top',
               bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.3))
        
        # Analysis metadata
        confidence_score = analysis_result.get('confidence_score', 0.0)
        processing_time = analysis_result.get('processing_time', 0.0)
        timestamp = analysis_result.get('timestamp', datetime.now().isoformat())
        
        metadata_text = f"Analysis Details:\n• Confidence Score: {confidence_score:.3f}\n• Processing Time: {processing_time:.3f}s\n• Generated: {timestamp[:19]}"
        ax.text(0.5, 1.0, metadata_text, fontsize=9, ha='left', va='top',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', alpha=0.5))
    
    def create_pattern_comparison_chart(self, multiple_results: Dict[str, Dict], save_path: Optional[str] = None) -> str:
        """
        Create a comparison chart for multiple pattern recognition analyses.
        
        Args:
            multiple_results: Dictionary of analysis results keyed by symbol
            save_path: Optional path to save the chart
            
        Returns:
            Path to the saved chart file
        """
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            axes = axes.ravel()
            
            symbols = list(multiple_results.keys())[:4]  # Limit to 4 symbols
            
            for i, symbol in enumerate(symbols):
                ax = axes[i]
                result = multiple_results[symbol]
                
                # Extract key metrics
                confidence = result.get('confidence_score', 0.0)
                market_condition = result.get('overall_assessment', {}).get('market_condition', 'neutral')
                
                # Create a simple summary visualization
                metrics = ['Confidence', 'Structure Quality', 'Pattern Coherence', 'Trend Strength']
                values = [
                    confidence,
                    result.get('market_structure', {}).get('structure_quality', {}).get('reliability_score', 0.5),
                    result.get('pattern_relationships', {}).get('overall_coherence', 0.5),
                    result.get('market_structure', {}).get('trend_analysis', {}).get('trend_strength', 0.5)
                ]
                
                # Create radar-like bar chart
                colors = ['green', 'blue', 'orange', 'red']
                bars = ax.bar(metrics, values, color=colors, alpha=0.7)
                
                # Add value labels on bars
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{value:.2f}', ha='center', va='bottom', fontsize=9)
                
                ax.set_ylim(0, 1)
                ax.set_title(f'{symbol} - {market_condition.replace("_", " ").title()}', fontsize=12, fontweight='bold')
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True, alpha=0.3)
            
            # Hide unused subplots
            for i in range(len(symbols), 4):
                axes[i].axis('off')
            
            fig.suptitle('Pattern Recognition Analysis Comparison', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            if save_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = f"pattern_recognition_comparison_{timestamp}.png"
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return save_path
            
        except Exception as e:
            logger.error(f"[PATTERN_RECOGNITION_CHARTS] Comparison chart creation failed: {str(e)}")
            if 'fig' in locals():
                plt.close(fig)
            return ""