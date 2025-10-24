#!/usr/bin/env python3
"""
Support/Resistance Agent - Chart Visualization Module

Provides comprehensive chart visualizations for support/resistance analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

from .processor import SupportResistanceProcessor
from .agent import SupportResistanceAgent

class SupportResistanceCharts:
    """
    Chart visualization for Support/Resistance Agent
    
    Creates professional charts showing price action, support/resistance levels,
    volume profile, and analysis insights
    """
    
    def __init__(self):
        self.default_style = {
            'figure_size': (16, 12),
            'dpi': 100,
            'font_size': 10,
            'title_font_size': 14,
            'line_width': 1.5,
            'alpha': 0.7
        }
        
        # Color scheme
        self.colors = {
            'price_line': '#2E86AB',
            'support_color': '#A23B72',
            'resistance_color': '#F18F01',
            'volume_bars': '#C73E1D',
            'volume_profile': '#4A4A4A',
            'background': '#FAFAFA',
            'grid': '#E0E0E0',
            'text': '#333333',
            'high_reliability': '#2ECC71',
            'medium_reliability': '#F39C12',
            'low_reliability': '#E74C3C'
        }
    
    def create_comprehensive_chart(self, data: pd.DataFrame, analysis_results: Dict[str, Any], 
                                 symbol: str = "STOCK", save_path: Optional[str] = None) -> plt.Figure:
        """
        Create comprehensive support/resistance chart with all components
        
        Args:
            data: OHLCV DataFrame
            analysis_results: Results from SupportResistanceAgent.analyze()
            symbol: Stock symbol for title
            save_path: Optional path to save chart
            
        Returns:
            matplotlib Figure object
        """
        
        # Create figure with subplots
        fig = plt.figure(figsize=self.default_style['figure_size'], 
                        dpi=self.default_style['dpi'])
        fig.patch.set_facecolor(self.colors['background'])
        
        # Create grid layout: main price chart (70%), volume profile (15%), volume bars (15%)
        gs = fig.add_gridspec(3, 2, height_ratios=[0.7, 0.15, 0.15], width_ratios=[0.85, 0.15],
                             hspace=0.1, wspace=0.05)
        
        # Main price chart
        ax_main = fig.add_subplot(gs[0, 0])
        self._plot_price_with_levels(ax_main, data, analysis_results, symbol)
        
        # Volume profile (right side of main chart)
        ax_vol_profile = fig.add_subplot(gs[0, 1])
        self._plot_volume_profile(ax_vol_profile, data, analysis_results)
        
        # Volume bars (bottom)
        ax_volume = fig.add_subplot(gs[1, 0])
        self._plot_volume_bars(ax_volume, data)
        
        # Analysis summary (bottom right)
        ax_summary = fig.add_subplot(gs[1:, 1])
        self._plot_analysis_summary(ax_summary, analysis_results)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.default_style['dpi'], 
                       bbox_inches='tight', facecolor=self.colors['background'])
        
        return fig
    
    def _plot_price_with_levels(self, ax: plt.Axes, data: pd.DataFrame, 
                               analysis_results: Dict[str, Any], symbol: str):
        """Plot main price chart with support/resistance levels"""
        
        # Plot price line
        ax.plot(data.index, data['close'], color=self.colors['price_line'],
               linewidth=self.default_style['line_width'], label='Close Price')
        
        # Add support/resistance levels
        self._add_support_resistance_levels(ax, data, analysis_results)
        
        # Add current position indicator
        current_analysis = analysis_results.get('current_position_analysis', {})
        if 'current_price' in current_analysis:
            current_price = current_analysis['current_price']
            ax.axhline(y=current_price, color='red', linestyle='--', alpha=0.8,
                      linewidth=2, label=f'Current: ${current_price:.2f}')
        
        # Formatting
        ax.set_title(f'{symbol} - Support/Resistance Analysis', 
                    fontsize=self.default_style['title_font_size'], 
                    fontweight='bold', color=self.colors['text'])
        ax.set_ylabel('Price ($)', fontsize=self.default_style['font_size'])
        ax.grid(True, alpha=0.3, color=self.colors['grid'])
        ax.legend(loc='upper left')
        
        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # Set background
        ax.set_facecolor('white')
    
    def _add_support_resistance_levels(self, ax: plt.Axes, data: pd.DataFrame, 
                                     analysis_results: Dict[str, Any]):
        """Add support and resistance level lines to chart"""
        
        validated_levels = analysis_results.get('detailed_analysis', {}).get('validated_levels', [])
        
        if not validated_levels:
            return
        
        date_min, date_max = data.index[0], data.index[-1]
        
        for level in validated_levels:
            price = level['price']
            level_type = level['type']
            reliability = level['reliability']
            success_rate = level['success_rate']
            
            # Choose color based on type and reliability
            if level_type == 'support':
                base_color = self.colors['support_color']
                label_prefix = 'Support'
            elif level_type == 'resistance':
                base_color = self.colors['resistance_color']
                label_prefix = 'Resistance'
            else:  # both
                base_color = '#8E44AD'  # Purple for dual levels
                label_prefix = 'S/R'
            
            # Adjust alpha based on reliability
            reliability_alpha = {
                'very_high': 0.9,
                'high': 0.8,
                'medium': 0.6,
                'low': 0.4,
                'very_low': 0.3
            }.get(reliability, 0.5)
            
            # Draw level line
            ax.axhline(y=price, color=base_color, alpha=reliability_alpha,
                      linewidth=2.5, linestyle='-',
                      label=f'{label_prefix} ${price:.2f} ({reliability})')
            
            # Add level zone (price tolerance area)
            tolerance = price * 0.01  # 1% zone
            rect = Rectangle((mdates.date2num(date_min), price - tolerance),
                           mdates.date2num(date_max) - mdates.date2num(date_min),
                           2 * tolerance,
                           alpha=0.1, facecolor=base_color, edgecolor='none')
            ax.add_patch(rect)
            
            # Add text annotation for strong levels
            if reliability in ['high', 'very_high']:
                ax.annotate(f'{label_prefix}: ${price:.2f}\n{success_rate:.0%} success',
                          xy=(date_max, price), xytext=(10, 0),
                          textcoords='offset points', fontsize=8,
                          bbox=dict(boxstyle='round,pad=0.3', facecolor=base_color, alpha=0.3),
                          verticalalignment='center')
    
    def _plot_volume_profile(self, ax: plt.Axes, data: pd.DataFrame, 
                            analysis_results: Dict[str, Any]):
        """Plot volume profile (volume at price) on right side"""
        
        vap_analysis = analysis_results.get('detailed_analysis', {}).get('volume_profile', {})
        
        if 'error' in vap_analysis or not vap_analysis.get('volume_profile'):
            ax.text(0.5, 0.5, 'Volume Profile\nNot Available', 
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=10, color=self.colors['text'])
            ax.set_xticks([])
            ax.set_yticks([])
            return
        
        volume_profile = vap_analysis['volume_profile']
        
        # Extract data for plotting
        prices = [level['price_level'] for level in volume_profile]
        volumes = [level['volume'] for level in volume_profile]
        
        # Normalize volumes for plotting width
        max_volume = max(volumes) if volumes else 1
        normalized_volumes = [v / max_volume for v in volumes]
        
        # Create horizontal bars
        ax.barh(prices, normalized_volumes, height=np.diff(prices).mean() if len(prices) > 1 else 10,
               alpha=0.7, color=self.colors['volume_profile'], edgecolor='none')
        
        # Highlight significant volume nodes
        significant_levels = vap_analysis.get('significant_volume_levels', [])
        for level in significant_levels:
            price = level['price_level']
            volume = level['volume']
            norm_vol = volume / max_volume
            
            ax.barh(price, norm_vol, height=np.diff(prices).mean() if len(prices) > 1 else 10,
                   alpha=0.9, color=self.colors['resistance_color'], edgecolor='black', linewidth=1)
        
        # Format axis
        ax.set_ylim(data['low'].min() * 0.995, data['high'].max() * 1.005)
        ax.set_title('Volume\nProfile', fontsize=10, fontweight='bold')
        ax.set_xlabel('Vol', fontsize=8)
        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
        ax.grid(True, alpha=0.3)
    
    def _plot_volume_bars(self, ax: plt.Axes, data: pd.DataFrame):
        """Plot volume bars at bottom"""
        
        # Plot volume bars
        colors = ['green' if close >= open_price else 'red' 
                 for close, open_price in zip(data['close'], data['open'])]
        
        ax.bar(data.index, data['volume'], color=colors, alpha=0.6, width=0.8)
        
        # Format
        ax.set_ylabel('Volume', fontsize=self.default_style['font_size'])
        ax.grid(True, alpha=0.3, color=self.colors['grid'])
        ax.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
        
        # Share x-axis with main chart
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    def _plot_analysis_summary(self, ax: plt.Axes, analysis_results: Dict[str, Any]):
        """Plot analysis summary and insights"""
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        # Get key data
        summary = analysis_results.get('analysis_summary', {})
        current_position = analysis_results.get('current_position', {})
        recommendations = analysis_results.get('recommendations', [])
        
        y_pos = 0.95
        line_height = 0.08
        
        # Title
        ax.text(0.5, y_pos, 'Analysis Summary', fontsize=12, fontweight='bold',
               ha='center', transform=ax.transAxes)
        y_pos -= line_height * 1.5
        
        # Quality Score
        quality_score = summary.get('analysis_quality_score', 0)
        quality_color = (self.colors['high_reliability'] if quality_score >= 80 
                        else self.colors['medium_reliability'] if quality_score >= 60 
                        else self.colors['low_reliability'])
        
        ax.text(0.05, y_pos, f'Quality Score: {quality_score:.0f}/100', 
               fontsize=10, color=quality_color, fontweight='bold',
               transform=ax.transAxes)
        y_pos -= line_height
        
        # Levels Found
        total_levels = summary.get('total_validated_levels', 0)
        support_count = summary.get('support_levels_found', 0)
        resistance_count = summary.get('resistance_levels_found', 0)
        
        ax.text(0.05, y_pos, f'Levels Found: {total_levels}', fontsize=10,
               transform=ax.transAxes)
        y_pos -= line_height * 0.7
        
        ax.text(0.05, y_pos, f'  Support: {support_count}', fontsize=9,
               color=self.colors['support_color'], transform=ax.transAxes)
        y_pos -= line_height * 0.7
        
        ax.text(0.05, y_pos, f'  Resistance: {resistance_count}', fontsize=9,
               color=self.colors['resistance_color'], transform=ax.transAxes)
        y_pos -= line_height * 1.2
        
        # Current Position
        range_position = current_position.get('range_position_classification', 'unknown')
        current_price = current_position.get('current_price')
        
        if current_price:
            ax.text(0.05, y_pos, f'Current: ${current_price:.2f}', fontsize=10,
                   fontweight='bold', transform=ax.transAxes)
            y_pos -= line_height * 0.8
        
        position_text = range_position.replace('_', ' ').title()
        position_color = (self.colors['support_color'] if 'support' in range_position 
                         else self.colors['resistance_color'] if 'resistance' in range_position 
                         else self.colors['text'])
        
        ax.text(0.05, y_pos, f'Position: {position_text}', fontsize=9,
               color=position_color, transform=ax.transAxes)
        y_pos -= line_height * 1.2
        
        # Distance to levels
        support_distance = current_position.get('support_distance_percentage')
        resistance_distance = current_position.get('resistance_distance_percentage')
        
        if support_distance and support_distance != float('inf'):
            ax.text(0.05, y_pos, f'Support: {support_distance:.1f}% away', 
                   fontsize=9, color=self.colors['support_color'],
                   transform=ax.transAxes)
            y_pos -= line_height * 0.7
        
        if resistance_distance and resistance_distance != float('inf'):
            ax.text(0.05, y_pos, f'Resistance: {resistance_distance:.1f}% away', 
                   fontsize=9, color=self.colors['resistance_color'],
                   transform=ax.transAxes)
            y_pos -= line_height * 1.2
        
        # Top Recommendation
        if recommendations:
            top_rec = recommendations[0]
            action = top_rec.get('action', '').replace('_', ' ').title()
            reason = top_rec.get('reason', '')
            priority = top_rec.get('priority', 'medium')
            
            priority_color = (self.colors['low_reliability'] if priority == 'high' 
                            else self.colors['medium_reliability'] if priority == 'medium' 
                            else self.colors['text'])
            
            ax.text(0.05, y_pos, 'Top Recommendation:', fontsize=10, 
                   fontweight='bold', transform=ax.transAxes)
            y_pos -= line_height * 0.8
            
            # Wrap long text
            if len(reason) > 30:
                words = reason.split()
                line1 = ' '.join(words[:5])
                line2 = ' '.join(words[5:10]) if len(words) > 5 else ''
                
                ax.text(0.05, y_pos, f'{action}: {line1}', fontsize=9,
                       color=priority_color, transform=ax.transAxes)
                if line2:
                    y_pos -= line_height * 0.6
                    ax.text(0.05, y_pos, line2, fontsize=9,
                           color=priority_color, transform=ax.transAxes)
            else:
                ax.text(0.05, y_pos, f'{action}: {reason}', fontsize=9,
                       color=priority_color, transform=ax.transAxes, wrap=True)
    
    def create_levels_strength_chart(self, analysis_results: Dict[str, Any], 
                                   save_path: Optional[str] = None) -> plt.Figure:
        """Create chart showing level strength analysis"""
        
        level_ratings = analysis_results.get('detailed_analysis', {}).get('level_ratings', {})
        individual_ratings = level_ratings.get('individual_ratings', [])
        
        if not individual_ratings:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'No level ratings available', ha='center', va='center',
                   fontsize=16, transform=ax.transAxes)
            return fig
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        fig.patch.set_facecolor(self.colors['background'])
        
        # Top chart: Level strength by price
        prices = [level['price'] for level in individual_ratings]
        scores = [level['overall_score'] for level in individual_ratings]
        types = [level['type'] for level in individual_ratings]
        
        colors_by_type = [self.colors['support_color'] if t == 'support' 
                         else self.colors['resistance_color'] if t == 'resistance'
                         else '#8E44AD' for t in types]
        
        bars1 = ax1.bar(range(len(prices)), scores, color=colors_by_type, alpha=0.7)
        
        # Add price labels
        ax1.set_xticks(range(len(prices)))
        ax1.set_xticklabels([f'${p:.2f}\n{t}' for p, t in zip(prices, types)], fontsize=9)
        ax1.set_ylabel('Strength Score', fontsize=12)
        ax1.set_title('Support/Resistance Level Strength Analysis', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 100)
        
        # Add score labels on bars
        for bar, score in zip(bars1, scores):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{score:.0f}', ha='center', va='bottom', fontsize=9)
        
        # Bottom chart: Score breakdown
        if len(individual_ratings) > 0:
            sample_level = individual_ratings[0]  # Use first level as example
            
            components = ['success_score', 'volume_score', 'test_frequency_score', 'recency_score']
            component_names = ['Success Rate', 'Volume Support', 'Test Frequency', 'Recency']
            component_values = [sample_level.get(comp, 0) for comp in components]
            
            bars2 = ax2.bar(component_names, component_values, 
                          color=['#3498DB', '#E74C3C', '#F39C12', '#2ECC71'], alpha=0.7)
            
            ax2.set_ylabel('Score Component', fontsize=12)
            ax2.set_title(f'Score Breakdown for ${sample_level["price"]:.2f} ({sample_level["type"]})', 
                         fontsize=12)
            ax2.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, value in zip(bars2, component_values):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{value:.1f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.default_style['dpi'], 
                       bbox_inches='tight', facecolor=self.colors['background'])
        
        return fig
    
    def create_quick_levels_chart(self, data: pd.DataFrame, analysis_results: Dict[str, Any],
                                symbol: str = "STOCK", save_path: Optional[str] = None) -> plt.Figure:
        """Create simplified chart showing just price and key levels"""
        
        fig, ax = plt.subplots(figsize=(14, 8))
        fig.patch.set_facecolor(self.colors['background'])
        
        # Plot price
        ax.plot(data.index, data['close'], color=self.colors['price_line'],
               linewidth=2, label='Close Price')
        
        # Add support/resistance levels
        self._add_support_resistance_levels(ax, data, analysis_results)
        
        # Current price
        current_analysis = analysis_results.get('current_position_analysis', {})
        if 'current_price' in current_analysis:
            current_price = current_analysis['current_price']
            ax.axhline(y=current_price, color='red', linestyle='--', alpha=0.8,
                      linewidth=2, label=f'Current: ${current_price:.2f}')
        
        # Formatting
        ax.set_title(f'{symbol} - Key Support & Resistance Levels', 
                    fontsize=16, fontweight='bold', color=self.colors['text'])
        ax.set_ylabel('Price ($)', fontsize=12)
        ax.grid(True, alpha=0.3, color=self.colors['grid'])
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Format dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.default_style['dpi'], 
                       bbox_inches='tight', facecolor=self.colors['background'])
        
        return fig

def demo_chart_creation():
    """Demonstrate chart creation with sample data"""
    print("📊 Creating Support/Resistance Chart Demo")
    print("=" * 60)
    
    # Create sample data
    dates = pd.date_range(start='2024-01-01', end='2024-10-20', freq='D')
    np.random.seed(42)
    
    base_price = 2400
    support_level = 2350
    resistance_level = 2450
    
    prices = []
    current_price = base_price
    
    for i in range(len(dates)):
        change = np.random.normal(0, 15)
        new_price = current_price + change
        
        # Bounce off levels
        if new_price <= support_level and np.random.random() > 0.3:
            new_price = support_level + np.random.uniform(5, 20)
        elif new_price >= resistance_level and np.random.random() > 0.3:
            new_price = resistance_level - np.random.uniform(5, 20)
        
        prices.append(new_price)
        current_price = new_price
    
    # Create OHLCV data
    data = pd.DataFrame({
        'open': [p + np.random.normal(0, 5) for p in prices],
        'high': [p + abs(np.random.normal(8, 4)) for p in prices],
        'low': [p - abs(np.random.normal(8, 4)) for p in prices],
        'close': prices,
        'volume': [int(np.random.lognormal(np.log(1500000), 0.3)) for _ in prices]
    }, index=dates)
    
    # Ensure realistic OHLC
    data['high'] = np.maximum(data[['open', 'close']].max(axis=1), data['high'])
    data['low'] = np.minimum(data[['open', 'close']].min(axis=1), data['low'])
    
    print(f"✅ Created sample data: {len(data)} days")
    
    # Run analysis
    agent = SupportResistanceAgent()
    analysis_results = agent.analyze(data, symbol='DEMO')
    
    if 'error' in analysis_results:
        print(f"❌ Analysis failed: {analysis_results['error']}")
        return
    
    print("✅ Analysis completed successfully")
    
    # Create charts
    chart_maker = SupportResistanceCharts()
    
    print("📈 Creating comprehensive chart...")
    fig1 = chart_maker.create_comprehensive_chart(
        data, analysis_results, symbol='DEMO',
        save_path='support_resistance_comprehensive_demo.png'
    )
    
    print("📊 Creating levels strength chart...")
    fig2 = chart_maker.create_levels_strength_chart(
        analysis_results,
        save_path='support_resistance_strength_demo.png'
    )
    
    print("📈 Creating quick levels chart...")
    fig3 = chart_maker.create_quick_levels_chart(
        data, analysis_results, symbol='DEMO',
        save_path='support_resistance_quick_demo.png'
    )
    
    print("✅ All charts created successfully!")
    print("   - support_resistance_comprehensive_demo.png")
    print("   - support_resistance_strength_demo.png") 
    print("   - support_resistance_quick_demo.png")
    
    # Show plots
    plt.show()

if __name__ == "__main__":
    demo_chart_creation()