#!/usr/bin/env python3
"""
Pattern Detection Charts - Visualization Module

This module generates advanced charts for pattern detection analysis including:
- Pattern overlay charts with detected patterns highlighted
- Pattern completion status charts
- Key level identification charts
- Pattern confluence visualization
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
import os

logger = logging.getLogger(__name__)

class PatternDetectionChartGenerator:
    """
    Advanced chart generator for pattern detection analysis.
    
    Generates comprehensive visualizations including:
    - Pattern overlay charts with detected patterns
    - Pattern formation stage visualization
    - Key level analysis charts
    - Pattern confluence indicators
    """
    
    def __init__(self):
        self.name = "pattern_detection_charts"
        self.version = "1.0.0"
    
    def generate_pattern_detection_charts(
        self, 
        stock_data: pd.DataFrame, 
        pattern_data: Dict[str, Any],
        symbol: str = "STOCK",
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive pattern detection charts.
        
        Args:
            stock_data: DataFrame with OHLCV data
            pattern_data: Pattern detection analysis results
            symbol: Stock symbol for chart titles
            save_path: Optional path to save charts
            
        Returns:
            Dictionary containing chart data and metadata
        """
        try:
            logger.info(f"[PATTERN_DETECTION_CHARTS] Generating charts for {symbol}")
            
            if stock_data is None or stock_data.empty:
                return self._build_error_result("No stock data provided for charting")
            
            if not pattern_data.get('success', False):
                return self._build_error_result("Pattern detection analysis failed")
            
            charts_data = {}
            
            # 1. Main Pattern Detection Chart
            main_chart = self._create_pattern_overlay_chart(stock_data, pattern_data, symbol)
            charts_data['pattern_overlay_chart'] = main_chart
            
            # 2. Pattern Formation Stage Chart
            formation_chart = self._create_formation_stage_chart(pattern_data, symbol)
            charts_data['formation_stage_chart'] = formation_chart
            
            # 3. Key Levels Chart
            levels_chart = self._create_key_levels_chart(stock_data, pattern_data, symbol)
            charts_data['key_levels_chart'] = levels_chart
            
            # 4. Pattern Confluence Heatmap
            confluence_chart = self._create_confluence_heatmap(pattern_data, symbol)
            charts_data['confluence_heatmap'] = confluence_chart
            
            # 5. Pattern Quality Dashboard
            quality_dashboard = self._create_quality_dashboard(pattern_data, symbol)
            charts_data['quality_dashboard'] = quality_dashboard
            
            # Save charts if path provided
            if save_path and os.path.exists(os.path.dirname(save_path)):
                self._save_charts(charts_data, save_path, symbol)
            
            result = {
                'success': True,
                'agent_name': self.name,
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'charts_generated': len(charts_data),
                'charts_data': charts_data,
                'total_patterns_visualized': pattern_data.get('total_patterns_detected', 0)
            }
            
            logger.info(f"[PATTERN_DETECTION_CHARTS] Generated {len(charts_data)} charts for {symbol}")
            return result
            
        except Exception as e:
            logger.error(f"[PATTERN_DETECTION_CHARTS] Chart generation failed: {e}")
            return self._build_error_result(str(e))
    
    def _create_pattern_overlay_chart(
        self, 
        stock_data: pd.DataFrame, 
        pattern_data: Dict[str, Any], 
        symbol: str
    ) -> Dict[str, Any]:
        """Create main candlestick chart with pattern overlays"""
        try:
            fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                row_width=[0.5, 0.3, 0.2],
                subplot_titles=('Price Action with Pattern Detection', 'Volume', 'Pattern Quality Score'),
                specs=[[{"secondary_y": True}],
                       [{"secondary_y": False}],
                       [{"secondary_y": False}]]
            )
            
            # Add candlestick chart
            fig.add_trace(
                go.Candlestick(
                    x=stock_data.index,
                    open=stock_data['open'],
                    high=stock_data['high'],
                    low=stock_data['low'],
                    close=stock_data['close'],
                    name='Price',
                    showlegend=False
                ),
                row=1, col=1
            )
            
            # Add detected patterns overlay
            patterns = pattern_data.get('detected_patterns', [])
            if patterns:
                self._add_pattern_overlays(fig, stock_data, patterns, row=1)
            
            # Add key levels
            key_levels = pattern_data.get('key_levels', {})
            if key_levels:
                self._add_key_level_lines(fig, stock_data, key_levels, row=1)
            
            # Add volume
            if 'volume' in stock_data.columns:
                colors = ['red' if close < open else 'green' 
                         for close, open in zip(stock_data['close'], stock_data['open'])]
                
                fig.add_trace(
                    go.Bar(
                        x=stock_data.index,
                        y=stock_data['volume'],
                        name='Volume',
                        marker_color=colors,
                        showlegend=False
                    ),
                    row=2, col=1
                )
            
            # Add pattern quality scores over time
            quality_scores = self._calculate_rolling_pattern_quality(stock_data, patterns)
            if quality_scores:
                fig.add_trace(
                    go.Scatter(
                        x=stock_data.index,
                        y=quality_scores,
                        mode='lines',
                        name='Pattern Quality',
                        line=dict(color='purple', width=2),
                        showlegend=False
                    ),
                    row=3, col=1
                )
            
            # Update layout
            fig.update_layout(
                title=f'{symbol} - Pattern Detection Analysis',
                xaxis_title='Date',
                height=800,
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=1.01
                )
            )
            
            fig.update_yaxes(title_text="Price", row=1, col=1)
            fig.update_yaxes(title_text="Volume", row=2, col=1)
            fig.update_yaxes(title_text="Quality Score", row=3, col=1)
            fig.update_xaxes(title_text="Date", row=3, col=1)
            
            return {
                'chart_type': 'pattern_overlay',
                'title': f'{symbol} Pattern Detection Analysis',
                'data': fig.to_json(),
                'patterns_shown': len(patterns),
                'key_levels_shown': len(key_levels)
            }
            
        except Exception as e:
            logger.error(f"[PATTERN_DETECTION_CHARTS] Pattern overlay chart failed: {e}")
            return {'error': str(e)}
    
    def _add_pattern_overlays(self, fig, stock_data: pd.DataFrame, patterns: List[Dict], row: int):
        """Add pattern overlays to the chart"""
        try:
            for i, pattern in enumerate(patterns):
                pattern_name = pattern.get('pattern_name', f'Pattern {i+1}')
                pattern_type = pattern.get('pattern_type', 'unknown')
                completion = pattern.get('completion_percentage', 0)
                reliability = pattern.get('reliability', 'unknown')
                
                # Get pattern color based on type and reliability
                if pattern_type == 'reversal':
                    color = 'red' if reliability == 'high' else 'orange'
                elif pattern_type == 'continuation':
                    color = 'blue' if reliability == 'high' else 'lightblue'
                else:
                    color = 'gray'
                
                # Add pattern annotation
                if hasattr(stock_data.index, 'to_list'):
                    mid_date = stock_data.index[len(stock_data) // 2]
                else:
                    mid_date = stock_data.index.iloc[len(stock_data) // 2] if hasattr(stock_data.index, 'iloc') else stock_data.index[len(stock_data) // 2]
                
                mid_price = (stock_data['high'].max() + stock_data['low'].min()) / 2
                
                fig.add_annotation(
                    x=mid_date,
                    y=mid_price + (i * 0.02 * mid_price),  # Offset annotations
                    text=f"{pattern_name}<br>({completion:.0f}% - {reliability})",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor=color,
                    bgcolor=color,
                    bordercolor=color,
                    font=dict(color="white", size=10),
                    row=row, col=1
                )
                
                # Add pattern-specific visualizations
                self._add_specific_pattern_lines(fig, stock_data, pattern, color, row)
                
        except Exception as e:
            logger.error(f"[PATTERN_DETECTION_CHARTS] Pattern overlay addition failed: {e}")
    
    def _add_specific_pattern_lines(self, fig, stock_data: pd.DataFrame, pattern: Dict, color: str, row: int):
        """Add pattern-specific trend lines and shapes"""
        try:
            pattern_data = pattern.get('pattern_data', {})
            pattern_name = pattern.get('pattern_name', '')
            
            # Triangle patterns - add trend lines
            if 'triangle' in pattern_name.lower():
                if 'high_trend' in pattern_data and 'low_trend' in pattern_data:
                    high_trend = pattern_data['high_trend']
                    low_trend = pattern_data['low_trend']
                    
                    # Add trend lines (simplified)
                    x_values = stock_data.index[-len(high_trend):]
                    
                    fig.add_trace(
                        go.Scatter(
                            x=x_values,
                            y=high_trend,
                            mode='lines',
                            line=dict(color=color, dash='dash'),
                            name=f'{pattern_name} Upper',
                            showlegend=False
                        ),
                        row=row, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(
                            x=x_values,
                            y=low_trend,
                            mode='lines',
                            line=dict(color=color, dash='dash'),
                            name=f'{pattern_name} Lower',
                            showlegend=False
                        ),
                        row=row, col=1
                    )
            
            # Channel patterns - add support/resistance lines
            elif 'channel' in pattern_name.lower() or 'rectangle' in pattern_name.lower():
                if 'resistance_level' in pattern_data and 'support_level' in pattern_data:
                    resistance = pattern_data['resistance_level']
                    support = pattern_data['support_level']
                    
                    fig.add_hline(
                        y=resistance,
                        line=dict(color=color, dash='dot', width=2),
                        annotation_text=f"Resistance: {resistance:.2f}",
                        row=row, col=1
                    )
                    
                    fig.add_hline(
                        y=support,
                        line=dict(color=color, dash='dot', width=2),
                        annotation_text=f"Support: {support:.2f}",
                        row=row, col=1
                    )
            
            # Head and shoulders - add neckline
            elif 'head' in pattern_name.lower() and 'shoulders' in pattern_name.lower():
                if 'neckline_level' in pattern_data:
                    neckline = pattern_data['neckline_level']
                    
                    fig.add_hline(
                        y=neckline,
                        line=dict(color=color, dash='dashdot', width=2),
                        annotation_text=f"Neckline: {neckline:.2f}",
                        row=row, col=1
                    )
            
        except Exception as e:
            logger.error(f"[PATTERN_DETECTION_CHARTS] Pattern-specific lines failed: {e}")
    
    def _add_key_level_lines(self, fig, stock_data: pd.DataFrame, key_levels: Dict, row: int):
        """Add key level horizontal lines"""
        try:
            current_price = key_levels.get('current_price', 0)
            resistance = key_levels.get('nearest_resistance')
            support = key_levels.get('nearest_support')
            breakout = key_levels.get('breakout_level')
            
            if resistance and resistance != current_price:
                fig.add_hline(
                    y=resistance,
                    line=dict(color='red', dash='dash', width=2),
                    annotation_text=f"Key Resistance: {resistance:.2f}",
                    row=row, col=1
                )
            
            if support and support != current_price:
                fig.add_hline(
                    y=support,
                    line=dict(color='green', dash='dash', width=2),
                    annotation_text=f"Key Support: {support:.2f}",
                    row=row, col=1
                )
            
            if breakout and breakout != current_price:
                fig.add_hline(
                    y=breakout,
                    line=dict(color='purple', dash='solid', width=2),
                    annotation_text=f"Breakout Level: {breakout:.2f}",
                    row=row, col=1
                )
                
        except Exception as e:
            logger.error(f"[PATTERN_DETECTION_CHARTS] Key level lines failed: {e}")
    
    def _calculate_rolling_pattern_quality(self, stock_data: pd.DataFrame, patterns: List[Dict]) -> List[float]:
        """Calculate rolling pattern quality scores"""
        try:
            quality_scores = []
            data_length = len(stock_data)
            
            for i in range(data_length):
                # Simple quality score based on pattern presence and reliability
                quality = 0.3  # Base quality
                
                for pattern in patterns:
                    completion = pattern.get('completion_percentage', 0) / 100
                    reliability_score = 0.8 if pattern.get('reliability') == 'high' else 0.5
                    quality += completion * reliability_score * 0.1
                
                quality_scores.append(min(1.0, quality))
            
            return quality_scores
            
        except Exception as e:
            logger.error(f"[PATTERN_DETECTION_CHARTS] Rolling quality calculation failed: {e}")
            return []
    
    def _create_formation_stage_chart(self, pattern_data: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """Create pattern formation stage visualization"""
        try:
            formation_stage = pattern_data.get('formation_stage', {})
            pattern_summary = pattern_data.get('pattern_summary', {})
            
            # Create gauge chart for pattern maturity
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=self._get_maturity_score(formation_stage),
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Pattern Formation Stage"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 33], 'color': "lightgray"},
                        {'range': [33, 66], 'color': "gray"},
                        {'range': [66, 100], 'color': "lightgreen"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            
            fig.update_layout(
                title=f'{symbol} - Pattern Formation Stage',
                height=400
            )
            
            return {
                'chart_type': 'formation_stage',
                'title': f'{symbol} Pattern Formation Stage',
                'data': fig.to_json(),
                'stage': formation_stage.get('primary_stage', 'unknown'),
                'maturity': formation_stage.get('pattern_maturity', 'unknown')
            }
            
        except Exception as e:
            logger.error(f"[PATTERN_DETECTION_CHARTS] Formation stage chart failed: {e}")
            return {'error': str(e)}
    
    def _get_maturity_score(self, formation_stage: Dict[str, Any]) -> float:
        """Convert formation stage to numeric score"""
        maturity = formation_stage.get('pattern_maturity', 'early')
        if maturity == 'mature':
            return 85
        elif maturity == 'developing':
            return 55
        else:
            return 25
    
    def _create_key_levels_chart(self, stock_data: pd.DataFrame, pattern_data: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """Create key levels analysis chart"""
        try:
            key_levels = pattern_data.get('key_levels', {})
            current_price = key_levels.get('current_price', stock_data['close'].iloc[-1])
            
            # Create horizontal bar chart for key levels
            levels_data = []
            colors = []
            
            if key_levels.get('nearest_resistance'):
                levels_data.append(('Resistance', key_levels['nearest_resistance']))
                colors.append('red')
            
            levels_data.append(('Current Price', current_price))
            colors.append('blue')
            
            if key_levels.get('nearest_support'):
                levels_data.append(('Support', key_levels['nearest_support']))
                colors.append('green')
            
            if key_levels.get('breakout_level') and key_levels['breakout_level'] != current_price:
                levels_data.append(('Breakout Level', key_levels['breakout_level']))
                colors.append('purple')
            
            if levels_data:
                labels, values = zip(*levels_data)
                
                fig = go.Figure(go.Bar(
                    y=labels,
                    x=values,
                    orientation='h',
                    marker_color=colors,
                    text=[f'{v:.2f}' for v in values],
                    textposition='inside'
                ))
                
                fig.update_layout(
                    title=f'{symbol} - Key Price Levels from Patterns',
                    xaxis_title='Price Level',
                    height=300,
                    showlegend=False
                )
                
                return {
                    'chart_type': 'key_levels',
                    'title': f'{symbol} Key Price Levels',
                    'data': fig.to_json(),
                    'levels_count': len(levels_data)
                }
            else:
                return {'error': 'No key levels data available'}
                
        except Exception as e:
            logger.error(f"[PATTERN_DETECTION_CHARTS] Key levels chart failed: {e}")
            return {'error': str(e)}
    
    def _create_confluence_heatmap(self, pattern_data: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """Create pattern confluence heatmap"""
        try:
            patterns = pattern_data.get('detected_patterns', [])
            pattern_summary = pattern_data.get('pattern_summary', {})
            
            if not patterns:
                return {'error': 'No patterns available for confluence analysis'}
            
            # Create matrix for pattern relationships
            pattern_names = [p.get('pattern_name', f'Pattern {i+1}') for i, p in enumerate(patterns)]
            pattern_types = [p.get('pattern_type', 'unknown') for p in patterns]
            
            # Calculate confluence scores
            confluence_matrix = []
            for i, pattern1 in enumerate(patterns):
                row = []
                for j, pattern2 in enumerate(patterns):
                    if i == j:
                        score = 1.0  # Perfect self-correlation
                    else:
                        # Calculate confluence based on pattern compatibility
                        score = self._calculate_pattern_confluence(pattern1, pattern2)
                    row.append(score)
                confluence_matrix.append(row)
            
            fig = go.Figure(data=go.Heatmap(
                z=confluence_matrix,
                x=pattern_names,
                y=pattern_names,
                colorscale='RdYlBu_r',
                text=[[f'{val:.2f}' for val in row] for row in confluence_matrix],
                texttemplate="%{text}",
                textfont={"size": 10},
                colorbar=dict(title="Confluence Score")
            ))
            
            fig.update_layout(
                title=f'{symbol} - Pattern Confluence Analysis',
                height=400,
                width=500
            )
            
            return {
                'chart_type': 'confluence_heatmap',
                'title': f'{symbol} Pattern Confluence',
                'data': fig.to_json(),
                'patterns_analyzed': len(patterns)
            }
            
        except Exception as e:
            logger.error(f"[PATTERN_DETECTION_CHARTS] Confluence heatmap failed: {e}")
            return {'error': str(e)}
    
    def _calculate_pattern_confluence(self, pattern1: Dict, pattern2: Dict) -> float:
        """Calculate confluence score between two patterns"""
        try:
            # Same pattern type increases confluence
            type_match = 0.3 if pattern1.get('pattern_type') == pattern2.get('pattern_type') else 0.1
            
            # Both high reliability increases confluence
            rel1 = pattern1.get('reliability', 'low')
            rel2 = pattern2.get('reliability', 'low')
            reliability_score = 0.4 if (rel1 == 'high' and rel2 == 'high') else 0.2
            
            # Similar completion percentages increase confluence
            comp1 = pattern1.get('completion_percentage', 0)
            comp2 = pattern2.get('completion_percentage', 0)
            completion_diff = abs(comp1 - comp2) / 100
            completion_score = 0.3 * (1 - completion_diff)
            
            return min(1.0, type_match + reliability_score + completion_score)
            
        except Exception:
            return 0.1
    
    def _create_quality_dashboard(self, pattern_data: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """Create pattern quality dashboard"""
        try:
            patterns = pattern_data.get('detected_patterns', [])
            pattern_summary = pattern_data.get('pattern_summary', {})
            confidence_score = pattern_data.get('confidence_score', 0)
            
            # Create subplot dashboard
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Pattern Types Distribution', 'Reliability Scores', 
                               'Completion Status', 'Overall Confidence'),
                specs=[[{"type": "pie"}, {"type": "bar"}],
                       [{"type": "bar"}, {"type": "indicator"}]]
            )
            
            # Pattern types pie chart
            if patterns:
                type_counts = {}
                for pattern in patterns:
                    ptype = pattern.get('pattern_type', 'unknown')
                    type_counts[ptype] = type_counts.get(ptype, 0) + 1
                
                fig.add_trace(
                    go.Pie(
                        labels=list(type_counts.keys()),
                        values=list(type_counts.values()),
                        name="Pattern Types"
                    ),
                    row=1, col=1
                )
                
                # Reliability bar chart
                reliability_counts = {}
                for pattern in patterns:
                    rel = pattern.get('reliability', 'unknown')
                    reliability_counts[rel] = reliability_counts.get(rel, 0) + 1
                
                fig.add_trace(
                    go.Bar(
                        x=list(reliability_counts.keys()),
                        y=list(reliability_counts.values()),
                        name="Reliability",
                        marker_color='lightblue'
                    ),
                    row=1, col=2
                )
                
                # Completion status
                completion_data = [p.get('completion_percentage', 0) for p in patterns]
                pattern_names = [p.get('pattern_name', f'P{i+1}') for i, p in enumerate(patterns)]
                
                fig.add_trace(
                    go.Bar(
                        x=pattern_names,
                        y=completion_data,
                        name="Completion %",
                        marker_color='lightgreen'
                    ),
                    row=2, col=1
                )
            
            # Overall confidence indicator
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=confidence_score * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Confidence"},
                    gauge={'axis': {'range': [None, 100]},
                           'bar': {'color': "darkblue"},
                           'steps': [{'range': [0, 50], 'color': "lightgray"},
                                   {'range': [50, 80], 'color': "gray"},
                                   {'range': [80, 100], 'color': "lightgreen"}]}
                ),
                row=2, col=2
            )
            
            fig.update_layout(
                title=f'{symbol} - Pattern Quality Dashboard',
                height=600,
                showlegend=False
            )
            
            return {
                'chart_type': 'quality_dashboard',
                'title': f'{symbol} Pattern Quality Dashboard',
                'data': fig.to_json(),
                'patterns_analyzed': len(patterns),
                'confidence_score': confidence_score
            }
            
        except Exception as e:
            logger.error(f"[PATTERN_DETECTION_CHARTS] Quality dashboard failed: {e}")
            return {'error': str(e)}
    
    def _save_charts(self, charts_data: Dict, save_path: str, symbol: str):
        """Save charts to files"""
        try:
            base_path = os.path.splitext(save_path)[0]
            
            for chart_name, chart_info in charts_data.items():
                if 'data' in chart_info and not chart_info.get('error'):
                    chart_path = f"{base_path}_{chart_name}_{symbol}.html"
                    
                    # Convert JSON back to figure and save
                    import plotly.io as pio
                    fig = pio.from_json(chart_info['data'])
                    fig.write_html(chart_path)
                    
            logger.info(f"[PATTERN_DETECTION_CHARTS] Charts saved to {base_path}_*_{symbol}.html")
            
        except Exception as e:
            logger.error(f"[PATTERN_DETECTION_CHARTS] Chart saving failed: {e}")
    
    def _build_error_result(self, error_message: str) -> Dict[str, Any]:
        """Build error result dictionary"""
        return {
            'success': False,
            'agent_name': self.name,
            'error': error_message,
            'timestamp': datetime.now().isoformat(),
            'charts_generated': 0
        }