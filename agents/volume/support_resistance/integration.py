#!/usr/bin/env python3
"""
Support/Resistance Agent - Integration Example

Shows how to integrate the Support/Resistance Agent into the main application
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

from .agent import SupportResistanceAgent
from .charts import SupportResistanceCharts

class SupportResistanceRegistry:
    """
    Registry for Support/Resistance Agent integration
    
    Provides standardized interface for the main application
    """
    
    def __init__(self):
        self.agent_info = {
            'name': 'Support/Resistance Agent',
            'version': '1.0.0',
            'description': 'Identifies volume-validated support and resistance levels',
            'category': 'Technical Analysis',
            'data_requirements': {
                'minimum_days': 90,
                'preferred_days': 180,
                'required_columns': ['open', 'high', 'low', 'close', 'volume']
            },
            'outputs': {
                'support_levels': 'Validated support levels with strength ratings',
                'resistance_levels': 'Validated resistance levels with strength ratings',
                'current_position': 'Current price position relative to key levels',
                'trading_implications': 'Risk levels, targets, and strategy suggestions',
                'recommendations': 'Actionable trading recommendations'
            }
        }
        
        self.agent = SupportResistanceAgent()
        self.chart_maker = SupportResistanceCharts()
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get agent information for registry"""
        return self.agent_info
    
    def analyze_stock(self, data: pd.DataFrame, symbol: str, 
                     include_charts: bool = False, 
                     save_charts_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze stock with Support/Resistance Agent
        
        Args:
            data: OHLCV DataFrame
            symbol: Stock symbol
            include_charts: Whether to generate charts
            save_charts_path: Path to save charts (if any)
            
        Returns:
            Comprehensive analysis results
        """
        
        # Run analysis
        analysis_results = self.agent.analyze(data, symbol=symbol)
        
        # Add chart generation if requested
        if include_charts and 'error' not in analysis_results:
            try:
                charts = {}
                
                # Comprehensive chart
                comprehensive_chart = self.chart_maker.create_comprehensive_chart(
                    data, analysis_results, symbol=symbol,
                    save_path=f"{save_charts_path}/{symbol}_support_resistance_comprehensive.png" if save_charts_path else None
                )
                charts['comprehensive'] = comprehensive_chart
                
                # Quick levels chart
                quick_chart = self.chart_maker.create_quick_levels_chart(
                    data, analysis_results, symbol=symbol,
                    save_path=f"{save_charts_path}/{symbol}_support_resistance_levels.png" if save_charts_path else None
                )
                charts['levels'] = quick_chart
                
                # Strength analysis chart
                strength_chart = self.chart_maker.create_levels_strength_chart(
                    analysis_results,
                    save_path=f"{save_charts_path}/{symbol}_support_resistance_strength.png" if save_charts_path else None
                )
                charts['strength'] = strength_chart
                
                analysis_results['charts'] = charts
                
            except Exception as e:
                analysis_results['chart_error'] = f"Chart generation failed: {str(e)}"
        
        return analysis_results
    
    def get_quick_levels(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Get just the key levels without full analysis"""
        try:
            support_levels = self.agent.get_key_levels(data, 'support')
            resistance_levels = self.agent.get_key_levels(data, 'resistance')
            current_position = self.agent.get_current_position_analysis(data)
            
            return {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'support_levels': support_levels,
                'resistance_levels': resistance_levels,
                'current_position': current_position,
                'total_levels': len(support_levels) + len(resistance_levels)
            }
        except Exception as e:
            return {
                'symbol': symbol,
                'error': f"Quick levels analysis failed: {str(e)}"
            }
    
    def get_trading_signals(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Get trading signals and implications"""
        try:
            implications = self.agent.get_trading_implications(data)
            position_analysis = self.agent.get_current_position_analysis(data)
            
            # Generate simplified signals
            signals = []
            current_price = position_analysis.get('current_price', 0)
            range_position = position_analysis.get('range_position_classification', 'unknown')
            
            if range_position == 'near_support':
                signals.append({
                    'type': 'BUY_SIGNAL',
                    'strength': 'MEDIUM',
                    'reason': 'Price near key support level',
                    'current_price': current_price,
                    'target': implications.get('target_levels', {}).get('resistance_target', {}).get('price'),
                    'stop_loss': implications.get('risk_levels', {}).get('support_break', {}).get('price')
                })
            
            elif range_position == 'near_resistance':
                signals.append({
                    'type': 'SELL_SIGNAL',
                    'strength': 'MEDIUM', 
                    'reason': 'Price near key resistance level',
                    'current_price': current_price,
                    'risk_reward_ratio': implications.get('risk_reward_ratio')
                })
            
            elif range_position == 'middle_range':
                signals.append({
                    'type': 'WAIT',
                    'strength': 'LOW',
                    'reason': 'Price in middle of range - wait for clear direction',
                    'current_price': current_price
                })
            
            return {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'signals': signals,
                'range_position': range_position,
                'trading_strategy': implications.get('trading_strategy', 'unknown'),
                'risk_reward_ratio': implications.get('risk_reward_ratio')
            }
            
        except Exception as e:
            return {
                'symbol': symbol,
                'error': f"Trading signals analysis failed: {str(e)}"
            }
    
    def batch_analyze(self, stock_data: Dict[str, pd.DataFrame], 
                     include_charts: bool = False) -> Dict[str, Any]:
        """
        Analyze multiple stocks in batch
        
        Args:
            stock_data: Dict mapping symbol -> OHLCV DataFrame
            include_charts: Whether to generate charts
            
        Returns:
            Dict mapping symbol -> analysis results
        """
        
        results = {}
        
        for symbol, data in stock_data.items():
            print(f"Analyzing {symbol}...")
            
            try:
                analysis = self.analyze_stock(data, symbol, include_charts=include_charts)
                results[symbol] = analysis
                
                # Quick summary
                if 'error' not in analysis:
                    summary = analysis.get('analysis_summary', {})
                    levels_found = summary.get('total_validated_levels', 0)
                    quality_score = summary.get('analysis_quality_score', 0)
                    print(f"  ✅ {symbol}: {levels_found} levels, quality {quality_score}/100")
                else:
                    print(f"  ❌ {symbol}: {analysis['error']}")
                    
            except Exception as e:
                results[symbol] = {'error': f"Analysis failed: {str(e)}"}
                print(f"  ❌ {symbol}: Analysis failed")
        
        return results
    
    def create_portfolio_summary(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary across multiple stocks"""
        
        summary = {
            'total_stocks': len(analysis_results),
            'successful_analyses': 0,
            'failed_analyses': 0,
            'average_quality_score': 0,
            'stocks_near_support': [],
            'stocks_near_resistance': [],
            'high_quality_analyses': [],
            'trading_opportunities': []
        }
        
        quality_scores = []
        
        for symbol, analysis in analysis_results.items():
            if 'error' in analysis:
                summary['failed_analyses'] += 1
                continue
                
            summary['successful_analyses'] += 1
            
            # Quality metrics
            analysis_summary = analysis.get('analysis_summary', {})
            quality_score = analysis_summary.get('analysis_quality_score', 0)
            quality_scores.append(quality_score)
            
            if quality_score >= 80:
                summary['high_quality_analyses'].append({
                    'symbol': symbol,
                    'quality_score': quality_score,
                    'levels_found': analysis_summary.get('total_validated_levels', 0)
                })
            
            # Position analysis
            current_position = analysis.get('current_position', {})
            range_position = current_position.get('range_position_classification', 'unknown')
            
            if range_position == 'near_support':
                summary['stocks_near_support'].append({
                    'symbol': symbol,
                    'current_price': current_position.get('current_price'),
                    'support_distance_pct': current_position.get('support_distance_percentage')
                })
            elif range_position == 'near_resistance':
                summary['stocks_near_resistance'].append({
                    'symbol': symbol,
                    'current_price': current_position.get('current_price'),
                    'resistance_distance_pct': current_position.get('resistance_distance_percentage')
                })
            
            # Trading opportunities
            recommendations = analysis.get('recommendations', [])
            for rec in recommendations:
                if rec.get('priority') == 'high' and rec.get('type') in ['entry_opportunity', 'exit_opportunity']:
                    summary['trading_opportunities'].append({
                        'symbol': symbol,
                        'type': rec['type'],
                        'action': rec['action'],
                        'reason': rec['reason']
                    })
        
        # Calculate averages
        if quality_scores:
            summary['average_quality_score'] = sum(quality_scores) / len(quality_scores)
        
        return summary

def demo_integration():
    """Demonstrate integration usage"""
    print("🔧 Support/Resistance Agent Integration Demo")
    print("=" * 70)
    
    # Initialize registry
    registry = SupportResistanceRegistry()
    
    print("📋 Agent Information:")
    info = registry.get_agent_info()
    print(f"   Name: {info['name']} v{info['version']}")
    print(f"   Description: {info['description']}")
    print(f"   Data Requirements: {info['data_requirements']['minimum_days']} days minimum")
    
    # Create sample multi-stock data
    symbols = ['STOCK_A', 'STOCK_B', 'STOCK_C']
    stock_data = {}
    
    for symbol in symbols:
        # Create different market scenarios
        dates = pd.date_range(start='2024-01-01', end='2024-10-20', freq='D')
        np.random.seed(hash(symbol) % 1000)  # Different seed per stock
        
        if symbol == 'STOCK_A':
            # Range-bound
            prices = []
            current_price = 2400
            support, resistance = 2350, 2450
            
            for _ in range(len(dates)):
                change = np.random.normal(0, 15)
                new_price = current_price + change
                if new_price <= support and np.random.random() > 0.3:
                    new_price = support + np.random.uniform(5, 20)
                elif new_price >= resistance and np.random.random() > 0.3:
                    new_price = resistance - np.random.uniform(5, 20)
                prices.append(new_price)
                current_price = new_price
                
        elif symbol == 'STOCK_B':
            # Uptrend
            base_price = 1800
            prices = []
            for i in range(len(dates)):
                trend = i * 0.8
                noise = np.random.normal(0, 20)
                prices.append(base_price + trend + noise)
                
        else:  # STOCK_C
            # Volatile
            prices = []
            current_price = 3000
            for _ in range(len(dates)):
                change = np.random.normal(0, 40)
                prices.append(current_price + change)
                current_price = prices[-1]
        
        # Create OHLCV data
        data = pd.DataFrame({
            'open': [p + np.random.normal(0, 5) for p in prices],
            'high': [p + abs(np.random.normal(10, 5)) for p in prices],
            'low': [p - abs(np.random.normal(10, 5)) for p in prices],
            'close': prices,
            'volume': [int(np.random.lognormal(np.log(2000000), 0.4)) for _ in prices]
        }, index=dates)
        
        # Fix OHLC relationships
        data['high'] = np.maximum(data[['open', 'close']].max(axis=1), data['high'])
        data['low'] = np.minimum(data[['open', 'close']].min(axis=1), data['low'])
        
        stock_data[symbol] = data
    
    print(f"\n📊 Created sample data for {len(symbols)} stocks")
    
    # Batch analysis
    print("\n🔍 Running batch analysis...")
    analysis_results = registry.batch_analyze(stock_data, include_charts=False)
    
    # Portfolio summary
    print("\n📈 Creating portfolio summary...")
    portfolio_summary = registry.create_portfolio_summary(analysis_results)
    
    print(f"\n📋 Portfolio Summary:")
    print(f"   Total stocks analyzed: {portfolio_summary['total_stocks']}")
    print(f"   Successful analyses: {portfolio_summary['successful_analyses']}")
    print(f"   Average quality score: {portfolio_summary['average_quality_score']:.1f}/100")
    print(f"   High-quality analyses: {len(portfolio_summary['high_quality_analyses'])}")
    print(f"   Stocks near support: {len(portfolio_summary['stocks_near_support'])}")
    print(f"   Stocks near resistance: {len(portfolio_summary['stocks_near_resistance'])}")
    print(f"   Trading opportunities: {len(portfolio_summary['trading_opportunities'])}")
    
    # Show trading opportunities
    if portfolio_summary['trading_opportunities']:
        print(f"\n💡 Trading Opportunities:")
        for opp in portfolio_summary['trading_opportunities'][:3]:
            action = opp['action'].replace('_', ' ').title()
            print(f"   • {opp['symbol']}: {action} - {opp['reason']}")
    
    # Show stocks near key levels
    if portfolio_summary['stocks_near_support']:
        print(f"\n📉 Stocks Near Support:")
        for stock in portfolio_summary['stocks_near_support']:
            print(f"   • {stock['symbol']}: ${stock['current_price']:.2f} ({stock['support_distance_pct']:.1f}% from support)")
    
    if portfolio_summary['stocks_near_resistance']:
        print(f"\n📈 Stocks Near Resistance:")
        for stock in portfolio_summary['stocks_near_resistance']:
            print(f"   • {stock['symbol']}: ${stock['current_price']:.2f} ({stock['resistance_distance_pct']:.1f}% from resistance)")
    
    print(f"\n✅ Integration demo completed successfully!")
    
    # Example of getting quick signals
    print(f"\n⚡ Quick Trading Signals Example:")
    for symbol in symbols[:2]:
        signals = registry.get_trading_signals(stock_data[symbol], symbol)
        if 'error' not in signals:
            for signal in signals['signals']:
                signal_type = signal['type']
                reason = signal['reason']
                print(f"   • {symbol}: {signal_type} - {reason}")

# Alias for backwards compatibility
SupportResistanceIntegration = SupportResistanceRegistry

if __name__ == "__main__":
    demo_integration()