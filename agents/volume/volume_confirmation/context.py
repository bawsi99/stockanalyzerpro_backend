#!/usr/bin/env python3
"""
Volume Confirmation Agent - Context Formatting Module

This module formats analysis data into structured context for the Volume Confirmation Agent prompt.
"""

from typing import Dict, List, Any
from datetime import datetime

class VolumeConfirmationContextFormatter:
    """
    Context formatter for Volume Confirmation Agent
    
    Formats processed volume confirmation data into structured context for LLM analysis
    """
    
    def __init__(self):
        pass
    
    def format_context(self, analysis_data: Dict[str, Any], 
                      stock_symbol: str = "STOCK", 
                      company_name: str = "Company", 
                      sector: str = "Sector") -> str:
        """
        Format volume confirmation analysis data into structured context
        
        Args:
            analysis_data: Processed analysis data from VolumeConfirmationProcessor
            stock_symbol: Stock symbol
            company_name: Company name
            sector: Company sector
            
        Returns:
            Formatted context string for LLM prompt
        """
        if 'error' in analysis_data:
            return self._format_error_context(analysis_data, stock_symbol)
        
        context_parts = []
        
        # Header information
        context_parts.append(f"**Stock**: {stock_symbol} ({company_name})")
        context_parts.append(f"**Sector**: {sector}")
        context_parts.append(f"**Analysis Period**: {analysis_data.get('data_period', 'unknown')}")
        context_parts.append(f"**Data Quality**: {analysis_data.get('data_quality', 'unknown')}")
        context_parts.append("")
        
        # Price-Volume Correlation Analysis
        correlation_data = analysis_data.get('price_volume_correlation', {})
        if 'error' not in correlation_data:
            context_parts.append("## Price-Volume Correlation:")
            context_parts.append(f"- **Correlation Coefficient**: {correlation_data.get('correlation_coefficient', 0):.3f}")
            context_parts.append(f"- **Correlation Strength**: {correlation_data.get('correlation_strength', 'unknown')}")
            context_parts.append(f"- **Correlation Direction**: {correlation_data.get('correlation_direction', 'unknown')}")
        else:
            context_parts.append("## Price-Volume Correlation:")
            context_parts.append("- **Status**: Analysis unavailable")
        context_parts.append("")
        
        # Volume Analysis
        volume_averages = analysis_data.get('volume_averages', {})
        if 'error' not in volume_averages:
            context_parts.append("## Volume Analysis:")
            context_parts.append(f"- **Current Volume**: {volume_averages.get('current_volume', 0):,}")
            context_parts.append(f"- **20-Day Volume Average**: {volume_averages.get('volume_20d_avg', 0):,}")
            context_parts.append(f"- **Volume vs 20D Avg**: {volume_averages.get('volume_vs_20d', 1.0):.2f}x")
        context_parts.append("")
        
        # Recent Confirmation Signals (Top 3 Most Significant)
        recent_movements = analysis_data.get('recent_movements', [])
        if recent_movements:
            context_parts.append("## Recent Volume Confirmation Signals:")
            
            # Sort by significance and take top 3
            significance_order = {'high': 3, 'medium': 2, 'low': 1}
            sorted_movements = sorted(recent_movements, 
                                    key=lambda x: significance_order.get(x.get('significance', 'low'), 1), 
                                    reverse=True)[:3]
            
            for i, movement in enumerate(sorted_movements, 1):
                if 'error' not in movement:
                    context_parts.append(f"**Signal {i}** ({movement.get('date', 'unknown')}):")
                    context_parts.append(f"  - Price Movement: {movement.get('price_change_pct', 0):.2f}% {movement.get('price_move', 'unknown')}")
                    context_parts.append(f"  - Volume Response: {movement.get('volume_response', 'unknown')}")
                    context_parts.append(f"  - Volume Ratio: {movement.get('volume_ratio', 1.0):.2f}x")
                    context_parts.append("")
            
            # Summary of recent signals
            confirming_count = len([m for m in recent_movements if m.get('volume_response') == 'confirming'])
            diverging_count = len([m for m in recent_movements if m.get('volume_response') == 'diverging'])
            
            context_parts.append("**Recent Signals Summary:**")
            context_parts.append(f"- Confirming Signals: {confirming_count}")
            context_parts.append(f"- Diverging Signals: {diverging_count}")
            context_parts.append(f"- Total Analyzed: {len(recent_movements)}")
        else:
            context_parts.append("## Recent Volume Confirmation Signals:")
            context_parts.append("- No significant signals detected in recent period")
        context_parts.append("")
        
        # Trend Support Analysis
        trend_support = analysis_data.get('trend_support', {})
        if 'error' not in trend_support:
            context_parts.append("## Trend Support Analysis:")
            context_parts.append(f"- **Current Trend**: {trend_support.get('current_trend', 'unknown')}")
            context_parts.append(f"- **Uptrend Volume Support**: {trend_support.get('uptrend_volume_support', 'unknown')}")
            context_parts.append(f"- **Downtrend Volume Support**: {trend_support.get('downtrend_volume_support', 'unknown')}")
        else:
            context_parts.append("## Trend Support Analysis:")
            context_parts.append("- **Status**: Analysis unavailable")
        context_parts.append("")
        
        
        # Analysis metadata
        context_parts.append("## Analysis Metadata:")
        context_parts.append(f"- **Generated**: {analysis_data.get('analysis_timestamp', 'unknown')}")
        context_parts.append(f"- **Data Range**: {analysis_data.get('data_range', 'unknown')}")
        
        return "\n".join(context_parts)
    
    def _format_error_context(self, analysis_data: Dict[str, Any], stock_symbol: str) -> str:
        """Format context when analysis has errors"""
        error_msg = analysis_data.get('error', 'Unknown error occurred')
        data_length = analysis_data.get('data_length', 0)
        
        context_parts = [
            f"**Stock**: {stock_symbol}",
            f"**Analysis Status**: Error",
            f"**Error Details**: {error_msg}",
            f"**Available Data Points**: {data_length}",
            "",
            "## Volume Confirmation Analysis:",
            "- Unable to perform analysis due to data limitations",
            "- Minimum 20 data points required for reliable correlation analysis",
            "- Please provide more historical data for comprehensive volume confirmation analysis"
        ]
        
        return "\n".join(context_parts)
    
    def format_focused_context(self, analysis_data: Dict[str, Any], 
                             stock_symbol: str = "STOCK") -> str:
        """
        Format a focused context highlighting key volume confirmation insights
        
        Args:
            analysis_data: Processed analysis data
            stock_symbol: Stock symbol
            
        Returns:
            Concise context focusing on key confirmation signals
        """
        if 'error' in analysis_data:
            return f"**{stock_symbol}**: Analysis unavailable - {analysis_data.get('error', 'unknown error')}"
        
        # Extract key metrics
        overall = analysis_data.get('overall_assessment', {})
        correlation = analysis_data.get('price_volume_correlation', {})
        volume_avg = analysis_data.get('volume_averages', {})
        recent_signals = analysis_data.get('recent_movements', [])
        
        # Build focused summary
        status = overall.get('confirmation_status', 'unknown')
        strength = overall.get('confirmation_strength', 'unknown')
        confidence = overall.get('confidence_score', 0)
        
        corr_coef = correlation.get('correlation_coefficient', 0) if 'error' not in correlation else 0
        current_volume_ratio = volume_avg.get('volume_vs_20d', 1.0) if 'error' not in volume_avg else 1.0
        
        confirming_signals = len([s for s in recent_signals if s.get('volume_response') == 'confirming'])
        diverging_signals = len([s for s in recent_signals if s.get('volume_response') == 'diverging'])
        
        focused_context = f"""**{stock_symbol} - Volume Confirmation Analysis**
        
**Current Status**: {status.replace('_', ' ').title()} ({strength} strength, {confidence}% confidence)

**Key Metrics**:
- Price-Volume Correlation: {corr_coef:.3f}
- Current Volume vs 20D Average: {current_volume_ratio:.2f}x
- Recent Confirming Signals: {confirming_signals}
- Recent Diverging Signals: {diverging_signals}

**Analysis Focus**: Determine if recent price movements have proper volume backing for trend confirmation."""
        
        return focused_context


def test_volume_confirmation_context():
    """Test function for Volume Confirmation Context Formatter"""
    print("📝 Testing Volume Confirmation Context Formatter")
    print("=" * 60)
    
    # Import and use the processor for test data
    from volume_confirmation_processor import VolumeConfirmationProcessor
    import pandas as pd
    import numpy as np
    
    # Create sample data
    dates = pd.date_range(start='2024-08-01', end='2024-09-20', freq='D')
    np.random.seed(42)
    
    base_price = 2450
    price_trend = np.cumsum(np.random.normal(0.8, 12, len(dates)))
    prices = base_price + price_trend
    
    sample_data = pd.DataFrame({
        'open': prices + np.random.normal(0, 4, len(dates)),
        'high': prices + np.abs(np.random.normal(8, 6, len(dates))),
        'low': prices - np.abs(np.random.normal(8, 6, len(dates))),
        'close': prices,
        'volume': np.abs(np.random.lognormal(14.3, 0.7, len(dates)))
    }, index=dates)
    
    # Ensure realistic OHLC relationships
    sample_data['high'] = np.maximum(sample_data[['open', 'close']].max(axis=1), sample_data['high'])
    sample_data['low'] = np.minimum(sample_data[['open', 'close']].min(axis=1), sample_data['low'])
    
    print(f"✅ Created sample data: {len(sample_data)} days")
    
    # Process data
    processor = VolumeConfirmationProcessor()
    analysis_data = processor.process_volume_confirmation_data(sample_data)
    
    if 'error' in analysis_data:
        print(f"❌ Data processing failed: {analysis_data['error']}")
        return False
    
    print("✅ Data processing completed")
    
    # Format context
    context_formatter = VolumeConfirmationContextFormatter()
    
    # Test full context
    full_context = context_formatter.format_context(
        analysis_data, "TESTSTOCK", "Test Company", "Technology"
    )
    
    print(f"✅ Full context generated: {len(full_context)} characters")
    
    # Test focused context
    focused_context = context_formatter.format_focused_context(analysis_data, "TESTSTOCK")
    
    print(f"✅ Focused context generated: {len(focused_context)} characters")
    
    # Save contexts for inspection
    with open("test_volume_confirmation_context_full.txt", "w") as f:
        f.write("FULL VOLUME CONFIRMATION CONTEXT\n")
        f.write("=" * 80 + "\n\n")
        f.write(full_context)
    
    with open("test_volume_confirmation_context_focused.txt", "w") as f:
        f.write("FOCUSED VOLUME CONFIRMATION CONTEXT\n")
        f.write("=" * 80 + "\n\n")
        f.write(focused_context)
    
    print("💾 Context files saved:")
    print("   - test_volume_confirmation_context_full.txt")
    print("   - test_volume_confirmation_context_focused.txt")
    
    # Display sample of context
    print(f"\n📄 Sample Context (first 500 chars):")
    print(full_context[:500] + "..." if len(full_context) > 500 else full_context)
    
    return True

# Alias for backwards compatibility
VolumeConfirmationContext = VolumeConfirmationContextFormatter

if __name__ == "__main__":
    test_volume_confirmation_context()