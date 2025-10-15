#!/usr/bin/env python3
"""
Resilient Market Structure Chart Generator

This module adds comprehensive error handling and resilience features:
- Fallback chart generation if complex visualization fails
- Graceful degradation for missing data (volume, incomplete price data)
- Retry mechanisms for chart encoding failures
- Validation of chart output before LLM integration
- Comprehensive error recovery and logging
- Data quality checks and automatic fixes
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional, Union
import os
import logging
from pathlib import Path
import json
import traceback
from dataclasses import dataclass
from enum import Enum
import time
from functools import wraps

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChartGenerationLevel(Enum):
    """Chart generation complexity levels for graceful degradation"""
    FULL = "full"              # All enhanced features
    STANDARD = "standard"      # Basic + some enhancements  
    MINIMAL = "minimal"        # Only essential elements
    EMERGENCY = "emergency"    # Bare minimum chart

@dataclass
class ChartValidationResult:
    """Result of chart validation"""
    is_valid: bool
    file_size: int
    issues: List[str]
    warnings: List[str]
    quality_score: float  # 0-100

@dataclass 
class DataQualityReport:
    """Data quality assessment result"""
    is_acceptable: bool
    completeness_score: float  # 0-100
    issues: List[str]
    fixes_applied: List[str]
    original_length: int
    processed_length: int

def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """Decorator for retrying failed operations"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying in {delay}s...")
                        time.sleep(delay * (attempt + 1))  # Exponential backoff
                    else:
                        logger.error(f"All {max_retries} attempts failed for {func.__name__}")
            raise last_exception
        return wrapper
    return decorator

class ResilientMarketStructureCharts:
    """
    Resilient chart generator with comprehensive error handling and graceful degradation.
    """
    
    def __init__(self, output_dir: str = "resilient_charts"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for different chart types
        (self.output_dir / "full").mkdir(exist_ok=True)
        (self.output_dir / "fallback").mkdir(exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)
        
        # Enhanced color scheme
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
            'fibonacci': '#ffa500',
            'trend_channel': '#9370db',
            'break_line': '#ff6347',
            'phase_highlight': '#ffd700',
            'price_label': '#000000'
        }
        
        self.fib_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
        self.generation_stats = {
            'attempts': 0,
            'successes': 0,
            'fallbacks': 0,
            'failures': 0
        }
        
        plt.style.use('seaborn-v0_8-darkgrid')
    
    def create_resilient_chart(self, 
                             stock_data: pd.DataFrame, 
                             analysis_data: Dict[str, Any], 
                             symbol: str, 
                             scenario: str) -> Tuple[Optional[str], ChartValidationResult]:
        """
        Create resilient chart with fallback mechanisms and validation.
        
        Returns:
            Tuple of (chart_path, validation_result)
        """
        self.generation_stats['attempts'] += 1
        
        # Step 1: Validate and clean input data
        try:
            data_quality = self._assess_data_quality(stock_data, analysis_data)
            if not data_quality.is_acceptable:
                logger.error(f"Data quality too poor for {symbol}: {data_quality.issues}")
                return None, ChartValidationResult(False, 0, data_quality.issues, [], 0)
                
            # Apply data fixes if needed
            stock_data, analysis_data = self._apply_data_fixes(stock_data, analysis_data, data_quality)
            
        except Exception as e:
            logger.error(f"Data validation failed for {symbol}: {e}")
            return None, ChartValidationResult(False, 0, [f"Data validation error: {e}"], [], 0)
        
        # Step 2: Try different chart generation levels
        chart_levels = [
            ChartGenerationLevel.FULL,
            ChartGenerationLevel.STANDARD,
            ChartGenerationLevel.MINIMAL,
            ChartGenerationLevel.EMERGENCY
        ]
        
        last_error = None
        for level in chart_levels:
            try:
                logger.info(f"Attempting {level.value} chart generation for {symbol}")
                
                chart_path = self._generate_chart_at_level(
                    stock_data, analysis_data, symbol, scenario, level
                )
                
                if chart_path:
                    # Validate generated chart
                    validation = self._validate_chart(chart_path)
                    
                    if validation.is_valid:
                        self.generation_stats['successes'] += 1
                        if level != ChartGenerationLevel.FULL:
                            self.generation_stats['fallbacks'] += 1
                        
                        logger.info(f"Successfully generated {level.value} chart for {symbol}")
                        return chart_path, validation
                    else:
                        logger.warning(f"Chart validation failed for {symbol}: {validation.issues}")
                        # Try next level
                        continue
                        
            except Exception as e:
                last_error = e
                logger.warning(f"{level.value} generation failed for {symbol}: {e}")
                continue
        
        # All levels failed
        self.generation_stats['failures'] += 1
        logger.error(f"All chart generation levels failed for {symbol}. Last error: {last_error}")
        
        return None, ChartValidationResult(
            False, 0, 
            [f"All generation levels failed. Last error: {str(last_error)}"], 
            [], 0
        )
    
    def _assess_data_quality(self, stock_data: pd.DataFrame, analysis_data: Dict[str, Any]) -> DataQualityReport:
        """Assess quality of input data and identify issues"""
        
        issues = []
        fixes_applied = []
        original_length = len(stock_data) if stock_data is not None else 0
        
        # Check stock data
        if stock_data is None or stock_data.empty:
            issues.append("Stock data is empty or None")
            return DataQualityReport(False, 0, issues, fixes_applied, 0, 0)
        
        # Check required columns
        required_cols = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in stock_data.columns]
        if missing_cols:
            issues.append(f"Missing required columns: {missing_cols}")
        
        # Check for sufficient data length
        if len(stock_data) < 10:
            issues.append(f"Insufficient data length: {len(stock_data)} rows (minimum 10)")
        
        # Check for data consistency
        if not stock_data['high'].ge(stock_data['low']).all():
            issues.append("High prices below low prices detected")
        
        if not stock_data['high'].ge(stock_data['close']).all():
            issues.append("High prices below close prices detected")
            
        if not stock_data['low'].le(stock_data['close']).all():
            issues.append("Low prices above close prices detected")
        
        # Check for missing values
        null_counts = stock_data[required_cols].isnull().sum()
        if null_counts.sum() > 0:
            issues.append(f"Null values detected: {null_counts.to_dict()}")
        
        # Check analysis data
        if not analysis_data or not isinstance(analysis_data, dict):
            issues.append("Analysis data is empty or invalid")
        else:
            # Check swing points
            swing_points = analysis_data.get('swing_points', {})
            if not swing_points.get('swing_highs') and not swing_points.get('swing_lows'):
                issues.append("No swing points found in analysis data")
        
        # Calculate completeness score
        completeness_score = 100.0
        if missing_cols:
            completeness_score -= len(missing_cols) * 20
        if null_counts.sum() > 0:
            completeness_score -= min(null_counts.sum() / len(stock_data) * 100, 30)
        if len(stock_data) < 50:  # Prefer longer datasets
            completeness_score -= (50 - len(stock_data)) * 0.5
        
        completeness_score = max(0, completeness_score)
        
        # Determine if data is acceptable
        is_acceptable = len(issues) == 0 or (
            completeness_score >= 60 and 
            len(stock_data) >= 10 and 
            not missing_cols
        )
        
        return DataQualityReport(
            is_acceptable=is_acceptable,
            completeness_score=completeness_score,
            issues=issues,
            fixes_applied=fixes_applied,
            original_length=original_length,
            processed_length=len(stock_data)
        )
    
    def _apply_data_fixes(self, stock_data: pd.DataFrame, analysis_data: Dict[str, Any], 
                         quality_report: DataQualityReport) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Apply automatic fixes to data issues"""
        
        fixes_applied = []
        
        # Create copy to avoid modifying original
        stock_data = stock_data.copy()
        
        try:
            # Fix 1: Forward fill small gaps in data
            required_cols = ['open', 'high', 'low', 'close']
            for col in required_cols:
                if col in stock_data.columns:
                    null_count = stock_data[col].isnull().sum()
                    if 0 < null_count <= 5:  # Only fix small gaps
                        stock_data[col] = stock_data[col].fillna(method='ffill')
                        fixes_applied.append(f"Forward filled {null_count} null values in {col}")
            
            # Fix 2: Correct impossible price relationships
            inconsistent_rows = ~stock_data['high'].ge(stock_data['low'])
            if inconsistent_rows.any():
                # Swap high and low where inconsistent
                stock_data.loc[inconsistent_rows, ['high', 'low']] = \
                    stock_data.loc[inconsistent_rows, ['low', 'high']].values
                fixes_applied.append(f"Fixed {inconsistent_rows.sum()} rows with high < low")
            
            # Fix 3: Ensure close prices are within high/low range
            close_too_high = stock_data['close'] > stock_data['high']
            if close_too_high.any():
                stock_data.loc[close_too_high, 'close'] = stock_data.loc[close_too_high, 'high']
                fixes_applied.append(f"Adjusted {close_too_high.sum()} close prices above high")
            
            close_too_low = stock_data['close'] < stock_data['low']
            if close_too_low.any():
                stock_data.loc[close_too_low, 'close'] = stock_data.loc[close_too_low, 'low']
                fixes_applied.append(f"Adjusted {close_too_low.sum()} close prices below low")
            
            # Fix 4: Handle volume data if present
            if 'volume' in stock_data.columns:
                # Replace negative volumes with 0
                negative_vol = stock_data['volume'] < 0
                if negative_vol.any():
                    stock_data.loc[negative_vol, 'volume'] = 0
                    fixes_applied.append(f"Fixed {negative_vol.sum()} negative volume values")
                
                # Fill null volumes with median
                null_vol = stock_data['volume'].isnull()
                if null_vol.any():
                    median_vol = stock_data['volume'].median()
                    stock_data.loc[null_vol, 'volume'] = median_vol
                    fixes_applied.append(f"Filled {null_vol.sum()} null volumes with median")
            
            # Fix 5: Ensure analysis data has minimum structure
            if 'swing_points' not in analysis_data:
                analysis_data['swing_points'] = {'swing_highs': [], 'swing_lows': [], 'total_swings': 0}
                fixes_applied.append("Added missing swing_points structure")
            
            if 'bos_choch_analysis' not in analysis_data:
                analysis_data['bos_choch_analysis'] = {
                    'bos_events': [], 
                    'choch_events': [], 
                    'structural_bias': 'neutral'
                }
                fixes_applied.append("Added missing BOS/CHOCH structure")
            
            quality_report.fixes_applied.extend(fixes_applied)
            
            if fixes_applied:
                logger.info(f"Applied {len(fixes_applied)} data fixes: {fixes_applied}")
            
        except Exception as e:
            logger.warning(f"Data fixes failed: {e}")
            # Return original data if fixes fail
            pass
        
        return stock_data, analysis_data
    
    @retry_on_failure(max_retries=2, delay=0.5)
    def _generate_chart_at_level(self, stock_data: pd.DataFrame, analysis_data: Dict[str, Any], 
                               symbol: str, scenario: str, level: ChartGenerationLevel) -> Optional[str]:
        """Generate chart at specified complexity level"""
        
        try:
            if level == ChartGenerationLevel.FULL:
                return self._generate_full_chart(stock_data, analysis_data, symbol, scenario)
            elif level == ChartGenerationLevel.STANDARD:
                return self._generate_standard_chart(stock_data, analysis_data, symbol, scenario)
            elif level == ChartGenerationLevel.MINIMAL:
                return self._generate_minimal_chart(stock_data, analysis_data, symbol, scenario)
            else:  # EMERGENCY
                return self._generate_emergency_chart(stock_data, analysis_data, symbol, scenario)
                
        except Exception as e:
            logger.error(f"Chart generation at {level.value} failed: {e}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            raise
    
    def _generate_full_chart(self, stock_data: pd.DataFrame, analysis_data: Dict[str, Any], 
                           symbol: str, scenario: str) -> str:
        """Generate full-featured chart with all enhancements"""
        
        # Import and use enhanced generator
        from enhanced_chart_generator import EnhancedMarketStructureCharts
        
        enhanced_gen = EnhancedMarketStructureCharts(output_dir=str(self.output_dir / "full"))
        chart_path = enhanced_gen.create_enhanced_chart(stock_data, analysis_data, symbol, scenario)
        
        if not chart_path:
            raise Exception("Enhanced chart generation returned None")
        
        return chart_path
    
    def _generate_standard_chart(self, stock_data: pd.DataFrame, analysis_data: Dict[str, Any], 
                               symbol: str, scenario: str) -> str:
        """Generate standard chart with basic enhancements"""
        
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, height_ratios=[3, 1, 1], width_ratios=[3, 1], 
                            hspace=0.3, wspace=0.2)
        
        ax_main = fig.add_subplot(gs[0, :])
        ax_volume = fig.add_subplot(gs[1, :], sharex=ax_main)
        ax_summary = fig.add_subplot(gs[2, :])
        
        fig.suptitle(f'Standard Market Structure - {symbol} ({scenario})', 
                    fontsize=16, fontweight='bold')
        
        # Plot basic elements
        self._plot_basic_price_action(ax_main, stock_data, analysis_data)
        self._plot_basic_volume(ax_volume, stock_data)
        self._plot_basic_summary(ax_summary, analysis_data, scenario)
        
        # Format axes
        self._format_basic_axes(ax_main, ax_volume, stock_data)
        
        # Save chart
        filename = f"{symbol}_{scenario}_standard_structure.png"
        filepath = self.output_dir / "fallback" / filename
        
        plt.savefig(filepath, dpi=200, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close(fig)
        
        return str(filepath)
    
    def _generate_minimal_chart(self, stock_data: pd.DataFrame, analysis_data: Dict[str, Any], 
                              symbol: str, scenario: str) -> str:
        """Generate minimal chart with essential elements only"""
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        fig.suptitle(f'Minimal Market Structure - {symbol}', fontsize=14, fontweight='bold')
        
        # Plot only essential elements
        dates = pd.to_datetime(stock_data.index)
        ax.plot(dates, stock_data['close'], color=self.colors['price'], linewidth=2, label='Price')
        
        # Add swing points if available
        swing_points = analysis_data.get('swing_points', {})
        swing_highs = swing_points.get('swing_highs', [])
        swing_lows = swing_points.get('swing_lows', [])
        
        for swing in swing_highs:
            try:
                date = pd.to_datetime(swing['date'])
                price = swing['price']
                ax.scatter(date, price, color=self.colors['swing_high'], 
                         s=50, marker='^', alpha=0.7, zorder=5)
            except:
                continue
                
        for swing in swing_lows:
            try:
                date = pd.to_datetime(swing['date'])
                price = swing['price']
                ax.scatter(date, price, color=self.colors['swing_low'], 
                         s=50, marker='v', alpha=0.7, zorder=5)
            except:
                continue
        
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format axes
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:.2f}'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # Save chart
        filename = f"{symbol}_{scenario}_minimal_structure.png"
        filepath = self.output_dir / "fallback" / filename
        
        plt.savefig(filepath, dpi=150, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close(fig)
        
        return str(filepath)
    
    def _generate_emergency_chart(self, stock_data: pd.DataFrame, analysis_data: Dict[str, Any], 
                                symbol: str, scenario: str) -> str:
        """Generate emergency fallback chart - bare minimum"""
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        fig.suptitle(f'Emergency Chart - {symbol}', fontsize=12)
        
        # Plot only price line
        try:
            dates = pd.to_datetime(stock_data.index)
            ax.plot(dates, stock_data['close'], color='blue', linewidth=1)
            ax.set_ylabel('Price')
            ax.grid(True, alpha=0.5)
            
        except Exception as e:
            # Last resort - plot with simple index
            ax.plot(stock_data['close'].values, color='blue', linewidth=1)
            ax.set_ylabel('Price')
            ax.set_xlabel('Time Index')
            ax.grid(True, alpha=0.5)
        
        # Save chart
        filename = f"{symbol}_{scenario}_emergency.png"
        filepath = self.output_dir / "fallback" / filename
        
        plt.savefig(filepath, dpi=100, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        return str(filepath)
    
    def _plot_basic_price_action(self, ax, stock_data: pd.DataFrame, analysis_data: Dict):
        """Plot basic price action with swing points"""
        
        dates = pd.to_datetime(stock_data.index)
        
        # Plot price
        ax.plot(dates, stock_data['close'], color=self.colors['price'], 
               linewidth=2, label='Close Price')
        
        # Fill high/low
        ax.fill_between(dates, stock_data['low'], stock_data['high'], 
                       alpha=0.1, color=self.colors['price_fill'])
        
        # Plot swing points
        swing_points = analysis_data.get('swing_points', {})
        
        for swing in swing_points.get('swing_highs', []):
            try:
                ax.scatter(pd.to_datetime(swing['date']), swing['price'], 
                         c=self.colors['swing_high'], s=60, marker='^', alpha=0.8)
            except:
                continue
                
        for swing in swing_points.get('swing_lows', []):
            try:
                ax.scatter(pd.to_datetime(swing['date']), swing['price'], 
                         c=self.colors['swing_low'], s=60, marker='v', alpha=0.8)
            except:
                continue
        
        ax.set_title('Price Action & Market Structure', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_basic_volume(self, ax, stock_data: pd.DataFrame):
        """Plot basic volume"""
        
        if 'volume' not in stock_data.columns:
            ax.text(0.5, 0.5, 'Volume Data Not Available', 
                   transform=ax.transAxes, ha='center', va='center')
            return
        
        dates = pd.to_datetime(stock_data.index)
        volumes = stock_data['volume']
        
        ax.bar(dates, volumes, alpha=0.6, color='gray', width=0.8)
        ax.set_title('Volume', fontsize=12)
        ax.ticklabel_format(style='plain', axis='y')
    
    def _plot_basic_summary(self, ax, analysis_data: Dict, scenario: str):
        """Plot basic analysis summary"""
        
        ax.axis('off')
        
        # Extract basic metrics
        swing_points = analysis_data.get('swing_points', {})
        total_swings = swing_points.get('total_swings', 0)
        
        trend_analysis = analysis_data.get('trend_analysis', {})
        trend_direction = trend_analysis.get('trend_direction', 'unknown')
        
        bos_choch = analysis_data.get('bos_choch_analysis', {})
        bos_count = len(bos_choch.get('bos_events', []))
        structural_bias = bos_choch.get('structural_bias', 'neutral')
        
        summary_text = f"""
MARKET STRUCTURE SUMMARY - {scenario.upper()}

Trend Direction: {trend_direction.title()}
Total Swing Points: {total_swings}
BOS Events: {bos_count}
Structural Bias: {structural_bias.title()}
        """.strip()
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    def _format_basic_axes(self, ax_main, ax_volume, stock_data):
        """Basic axis formatting"""
        
        dates = pd.to_datetime(stock_data.index)
        
        # Format dates
        ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax_volume.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        
        # Format prices
        ax_main.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:.2f}'))
        
        # Rotate labels
        plt.setp(ax_main.xaxis.get_majorticklabels(), rotation=45)
        plt.setp(ax_volume.xaxis.get_majorticklabels(), rotation=45)
    
    @retry_on_failure(max_retries=3, delay=0.5)
    def _validate_chart(self, chart_path: str) -> ChartValidationResult:
        """Validate generated chart"""
        
        issues = []
        warnings = []
        quality_score = 100.0
        
        try:
            # Check if file exists
            if not os.path.exists(chart_path):
                issues.append("Chart file does not exist")
                return ChartValidationResult(False, 0, issues, warnings, 0)
            
            # Check file size
            file_size = os.path.getsize(chart_path)
            if file_size == 0:
                issues.append("Chart file is empty (0 bytes)")
            elif file_size < 1000:  # Less than 1KB is suspicious
                warnings.append(f"Chart file very small: {file_size} bytes")
                quality_score -= 10
            elif file_size > 10 * 1024 * 1024:  # Over 10MB is too large
                warnings.append(f"Chart file very large: {file_size / 1024 / 1024:.1f} MB")
                quality_score -= 5
            
            # Try to validate it's a proper image file (basic check)
            try:
                with open(chart_path, 'rb') as f:
                    header = f.read(8)
                    if not header.startswith(b'\x89PNG\r\n\x1a\n'):
                        issues.append("File does not appear to be a valid PNG image")
            except Exception as e:
                warnings.append(f"Could not verify PNG header: {e}")
                quality_score -= 5
            
            # Check filename format
            filename = Path(chart_path).name
            if not filename.endswith('.png'):
                warnings.append("Chart filename doesn't end with .png")
                quality_score -= 5
            
            # All validations passed if no issues
            is_valid = len(issues) == 0
            
            return ChartValidationResult(
                is_valid=is_valid,
                file_size=file_size,
                issues=issues,
                warnings=warnings,
                quality_score=max(0, quality_score)
            )
            
        except Exception as e:
            issues.append(f"Chart validation failed: {e}")
            return ChartValidationResult(False, 0, issues, warnings, 0)
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """Get chart generation statistics"""
        
        total = self.generation_stats['attempts']
        if total == 0:
            return self.generation_stats
        
        return {
            **self.generation_stats,
            'success_rate': self.generation_stats['successes'] / total * 100,
            'fallback_rate': self.generation_stats['fallbacks'] / total * 100,
            'failure_rate': self.generation_stats['failures'] / total * 100
        }
    
    def save_error_log(self, symbol: str, error_info: Dict[str, Any]):
        """Save detailed error information for debugging"""
        
        log_file = self.output_dir / "logs" / f"error_log_{symbol}_{int(time.time())}.json"
        
        try:
            with open(log_file, 'w') as f:
                json.dump(error_info, f, indent=2, default=str)
            logger.info(f"Error log saved: {log_file}")
        except Exception as e:
            logger.error(f"Failed to save error log: {e}")


# Test function for resilient chart generation
def test_resilient_charts():
    """Test resilient chart generation with various data quality scenarios"""
    
    logger.info("Testing Resilient Market Structure Charts...")
    
    # Create resilient chart generator
    resilient_gen = ResilientMarketStructureCharts(output_dir="resilient_charts_test")
    
    # Import mock data
    from test_chart_generation import create_mock_data_scenarios
    scenarios = create_mock_data_scenarios()
    
    # Test with original good data
    logger.info("\n" + "="*60)
    logger.info("Testing with GOOD DATA")
    logger.info("="*60)
    
    for stock_data, analysis_data, symbol, scenario in scenarios[:2]:  # Test 2 scenarios
        chart_path, validation = resilient_gen.create_resilient_chart(
            stock_data, analysis_data, symbol, scenario
        )
        
        if chart_path:
            logger.info(f"✅ Success: {chart_path}")
            logger.info(f"📊 Validation: Quality {validation.quality_score:.1f}, Size {validation.file_size:,} bytes")
        else:
            logger.error(f"❌ Failed: {validation.issues}")
    
    # Test with corrupted data
    logger.info("\n" + "="*60)
    logger.info("Testing with CORRUPTED DATA")
    logger.info("="*60)
    
    stock_data_bad, analysis_data_bad, symbol, scenario = scenarios[0]
    
    # Corrupt the data in various ways
    test_cases = [
        ("missing_volume", lambda df, ad: (df.drop('volume', axis=1, errors='ignore'), ad)),
        ("null_values", lambda df, ad: (df.mask(df.index % 10 == 0), ad)),  # Add nulls
        ("wrong_high_low", lambda df, ad: (df.assign(high=df['low'], low=df['high']), ad)),  # Swap high/low
        ("empty_swings", lambda df, ad: (df, {'swing_points': {'swing_highs': [], 'swing_lows': []}})),
        ("very_short_data", lambda df, ad: (df.head(5), ad)),  # Very short dataset
    ]
    
    for test_name, corruptor in test_cases:
        logger.info(f"\nTesting {test_name}...")
        
        try:
            corrupted_stock, corrupted_analysis = corruptor(stock_data_bad.copy(), analysis_data_bad.copy())
            
            chart_path, validation = resilient_gen.create_resilient_chart(
                corrupted_stock, corrupted_analysis, f"TEST_{test_name.upper()}", "corrupted_test"
            )
            
            if chart_path:
                logger.info(f"✅ Resilient success: {chart_path}")
                logger.info(f"📊 Quality: {validation.quality_score:.1f}")
                if validation.warnings:
                    logger.info(f"⚠️  Warnings: {validation.warnings}")
            else:
                logger.info(f"❌ Expected failure: {validation.issues}")
                
        except Exception as e:
            logger.error(f"💥 Unexpected error in {test_name}: {e}")
    
    # Print final statistics
    stats = resilient_gen.get_generation_stats()
    logger.info("\n" + "="*60)
    logger.info("RESILIENT CHART GENERATION STATISTICS")
    logger.info("="*60)
    logger.info(f"Total Attempts: {stats['attempts']}")
    logger.info(f"Successes: {stats['successes']}")
    logger.info(f"Fallbacks: {stats['fallbacks']}")
    logger.info(f"Failures: {stats['failures']}")
    if stats['attempts'] > 0:
        logger.info(f"Success Rate: {stats.get('success_rate', 0):.1f}%")
        logger.info(f"Fallback Rate: {stats.get('fallback_rate', 0):.1f}%")
    
    logger.info("\n✨ Resilient chart generation test completed!")


if __name__ == "__main__":
    test_resilient_charts()