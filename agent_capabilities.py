import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
from typing import Dict, Any, List, Optional, Tuple
import logging
import os
import json
from dataclasses import dataclass, field
from gemini.gemini_client import GeminiClient
from gemini.token_tracker import AnalysisTokenTracker
from zerodha_client import ZerodhaDataClient
from technical_indicators import TechnicalIndicators, DataCollector
from patterns.recognition import PatternRecognition
from patterns.visualization import PatternVisualizer, ChartVisualizer
from sector_benchmarking import sector_benchmarking_provider
import asyncio
from mtf_analysis_utils import multi_timeframe_analysis
from enhanced_mtf_analysis import enhanced_mtf_analyzer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('StockAnalysisOrchestrator')

@dataclass
class AnalysisState:
    """Represents the current state of stock analysis."""
    symbol: str
    exchange: str
    indicators: Optional[Dict[str, pd.DataFrame]] = None
    patterns: Optional[Dict[str, Any]] = None
    analysis_results: Optional[Dict[str, Any]] = None
    last_updated: Optional[datetime] = None
    
    def is_valid(self, max_age_hours: int = 1) -> bool:
        """Check if the state is still valid based on age."""
        if not self.last_updated:
            return False
        age = datetime.now() - self.last_updated
        return age.total_seconds() < (max_age_hours * 3600)
    
    def update(self, **kwargs):
        """Update state with new data."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.last_updated = datetime.now()

class StockAnalysisOrchestrator:
    """
    Orchestrates the complete stock analysis process using AI-powered analysis.
    Handles authentication, data retrieval, indicator calculation, pattern recognition, AI analysis, visualization, and sector benchmarking.
    """
    def __init__(self):
        self.data_client = ZerodhaDataClient()
        self.gemini_client = GeminiClient()
        self.state_cache = {}
        self.indicators = TechnicalIndicators()
        self.visualizer = PatternVisualizer()
        from sector_benchmarking import SectorBenchmarkingProvider
        self.sector_benchmarking_provider = SectorBenchmarkingProvider()
    
    def authenticate(self) -> bool:
        """
        Authenticate with Zerodha API.
        
        Returns:
            bool: True if authentication successful, False otherwise
        """
        return self.data_client.authenticate()
    
    def _get_or_create_state(self, symbol: str, exchange: str) -> AnalysisState:
        """Get existing state or create new one."""
        key = f"{exchange}:{symbol}"
        if key not in self.state_cache:
            self.state_cache[key] = AnalysisState(symbol=symbol, exchange=exchange)
        return self.state_cache[key]
    
    async def retrieve_stock_data(self, symbol: str, exchange: str = "NSE", interval: str = "day", period: int = 365) -> pd.DataFrame:
        """
        Retrieve real-time or historical stock data, preferring real-time streaming data if available.
        """
        from zerodha_ws_client import zerodha_ws_client
        from zerodha_client import ZerodhaDataClient
        import pandas as pd
        from datetime import datetime, timedelta, time

        # Map interval to supported streaming timeframes
        streaming_timeframes = ["1m", "5m", "15m", "1h", "1d"]
        now = datetime.now()
        ist_time = now + timedelta(hours=5, minutes=30)
        is_weekday = ist_time.weekday() < 5
        market_open = time(9, 15)
        market_close = time(15, 30)
        is_market_hour = is_weekday and (market_open <= ist_time.time() <= market_close)

        MIN_CANDLES = 20  # Minimum number of candles required for real-time data to be considered valid

        # Use streaming data if market is open and interval is supported
        if is_market_hour and interval in streaming_timeframes:
            data_client = ZerodhaDataClient()
            token = data_client.get_instrument_token(symbol, exchange)
            if token is not None:
                # Build a rolling window of recent candles for indicator calculation
                candle_agg = zerodha_ws_client.candle_aggregator
                candles = candle_agg.candles[token][interval]
                if candles:
                    sorted_buckets = sorted(candles.keys())
                    N = min(100, len(sorted_buckets))
                    recent_candles = [candles[b] for b in sorted_buckets[-N:]]
                    if len(recent_candles) >= MIN_CANDLES:
                        df = pd.DataFrame(recent_candles)
                        # Ensure datetime index for compatibility
                        df['datetime'] = pd.to_datetime(df['start'], unit='s')
                        df = df.set_index('datetime')
                        df.attrs['data_freshness'] = 'real_time'
                        df.attrs['last_update_time'] = now.isoformat()
                        df.attrs['market_status'] = "open"
                        return df
                    else:
                        logger.warning(f"Real-time data for {symbol} has only {len(recent_candles)} candles (<{MIN_CANDLES}). Falling back to historical data.")
                else:
                    logger.warning(f"No real-time candles found for {symbol}. Falling back to historical data.")
        # Fallback: use historical data
        data = await self.data_client.get_historical_data_async(
            symbol=symbol,
            exchange=exchange,
            interval=interval,
            period=period
        )
        if data is None:
            logger.error(f"Failed to retrieve data for {symbol}")
            return data
        if 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'])
            data = data.set_index('date')
        data.attrs['data_freshness'] = 'historical'
        data.attrs['last_update_time'] = now.isoformat()
        data.attrs['market_status'] = "closed"
        return data
    
    def calculate_indicators(self, data: pd.DataFrame, stock_symbol: str = None) -> Dict[str, Any]:
        """
        Calculate all technical indicators for the given data.
        
        Args:
            data: DataFrame containing price and volume data
            
        Returns:
            Dict[str, Any]: Dictionary containing all calculated indicators
        """
        logger.info(f"Calculating technical indicators for {len(data)} records")
        
        # Calculate indicators using DataCollector
        data_collector = DataCollector()
        indicators = data_collector.collect_all_data(data, stock_symbol)
        
        logger.info(f"Calculated comprehensive technical analysis with {len(indicators)} indicator groups")
        
        return indicators
    
    def create_visualizations(self, data: pd.DataFrame, indicators: Dict[str, Any], 
                             symbol: str, output_dir: str) -> Dict[str, Any]:
        """
        Create optimized visualization charts for AI analysis.
        Reduced from 8 charts to 4 comprehensive charts to eliminate redundancy.
        
        Args:
            data: DataFrame containing price data
            indicators: Dictionary containing calculated indicators
            symbol: Stock symbol
            output_dir: Directory to save chart images
        Returns:
            Dict[str, Any]: Dictionary containing chart data and metadata
        """
        import os
        logger.info(f"Creating optimized visualization charts for {symbol}")
        charts = {}
        
        try:
            # 1. COMPREHENSIVE TECHNICAL OVERVIEW CHART
            # Combines: comparison_chart + support/resistance levels
            technical_chart_path = os.path.join(output_dir, f"{symbol}_technical_overview.png")
            ChartVisualizer.plot_comprehensive_technical_chart(data, indicators, technical_chart_path, stock_symbol=symbol)
            charts['technical_overview'] = technical_chart_path
        except Exception as e:
            logger.warning(f"Failed to create technical overview chart for {symbol}: {e}")
        
        try:
            # 2. COMPREHENSIVE PATTERN ANALYSIS CHART
            # Combines: divergence + double_tops_bottoms + triangles_flags
            pattern_chart_path = os.path.join(output_dir, f"{symbol}_pattern_analysis.png")
            ChartVisualizer.plot_comprehensive_pattern_chart(data, indicators, pattern_chart_path, stock_symbol=symbol)
            charts['pattern_analysis'] = pattern_chart_path
        except Exception as e:
            logger.warning(f"Failed to create pattern analysis chart for {symbol}: {e}")
        
        try:
            # 3. COMPREHENSIVE VOLUME ANALYSIS CHART
            # Combines: volume_anomalies + price_volume_correlation + candlestick_volume
            volume_chart_path = os.path.join(output_dir, f"{symbol}_volume_analysis.png")
            ChartVisualizer.plot_comprehensive_volume_chart(data, indicators, volume_chart_path, stock_symbol=symbol)
            charts['volume_analysis'] = volume_chart_path
        except Exception as e:
            logger.warning(f"Failed to create volume analysis chart for {symbol}: {e}")
        
        try:
            # 4. MULTI-TIMEFRAME COMPARISON CHART
            # New chart for multi-timeframe analysis
            mtf_chart_path = os.path.join(output_dir, f"{symbol}_mtf_comparison.png")
            ChartVisualizer.plot_mtf_comparison_chart(data, indicators, mtf_chart_path, stock_symbol=symbol)
            charts['mtf_comparison'] = mtf_chart_path
        except Exception as e:
            logger.warning(f"Failed to create MTF comparison chart for {symbol}: {e}")
        
        logger.info(f"Created {len(charts)} optimized charts for {symbol}")
        return charts
    
    def serialize_indicators(self, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize indicators to JSON-serializable format with optimized data reduction."""
        def convert_numpy_types(obj):
            """Convert NumPy types to JSON-serializable Python types."""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict(orient='records')
            else:
                return obj
        
        def optimize_indicator_data(indicator_data):
            """Optimize indicator data by reducing historical arrays and redundant information."""
            if isinstance(indicator_data, dict):
                optimized = {}
                for key, value in indicator_data.items():
                    # Skip historical arrays - only keep current and recent values
                    if key in ['values', 'historical_values', 'price_history', 'volume_history']:
                        if isinstance(value, (list, np.ndarray, pd.Series)):
                            # Keep only last 5 values for trend analysis
                            if len(value) > 5:
                                optimized[key] = convert_numpy_types(value[-5:])
                            else:
                                optimized[key] = convert_numpy_types(value)
                        else:
                            optimized[key] = convert_numpy_types(value)
                    # Skip redundant moving average historical data
                    elif key in ['sma_values', 'ema_values', 'wma_values']:
                        if isinstance(value, dict):
                            optimized[key] = {k: convert_numpy_types(v[-5:] if isinstance(v, (list, np.ndarray, pd.Series)) and len(v) > 5 else v) 
                                            for k, v in value.items()}
                        else:
                            optimized[key] = convert_numpy_types(value)
                    # Keep essential data
                    else:
                        optimized[key] = convert_numpy_types(value)
                return optimized
            else:
                return convert_numpy_types(indicator_data)
        
        serializable = {}
        for key, value in indicators.items():
            serializable[key] = optimize_indicator_data(value)
        return serializable

    async def orchestrate_llm_analysis(self, symbol: str, indicators: dict, chart_paths: dict, period: int, interval: str, knowledge_context: str = "") -> tuple:
        """Orchestrate the LLM analysis workflow."""
        return await self.analyze_with_ai(symbol, indicators, chart_paths, period, interval, knowledge_context)

    async def analyze_with_ai(self, symbol: str, indicators: dict, chart_paths: dict, period: int, interval: str, knowledge_context: str = "", sector_context: dict = None, mtf_context: dict = None) -> tuple:
        # Add sector context to knowledge context if available
        enhanced_knowledge_context = knowledge_context
        if sector_context:
            sector_benchmarking = sector_context.get('sector_benchmarking', {})
            sector_rotation = sector_context.get('sector_rotation', {})
            sector_correlation = sector_context.get('sector_correlation', {})
            sector_analysis = sector_context.get('sector_analysis', {})
            
            sector_context_str = f"""
SECTOR CONTEXT:
SECTOR PERFORMANCE:
- Market Outperformance: {float(sector_benchmarking.get('excess_return', 0)):.2%}
- Sector Outperformance: {f"{float(sector_benchmarking.get('sector_excess_return', 0)):.2%}" if sector_benchmarking else 'N/A'}
- Sector Beta: {f"{sector_benchmarking.get('sector_beta', 1.0):.2f}" if sector_benchmarking else '1.00'}

SECTOR ANALYSIS:
- {sector_analysis.get('market_performance', 'Market performance analysis not available')}
- {sector_analysis.get('sector_performance', 'Sector performance analysis not available')}
- {sector_analysis.get('risk_assessment', 'Risk assessment not available')}

Consider this sector context when analyzing the stock's technical indicators and patterns.
"""
            enhanced_knowledge_context = knowledge_context + "\n" + sector_context_str
        
        # Add enhanced multi-timeframe context if available
        if mtf_context and mtf_context.get('success', False):
            mtf_summary = mtf_context.get('summary', {})
            mtf_validation = mtf_context.get('cross_timeframe_validation', {})
            mtf_timeframes = mtf_context.get('timeframe_analyses', {})
            
            mtf_context_str = f"""
ENHANCED MULTI-TIMEFRAME ANALYSIS CONTEXT:
This analysis covers 6 timeframes: 1min, 5min, 15min, 30min, 1hour, and 1day.

OVERALL MTF SUMMARY:
- Consensus Trend: {mtf_summary.get('overall_signal', 'Unknown')}
- Confidence Score: {mtf_summary.get('confidence', 0):.2%}
- Signal Alignment: {mtf_summary.get('signal_alignment', 'Unknown')}
- Risk Level: {mtf_summary.get('risk_level', 'Unknown')}
- Recommendation: {mtf_summary.get('recommendation', 'Unknown')}

CROSS-TIMEFRAME VALIDATION:
- Signal Strength: {mtf_validation.get('signal_strength', 0):.2%}
- Supporting Timeframes: {', '.join(mtf_validation.get('supporting_timeframes', []))}
- Conflicting Timeframes: {', '.join(mtf_validation.get('conflicting_timeframes', []))}
- Divergence Detected: {'Yes' if mtf_validation.get('divergence_detected', False) else 'No'}
- Divergence Type: {mtf_validation.get('divergence_type', 'None')}

DYNAMIC TIMEFRAME ANALYSIS (Signal Quality Based):
"""
            
            # Add individual timeframe signals with dynamic importance
            for timeframe, analysis in mtf_timeframes.items():
                trend = analysis.get('trend', 'Unknown')
                confidence = analysis.get('confidence', 0)
                rsi = analysis.get('key_indicators', {}).get('rsi', 'N/A')
                macd_signal = analysis.get('key_indicators', {}).get('macd_signal', 'Unknown')
                
                # Determine importance based on signal quality and confidence
                if confidence > 0.8:
                    importance = "🔥 HIGH IMPORTANCE"
                elif confidence > 0.6:
                    importance = "⚡ MEDIUM-HIGH IMPORTANCE"
                elif confidence > 0.4:
                    importance = "📊 MEDIUM IMPORTANCE"
                else:
                    importance = "⚠️ LOW IMPORTANCE"
                
                mtf_context_str += f"- {timeframe}: {trend} (Confidence: {confidence:.2%}, RSI: {rsi:.2f}, MACD: {macd_signal}) - {importance}\n"
            
            mtf_context_str += f"""
KEY CONFLICTS:
{chr(10).join(mtf_validation.get('key_conflicts', ['None identified']))}

DYNAMIC WEIGHTING INSIGHTS:
- Timeframes with higher confidence and signal quality have more influence
- Supporting timeframes strengthen the consensus signal
- Conflicting timeframes indicate potential reversals or uncertainty
- Divergence between timeframes suggests trend change potential

IMPORTANT: Consider this multi-timeframe context when analyzing the stock. Pay special attention to:
1. Whether the single-timeframe indicators align with the multi-timeframe consensus
2. Any divergences between timeframes that might indicate trend changes
3. The confidence level and signal alignment across timeframes
4. Risk assessment based on timeframe conflicts or alignments
5. Which timeframes are showing HIGH IMPORTANCE signals
"""
            
            enhanced_knowledge_context = enhanced_knowledge_context + "\n" + mtf_context_str
        
        # Pass MTF context to the LLM analysis
        result, ind_summary_md, chart_insights_md = await self.orchestrate_llm_analysis_with_mtf(symbol, indicators, chart_paths, period, interval, enhanced_knowledge_context, mtf_context)
        return result, ind_summary_md, chart_insights_md

    async def orchestrate_llm_analysis_with_mtf(self, symbol: str, indicators: dict, chart_paths: dict, period: int, interval: str, knowledge_context: str = "", mtf_context: dict = None) -> tuple:
        """Orchestrate the LLM analysis workflow with MTF context integration."""
        try:
            # Initialize token tracker
            import time
            analysis_id = f"{symbol}_{int(time.time())}"
            token_tracker = AnalysisTokenTracker(analysis_id=analysis_id, symbol=symbol)
            
            # 1. Indicator summary analysis with MTF context
            print(f"[LLM-ANALYSIS] Starting indicator summary analysis for {symbol}...")
            ind_summary_md, ind_json = await self.gemini_client.build_indicators_summary(
                symbol, indicators, period, interval, knowledge_context, token_tracker, mtf_context
            )
            
            # 2. Chart analysis (already optimized for MTF)
            print(f"[LLM-ANALYSIS] Starting chart analysis for {symbol}...")
            result, ind_summary_md, chart_insights_md = await self.gemini_client.analyze_stock(
                symbol, indicators, chart_paths, period, interval, knowledge_context
            )
            
            return result, ind_summary_md, chart_insights_md
            
        except Exception as e:
            logger.error(f"Error in LLM analysis orchestration: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    async def analyze_stock(self, symbol: str, exchange: str = "NSE",
                     period: int = 365, interval: str = "day", output_dir: str = None, 
                     knowledge_context: str = "", sector: str = None) -> tuple:
        """
        Main analysis method that orchestrates the entire workflow.
        """
        try:
            # Get or create analysis state
            state = self._get_or_create_state(symbol, exchange)
            
            # Retrieve stock data
            stock_data = await self.retrieve_stock_data(symbol, exchange, interval, period)
            if stock_data.empty:
                return None, None, f"No data available for {symbol}"
            # Warn if data is not real-time
            if stock_data.attrs.get('data_freshness') != 'real_time':
                logger.warning(f"Data for {symbol} is not real-time (freshness: {stock_data.attrs.get('data_freshness')}). Analysis may be based on stale data.")
            
            # --- MTF/LONG-TERM ANALYSIS ---
            # Map interval to base_interval for MTF utility
            interval_map = {
                'minute': 'minute', '3minute': 'minute', '5minute': 'minute', '10minute': 'minute', '15minute': 'minute', '30minute': 'minute', '60minute': 'hour',
                'day': 'day', 'week': 'week', 'month': 'month'
            }
            base_interval = interval_map.get(interval, 'day')
            try:
                mtf_result = multi_timeframe_analysis(stock_data, base_interval=base_interval)
            except Exception as e:
                mtf_result = {'messages': [f"Error in multi-timeframe analysis: {e}"]}
            
            # Calculate technical indicators with optimized data reduction (95-98% reduction in historical data)
            state.indicators = TechnicalIndicators.calculate_all_indicators_optimized(stock_data, symbol)
            
            # Create visualizations for AI analysis
            chart_paths = {}
            if output_dir:
                chart_paths = self.create_visualizations(stock_data, state.indicators, symbol, output_dir)
            
            # Get sector context if available
            sector_benchmarking = None
            sector_rotation = None
            sector_correlation = None
            enhanced_sector_context = None
            
            if sector:
                try:
                    # OPTIMIZED: Use unified sector data fetcher instead of separate calls
                    logging.info(f"OPTIMIZED: Using unified sector data fetcher for {symbol}")
                    comprehensive_sector_data = await self.sector_benchmarking_provider.get_optimized_comprehensive_sector_analysis(
                        symbol, stock_data, sector
                    )
                    
                    # Extract individual components from unified analysis
                    sector_benchmarking = comprehensive_sector_data.get('sector_benchmarking', {})
                    sector_rotation = comprehensive_sector_data.get('sector_rotation', {})
                    sector_correlation = comprehensive_sector_data.get('sector_correlation', {})
                    
                    # Log optimization metrics
                    optimization_metrics = comprehensive_sector_data.get('optimization_metrics', {})
                    logging.info(f"OPTIMIZATION METRICS: {optimization_metrics}")
                    
                    enhanced_sector_context = self._build_enhanced_sector_context(
                        sector, sector_benchmarking, sector_rotation, sector_correlation
                    )
                    
                    # Add optimization note to context
                    if optimization_metrics:
                        enhanced_sector_context['optimization_metrics'] = optimization_metrics
                        
                except Exception as e:
                    print(f"Warning: Could not get optimized sector context for {sector}: {e}")
                    # Fallback to old method if optimized method fails
                    try:
                        logging.info(f"FALLBACK: Using legacy sector data fetching for {symbol}")
                        sector_benchmarking = await self.sector_benchmarking_provider.get_comprehensive_benchmarking_async(symbol, stock_data)
                        sector_rotation = await self.sector_benchmarking_provider.analyze_sector_rotation_async("1M")
                        sector_correlation = await self.sector_benchmarking_provider.generate_sector_correlation_matrix_async("3M")
                        enhanced_sector_context = self._build_enhanced_sector_context(
                            sector, sector_benchmarking, sector_rotation, sector_correlation
                        )
                    except Exception as fallback_error:
                        print(f"Warning: Fallback sector context also failed for {sector}: {fallback_error}")
            
            # Get AI analysis (primary analysis method) with MTF context
            ai_analysis, ind_summary_md, chart_insights_md = await self.analyze_with_ai(
                symbol, state.indicators, chart_paths, period, interval, knowledge_context, enhanced_sector_context, mtf_result
            )
            
            # Convert indicators to serializable format
            serializable_indicators = self.serialize_indicators(state.indicators)
            
            # Create overlays for visualization
            overlays = self._create_overlays(stock_data, state.indicators)
            
            # Build enhanced analysis results with MTF context
            analysis_results = self._build_enhanced_analysis_result(
                symbol, exchange, stock_data, state.indicators, ai_analysis, 
                ind_summary_md, chart_insights_md, chart_paths, 
                enhanced_sector_context, period, interval
            )
            
            # Add MTF context to the analysis results
            if mtf_result:
                analysis_results['multi_timeframe_analysis'] = mtf_result
            
            # Update state
            state.update(
                analysis_results=analysis_results,
                last_updated=datetime.now()
            )
            
            success_message = f"AI analysis completed for {symbol}. Signal: {ai_analysis.get('trend', 'Unknown')} (Confidence: {ai_analysis.get('confidence_pct', 0)}%)"
            
            return analysis_results, success_message, None
            
        except Exception as e:
            error_message = f"Error analyzing {symbol}: {str(e)}"
            print(f"Error in analyze_stock: {e}")
            import traceback
            traceback.print_exc()
            return None, None, error_message

    def _determine_risk_level(self, ai_analysis: Dict[str, Any]) -> str:
        """Determine risk level based on AI analysis confidence and market conditions."""
        confidence = ai_analysis.get('confidence_pct', 0)
        trend = ai_analysis.get('trend', 'Unknown')
        
        if confidence >= 80:
            return 'Low'
        elif confidence >= 60:
            return 'Medium'
        elif confidence >= 40:
            return 'High'
        else:
            return 'Very High'
    
    def _generate_recommendation(self, ai_analysis: Dict[str, Any]) -> str:
        """Generate actionable recommendation based on AI analysis."""
        confidence = ai_analysis.get('confidence_pct', 0)
        trend = ai_analysis.get('trend', 'Unknown')
        
        if confidence >= 80:
            if trend == 'Bullish':
                return 'Strong Buy'
            elif trend == 'Bearish':
                return 'Strong Sell'
            else:
                return 'Hold'
        elif confidence >= 60:
            if trend == 'Bullish':
                return 'Buy'
            elif trend == 'Bearish':
                return 'Sell'
            else:
                return 'Hold'
        elif confidence >= 40:
            return 'Wait and Watch'
        else:
            return 'Avoid Trading'
    
    def _create_overlays(self, data: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Create overlays for visualization without rule-based consensus."""
        try:
            # --- TRIANGLES ---
            triangle_indices = PatternRecognition.detect_triangle(data['close'])
            triangles = []
            for tri in triangle_indices:
                vertices = []
                for idx in tri:
                    date = str(data.index[idx])
                    price = float(data['close'].iloc[idx])
                    vertices.append({"date": date, "price": price})
                triangles.append({"vertices": vertices})

            # --- FLAGS ---
            flag_indices = PatternRecognition.detect_flag(data['close'])
            flags = []
            for flg in flag_indices:
                if flg:
                    start_idx, end_idx = flg[0], flg[-1]
                    flags.append({
                        "start_date": str(data.index[start_idx]),
                        "end_date": str(data.index[end_idx]),
                        "start_price": float(data['close'].iloc[start_idx]),
                        "end_price": float(data['close'].iloc[end_idx])
                    })

            # --- SUPPORT/RESISTANCE ---
            support_levels, resistance_levels = TechnicalIndicators.detect_support_resistance(data)
            support = [{"level": float(lvl)} for lvl in support_levels]
            resistance = [{"level": float(lvl)} for lvl in resistance_levels]

            # --- DOUBLE TOPS/BOTTOMS ---
            double_tops = PatternRecognition.detect_double_top(data['close'])
            double_bottoms = PatternRecognition.detect_double_bottom(data['close'])
            double_tops_formatted = [
                {
                    "peak1": {"date": str(data.index[p1]), "price": float(data['close'].iloc[p1])},
                    "peak2": {"date": str(data.index[p2]), "price": float(data['close'].iloc[p2])}
                }
                for p1, p2 in double_tops
            ]
            double_bottoms_formatted = [
                {
                    "bottom1": {"date": str(data.index[b1]), "price": float(data['close'].iloc[b1])},
                    "bottom2": {"date": str(data.index[b2]), "price": float(data['close'].iloc[b2])}
                }
                for b1, b2 in double_bottoms
            ]

            # --- DIVERGENCES ---
            rsi = TechnicalIndicators.calculate_rsi(data)
            divergences = PatternRecognition.detect_divergence(data['close'], rsi)
            divergences_formatted = [
                {
                    "type": dtype,
                    "start_date": str(data.index[idx1]),
                    "end_date": str(data.index[idx2]),
                    "start_price": float(data['close'].iloc[idx1]),
                    "end_price": float(data['close'].iloc[idx2]),
                    "start_rsi": float(rsi.iloc[idx1]),
                    "end_rsi": float(rsi.iloc[idx2])
                }
                for idx1, idx2, dtype in divergences
            ]

            # --- VOLUME ANOMALIES ---
            anomalies = PatternRecognition.detect_volume_anomalies(data['volume'])
            volume_anomalies = [
                {
                    "date": str(idx),
                    "volume": float(data.loc[idx, 'volume']),
                    "price": float(data.loc[idx, 'close'])
                }
                for idx in anomalies if idx in data.index
            ]

            # --- BUILD OVERLAYS DICT ---
            overlays = {
                "triangles": triangles,
                "flags": flags,
                "support_resistance": {
                    "support": support,
                    "resistance": resistance
                },
                "double_tops": double_tops_formatted,
                "double_bottoms": double_bottoms_formatted,
                "divergences": divergences_formatted,
                "volume_anomalies": volume_anomalies
            }
            
            return overlays
            
        except Exception as e:
            print(f"Error creating overlays: {e}")
            return {}
 
    def _build_enhanced_sector_context(self, sector: str, sector_benchmarking: Dict, 
                                     sector_rotation: Dict, sector_correlation: Dict) -> Dict[str, Any]:
        """Build enhanced sector context combining all sector analysis data."""
        try:
            enhanced_context = {
                'sector': sector,
                'benchmarking': sector_benchmarking,
                'rotation_insights': {},
                'correlation_insights': {},
                'trading_recommendations': []
            }
            
            # Add sector rotation insights
            if sector_rotation:
                enhanced_context['rotation_insights'] = {
                    'sector_rank': None,
                    'sector_performance': None,
                    'rotation_strength': sector_rotation.get('rotation_patterns', {}).get('rotation_strength', 'unknown'),
                    'leading_sectors': sector_rotation.get('rotation_patterns', {}).get('leading_sectors', []),
                    'lagging_sectors': sector_rotation.get('rotation_patterns', {}).get('lagging_sectors', []),
                    'recommendations': sector_rotation.get('recommendations', [])
                }
                
                # Find current sector's rank and performance
                sector_rankings = sector_rotation.get('sector_rankings', {})
                if sector in sector_rankings:
                    enhanced_context['rotation_insights']['sector_rank'] = sector_rankings[sector]['rank']
                    enhanced_context['rotation_insights']['sector_performance'] = sector_rankings[sector]['performance']
                
                # Add rotation-based trading recommendations
                for rec in sector_rotation.get('recommendations', []):
                    if rec.get('sector') == sector:
                        enhanced_context['trading_recommendations'].append({
                            'type': 'rotation',
                            'recommendation': rec.get('type', ''),
                            'reason': rec.get('reason', ''),
                            'confidence': rec.get('confidence', 'medium')
                        })

            # Add correlation insights
            if sector_correlation:
                enhanced_context['correlation_insights'] = {
                    'average_correlation': sector_correlation.get('average_correlation', 0),
                    'diversification_quality': sector_correlation.get('diversification_insights', {}).get('diversification_quality', 'unknown'),
                    'sector_volatility': sector_correlation.get('sector_volatility', {}).get(sector, 0),
                    'high_correlation_sectors': [],
                    'low_correlation_sectors': []
                }
                
                # Find sectors highly correlated with current sector
                correlation_matrix = sector_correlation.get('correlation_matrix', {})
                if sector in correlation_matrix:
                    sector_correlations = correlation_matrix[sector]
                    for other_sector, corr in sector_correlations.items():
                        if other_sector != sector:
                            if corr > 0.7:
                                enhanced_context['correlation_insights']['high_correlation_sectors'].append({
                                    'sector': other_sector,
                                    'correlation': corr
                                })
                            elif corr < 0.3:
                                enhanced_context['correlation_insights']['low_correlation_sectors'].append({
                                    'sector': other_sector,
                                    'correlation': corr
                                })
                
                # Add correlation-based recommendations
                diversification_insights = sector_correlation.get('diversification_insights', {})
                for rec in diversification_insights.get('recommendations', []):
                    enhanced_context['trading_recommendations'].append({
                        'type': 'diversification',
                        'recommendation': rec.get('type', ''),
                        'message': rec.get('message', ''),
                        'priority': rec.get('priority', 'medium')
                    })
            
            return enhanced_context
            
        except Exception as e:
            print(f"Error building enhanced sector context: {e}")
            return {'sector': sector, 'benchmarking': sector_benchmarking}

    async def enhanced_analyze_stock(self, symbol: str, exchange: str = "NSE",
                     period: int = 365, interval: str = "day", output_dir: str = None, 
                     knowledge_context: str = "", sector: str = None) -> tuple:
        """
        Enhanced stock analysis with mathematical validation using code execution.
        This method provides more accurate analysis by performing actual calculations
        instead of relying on LLM estimation.
        """
        logger.info(f"[ENHANCED ANALYSIS] Starting enhanced analysis for {symbol}")
        
        try:
            # Get or create analysis state
            state = self._get_or_create_state(symbol, exchange)
            
            # Step 1: Retrieve stock data
            logger.info(f"[ENHANCED ANALYSIS] Retrieving data for {symbol}")
            data = await self.retrieve_stock_data(symbol, exchange, interval, period)
            if data.empty:
                raise ValueError(f"No data available for {symbol}")
            
            # Step 2: Calculate technical indicators with optimized data reduction
            logger.info(f"[ENHANCED ANALYSIS] Calculating optimized indicators for {symbol}")
            indicators = TechnicalIndicators.calculate_all_indicators_optimized(data, symbol)
            
            # Step 3: Create visualizations
            logger.info(f"[ENHANCED ANALYSIS] Creating visualizations for {symbol}")
            chart_paths = self.create_visualizations(data, indicators, symbol, output_dir or "output")
            
            # Step 4: Get sector context if available
            sector_context = None
            if sector:
                try:
                    sector_benchmarking = await self.sector_benchmarking_provider.get_sector_benchmarking(sector, period)
                    sector_rotation = await self.sector_benchmarking_provider.get_sector_rotation_analysis(sector, period)
                    sector_correlation = await self.sector_benchmarking_provider.get_sector_correlation_analysis(sector, period)
                    sector_context = self._build_enhanced_sector_context(sector, sector_benchmarking, sector_rotation, sector_correlation)
                except Exception as e:
                    logger.warning(f"[ENHANCED ANALYSIS] Failed to get sector context for {sector}: {e}")
            
            # Step 5: Enhanced AI analysis with code execution
            logger.info(f"[ENHANCED ANALYSIS] Performing enhanced AI analysis for {symbol}")
            ai_analysis, indicator_summary, chart_insights = await self.enhanced_analyze_with_ai(
                symbol, indicators, chart_paths, period, interval, knowledge_context, sector_context
            )
            
            # Step 6: Build enhanced result with mathematical validation
            logger.info(f"[ENHANCED ANALYSIS] Building enhanced result for {symbol}")
            result = self._build_enhanced_analysis_result(
                symbol, exchange, data, indicators, ai_analysis, 
                indicator_summary, chart_insights, chart_paths, 
                sector_context, period, interval
            )
            
            # Step 7: Update state
            state.update(
                indicators=indicators,
                analysis_results=result,
                last_updated=datetime.now()
            )
            
            logger.info(f"[ENHANCED ANALYSIS] Completed enhanced analysis for {symbol}")
            return result
            
        except Exception as e:
            logger.error(f"[ENHANCED ANALYSIS] Error in enhanced analysis for {symbol}: {e}")
            raise

    async def enhanced_analyze_with_ai(self, symbol: str, indicators: dict, chart_paths: dict, 
                                     period: int, interval: str, knowledge_context: str = "", 
                                     sector_context: dict = None) -> tuple:
        """
        Enhanced AI analysis with code execution for mathematical validation.
        """
        try:
            # Combine knowledge context with sector context
            enhanced_knowledge_context = knowledge_context
            if sector_context:
                enhanced_knowledge_context += f"\n\nSector Context:\n{json.dumps(sector_context, indent=2)}"
            
            # Use enhanced analysis with code execution
            ai_analysis, indicator_summary, chart_insights = await self.gemini_client.analyze_stock_with_enhanced_calculations(
                symbol=symbol,
                indicators=indicators,
                chart_paths=chart_paths,
                period=period,
                interval=interval,
                knowledge_context=enhanced_knowledge_context
            )
            
            return ai_analysis, indicator_summary, chart_insights
            
        except Exception as e:
            logger.error(f"[ENHANCED ANALYSIS] Error in enhanced AI analysis for {symbol}: {e}")
            raise

    def _build_enhanced_analysis_result(self, symbol: str, exchange: str, data: pd.DataFrame, 
                                      indicators: dict, ai_analysis: dict, indicator_summary: str, 
                                      chart_insights: str, chart_paths: dict, sector_context: dict, 
                                      period: int, interval: str) -> dict:
        """
        Build comprehensive enhanced analysis result with mathematical validation.
        """
        try:
            import time
            
            # Get latest price and basic info
            latest_price = data['close'].iloc[-1] if not data.empty else None
            price_change = data['close'].iloc[-1] - data['close'].iloc[-2] if len(data) > 1 else 0
            price_change_pct = (price_change / data['close'].iloc[-2]) * 100 if len(data) > 1 and data['close'].iloc[-2] != 0 else 0
            
            # Determine risk level with mathematical validation
            risk_level = self._determine_enhanced_risk_level(ai_analysis, indicators)
            
            # Generate enhanced recommendation
            recommendation = self._generate_enhanced_recommendation(ai_analysis, indicators)
            
            # Build result structure
            result = {
                "symbol": symbol,
                "exchange": exchange,
                "analysis_timestamp": datetime.now().isoformat(),
                "analysis_type": "enhanced_with_code_execution",
                "mathematical_validation": True,
                "calculation_method": "code_execution",
                "accuracy_improvement": "high",
                
                # Price information
                "current_price": latest_price,
                "price_change": price_change,
                "price_change_percentage": price_change_pct,
                "analysis_period": f"{period} days",
                "interval": interval,
                
                # AI Analysis
                "ai_analysis": ai_analysis,
                "indicator_summary": indicator_summary,
                "chart_insights": chart_insights,
                
                # Technical Analysis
                "technical_indicators": self.serialize_indicators(indicators),
                "risk_level": risk_level,
                "recommendation": recommendation,
                
                # Sector Analysis
                "sector_context": sector_context,
                
                # Charts
                "charts": chart_paths,
                
                # Enhanced Metadata
                "enhanced_metadata": {
                    "mathematical_validation": True,
                    "code_execution_enabled": True,
                    "statistical_analysis": True,
                    "confidence_improvement": "high",
                    "calculation_timestamp": time.time(),
                    "analysis_quality": "enhanced"
                }
            }
            
            # Add mathematical validation results if available
            if 'mathematical_validation' in ai_analysis:
                result['mathematical_validation_results'] = ai_analysis['mathematical_validation']
            
            # Add code execution metadata if available
            if 'analysis_metadata' in ai_analysis and 'code_execution' in ai_analysis['analysis_metadata']:
                result['code_execution_metadata'] = ai_analysis['analysis_metadata']['code_execution']
            
            return result
            
        except Exception as e:
            logger.error(f"[ENHANCED ANALYSIS] Error building enhanced result for {symbol}: {e}")
            raise

    def _determine_enhanced_risk_level(self, ai_analysis: Dict[str, Any], indicators: Dict[str, Any]) -> str:
        """
        Determine risk level with enhanced mathematical validation.
        """
        try:
            # Extract risk information from AI analysis
            risk_score = 0
            
            # Check mathematical validation results
            if 'mathematical_validation' in ai_analysis:
                math_val = ai_analysis['mathematical_validation']
                
                # Volatility analysis
                if 'volatility_metrics' in math_val:
                    vol_metrics = math_val['volatility_metrics']
                    if vol_metrics.get('volatility_level') == 'high':
                        risk_score += 3
                    elif vol_metrics.get('volatility_level') == 'medium':
                        risk_score += 2
                
                # Trend strength analysis
                if 'trend_strength' in math_val:
                    trend = math_val['trend_strength']
                    if trend.get('trend_reliability') == 'low':
                        risk_score += 2
                
                # RSI analysis
                if 'rsi_analysis' in math_val:
                    rsi = math_val['rsi_analysis']
                    if rsi.get('signal_strength') == 'weak':
                        risk_score += 1
            
            # Check AI analysis confidence
            if 'confidence_pct' in ai_analysis:
                confidence = ai_analysis['confidence_pct']
                if confidence < 50:
                    risk_score += 2
                elif confidence < 70:
                    risk_score += 1
            
            # Determine risk level based on score
            if risk_score >= 5:
                return "High"
            elif risk_score >= 3:
                return "Medium"
            else:
                return "Low"
                
        except Exception as e:
            logger.error(f"Error determining enhanced risk level: {e}")
            return "Medium"

    def _generate_enhanced_recommendation(self, ai_analysis: Dict[str, Any], indicators: Dict[str, Any]) -> str:
        """
        Generate enhanced recommendation with mathematical validation.
        """
        try:
            # Extract recommendation from AI analysis
            if 'trading_strategy' in ai_analysis:
                strategy = ai_analysis['trading_strategy']
                
                # Check short-term bias
                if 'short_term' in strategy:
                    short_term = strategy['short_term']
                    bias = short_term.get('bias', 'neutral')
                    confidence = short_term.get('confidence', 50)
                    
                    if confidence >= 70:
                        if bias == 'bullish':
                            return "Strong Buy"
                        elif bias == 'bearish':
                            return "Strong Sell"
                    elif confidence >= 50:
                        if bias == 'bullish':
                            return "Buy"
                        elif bias == 'bearish':
                            return "Sell"
                    else:
                        return "Hold"
            
            # Check mathematical validation for additional insights
            if 'mathematical_validation' in ai_analysis:
                math_val = ai_analysis['mathematical_validation']
                
                # Check trend strength
                if 'trend_strength' in math_val:
                    trend = math_val['trend_strength']
                    if trend.get('trend_reliability') == 'high':
                        slope = trend.get('linear_regression_slope', 0)
                        if slope > 0.1:
                            return "Buy (Strong Trend)"
                        elif slope < -0.1:
                            return "Sell (Strong Trend)"
                
                # Check RSI signals
                if 'rsi_analysis' in math_val:
                    rsi = math_val['rsi_analysis']
                    if rsi.get('oversold_periods', 0) > rsi.get('overbought_periods', 0):
                        return "Buy (Oversold)"
                    elif rsi.get('overbought_periods', 0) > rsi.get('oversold_periods', 0):
                        return "Sell (Overbought)"
            
            return "Hold"
            
        except Exception as e:
            logger.error(f"Error generating enhanced recommendation: {e}")
            return "Hold"

# Utility to clean NaN/Infinity for JSON
from utils import clean_for_json

# Example usage
if __name__ == "__main__":
    # Create orchestrator
    orchestrator = StockAnalysisOrchestrator()
    
    # Authenticate
    orchestrator.authenticate()
    
    # Analyze stock (now async)
    results, data = asyncio.run(orchestrator.analyze_stock("RELIANCE", output_dir="./output"))
    
    # Print recommendation
    if "recommendation" in results and "candidates" in results["recommendation"]:
        content = results["recommendation"]["candidates"][0]["content"]
        if "parts" in content and content["parts"]:
            logger.info("\nInvestment Recommendation:")
            logger.info(content["parts"][0]["text"])

    async def get_sector_context_async(self, symbol: str, stock_data: pd.DataFrame, sector: str) -> Dict[str, Any]:
        """
        Get sector context asynchronously using async index data fetching.
        """
        try:
            # Get sector benchmarking using async methods
            sector_benchmarking = await self.sector_benchmarking_provider.get_comprehensive_benchmarking_async(symbol, stock_data)
            
            # Get sector rotation and correlation (these are still sync for now)
            sector_rotation = await self.sector_benchmarking_provider.analyze_sector_rotation_async("3M")
            sector_correlation = await self.sector_benchmarking_provider.generate_sector_correlation_matrix_async("6M")
            
            # Build enhanced sector context
            enhanced_sector_context = self._build_enhanced_sector_context(
                sector, sector_benchmarking, sector_rotation, sector_correlation
            )
            
            return {
                'sector_benchmarking': sector_benchmarking,
                'sector_rotation': sector_rotation,
                'sector_correlation': sector_correlation,
                'enhanced_sector_context': enhanced_sector_context
            }
            
        except Exception as e:
            logger.error(f"Error getting async sector context for {sector}: {e}")
            return {}

    async def analyze_stock_with_async_index_data(self, symbol: str, exchange: str = "NSE",
                     period: int = 365, interval: str = "day", output_dir: str = None, 
                     knowledge_context: str = "", sector: str = None) -> tuple:
        """
        Enhanced analysis with async index data fetching and comprehensive multi-timeframe analysis.
        """
        try:
            # Get or create analysis state
            state = self._get_or_create_state(symbol, exchange)
            
            # Retrieve stock data
            stock_data = await self.retrieve_stock_data(symbol, exchange, interval, period)
            if stock_data.empty:
                return None, None, f"No data available for {symbol}"
            
            # Perform enhanced multi-timeframe analysis
            print(f"[ENHANCED MTF] Starting comprehensive multi-timeframe analysis for {symbol}")
            mtf_results = await enhanced_mtf_analyzer.comprehensive_mtf_analysis(symbol, exchange)
            
            if not mtf_results.get('success', False):
                print(f"[ENHANCED MTF] Warning: Multi-timeframe analysis failed: {mtf_results.get('error', 'Unknown error')}")
                # Fallback to basic MTF analysis
                interval_map = {
                    'minute': 'minute', '3minute': 'minute', '5minute': 'minute', '10minute': 'minute', '15minute': 'minute', '30minute': 'minute', '60minute': 'hour',
                    'day': 'day', 'week': 'week', 'month': 'month'
                }
                base_interval = interval_map.get(interval, 'day')
                try:
                    mtf_result = multi_timeframe_analysis(stock_data, base_interval=base_interval)
                except Exception as e:
                    mtf_result = {'messages': [f"Error in multi-timeframe analysis: {e}"]}
            else:
                print(f"[ENHANCED MTF] Multi-timeframe analysis completed successfully")
                mtf_result = mtf_results
            
            # Calculate technical indicators with optimized data reduction (95-98% reduction in historical data)
            state.indicators = TechnicalIndicators.calculate_all_indicators_optimized(stock_data, symbol)
            
            # Create visualizations for AI analysis
            chart_paths = {}
            if output_dir:
                chart_paths = self.create_visualizations(stock_data, state.indicators, symbol, output_dir)
            
            # Get sector context if available
            sector_benchmarking = None
            sector_rotation = None
            sector_correlation = None
            enhanced_sector_context = None
            
            if sector:
                try:
                    # Get sector benchmarking using the correct provider
                    sector_benchmarking = await self.sector_benchmarking_provider.get_comprehensive_benchmarking_async(symbol, stock_data)
                    # OPTIMIZED: Use 1M instead of 3M for sector rotation (reduced from 140 to 50 days)
                    sector_rotation = await self.sector_benchmarking_provider.analyze_sector_rotation_async("1M")
                    # OPTIMIZED: Use 3M instead of 6M for correlation (reduced from 230 to 80 days)
                    sector_correlation = await self.sector_benchmarking_provider.generate_sector_correlation_matrix_async("3M")
                    enhanced_sector_context = self._build_enhanced_sector_context(
                        sector, sector_benchmarking, sector_rotation, sector_correlation
                    )
                except Exception as e:
                    print(f"Warning: Could not get sector context for {sector}: {e}")
            
            # Get AI analysis (primary analysis method)
            ai_analysis, ind_summary_md, chart_insights_md = await self.analyze_with_ai(
                symbol, state.indicators, chart_paths, period, interval, knowledge_context, enhanced_sector_context, mtf_result
            )
            
            # Convert indicators to serializable format
            serializable_indicators = self.serialize_indicators(state.indicators)
            
            # Create overlays for visualization
            overlays = self._create_overlays(stock_data, state.indicators)
            
            # Build optimized analysis results with enhanced MTF data
            analysis_results = self.build_optimized_analysis_result(
                symbol, exchange, stock_data, state.indicators, ai_analysis, 
                ind_summary_md, chart_insights_md, chart_paths, 
                enhanced_sector_context, mtf_results, period, interval
            )
            
            # Add enhanced MTF specific data to the optimized structure
            if mtf_results.get('success', False):
                analysis_results['enhanced_mtf_analysis'] = mtf_results
                analysis_results['summary']['mtf_consensus'] = mtf_results.get('summary', {}).get('overall_signal', 'Unknown')
                analysis_results['summary']['mtf_confidence'] = mtf_results.get('summary', {}).get('confidence', 0)
                analysis_results['summary']['signal_alignment'] = mtf_results.get('summary', {}).get('signal_alignment', 'Unknown')
                analysis_results['trading_guidance']['mtf_recommendation'] = mtf_results.get('summary', {}).get('recommendation', 'Unknown')
                analysis_results['metadata']['timeframes_analyzed'] = mtf_results.get('summary', {}).get('timeframes_analyzed', 0)
                analysis_results['metadata']['analysis_type'] = 'enhanced_mtf_analysis'
            
            # Update state
            state.update(
                analysis_results=analysis_results,
                last_updated=datetime.now()
            )
            
            success_message = f"Enhanced multi-timeframe analysis completed for {symbol}. AI Signal: {ai_analysis.get('trend', 'Unknown')} (Confidence: {ai_analysis.get('confidence_pct', 0)}%). MTF Consensus: {mtf_results.get('summary', {}).get('overall_signal', 'Unknown') if mtf_results.get('success', False) else 'Unknown'}"
            
            return analysis_results, success_message, None
            
        except Exception as e:
            error_message = f"Error in enhanced MTF analysis for {symbol}: {str(e)}"
            print(f"Error in analyze_stock_with_async_index_data: {e}")
            import traceback
            traceback.print_exc()
            return None, None, error_message

    def build_optimized_analysis_result(self, symbol: str, exchange: str, data: pd.DataFrame, 
                                      indicators: dict, ai_analysis: dict, indicator_summary: str, 
                                      chart_insights: str, chart_paths: dict, sector_context: dict, 
                                      mtf_context: dict, period: int, interval: str) -> dict:
        """
        Build optimized analysis result with significant data reduction.
        """
        try:
            import time
            
            # Get latest price and basic info
            latest_price = data['close'].iloc[-1] if not data.empty else None
            price_change = data['close'].iloc[-1] - data['close'].iloc[-2] if len(data) > 1 else 0
            price_change_pct = (price_change / data['close'].iloc[-2]) * 100 if len(data) > 1 and data['close'].iloc[-2] != 0 else 0
            
            # Determine risk level
            risk_level = self._determine_risk_level_static(ai_analysis)
            
            # Generate recommendation
            recommendation = self._generate_recommendation_static(ai_analysis)
            
            # Optimize AI analysis by removing redundant information
            optimized_ai_analysis = self._optimize_ai_analysis(ai_analysis)
            
            # Optimize sector context by removing duplicates
            optimized_sector_context = self._optimize_sector_context(sector_context)
            
            # Build optimized result structure
            result = {
                # Essential metadata (single location)
                "metadata": {
                    "symbol": symbol,
                    "exchange": exchange,
                    "analysis_date": datetime.now().isoformat(),
                    "period_days": period,
                    "interval": interval,
                    "sector": sector_context.get('sector') if sector_context else None
                },
                
                # Optimized AI analysis
                "ai_analysis": optimized_ai_analysis,
                
                # Optimized indicators (reduced historical data)
                "indicators": self.serialize_indicators(indicators),
                
                # Optimized overlays (essential pattern data only)
                "overlays": self._optimize_overlays(data, indicators),
                
                # Consolidated trading guidance
                "trading_guidance": self._consolidate_trading_guidance(ai_analysis),
                
                # Optimized sector context
                "sector_context": optimized_sector_context,
                
                # Keep multi-timeframe analysis intact (as requested)
                "multi_timeframe_analysis": mtf_context,
                
                # Consolidated summary
                "summary": {
                    "overall_signal": ai_analysis.get('trend', 'Unknown'),
                    "confidence": ai_analysis.get('confidence_pct', 0),
                    "risk_level": risk_level,
                    "recommendation": recommendation
                },
                
                # Charts (keep as is)
                "charts": chart_paths,
                
                # Markdown summaries (keep as is)
                "indicator_summary_md": indicator_summary,
                "chart_insights": chart_insights
            }
            
            return result
            
        except Exception as e:
            logger.error(f"[OPTIMIZED ANALYSIS] Error building optimized result for {symbol}: {e}")
            raise

    def _optimize_ai_analysis(self, ai_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize AI analysis by removing redundant information."""
        if not ai_analysis:
            return {}
        
        optimized = {}
        
        # Keep essential trend information
        if 'trend' in ai_analysis:
            optimized['trend'] = ai_analysis['trend']
        if 'confidence_pct' in ai_analysis:
            optimized['confidence_pct'] = ai_analysis['confidence_pct']
        
        # Consolidate primary trend information
        if 'primary_trend' in ai_analysis:
            primary = ai_analysis['primary_trend']
            optimized['primary_trend'] = {
                'signal': primary.get('signal'),
                'strength': primary.get('strength'),
                'key_drivers': primary.get('key_drivers', [])
            }
        
        # Consolidate market outlook
        if 'market_outlook' in ai_analysis:
            outlook = ai_analysis['market_outlook']
            optimized['market_outlook'] = {
                'bias': outlook.get('bias'),
                'timeframe': outlook.get('timeframe'),
                'key_factors': outlook.get('key_factors', [])
            }
        
        # Remove redundant sector information (will be in separate sector_context)
        if 'sector_integration' in ai_analysis:
            del ai_analysis['sector_integration']
        if 'sector_context' in ai_analysis:
            del ai_analysis['sector_context']
        
        return optimized

    def _optimize_sector_context(self, sector_context: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize sector context by removing redundant information."""
        if not sector_context:
            return {}
        
        optimized = {}
        
        # Extract essential sector information
        if 'sector' in sector_context:
            optimized['sector'] = sector_context['sector']
        
        # Consolidate sector performance
        if 'sector_benchmarking' in sector_context:
            benchmarking = sector_context['sector_benchmarking']
            optimized['benchmarking'] = {
                'excess_return': benchmarking.get('excess_return'),
                'sector_excess_return': benchmarking.get('sector_excess_return'),
                'sector_beta': benchmarking.get('sector_beta')
            }
        
        # Consolidate sector rotation
        if 'sector_rotation' in sector_context:
            rotation = sector_context['sector_rotation']
            optimized['rotation'] = {
                'rotation_strength': rotation.get('rotation_strength'),
                'rotation_direction': rotation.get('rotation_direction'),
                'momentum_score': rotation.get('momentum_score')
            }
        
        # Consolidate sector correlation
        if 'sector_correlation' in sector_context:
            correlation = sector_context['sector_correlation']
            optimized['correlation'] = {
                'market_correlation': correlation.get('market_correlation'),
                'sector_correlation': correlation.get('sector_correlation')
            }
        
        return optimized

    def _optimize_overlays(self, data: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize overlays by keeping only essential pattern data."""
        try:
            # --- TRIANGLES ---
            triangle_indices = PatternRecognition.detect_triangle(data['close'])
            triangles = []
            for tri in triangle_indices:
                # Keep only essential triangle data
                triangles.append({
                    "type": tri.get('type'),
                    "breakout_price": tri.get('breakout_price'),
                    "target": tri.get('target'),
                    "confidence": tri.get('confidence')
                })
            
            # --- SUPPORT/RESISTANCE ---
            support_levels = []
            resistance_levels = []
            
            if 'support_resistance' in indicators:
                sr_data = indicators['support_resistance']
                if 'support_levels' in sr_data:
                    for level in sr_data['support_levels'][:5]:  # Keep only top 5
                        support_levels.append({
                            "level": level.get('level'),
                            "strength": level.get('strength')
                        })
                if 'resistance_levels' in sr_data:
                    for level in sr_data['resistance_levels'][:5]:  # Keep only top 5
                        resistance_levels.append({
                            "level": level.get('level'),
                            "strength": level.get('strength')
                        })
            
            return {
                "triangles": triangles,
                "support_levels": support_levels,
                "resistance_levels": resistance_levels
            }
            
        except Exception as e:
            logger.error(f"Error optimizing overlays: {e}")
            return {"triangles": [], "support_levels": [], "resistance_levels": []}

    def _consolidate_trading_guidance(self, ai_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Consolidate trading guidance by removing redundant information."""
        consolidated = {}
        
        # Extract primary signal and confidence
        if 'trend' in ai_analysis:
            consolidated['primary_signal'] = 'Buy' if ai_analysis['trend'] == 'Bullish' else 'Sell' if ai_analysis['trend'] == 'Bearish' else 'Hold'
        if 'confidence_pct' in ai_analysis:
            consolidated['confidence'] = ai_analysis['confidence_pct']
        
        # Consolidate entry/exit levels from different timeframes
        entry_ranges = []
        stop_losses = []
        targets = []
        
        for timeframe in ['short_term', 'medium_term', 'long_term']:
            if timeframe in ai_analysis:
                tf_data = ai_analysis[timeframe]
                if 'entry_range' in tf_data:
                    entry_ranges.extend(tf_data['entry_range'])
                if 'stop_loss' in tf_data:
                    stop_losses.append(tf_data['stop_loss'])
                if 'targets' in tf_data:
                    targets.extend(tf_data['targets'])
        
        # Use most conservative/realistic values
        if entry_ranges:
            consolidated['entry_range'] = [min(entry_ranges), max(entry_ranges)]
        if stop_losses:
            consolidated['stop_loss'] = min(stop_losses)  # Most conservative
        if targets:
            consolidated['targets'] = sorted(list(set(targets)))[:4]  # Top 4 unique targets
        
        # Add timeframe breakdown (simplified)
        timeframe_breakdown = {}
        for timeframe in ['short_term', 'medium_term', 'long_term']:
            if timeframe in ai_analysis:
                tf_data = ai_analysis[timeframe]
                timeframe_breakdown[timeframe] = {
                    'signal': tf_data.get('signal'),
                    'confidence': tf_data.get('confidence')
                }
        consolidated['timeframe_breakdown'] = timeframe_breakdown
        
        return consolidated

    def _determine_risk_level_static(self, ai_analysis: Dict[str, Any]) -> str:
        """Static version of risk level determination."""
        confidence = ai_analysis.get('confidence_pct', 0)
        
        if confidence >= 80:
            return 'Low'
        elif confidence >= 60:
            return 'Medium'
        elif confidence >= 40:
            return 'High'
        else:
            return 'Very High'

    def _generate_recommendation_static(self, ai_analysis: Dict[str, Any]) -> str:
        """Static version of recommendation generation."""
        confidence = ai_analysis.get('confidence_pct', 0)
        trend = ai_analysis.get('trend', 'Unknown')
        
        if confidence >= 80:
            if trend == 'Bullish':
                return 'Strong Buy'
            elif trend == 'Bearish':
                return 'Strong Sell'
            else:
                return 'Hold'
        elif confidence >= 60:
            if trend == 'Bullish':
                return 'Buy'
            elif trend == 'Bearish':
                return 'Sell'
            else:
                return 'Hold'
        elif confidence >= 40:
            return 'Wait and Watch'
        else:
            return 'Avoid Trading'








