import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
from typing import Dict, Any, List, Optional, Tuple
import logging
import os
import json
from dataclasses import dataclass, field
from gemini.gemini_client import GeminiClient
from zerodha_client import ZerodhaDataClient
from technical_indicators import (
    TechnicalIndicators, 
    IndicatorComparisonAnalyzer,
    DataCollector,
)
from patterns.recognition import PatternRecognition
from patterns.visualization import PatternVisualizer, ChartVisualizer
from sector_benchmarking import sector_benchmarking_provider

import asyncio

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
    consensus: Optional[Dict[str, Any]] = None
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
    Orchestrates the complete stock analysis process including data retrieval,
    technical analysis, pattern recognition, and AI-powered insights.
    """
    
    def __init__(self):
        self.data_client = ZerodhaDataClient()
        self.gemini_client = GeminiClient()
        self.analyzer = IndicatorComparisonAnalyzer()
        self.state_cache = {}
        
        # Initialize technical indicators
        self.indicators = TechnicalIndicators()
        self.visualizer = PatternVisualizer()
    
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
    
    def retrieve_stock_data(self, symbol: str, exchange: str = "NSE", 
                           interval: str = "day", period: int = 365) -> pd.DataFrame:
        """
        Retrieve historical stock data.
        
        Args:
            symbol: Stock symbol
            exchange: Exchange code
            interval: Candle interval
            period: Number of days of historical data
            
        Returns:
            pd.DataFrame: DataFrame containing historical data
        """
        # Calculate date range with proper market timing
        now = datetime.now()
        ist_time = now + timedelta(hours=5, minutes=30)  # Convert to IST
        
        # Check if market is currently open (9:15 AM to 3:30 PM IST, Monday to Friday)
        is_weekday = ist_time.weekday() < 5  # 0-4 are Monday to Friday
        is_market_hour = False
        if is_weekday:
            market_open = time(9, 15)  # 9:15 AM IST
            market_close = time(15, 30)  # 3:30 PM IST
            is_market_hour = market_open <= ist_time.time() <= market_close
        
        # Determine the end date for data retrieval
        if is_weekday and is_market_hour:
            # Market is open, include today's data
            to_date = now
            data_freshness = "real_time"
        else:
            # Market is closed, use last trading day
            to_date = now - timedelta(days=1)
            # Adjust for weekends
            while to_date.weekday() > 4:  # Saturday = 5, Sunday = 6
                to_date = to_date - timedelta(days=1)
            data_freshness = "last_close"

        from_date = to_date - timedelta(days=period)
        
        logger.info(f"Retrieving data for {symbol} from {from_date.strftime('%Y-%m-%d')} to {to_date.strftime('%Y-%m-%d')} (freshness: {data_freshness})")
        
        # Retrieve data
        data = self.data_client.get_historical_data(
            symbol=symbol,
            exchange=exchange,
            interval=interval,
            from_date=from_date,
            to_date=to_date,
            period=period
        )
        
        if data is None:
            logger.error(f"Failed to retrieve data for {symbol}")
            return data
        
        # Always prepare data here
        if 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'])
            data = data.set_index('date')
        
        # Add metadata about data freshness
        data.attrs['data_freshness'] = data_freshness
        data.attrs['last_update_time'] = to_date.isoformat()
        data.attrs['market_status'] = "open" if is_weekday and is_market_hour else "closed"
        
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
        Create all visualization charts and prepare them for AI analysis.
        Args:
            data: DataFrame containing price data
            indicators: Dictionary containing calculated indicators
            symbol: Stock symbol
            output_dir: Directory to save chart images
        Returns:
            Dict[str, Any]: Dictionary containing chart data and metadata
        """
        import os
        logger.info(f"Creating visualization charts for {symbol}")
        charts = {}
        # 1. Multi-panel Technical Analysis Comparison Chart
        comparison_chart_path = os.path.join(output_dir, f"{symbol}_comparison_chart.png")
        ChartVisualizer.plot_comparison_chart(data, indicators, comparison_chart_path, stock_symbol=symbol)
        charts['comparison_chart'] = comparison_chart_path
        # 2. Divergence Chart (RSI)
        divergence_chart_path = os.path.join(output_dir, f"{symbol}_divergence.png")
        rsi = TechnicalIndicators.calculate_rsi(data)
        divergences = PatternRecognition.detect_divergence(data['close'], rsi)
        ChartVisualizer.plot_divergence_chart(data['close'], rsi, divergences, divergence_chart_path, title=f"{symbol} Divergences")
        charts['divergence'] = divergence_chart_path
        # 3. Double Tops/Bottoms Chart
        double_tops_bottoms_chart_path = os.path.join(output_dir, f"{symbol}_double_tops_bottoms.png")
        double_tops = PatternRecognition.detect_double_top(data['close'])
        double_bottoms = PatternRecognition.detect_double_bottom(data['close'])
        ChartVisualizer.plot_double_tops_bottoms_chart(data['close'], double_tops, double_bottoms, double_tops_bottoms_chart_path, title=f"{symbol} Double Tops/Bottoms")
        charts['double_tops_bottoms'] = double_tops_bottoms_chart_path
        # 4. Support & Resistance Chart
        support_resistance_chart_path = os.path.join(output_dir, f"{symbol}_support_resistance.png")
        support, resistance = TechnicalIndicators.detect_support_resistance(data)
        ChartVisualizer.plot_support_resistance_chart(data['close'], support, resistance, support_resistance_chart_path, title=f"{symbol} Support & Resistance")
        charts['support_resistance'] = support_resistance_chart_path
        # 5. Triangles & Flags Chart
        triangles_flags_chart_path = os.path.join(output_dir, f"{symbol}_triangles_flags.png")
        triangles = PatternRecognition.detect_triangle(data['close'])
        flags = PatternRecognition.detect_flag(data['close'])
        ChartVisualizer.plot_triangles_flags_chart(data['close'], triangles, flags, triangles_flags_chart_path, title=f"{symbol} Triangles & Flags")
        charts['triangles_flags'] = triangles_flags_chart_path
        # 6. Volume Anomalies Chart
        volume_anomalies_chart_path = os.path.join(output_dir, f"{symbol}_volume_anomalies.png")
        anomalies = PatternRecognition.detect_volume_anomalies(data['volume'])
        ChartVisualizer.plot_volume_anomalies_chart(data['volume'], anomalies, volume_anomalies_chart_path, title=f"{symbol} Volume Anomalies")
        charts['volume_anomalies'] = volume_anomalies_chart_path
        
        # 7. Price-Volume Correlation Chart (NEW)
        price_volume_chart_path = os.path.join(output_dir, f"{symbol}_price_volume_correlation.png")
        ChartVisualizer.plot_price_volume_correlation(data, anomalies, price_volume_chart_path, title=f"{symbol} Price-Volume Correlation")
        charts['price_volume_correlation'] = price_volume_chart_path
        
        # 8. Candlestick with Volume Chart (NEW)
        candlestick_volume_chart_path = os.path.join(output_dir, f"{symbol}_candlestick_volume.png")
        ChartVisualizer.plot_candlestick_with_volume(data, anomalies, candlestick_volume_chart_path, title=f"{symbol} Price & Volume Analysis")
        charts['candlestick_volume'] = candlestick_volume_chart_path
        
        logger.info(f"Created {len(charts)} charts for {symbol}")
        return charts
    
    def compare_indicators(self, indicators: Dict[str, Any], state: AnalysisState = None) -> Dict[str, Any]:
        """
        Compare multiple indicators to determine consensus signals.
        
        Args:
            indicators: Dictionary containing calculated indicators
            
        Returns:
            Dict[str, Any]: Consensus analysis results
        """
        logger.info("Comparing technical indicators for consensus signals")
        if state and state.consensus is not None:
            return state.consensus
        consensus = self.analyzer.analyze_indicator_consensus(indicators)
        logger.info(f"Consensus signal: {consensus['overall_signal']} ({consensus['signal_strength']})")  
        if state is not None:
            state.consensus = consensus
        return consensus
    
    @staticmethod
    def serialize_indicators(indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize indicators to JSON-serializable format."""
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
        
        serializable = {}
        for key, value in indicators.items():
            serializable[key] = convert_numpy_types(value)
        return serializable
    
    async def orchestrate_llm_analysis(self, symbol: str, indicators: dict, chart_paths: dict, period: int, interval: str, knowledge_context: str = "") -> tuple:
        result, ind_summary_md, chart_insights_md = await self.gemini_client.analyze_stock(symbol, indicators, chart_paths, period, interval, knowledge_context)
        return result, ind_summary_md, chart_insights_md

    async def analyze_with_ai(self, stock_data, indicators, charts, period, interval, knowledge_context: str = "", sector_context: dict = None) -> tuple:
        # Add sector context to knowledge context if available
        enhanced_knowledge_context = knowledge_context
        if sector_context:
            sector_info = sector_context.get('sector_info', {})
            market_benchmarking = sector_context.get('market_benchmarking', {})
            sector_benchmarking = sector_context.get('sector_benchmarking', {})
            sector_analysis = sector_context.get('analysis_summary', {})
            
            sector_context_str = f"""
SECTOR CONTEXT:
- Sector: {sector_info.get('sector_name', 'Unknown')} ({sector_info.get('sector', 'Unknown')})
- Sector Index: {sector_info.get('sector_index', 'N/A')}
- Sector Stocks: {sector_info.get('sector_stocks_count', 0)} stocks

SECTOR PERFORMANCE:
- Market Outperformance: {float(market_benchmarking.get('excess_return', 0)):.2%}
- Sector Outperformance: {f"{float(sector_benchmarking.get('sector_excess_return', 0)):.2%}" if sector_benchmarking else 'N/A'}
- Sector Beta: {f"{sector_benchmarking.get('sector_beta', 1.0):.2f}" if sector_benchmarking else '1.00'}

SECTOR ANALYSIS:
- {sector_analysis.get('market_performance', 'Market performance analysis not available')}
- {sector_analysis.get('sector_performance', 'Sector performance analysis not available')}
- {sector_analysis.get('risk_assessment', 'Risk assessment not available')}

Consider this sector context when analyzing the stock's technical indicators and patterns.
"""
            enhanced_knowledge_context = knowledge_context + "\n" + sector_context_str
        
        result, ind_summary_md, chart_insights_md = await self.orchestrate_llm_analysis(stock_data['symbol'], indicators, charts, period, interval, enhanced_knowledge_context)
        return result, ind_summary_md, chart_insights_md
    
    async def analyze_stock(self, symbol: str, exchange: str = "NSE",
                     period: int = 365, interval: str = "day", output_dir: str = None, 
                     knowledge_context: str = "", sector: str = None) -> tuple:
        """
        Analyze a stock using all available data and methods.
        Now supports sector-specific analysis.
        """
        state = self._get_or_create_state(symbol, exchange)
        data = self.retrieve_stock_data(symbol, exchange, interval, period)
        if data is None:
            return {}, pd.DataFrame()

        # Calculate indicators if not already done or if data has changed
        if state.indicators is None or not state.is_valid():
            # Pass stock symbol for sector-aware calculations
            indicators = self.calculate_indicators(data, symbol)
            state.update(indicators=indicators)
            state.consensus = None

        # Get indicator consensus (cached if possible)
        consensus = self.compare_indicators(state.indicators, state)

        # Prepare stock data for AI analysis
        current_price = float(data['close'].iloc[-1])
        data_freshness = data.attrs.get('data_freshness', 'unknown')
        last_update_time = data.attrs.get('last_update_time', 'unknown')
        market_status = data.attrs.get('market_status', 'unknown')

        stock_data = {
            "symbol": symbol,
            "exchange": exchange,
            "current_price": current_price,
            "price_data_info": {
                "freshness": data_freshness,
                "last_update": last_update_time,
                "market_status": market_status,
                "data_source": "historical_close" if data_freshness == "last_close" else "real_time",
                "note": "Price is from last available trading data. Real-time quotes available during market hours (9:15 AM - 3:30 PM IST, Mon-Fri)."
            },
            "period": f"{period} days",
            "date_range": f"{data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}",
        }

        # --- CHART VISUALIZATIONS ---
        chart_paths = None
        if output_dir is not None:
            chart_paths = self.create_visualizations(data, state.indicators, symbol, output_dir)
            
            # Add Phase 2 pattern visualizations
            head_and_shoulders = PatternRecognition.detect_head_and_shoulders(data['close'])
            inverse_head_and_shoulders = PatternRecognition.detect_inverse_head_and_shoulders(data['close'])
            cup_and_handle = PatternRecognition.detect_cup_and_handle(data['close'])
            
            if head_and_shoulders:
                ChartVisualizer.plot_head_and_shoulders_pattern(
                    data, head_and_shoulders, 
                    os.path.join(output_dir, 'head_and_shoulders.png')
                )
            
            if inverse_head_and_shoulders:
                ChartVisualizer.plot_inverse_head_and_shoulders_pattern(
                    data, inverse_head_and_shoulders, 
                    os.path.join(output_dir, 'inverse_head_and_shoulders.png')
                )
            
            if cup_and_handle:
                ChartVisualizer.plot_cup_and_handle_pattern(
                    data, cup_and_handle, 
                    os.path.join(output_dir, 'cup_and_handle.png')
                )
            
            # Multi-timeframe analysis chart
            if 'multi_timeframe' in state.indicators and 'error' not in state.indicators['multi_timeframe']:
                ChartVisualizer.plot_multi_timeframe_analysis(
                    data, state.indicators['multi_timeframe'], 
                    os.path.join(output_dir, 'multi_timeframe_analysis.png')
                )
            
            # Phase 3 complex pattern visualizations
            triple_tops = PatternRecognition.detect_triple_top(data['close'])
            triple_bottoms = PatternRecognition.detect_triple_bottom(data['close'])
            wedge_patterns = PatternRecognition.detect_wedge_patterns(data['close'])
            channel_patterns = PatternRecognition.detect_channel_patterns(data['close'])
            
            if triple_tops:
                ChartVisualizer.plot_triple_top_pattern(
                    data, triple_tops, 
                    os.path.join(output_dir, 'triple_tops.png')
                )
            
            if triple_bottoms:
                ChartVisualizer.plot_triple_bottom_pattern(
                    data, triple_bottoms, 
                    os.path.join(output_dir, 'triple_bottoms.png')
                )
            
            if wedge_patterns:
                ChartVisualizer.plot_wedge_patterns(
                    data, wedge_patterns, 
                    os.path.join(output_dir, 'wedge_patterns.png')
                )
            
            if channel_patterns:
                ChartVisualizer.plot_channel_patterns(
                    data, channel_patterns, 
                    os.path.join(output_dir, 'channel_patterns.png')
                )

        # Get hybrid sector analysis (optimized + comprehensive inter-sector relationships)
        hybrid_sector_analysis = sector_benchmarking_provider.get_hybrid_stock_analysis(symbol, data)
        
        # Extract components for backward compatibility
        sector_benchmarking = hybrid_sector_analysis.get('stock_specific_analysis', {}) if hybrid_sector_analysis else {}
        sector_rotation = hybrid_sector_analysis.get('comprehensive_sector_context', {}).get('sector_rotation_context', {}) if hybrid_sector_analysis else {}
        sector_correlation = hybrid_sector_analysis.get('comprehensive_sector_context', {}).get('correlation_insights', {}) if hybrid_sector_analysis else {}
        
        # Enhanced sector context for LLM
        enhanced_sector_context = self._build_enhanced_sector_context(
            sector, sector_benchmarking, sector_rotation, sector_correlation
        )

        # Get AI analysis (async) - now with sector context
        ai_analysis, ind_summary_md, chart_insights_md = await self.analyze_with_ai(
            stock_data, state.indicators, chart_paths, period, interval, knowledge_context, enhanced_sector_context
        )

        # Convert indicators to serializable format
        serializable_indicators = self.serialize_indicators(state.indicators)

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

        # --- ADVANCED PATTERNS (PHASE 2) ---
        head_and_shoulders = PatternRecognition.detect_head_and_shoulders(data['close'])
        inverse_head_and_shoulders = PatternRecognition.detect_inverse_head_and_shoulders(data['close'])
        cup_and_handle = PatternRecognition.detect_cup_and_handle(data['close'])
        
        # --- COMPLEX PATTERNS (PHASE 3) ---
        triple_tops = PatternRecognition.detect_triple_top(data['close'])
        triple_bottoms = PatternRecognition.detect_triple_bottom(data['close'])
        wedge_patterns = PatternRecognition.detect_wedge_patterns(data['close'])
        channel_patterns = PatternRecognition.detect_channel_patterns(data['close'])
        
        # Format advanced patterns for output
        advanced_patterns = {
            "head_and_shoulders": head_and_shoulders,
            "inverse_head_and_shoulders": inverse_head_and_shoulders,
            "cup_and_handle": cup_and_handle,
            "triple_tops": triple_tops,
            "triple_bottoms": triple_bottoms,
            "wedge_patterns": wedge_patterns,
            "channel_patterns": channel_patterns
        }

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

        analysis_results = {
            'consensus': consensus,
            'indicators': serializable_indicators,
            'overlays': overlays,
            'ai_analysis': ai_analysis,
            'indicator_summary_md': ind_summary_md,
            'chart_insights': chart_insights_md,
            'sector_benchmarking': sector_benchmarking,
            'summary': {
                'overall_signal': consensus['overall_signal'],
                'signal_strength': consensus['signal_strength'],
                'bullish_percentage': consensus['bullish_percentage'],
                'bearish_percentage': consensus['bearish_percentage'],
                'neutral_percentage': consensus['neutral_percentage']
            }
        }
        state.update(analysis_results=analysis_results)
        return analysis_results, data

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
            logging.error(f"Error building enhanced sector context: {e}")
            return {'sector': sector, 'benchmarking': sector_benchmarking}

# Utility to clean NaN/Infinity for JSON
def clean_for_json(obj):
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    elif isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(v) for v in obj]
    else:
        return obj

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

