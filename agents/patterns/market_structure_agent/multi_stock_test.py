#!/usr/bin/env python3
"""
Multi-Stock Test - Complete LLM Workflow Demonstration

This test demonstrates the complete intended flow for the market structure agent:
1. Generate market structure data for multiple stocks
2. Create optimized chart images
3. Build enhanced prompts with visual context
4. Send images and prompts to LLM
5. Process and validate LLM responses
6. Display comprehensive results

Features:
- Real LLM integration with multimodal capabilities
- Multiple stock scenarios (tech, financial, utility, energy, healthcare)
- Chart generation with different market patterns
- Enhanced prompt templates with visual context
- Structured response parsing and validation
- Performance metrics and timing analysis
"""

import os
import sys
import asyncio
import logging
import json
import time
from typing import Dict, Any, List, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np

# Import pandas with fallback
try:
    import pandas as pd
except ImportError:
    pd = None

# Load environment variables
try:
    import dotenv
    # Load .env file from the backend/config directory
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'config', '.env')
    dotenv.load_dotenv(dotenv_path=env_path)
    print(f"✅ Environment variables loaded from: {env_path}")
except ImportError:
    print("⚠️ python-dotenv not available, using system environment variables")

# Add the backend directory to sys.path for imports
backend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..')
sys.path.append(backend_dir)

# Import agent components
try:
    from .integrated_market_structure_agent import IntegratedMarketStructureAgent
    from .production_optimizations import ProductionOptimizedMarketStructureAgent
    print("✅ Integrated market structure agents imported successfully")
except ImportError as e:
    print(f"⚠️ Integrated agent import failed: {e}")
    IntegratedMarketStructureAgent = None
    ProductionOptimizedMarketStructureAgent = None

# Import data clients for real data
try:
    from core.orchestrator import StockAnalysisOrchestrator
    from zerodha.client import ZerodhaDataClient
    print("✅ Real data client imports successful")
except ImportError as e:
    print(f"⚠️ Real data client import failed: {e}")
    print("Will fall back to synthetic data if real data is not available")
    StockAnalysisOrchestrator = None
    ZerodhaDataClient = None

# Import LLM backend
try:
    from llm import LLMClient
    print("✅ LLM client imported successfully")
except ImportError as e:
    print(f"⚠️ LLM client import failed: {e}")
    LLMClient = None

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MultiStockLLMWorkflowTest:
    """
    Complete LLM workflow demonstration for market structure analysis.
    
    This test demonstrates the complete intended flow:
    1. Generate market structure data for multiple stocks
    2. Create optimized chart images
    3. Build enhanced prompts with visual context
    4. Send images and prompts to LLM
    5. Process and validate LLM responses
    6. Display comprehensive results
    """
    
    def __init__(self, output_dir: str = None):
        self.name = "multi_stock_llm_workflow_test"
        self.version = "1.0.0"
        
        # Setup output directory
        if output_dir is None:
            output_dir = "llm_workflow_test_results"
        
        if not os.path.isabs(output_dir):
            script_dir = Path(__file__).parent
            self.output_dir = script_dir / output_dir
        else:
            self.output_dir = Path(output_dir)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize agents (with fallbacks)
        self.integrated_agent = None
        self.production_agent = None
        self.llm_client = None
        
        # Initialize real data clients
        self.orchestrator = None
        self.zerodha_client = None
        self.use_real_data = False
        
        # Try to initialize real data clients
        if StockAnalysisOrchestrator and ZerodhaDataClient:
            try:
                self.orchestrator = StockAnalysisOrchestrator()
                self.zerodha_client = ZerodhaDataClient()
                self.use_real_data = True
                print("✅ Real data clients initialized successfully")
            except Exception as e:
                print(f"⚠️ Could not initialize real data clients: {e}")
                print("Will use synthetic data for testing")
        else:
            print("⚠️ Real data client classes not available - using synthetic data")
        
        # Try to initialize integrated agent
        if IntegratedMarketStructureAgent:
            try:
                self.integrated_agent = IntegratedMarketStructureAgent()
                print("✅ Integrated market structure agent initialized")
            except Exception as e:
                print(f"⚠️ Failed to initialize integrated agent: {e}")
        
        # Try to initialize production agent
        if ProductionOptimizedMarketStructureAgent:
            try:
                self.production_agent = ProductionOptimizedMarketStructureAgent()
                print("✅ Production optimized agent initialized")
            except Exception as e:
                print(f"⚠️ Failed to initialize production agent: {e}")
        
        # Try to initialize LLM client
        if LLMClient:
            try:
                self.llm_client = LLMClient()
                print("✅ LLM client initialized")
            except Exception as e:
                print(f"⚠️ Failed to initialize LLM client: {e}")
        
        # Test configuration
        self.test_session_id = f"llm_workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.results = []
        
        logger.info(f"Multi-Stock LLM Workflow Test initialized")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Test session ID: {self.test_session_id}")
    
    async def _get_real_stock_data(self, symbol: str, period_days: int) -> pd.DataFrame:
        """
        Get real stock data using the orchestrator or Zerodha client.
        Falls back to synthetic data if real data fetch fails.
        """
        try:
            logger.info(f"[MARKET_STRUCTURE_TESTER] Fetching real data for {symbol} ({period_days} days)")
            
            # Try using orchestrator first (preferred method)
            if self.orchestrator:
                try:
                    stock_data = await self.orchestrator.retrieve_stock_data(
                        symbol=symbol,
                        exchange="NSE",
                        interval="day",
                        period=period_days
                    )
                    
                    if stock_data is not None and len(stock_data) > 0:
                        logger.info(f"[MARKET_STRUCTURE_TESTER] Retrieved {len(stock_data)} days of real data via orchestrator for {symbol}")
                        return stock_data
                    else:
                        logger.warning(f"[MARKET_STRUCTURE_TESTER] Orchestrator returned empty data for {symbol}")
                        
                except Exception as e:
                    logger.warning(f"[MARKET_STRUCTURE_TESTER] Orchestrator failed for {symbol}: {e}")
            
            # Fallback to direct Zerodha client
            if self.zerodha_client:
                try:
                    # Skip explicit authentication - trust that environment variables are set correctly
                    # The client will use ZERODHA_ACCESS_TOKEN from .env if available
                    logger.info(f"[MARKET_STRUCTURE_TESTER] Using Zerodha client with environment credentials for {symbol}")
                    
                    # Check if async method exists
                    if hasattr(self.zerodha_client, 'get_historical_data_async'):
                        stock_data = await self.zerodha_client.get_historical_data_async(
                            symbol=symbol,
                            exchange="NSE",
                            interval="day",
                            period=period_days
                        )
                    else:
                        # Use sync method in executor
                        import asyncio
                        loop = asyncio.get_event_loop()
                        stock_data = await loop.run_in_executor(
                            None,
                            self.zerodha_client.get_historical_data,
                            symbol,
                            "NSE",
                            "day",
                            None,
                            None,
                            period_days
                        )
                    
                    if stock_data is not None and len(stock_data) > 0:
                        # Ensure proper data structure
                        required_columns = ['open', 'high', 'low', 'close', 'volume']
                        
                        # Handle date column properly
                        if 'date' not in stock_data.columns:
                            if stock_data.index.name == 'date' or hasattr(stock_data.index, 'date'):
                                stock_data = stock_data.reset_index()
                            else:
                                stock_data['date'] = stock_data.index
                                stock_data = stock_data.reset_index(drop=True)
                        
                        # Check for missing columns
                        missing_columns = [col for col in required_columns if col not in stock_data.columns]
                        if missing_columns:
                            logger.error(f"[MARKET_STRUCTURE_TESTER] Missing required columns for {symbol}: {missing_columns}")
                            return None
                        
                        # Sort by date and set as index for pattern analysis
                        stock_data = stock_data.sort_values('date').reset_index(drop=True)
                        stock_data = stock_data.set_index('date')
                        
                        logger.info(f"[MARKET_STRUCTURE_TESTER] Retrieved {len(stock_data)} days of real data via Zerodha for {symbol}")
                        logger.info(f"[MARKET_STRUCTURE_TESTER] Date range: {stock_data.index.min()} to {stock_data.index.max()}")
                        logger.info(f"[MARKET_STRUCTURE_TESTER] Volume range: {stock_data['volume'].min():,} to {stock_data['volume'].max():,}")
                        
                        return stock_data
                    else:
                        logger.warning(f"[MARKET_STRUCTURE_TESTER] Zerodha returned empty data for {symbol}")
                        
                except Exception as e:
                    logger.warning(f"[MARKET_STRUCTURE_TESTER] Zerodha client failed for {symbol}: {e}")
            
            # If all real data methods failed, return None (will trigger fallback to synthetic)
            logger.warning(f"[MARKET_STRUCTURE_TESTER] All real data sources failed for {symbol}, falling back to synthetic data")
            return None
            
        except Exception as e:
            logger.error(f"[MARKET_STRUCTURE_TESTER] Real data fetch failed for {symbol}: {e}")
            return None
    
    def generate_synthetic_market_data(self, symbol: str, scenario: str = "mixed", periods: int = 252) -> 'pd.DataFrame':
        """
        Generate synthetic market data with realistic patterns.
        
        Args:
            symbol: Stock symbol for seeding
            scenario: Market pattern scenario
            periods: Number of data points
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            import pandas as pd
        except ImportError:
            # Fallback data structure
            class SimpleDataFrame:
                def __init__(self, data):
                    self.data = data
                    self.index = list(range(len(data['close'])))
                    
                def __getitem__(self, key):
                    return self.data[key]
                    
                def __len__(self):
                    return len(self.data['close'])
            
            periods = 100
            base_price = 150.0
            data = {
                'open': [base_price + i * 0.5 for i in range(periods)],
                'high': [base_price + i * 0.5 + 2 for i in range(periods)],
                'low': [base_price + i * 0.5 - 2 for i in range(periods)],
                'close': [base_price + i * 0.5 + np.random.uniform(-1, 1) for i in range(periods)],
                'volume': [100000 + np.random.randint(-20000, 20000) for i in range(periods)]
            }
            return SimpleDataFrame(data)
        
        np.random.seed(hash(symbol) % 2147483647)
        
        base_price = 100.0 + (hash(symbol) % 200)
        dates = pd.date_range(start=datetime.now() - timedelta(days=periods), periods=periods, freq='D')
        
        if scenario == "uptrend":
            trend = np.linspace(0, 30, periods)
            noise = np.random.normal(0, 2, periods)
            swing_pattern = 4 * np.sin(np.linspace(0, 8*np.pi, periods))
        elif scenario == "downtrend":
            trend = np.linspace(0, -25, periods)
            noise = np.random.normal(0, 2, periods)
            swing_pattern = 3 * np.sin(np.linspace(0, 6*np.pi, periods))
        elif scenario == "sideways":
            trend = np.zeros(periods)
            noise = np.random.normal(0, 1, periods)
            swing_pattern = 5 * np.sin(np.linspace(0, 10*np.pi, periods))
        elif scenario == "volatile":
            trend = np.linspace(0, 15, periods)
            noise = np.random.normal(0, 5, periods)
            swing_pattern = 6 * np.sin(np.linspace(0, 12*np.pi, periods))
        else:  # mixed
            trend1 = np.linspace(0, 20, periods//2)
            trend2 = np.linspace(20, 5, periods//2)
            trend = np.concatenate([trend1, trend2])
            noise = np.random.normal(0, 2.5, periods)
            swing_pattern = 4 * np.sin(np.linspace(0, 9*np.pi, periods))
        
        prices = base_price + trend + noise + swing_pattern
        prices = np.maximum(prices, base_price * 0.3)  # Ensure positive prices
        
        # Generate OHLC from close prices
        highs = prices * (1 + np.abs(np.random.normal(0, 0.025, periods)))
        lows = prices * (1 - np.abs(np.random.normal(0, 0.025, periods)))
        opens = np.roll(prices, 1)
        opens[0] = prices[0]
        
        # Generate realistic volume
        volumes = np.random.lognormal(11, 0.6, periods).astype(int)
        
        return pd.DataFrame({
            'open': opens,
            'high': highs,
            'low': lows,
            'close': prices,
            'volume': volumes
        }, index=dates)
    
    async def demonstrate_complete_llm_workflow(self, symbol: str, scenario: str = "mixed") -> Dict[str, Any]:
        """
        Demonstrate the complete LLM workflow for a single stock.
        
        Steps:
        1. Generate/retrieve market data
        2. Process market structure analysis
        3. Generate optimized chart image
        4. Build enhanced multimodal prompt
        5. Send prompt and chart to LLM
        6. Parse and validate LLM response
        7. Return complete results
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"COMPLETE LLM WORKFLOW DEMONSTRATION: {symbol}")
        logger.info(f"{'='*60}")
        
        workflow_start = time.time()
        result = {
            'symbol': symbol,
            'scenario': scenario,
            'workflow_steps': [],
            'success': False,
            'error': None
        }
        
        try:
            # Step 1: Get real market data or generate synthetic
            step1_start = time.time()
            logger.info(f"\n🔵 STEP 1: Getting market data for {symbol}")
            
            data_source = "unknown"
            if self.use_real_data:
                market_data = await self._get_real_stock_data(symbol, 200)  # Default 200 days
                if market_data is not None and (not hasattr(market_data, 'empty') or not market_data.empty) and len(market_data) >= 50:
                    data_source = "real_market_data"
                    logger.info(f"Using REAL market data for {symbol} ({len(market_data)} days)")
                else:
                    logger.warning(f"Real data unavailable/insufficient for {symbol}, falling back to synthetic")
                    market_data = self.generate_synthetic_market_data(symbol, scenario, periods=200)
                    data_source = "synthetic_fallback"
            else:
                market_data = self.generate_synthetic_market_data(symbol, scenario, periods=200)
                data_source = "synthetic_only"
                logger.info(f"Using synthetic data for {symbol} ({scenario} pattern)")
            
            step1_time = time.time() - step1_start
            result['workflow_steps'].append({
                'step': 1,
                'name': 'Market Data Retrieval',
                'duration': step1_time,
                'success': market_data is not None,
                'details': f"Retrieved {len(market_data)} data points - Source: {data_source}"
            })
            
            logger.info(f"✅ Step 1 completed in {step1_time:.2f}s - Retrieved {len(market_data)} data points from {data_source}")
            
            if market_data is None or (hasattr(market_data, 'empty') and market_data.empty) or len(market_data) < 50:
                data_points = len(market_data) if market_data is not None else 0
                raise ValueError(f"Insufficient market data retrieved: {data_points} points")
            
            # Step 2: Market structure processing (using integrated agent if available)
            step2_start = time.time()
            logger.info(f"\n🔵 STEP 2: Processing market structure analysis")
            
            if self.integrated_agent:
                try:
                    analysis_result = await self.integrated_agent.analyze_market_structure_with_chart(
                        data=market_data,
                        symbol=symbol,
                        quality_level='high'
                    )
                    technical_analysis = analysis_result.get('technical_analysis', {})
                    chart_image = analysis_result.get('chart_image')
                    processing_success = analysis_result.get('success', False)
                except Exception as e:
                    logger.warning(f"⚠️ Integrated agent failed: {e}, using fallback processing")
                    technical_analysis = self._generate_fallback_technical_analysis(market_data, symbol)
                    chart_image = None
                    processing_success = True
            else:
                logger.info(f"Using fallback technical analysis (integrated agent not available)")
                technical_analysis = self._generate_fallback_technical_analysis(market_data, symbol)
                chart_image = None
                processing_success = True
            
            step2_time = time.time() - step2_start
            result['workflow_steps'].append({
                'step': 2,
                'name': 'Market Structure Analysis',
                'duration': step2_time,
                'success': processing_success,
                'details': f"Processed technical analysis with {len(technical_analysis)} components"
            })
            
            logger.info(f"✅ Step 2 completed in {step2_time:.2f}s - Technical analysis processed")
            
            # Step 3: Chart generation (if not already generated)
            step3_start = time.time()
            logger.info(f"\n🔵 STEP 3: Generating optimized chart image")
            
            if chart_image is None or (isinstance(chart_image, bytes) and len(chart_image) == 0):
                chart_image = self._generate_fallback_chart(market_data, symbol, technical_analysis)
            
            chart_success = chart_image is not None
            chart_size = len(chart_image) if chart_image else 0
            
            step3_time = time.time() - step3_start
            result['workflow_steps'].append({
                'step': 3,
                'name': 'Chart Generation',
                'duration': step3_time,
                'success': chart_success,
                'details': f"Generated chart: {chart_size:,} bytes" if chart_success else "Chart generation failed"
            })
            
            logger.info(f"✅ Step 3 completed in {step3_time:.2f}s - Chart: {'Generated' if chart_success else 'Failed'} ({chart_size:,} bytes)")
            
            # Step 4: Build enhanced multimodal prompt
            step4_start = time.time()
            logger.info(f"\n🔵 STEP 4: Building enhanced multimodal prompt")
            
            prompt = self._build_enhanced_prompt(technical_analysis, symbol, scenario, market_data)
            
            step4_time = time.time() - step4_start
            result['workflow_steps'].append({
                'step': 4,
                'name': 'Prompt Generation',
                'duration': step4_time,
                'success': True,
                'details': f"Generated prompt: {len(prompt):,} characters"
            })
            
            logger.info(f"✅ Step 4 completed in {step4_time:.2f}s - Prompt generated ({len(prompt):,} characters)")
            
            # Step 5: Send to LLM with multimodal support
            step5_start = time.time()
            logger.info(f"\n🔵 STEP 5: Sending prompt and chart to LLM")
            
            if self.llm_client and chart_image is not None and isinstance(chart_image, bytes) and len(chart_image) > 0:
                try:
                    # Convert chart bytes to PIL Image for LLM
                    from PIL import Image
                    import io
                    
                    chart_pil = Image.open(io.BytesIO(chart_image))
                    
                    llm_response = await self.llm_client.generate_with_images(
                        prompt=prompt,
                        images=[chart_pil],
                        model='auto'  # Let system choose best model
                    )
                    
                    llm_success = True
                    logger.info(f"✅ LLM responded with multimodal analysis")
                    
                except Exception as e:
                    logger.warning(f"⚠️ LLM multimodal call failed: {e}, using text-only fallback")
                    llm_response = await self._fallback_llm_analysis(prompt, technical_analysis)
                    llm_success = False
            else:
                logger.info(f"Using simulated LLM response (client not available or no chart)")
                llm_response = self._generate_simulated_llm_response(technical_analysis, symbol, scenario)
                llm_success = False  # Mark as simulated
            
            step5_time = time.time() - step5_start
            result['workflow_steps'].append({
                'step': 5,
                'name': 'LLM Analysis',
                'duration': step5_time,
                'success': llm_success,
                'details': f"{'Real' if llm_success else 'Simulated'} LLM analysis: {len(str(llm_response)):,} characters"
            })
            
            logger.info(f"✅ Step 5 completed in {step5_time:.2f}s - {'Real' if llm_success else 'Simulated'} LLM analysis")
            
            # Step 6: Parse and validate LLM response
            step6_start = time.time()
            logger.info(f"\n🔵 STEP 6: Parsing and validating LLM response")
            
            parsed_analysis = self._parse_llm_response(llm_response)
            validation_result = self._validate_llm_analysis(parsed_analysis, technical_analysis)
            
            step6_time = time.time() - step6_start
            result['workflow_steps'].append({
                'step': 6,
                'name': 'Response Processing',
                'duration': step6_time,
                'success': validation_result['valid'],
                'details': f"Parsed analysis - Valid: {validation_result['valid']}, Quality: {validation_result.get('quality_score', 0):.1f}/10"
            })
            
            logger.info(f"✅ Step 6 completed in {step6_time:.2f}s - Response validation: {'PASS' if validation_result['valid'] else 'FAIL'}")
            
            # Step 7: Compile complete results
            total_workflow_time = time.time() - workflow_start
            
            result.update({
                'success': True,
                'total_workflow_time': total_workflow_time,
                'market_data_points': len(market_data),
                'data_source': data_source,  # Track whether real or synthetic data was used
                'technical_analysis': technical_analysis,
                'chart_generated': chart_success,
                'chart_size_bytes': chart_size,
                'prompt_length': len(prompt),
                'llm_analysis': parsed_analysis,
                'llm_real_call': llm_success,
                'validation_result': validation_result,
                'performance_metrics': {
                    'total_time': total_workflow_time,
                    'data_generation_time': result['workflow_steps'][0]['duration'],
                    'analysis_time': result['workflow_steps'][1]['duration'],
                    'chart_time': result['workflow_steps'][2]['duration'],
                    'prompt_time': result['workflow_steps'][3]['duration'],
                    'llm_time': result['workflow_steps'][4]['duration'],
                    'validation_time': result['workflow_steps'][5]['duration']
                }
            })
            
            logger.info(f"\n🎉 COMPLETE WORKFLOW SUCCESS for {symbol}")
            logger.info(f"   Total Time: {total_workflow_time:.2f}s")
            logger.info(f"   Steps Completed: {len([s for s in result['workflow_steps'] if s['success']])}/6")
            logger.info(f"   Chart Generated: {'✅' if chart_success else '❌'}")
            logger.info(f"   LLM Analysis: {'✅ Real' if llm_success else '🔄 Simulated'}")
            logger.info(f"   Response Valid: {'✅' if validation_result['valid'] else '❌'}")
            
            return result
            
        except Exception as e:
            error_msg = str(e)
            result['error'] = error_msg
            result['total_workflow_time'] = time.time() - workflow_start
            
            logger.error(f"❌ WORKFLOW FAILED for {symbol}: {error_msg}")
            return result
    
    def _generate_fallback_technical_analysis(self, market_data, symbol: str) -> Dict[str, Any]:
        """Generate basic technical analysis when integrated agent is unavailable."""
        try:
            closes = market_data['close']
            highs = market_data['high']
            lows = market_data['low']
            
            # Basic trend analysis
            recent_closes = closes[-20:] if len(closes) >= 20 else closes
            trend_direction = 'bullish' if recent_closes[-1] > recent_closes[0] else 'bearish'
            
            # Simple swing point detection
            swing_count = 0
            for i in range(1, len(closes) - 1):
                if (closes[i] > closes[i-1] and closes[i] > closes[i+1]) or \
                   (closes[i] < closes[i-1] and closes[i] < closes[i+1]):
                    swing_count += 1
            
            return {
                'trend_analysis': {
                    'trend_direction': trend_direction,
                    'trend_strength': 'moderate',
                    'current_price': float(closes[-1]) if len(closes) > 0 else 0
                },
                'swing_points': {
                    'total_swings': swing_count,
                    'swing_density': swing_count / len(closes) if len(closes) > 0 else 0
                },
                'structure_quality': {
                    'quality_score': min(75, max(25, swing_count * 10)),
                    'quality_rating': 'good' if swing_count > 5 else 'moderate'
                },
                'key_levels': {
                    'support_levels': [float(min(lows[-10:])) if len(lows) >= 10 else float(min(lows))],
                    'resistance_levels': [float(max(highs[-10:])) if len(highs) >= 10 else float(max(highs))]
                },
                'confidence_score': 70
            }
        except Exception as e:
            logger.error(f"Fallback technical analysis failed: {e}")
            return {
                'trend_analysis': {'trend_direction': 'unknown'},
                'swing_points': {'total_swings': 0},
                'structure_quality': {'quality_score': 0},
                'confidence_score': 0
            }
    
    def _generate_fallback_chart(self, market_data, symbol: str, technical_analysis: Dict) -> bytes:
        """Generate a simple chart when integrated chart generation is unavailable."""
        try:
            import matplotlib.pyplot as plt
            import io
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Plot price data
            closes = market_data['close']
            dates = list(range(len(closes)))
            
            ax.plot(dates, closes, linewidth=2, label=f'{symbol} Close Price')
            ax.set_title(f'{symbol} - Market Structure Analysis', fontsize=16, fontweight='bold')
            ax.set_xlabel('Time Period', fontsize=12)
            ax.set_ylabel('Price', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Add support/resistance if available
            key_levels = technical_analysis.get('key_levels', {})
            if key_levels.get('support_levels'):
                for level in key_levels['support_levels'][:2]:  # Show max 2 levels
                    ax.axhline(y=level, color='green', linestyle='--', alpha=0.7, label=f'Support: {level:.2f}')
            
            if key_levels.get('resistance_levels'):
                for level in key_levels['resistance_levels'][:2]:  # Show max 2 levels
                    ax.axhline(y=level, color='red', linestyle='--', alpha=0.7, label=f'Resistance: {level:.2f}')
            
            plt.tight_layout()
            
            # Save to bytes
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
            buffer.seek(0)
            chart_bytes = buffer.read()
            
            plt.close()
            return chart_bytes
            
        except Exception as e:
            logger.error(f"Fallback chart generation failed: {e}")
            return b""  # Return empty bytes if chart generation fails
    
    def _build_enhanced_prompt(self, technical_analysis: Dict, symbol: str, scenario: str, market_data) -> str:
        """Build enhanced multimodal prompt for LLM analysis."""
        prompt = f"""# Market Structure Analysis Request for {symbol}

## Context
You are analyzing the market structure for {symbol} using a {scenario} market scenario with {len(market_data)} data points.

## Technical Analysis Data
{json.dumps(technical_analysis, indent=2, default=str)}

## Analysis Requirements

Please analyze the provided chart image and technical data to provide:

1. **Market Structure Assessment**:
   - Overall trend direction and strength
   - Key support and resistance levels
   - Swing point analysis and significance

2. **Technical Insights**:
   - Current market phase (accumulation, distribution, trending, consolidation)
   - BOS (Break of Structure) and CHOCH (Change of Character) signals
   - Volume analysis and price action confirmation

3. **Strategic Outlook**:
   - Potential entry and exit levels
   - Risk management considerations
   - Short-term and medium-term expectations

4. **Quality Assessment**:
   - Rate the overall market structure clarity (1-10)
   - Identify any conflicting signals
   - Confidence level in the analysis

## Response Format
Provide your analysis in a structured JSON format with clear sections for each requirement.

The chart image shows the current market structure with key levels marked. Please reference specific visual elements you observe."""
        
        return prompt
    
    def _generate_simulated_llm_response(self, technical_analysis: Dict, symbol: str, scenario: str) -> str:
        """Generate a realistic simulated LLM response for testing."""
        trend_direction = technical_analysis.get('trend_analysis', {}).get('trend_direction', 'unknown')
        swing_count = technical_analysis.get('swing_points', {}).get('total_swings', 0)
        quality_score = technical_analysis.get('structure_quality', {}).get('quality_score', 0)
        
        return f"""{{
    "market_structure_assessment": {{
        "overall_trend": "{trend_direction}",
        "trend_strength": "moderate",
        "market_phase": "trending",
        "swing_analysis": {{
            "total_swing_points": {swing_count},
            "swing_significance": "moderate to high",
            "structure_clarity": "good"
        }}
    }},
    "technical_insights": {{
        "current_phase": "price discovery",
        "bos_choch_signals": "present",
        "volume_confirmation": "adequate",
        "price_action": "consistent with {scenario} scenario"
    }},
    "strategic_outlook": {{
        "short_term": "continue monitoring key levels",
        "medium_term": "maintain {trend_direction} bias",
        "risk_management": "standard position sizing recommended"
    }},
    "quality_assessment": {{
        "structure_clarity_rating": {min(10, max(1, quality_score // 10))},
        "conflicting_signals": "minimal",
        "confidence_level": "high",
        "analysis_reliability": "good"
    }}
}}"""
    
    async def _fallback_llm_analysis(self, prompt: str, technical_analysis: Dict) -> str:
        """Fallback LLM analysis using text-only mode."""
        if self.llm_client:
            try:
                return await self.llm_client.generate(prompt=prompt, model='auto')
            except Exception as e:
                logger.warning(f"Text-only LLM call failed: {e}")
        
        # Return simulated response if LLM unavailable
        return self._generate_simulated_llm_response(technical_analysis, "Unknown", "mixed")
    
    def _parse_llm_response(self, llm_response: str) -> Dict[str, Any]:
        """Parse LLM response into structured data."""
        try:
            # Try to extract JSON from the response
            import re
            
            # Look for JSON blocks in the response
            json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
            json_matches = re.findall(json_pattern, str(llm_response), re.DOTALL)
            
            if json_matches:
                # Try to parse the largest JSON block
                largest_json = max(json_matches, key=len)
                parsed = json.loads(largest_json)
                return parsed
            else:
                # Fallback: create structured response from text
                return {
                    "raw_response": str(llm_response),
                    "parsed_success": False,
                    "analysis_type": "text_based",
                    "content_length": len(str(llm_response))
                }
                
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
            return {
                "raw_response": str(llm_response),
                "parsing_error": str(e),
                "parsed_success": False
            }
    
    def _validate_llm_analysis(self, parsed_analysis: Dict, technical_analysis: Dict) -> Dict[str, Any]:
        """Validate the parsed LLM analysis for quality and consistency."""
        validation_result = {
            'valid': False,
            'quality_score': 0,
            'validation_details': {}
        }
        
        try:
            score = 0
            max_score = 10
            details = {}
            
            # Check for required sections
            required_sections = ['market_structure_assessment', 'technical_insights', 'strategic_outlook', 'quality_assessment']
            sections_present = 0
            
            for section in required_sections:
                if section in parsed_analysis:
                    sections_present += 1
                    score += 2
            
            details['sections_present'] = f"{sections_present}/{len(required_sections)}"
            
            # Check for content quality
            if parsed_analysis.get('parsed_success', True):
                score += 1
                details['parsing_success'] = True
            
            # Check response length (good responses should be substantial)
            response_length = len(str(parsed_analysis))
            if response_length > 200:
                score += 1
                details['adequate_length'] = True
            
            # Check for specific analysis elements
            if 'market_structure_assessment' in parsed_analysis:
                msa = parsed_analysis['market_structure_assessment']
                if 'overall_trend' in msa or 'trend_direction' in msa:
                    score += 1
                    details['trend_analysis_present'] = True
            
            # Final validation
            validation_result['quality_score'] = score
            validation_result['valid'] = score >= 6  # Minimum threshold
            validation_result['validation_details'] = details
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            validation_result['validation_error'] = str(e)
            return validation_result
    
    async def run_multi_stock_demonstration(self, test_scenarios: List[Tuple[str, str]] = None) -> Dict[str, Any]:
        """
        Run the complete LLM workflow demonstration across multiple stocks and scenarios.
        
        Args:
            test_scenarios: List of (symbol, scenario) tuples
            
        Returns:
            Complete demonstration results
        """
        if test_scenarios is None:
            test_scenarios = [
                ("AAPL", "uptrend"),
                ("MSFT", "sideways"),
                ("GOOGL", "downtrend"),
                ("RELIANCE", "volatile"),
                ("INFY", "mixed")
            ]
        
        logger.info(f"\n{'='*80}")
        logger.info(f"MULTI-STOCK LLM WORKFLOW DEMONSTRATION")
        logger.info(f"{'='*80}")
        logger.info(f"Testing {len(test_scenarios)} scenarios:")
        for symbol, scenario in test_scenarios:
            logger.info(f"  - {symbol}: {scenario} pattern")
        
        demo_start = time.time()
        results = {
            'demonstration_metadata': {
                'start_time': datetime.now().isoformat(),
                'test_scenarios': test_scenarios,
                'total_scenarios': len(test_scenarios)
            },
            'individual_results': [],
            'summary_statistics': {},
            'success': False
        }
        
        # Execute each scenario
        for i, (symbol, scenario) in enumerate(test_scenarios, 1):
            logger.info(f"\n[{i}/{len(test_scenarios)}] Processing {symbol} ({scenario})...")
            
            try:
                scenario_result = await self.demonstrate_complete_llm_workflow(symbol, scenario)
                results['individual_results'].append(scenario_result)
                
                # Log summary
                if scenario_result['success']:
                    logger.info(f"✅ {symbol} completed in {scenario_result['total_workflow_time']:.2f}s")
                else:
                    logger.warning(f"⚠️ {symbol} failed: {scenario_result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                error_result = {
                    'symbol': symbol,
                    'scenario': scenario,
                    'success': False,
                    'error': str(e),
                    'total_workflow_time': 0
                }
                results['individual_results'].append(error_result)
                logger.error(f"❌ {symbol} failed with exception: {e}")
        
        # Calculate summary statistics
        total_time = time.time() - demo_start
        successful_demos = [r for r in results['individual_results'] if r.get('success', False)]
        failed_demos = [r for r in results['individual_results'] if not r.get('success', False)]
        
        results['summary_statistics'] = {
            'total_demonstration_time': total_time,
            'scenarios_attempted': len(test_scenarios),
            'scenarios_successful': len(successful_demos),
            'scenarios_failed': len(failed_demos),
            'success_rate_percent': (len(successful_demos) / len(test_scenarios) * 100) if test_scenarios else 0,
            'average_workflow_time': np.mean([r.get('total_workflow_time', 0) for r in successful_demos]) if successful_demos else 0
        }
        
        results['success'] = len(successful_demos) > 0
        results['demonstration_metadata']['end_time'] = datetime.now().isoformat()
        
        # Save results
        await self._save_demonstration_results(results)
        
        # Log final summary
        logger.info(f"\n{'='*80}")
        logger.info(f"DEMONSTRATION COMPLETE")
        logger.info(f"{'='*80}")
        logger.info(f"Total Time: {total_time:.2f}s")
        logger.info(f"Success Rate: {results['summary_statistics']['success_rate_percent']:.1f}%")
        logger.info(f"Successful Scenarios: {len(successful_demos)}/{len(test_scenarios)}")
        logger.info(f"Average Workflow Time: {results['summary_statistics']['average_workflow_time']:.2f}s")
        
        if successful_demos:
            logger.info(f"\n🎉 Successful demonstrations:")
            for result in successful_demos:
                logger.info(f"  ✅ {result['symbol']} ({result['scenario']}) - {result['total_workflow_time']:.2f}s")
        
        if failed_demos:
            logger.info(f"\n⚠️ Failed demonstrations:")
            for result in failed_demos:
                logger.info(f"  ❌ {result['symbol']} ({result['scenario']}) - {result.get('error', 'Unknown error')}")
        
        logger.info(f"\nDetailed results saved to: {self.output_dir}")
        
        return results
    
    async def _save_demonstration_results(self, results: Dict[str, Any]):
        """Save demonstration results to files."""
        try:
            # Save main results
            results_file = self.output_dir / f"llm_workflow_demonstration_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            # Save summary
            summary_file = self.output_dir / "demonstration_summary.txt"
            with open(summary_file, 'w') as f:
                f.write("Multi-Stock LLM Workflow Demonstration Summary\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Total Scenarios: {results['summary_statistics']['scenarios_attempted']}\n")
                f.write(f"Successful: {results['summary_statistics']['scenarios_successful']}\n")
                f.write(f"Failed: {results['summary_statistics']['scenarios_failed']}\n")
                f.write(f"Success Rate: {results['summary_statistics']['success_rate_percent']:.1f}%\n")
                f.write(f"Total Time: {results['summary_statistics']['total_demonstration_time']:.2f}s\n")
                f.write(f"Average Workflow Time: {results['summary_statistics']['average_workflow_time']:.2f}s\n")
            
            logger.info(f"Demonstration results saved to: {results_file}")
            
        except Exception as e:
            logger.error(f"Failed to save demonstration results: {e}")


# Helper function for running single demonstration
async def run_single_demonstration(symbol: str, scenario: str = "mixed"):
    """
    Run a single LLM workflow demonstration for quick testing.
    
    Args:
        symbol: Stock symbol to analyze
        scenario: Market scenario to simulate
        
    Returns:
        Demonstration result
    """
    workflow_test = MultiStockLLMWorkflowTest()
    return await workflow_test.demonstrate_complete_llm_workflow(symbol, scenario)


# Main execution function for standalone testing
async def main():
    """Main function for running the complete LLM workflow demonstration."""
    
    print("\n" + "="*80)
    print("MULTI-STOCK LLM WORKFLOW DEMONSTRATION")
    print("Complete Market Structure Analysis with Real LLM Integration")
    print("="*80)
    
    # Initialize the workflow test
    workflow_test = MultiStockLLMWorkflowTest()
    
    # Show configuration
    print(f"\nConfiguration:")
    print(f"  Real Data Available: {'✅ Yes' if workflow_test.use_real_data else '❌ No (Synthetic Only)'}")
    print(f"  Orchestrator: {'✅ Available' if workflow_test.orchestrator else '❌ Not Available'}")
    print(f"  Zerodha Client: {'✅ Available' if workflow_test.zerodha_client else '❌ Not Available'}")
    print(f"  Integrated Agent: {'✅ Available' if workflow_test.integrated_agent else '❌ Fallback'}")
    print(f"  Production Agent: {'✅ Available' if workflow_test.production_agent else '❌ Not Available'}")
    print(f"  LLM Client: {'✅ Available' if workflow_test.llm_client else '❌ Simulated'}")
    print(f"  Output Directory: {workflow_test.output_dir}")
    print(f"  Data Strategy: {'Real market data (with synthetic fallback)' if workflow_test.use_real_data else 'Synthetic data only'}")
    
    # Define test scenarios covering different market patterns with Indian stocks
    test_scenarios = [
        ("RELIANCE", "mixed"),     # Large cap - let real data determine pattern
        ("TCS", "mixed"),         # IT sector - let real data determine pattern
        ("INFY", "mixed"),        # IT sector - let real data determine pattern
        ("HDFCBANK", "mixed"),    # Banking sector - let real data determine pattern
        ("ITC", "mixed")          # FMCG sector - let real data determine pattern
    ]
    
    print(f"\nTest Scenarios ({len(test_scenarios)}):")
    for i, (symbol, scenario) in enumerate(test_scenarios, 1):
        print(f"  {i}. {symbol}: {scenario} pattern")
    
    # Run the complete demonstration
    try:
        demo_results = await workflow_test.run_multi_stock_demonstration(test_scenarios)
        
        # Display final results
        print("\n" + "="*80)
        print("FINAL DEMONSTRATION RESULTS")
        print("="*80)
        
        stats = demo_results['summary_statistics']
        print(f"Total Scenarios: {stats['scenarios_attempted']}")
        print(f"Successful: {stats['scenarios_successful']} ({stats['success_rate_percent']:.1f}%)")
        print(f"Failed: {stats['scenarios_failed']}")
        print(f"Total Time: {stats['total_demonstration_time']:.2f}s")
        print(f"Average Workflow Time: {stats['average_workflow_time']:.2f}s")
        
        # Performance breakdown
        successful_results = [r for r in demo_results['individual_results'] if r.get('success')]
        if successful_results:
            print(f"\n📊 Performance Breakdown (Successful Scenarios):")
            
            avg_times = {}
            for step_name in ['data_generation_time', 'analysis_time', 'chart_time', 'prompt_time', 'llm_time', 'validation_time']:
                times = [r['performance_metrics'][step_name] for r in successful_results if 'performance_metrics' in r]
                if times:
                    avg_times[step_name] = np.mean(times)
            
            print(f"  Data Generation: {avg_times.get('data_generation_time', 0):.3f}s")
            print(f"  Market Analysis: {avg_times.get('analysis_time', 0):.3f}s")
            print(f"  Chart Generation: {avg_times.get('chart_time', 0):.3f}s")
            print(f"  Prompt Building: {avg_times.get('prompt_time', 0):.3f}s")
            print(f"  LLM Processing: {avg_times.get('llm_time', 0):.3f}s")
            print(f"  Response Validation: {avg_times.get('validation_time', 0):.3f}s")
        
        # LLM integration summary
        real_llm_count = len([r for r in demo_results['individual_results'] if r.get('llm_real_call')])
        simulated_count = stats['scenarios_successful'] - real_llm_count
        
        # Data source statistics
        real_data_count = len([r for r in demo_results['individual_results'] if r.get('data_source') == 'real_market_data'])
        synthetic_fallback_count = len([r for r in demo_results['individual_results'] if r.get('data_source') == 'synthetic_fallback'])
        synthetic_only_count = len([r for r in demo_results['individual_results'] if r.get('data_source') == 'synthetic_only'])
        
        print(f"\n📊 Data Source Summary:")
        print(f"  Real Market Data: {real_data_count}")
        print(f"  Synthetic Fallback: {synthetic_fallback_count}")
        print(f"  Synthetic Only: {synthetic_only_count}")
        print(f"  Real Data Success Rate: {(real_data_count / len(test_scenarios) * 100):.1f}%")
        
        print(f"\n🤖 LLM Integration Summary:")
        print(f"  Real LLM Calls: {real_llm_count}")
        print(f"  Simulated Responses: {simulated_count}")
        print(f"  Charts Generated: {len([r for r in demo_results['individual_results'] if r.get('chart_generated')])}")
        
        print(f"\n📁 Results saved to: {workflow_test.output_dir}")
        print(f"\n🎉 Demonstration {'COMPLETED SUCCESSFULLY' if demo_results['success'] else 'COMPLETED WITH ISSUES'}")
        
        return demo_results
        
    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        import traceback
        print(f"\nTraceback:")
        print(traceback.format_exc())
        return {'success': False, 'error': str(e)}


# Allow running this script directly
if __name__ == "__main__":
    # Run the test
    results = asyncio.run(main())
