#!/usr/bin/env python3
"""
service_endpoints.py

Service Endpoints for Testing Individual Components

This module provides individual endpoints for testing each service component
in the stock analysis system.
"""

import os
import sys
import time
import json
import asyncio
import traceback
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path

# Add the backend directory to Python path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

# Try to import optional dependencies
try:
    import dotenv
    dotenv.load_dotenv()
except ImportError:
    pass

try:
    import pandas as pd
    import numpy as np
    from pandas import Timestamp
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

# FastAPI imports
from fastapi import FastAPI, HTTPException, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Local imports
from zerodha.client import ZerodhaDataClient
from ml.indicators.technical_indicators import TechnicalIndicators, IndianMarketMetricsProvider
from patterns.recognition import PatternRecognition
from agents.sector import SectorClassifier, enhanced_sector_classifier, SectorBenchmarkingProvider
from analysis.orchestrator import StockAnalysisOrchestrator
# Note: These modules may not have the expected classes, using functions instead
# from risk_scoring import RiskScorer
# from bayesian_scorer import BayesianScorer
# from market_regime import MarketRegimeDetector
# from backtesting import Backtester

# Create FastAPI app
app = FastAPI(
    title="Service Endpoints",
    description="Individual service endpoints for testing components",
    version="1.0.0"
)

# Load CORS origins from environment variable
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:8080,http://localhost:5173").split(",")
CORS_ORIGINS = [origin.strip() for origin in CORS_ORIGINS if origin.strip()]

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for requests
class DataRequest(BaseModel):
    symbol: str = Field(..., description="Stock symbol (e.g., RELIANCE)")
    exchange: str = Field(default="NSE", description="Exchange (default: NSE)")
    period: int = Field(default=365, description="Period in days (default: 365)")
    interval: str = Field(default="day", description="Data interval (default: day)")
    sector: str = Field(default="", description="Sector name (optional, will auto-detect if not provided)")

class TechnicalAnalysisRequest(BaseModel):
    symbol: str = Field(..., description="Stock symbol")
    exchange: str = Field(default="NSE", description="Exchange")
    period: int = Field(default=365, description="Period in days")
    interval: str = Field(default="day", description="Data interval")
    indicators: str = Field(default="all", description="Comma-separated indicators or 'all'")

class PatternRequest(BaseModel):
    symbol: str = Field(..., description="Stock symbol")
    exchange: str = Field(default="NSE", description="Exchange")
    period: int = Field(default=365, description="Period in days")
    interval: str = Field(default="day", description="Data interval")
    pattern_types: str = Field(default="all", description="Comma-separated pattern types or 'all'")

class SectorRequest(BaseModel):
    symbol: str = Field(..., description="Stock symbol")
    sector: str = Field(default="", description="Sector name (optional, will auto-detect if not provided)")

# ============================================================================
# HEALTH AND STATUS ENDPOINTS
# ============================================================================

@app.get("/health")
async def health_check():
    """Comprehensive health check for all services."""
    try:
        health_status = {
            "status": "healthy",
            "service": "Service Endpoints",
            "timestamp": datetime.now().isoformat(),
            "components": {}
        }
        
        # Check Zerodha client
        try:
            zerodha_client = ZerodhaDataClient()
            auth_status = zerodha_client.authenticate()
            health_status["components"]["zerodha_client"] = {
                "status": "authenticated" if auth_status else "not_authenticated",
                "configured": bool(os.getenv("ZERODHA_API_KEY") and os.getenv("ZERODHA_ACCESS_TOKEN"))
            }
        except Exception as e:
            health_status["components"]["zerodha_client"] = {
                "status": "error",
                "error": str(e)
            }
        
        # Check sector classifiers
        try:
            sector_classifier = SectorClassifier()
            sectors = sector_classifier.get_all_sectors()
            health_status["components"]["sector_classifier"] = {
                "status": "loaded",
                "sectors_count": len(sectors)
            }
        except Exception as e:
            health_status["components"]["sector_classifier"] = {
                "status": "error",
                "error": str(e)
            }
        
        return health_status
    except Exception as e:
        return {
            "status": "unhealthy",
            "service": "Service Endpoints",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/status")
async def service_status():
    """Detailed status of all service components."""
    return {
        "service": "Service Endpoints",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "health": "/health",
            "status": "/status",
            "data": {
                "fetch_data": "/data/fetch",
                "get_stock_info": "/data/stock-info/{symbol}",
                "get_market_status": "/data/market-status",
                "get_token_mapping": "/data/token-mapping"
            },
            "technical_analysis": {
                "calculate_indicators": "/technical/indicators",
                "get_market_metrics": "/technical/market-metrics",
                "get_enhanced_metrics": "/technical/enhanced-metrics"
            },
            "patterns": {
                "detect_patterns": "/patterns/detect",
                "get_candlestick_patterns": "/patterns/candlestick",
                "get_chart_patterns": "/patterns/chart",
                "get_volume_patterns": "/patterns/volume"
            },
            "sectors": {
                "get_sector_info": "/sectors/info",
                "get_sector_benchmarking": "/sectors/benchmarking",
                "get_sector_rotation": "/sectors/rotation",
                "get_sector_correlation": "/sectors/correlation",
                "get_sector_performance": "/sectors/performance"
            },
            "analysis": {
                "full_analysis": "/analysis/full",
                "enhanced_analysis": "/analysis/enhanced",
                "risk_assessment": "/analysis/risk",
                "backtesting": "/analysis/backtest"
            }
        }
    }

# ============================================================================
# DATA SERVICE ENDPOINTS
# ============================================================================

@app.post("/data/fetch")
async def fetch_stock_data(request: DataRequest):
    """Fetch historical stock data."""
    try:
        zerodha_client = ZerodhaDataClient()
        if not zerodha_client.authenticate():
            raise HTTPException(status_code=401, detail="Zerodha authentication failed")
        
        data = await zerodha_client.get_historical_data(
            symbol=request.symbol,
            exchange=request.exchange,
            interval=request.interval,
            period=request.period
        )
        
        if data is None or data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {request.symbol}")
        
        return {
            "symbol": request.symbol,
            "exchange": request.exchange,
            "period": request.period,
            "interval": request.interval,
            "data_points": len(data),
            "start_date": data.index[0].isoformat(),
            "end_date": data.index[-1].isoformat(),
            "last_price": float(data['close'].iloc[-1]),
            "last_volume": float(data['volume'].iloc[-1]),
            "sample_data": data.tail(5).to_dict('records')
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/data/stock-info/{symbol}")
async def get_stock_info(symbol: str, exchange: str = "NSE"):
    """Get basic stock information."""
    try:
        zerodha_client = ZerodhaDataClient()
        if not zerodha_client.authenticate():
            raise HTTPException(status_code=401, detail="Zerodha authentication failed")
        
        # Get token mapping
        token = zerodha_client.get_token_from_symbol(symbol, exchange)
        if not token:
            raise HTTPException(status_code=404, detail=f"Token not found for {symbol}")
        
        # Get recent data for basic info
        data = await zerodha_client.get_historical_data(
            symbol=symbol,
            exchange=exchange,
            interval="day",
            period=30
        )
        
        if data is None or data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {symbol}")
        
        return {
            "symbol": symbol,
            "exchange": exchange,
            "token": token,
            "last_price": float(data['close'].iloc[-1]),
            "last_volume": float(data['volume'].iloc[-1]),
            "price_change": float(data['close'].iloc[-1] - data['close'].iloc[-2]),
            "price_change_pct": float(((data['close'].iloc[-1] / data['close'].iloc[-2]) - 1) * 100),
            "high_30d": float(data['high'].max()),
            "low_30d": float(data['low'].min()),
            "avg_volume_30d": float(data['volume'].mean())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/data/market-status")
async def get_market_status():
    """Get current market status."""
    try:
        zerodha_client = ZerodhaDataClient()
        if not zerodha_client.authenticate():
            raise HTTPException(status_code=401, detail="Zerodha authentication failed")
        
        # Get NIFTY 50 data for market status
        nifty_data = await zerodha_client.get_historical_data(
            symbol="NIFTY 50",
            exchange="NSE",
            interval="day",
            period=5
        )
        
        if nifty_data is None or nifty_data.empty:
            return {"status": "unknown", "message": "Unable to fetch market data"}
        
        current_time = datetime.now()
        market_open = current_time.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close = current_time.replace(hour=15, minute=30, second=0, microsecond=0)
        
        is_market_open = market_open <= current_time <= market_close and current_time.weekday() < 5
        
        return {
            "market_status": "open" if is_market_open else "closed",
            "current_time": current_time.isoformat(),
            "market_open_time": market_open.isoformat(),
            "market_close_time": market_close.isoformat(),
            "nifty_50": {
                "last_price": float(nifty_data['close'].iloc[-1]),
                "price_change": float(nifty_data['close'].iloc[-1] - nifty_data['close'].iloc[-2]),
                "price_change_pct": float(((nifty_data['close'].iloc[-1] / nifty_data['close'].iloc[-2]) - 1) * 100)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/data/token-mapping")
async def get_token_mapping(symbol: str = None, token: int = None, exchange: str = "NSE"):
    """Get token to symbol mapping or vice versa."""
    try:
        zerodha_client = ZerodhaDataClient()
        if not zerodha_client.authenticate():
            raise HTTPException(status_code=401, detail="Zerodha authentication failed")
        
        if symbol:
            token_result = zerodha_client.get_token_from_symbol(symbol, exchange)
            return {
                "symbol": symbol,
                "token": token_result,
                "exchange": exchange,
                "mapping_type": "symbol_to_token"
            }
        elif token:
            symbol_result = zerodha_client.get_symbol_from_token(token, exchange)
            return {
                "symbol": symbol_result,
                "token": token,
                "exchange": exchange,
                "mapping_type": "token_to_symbol"
            }
        else:
            raise HTTPException(status_code=400, detail="Either symbol or token must be provided")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# TECHNICAL ANALYSIS ENDPOINTS
# ============================================================================

@app.post("/technical/indicators")
async def calculate_technical_indicators(request: TechnicalAnalysisRequest):
    """Calculate technical indicators for a stock."""
    try:
        # Fetch data
        zerodha_client = ZerodhaDataClient()
        if not zerodha_client.authenticate():
            raise HTTPException(status_code=401, detail="Zerodha authentication failed")
        
        data = await zerodha_client.get_historical_data(
            symbol=request.symbol,
            exchange=request.exchange,
            interval=request.interval,
            period=request.period
        )
        
        if data is None or data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {request.symbol}")
        
        # Calculate indicators
        if request.indicators.lower() == "all":
            indicators = TechnicalIndicators.calculate_all_indicators_optimized(data, request.symbol)
        else:
            # Calculate specific indicators
            indicator_list = [ind.strip() for ind in request.indicators.split(",")]
            indicators = {}
            
            for indicator in indicator_list:
                if hasattr(TechnicalIndicators, f"calculate_{indicator.lower()}"):
                    method = getattr(TechnicalIndicators, f"calculate_{indicator.lower()}")
                    indicators[indicator] = method(data)
                else:
                    indicators[indicator] = f"Indicator {indicator} not found"
        
        return {
            "symbol": request.symbol,
            "exchange": request.exchange,
            "period": request.period,
            "interval": request.interval,
            "data_points": len(data),
            "indicators": indicators,
            "last_price": float(data['close'].iloc[-1]),
            "calculation_time": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/technical/market-metrics")
async def get_market_metrics(request: DataRequest):
    """Get basic market metrics for a stock."""
    try:
        # Fetch data
        zerodha_client = ZerodhaDataClient()
        if not zerodha_client.authenticate():
            raise HTTPException(status_code=401, detail="Zerodha authentication failed")
        
        data = await zerodha_client.get_historical_data(
            symbol=request.symbol,
            exchange=request.exchange,
            interval=request.interval,
            period=request.period
        )
        
        if data is None or data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {request.symbol}")
        
        # Calculate market metrics
        market_metrics_provider = IndianMarketMetricsProvider()
        metrics = market_metrics_provider.get_basic_market_metrics(data)
        
        return {
            "symbol": request.symbol,
            "exchange": request.exchange,
            "period": request.period,
            "interval": request.interval,
            "market_metrics": metrics,
            "calculation_time": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/technical/enhanced-metrics")
async def get_enhanced_market_metrics(request: DataRequest):
    """Get enhanced market metrics with sector context."""
    try:
        # Fetch data
        zerodha_client = ZerodhaDataClient()
        if not zerodha_client.authenticate():
            raise HTTPException(status_code=401, detail="Zerodha authentication failed")
        
        data = await zerodha_client.get_historical_data(
            symbol=request.symbol,
            exchange=request.exchange,
            interval=request.interval,
            period=request.period
        )
        
        if data is None or data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {request.symbol}")
        
        # Calculate enhanced market metrics
        market_metrics_provider = IndianMarketMetricsProvider()
        metrics = market_metrics_provider.get_enhanced_market_metrics(data, request.symbol)
        
        return {
            "symbol": request.symbol,
            "exchange": request.exchange,
            "period": request.period,
            "interval": request.interval,
            "enhanced_market_metrics": metrics,
            "calculation_time": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# PATTERN RECOGNITION ENDPOINTS
# ============================================================================

@app.post("/patterns/detect")
async def detect_patterns(request: PatternRequest):
    """Detect all patterns for a stock."""
    try:
        # Fetch data
        zerodha_client = ZerodhaDataClient()
        if not zerodha_client.authenticate():
            raise HTTPException(status_code=401, detail="Zerodha authentication failed")
        
        data = await zerodha_client.get_historical_data(
            symbol=request.symbol,
            exchange=request.exchange,
            interval=request.interval,
            period=request.period
        )
        
        if data is None or data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {request.symbol}")
        
        # Detect patterns
        patterns = {}
        
        if request.pattern_types.lower() == "all" or "candlestick" in request.pattern_types.lower():
            patterns['candlestick_patterns'] = PatternRecognition.detect_candlestick_patterns(data)
        
        if request.pattern_types.lower() == "all" or "chart" in request.pattern_types.lower():
            patterns['double_tops'] = PatternRecognition.detect_double_top(data['close'])
            patterns['double_bottoms'] = PatternRecognition.detect_double_bottom(data['close'])
            patterns['head_and_shoulders'] = PatternRecognition.detect_head_and_shoulders(data['close'])
            patterns['triangles'] = PatternRecognition.detect_triangle(data['close'])
        
        if request.pattern_types.lower() == "all" or "volume" in request.pattern_types.lower():
            patterns['volume_patterns'] = PatternRecognition.detect_volume_patterns(data)
        
        return {
            "symbol": request.symbol,
            "exchange": request.exchange,
            "period": request.period,
            "interval": request.interval,
            "patterns": patterns,
            "detection_time": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/patterns/candlestick")
async def detect_candlestick_patterns(request: DataRequest):
    """Detect candlestick patterns for a stock."""
    try:
        # Fetch data
        zerodha_client = ZerodhaDataClient()
        if not zerodha_client.authenticate():
            raise HTTPException(status_code=401, detail="Zerodha authentication failed")
        
        data = await zerodha_client.get_historical_data(
            symbol=request.symbol,
            exchange=request.exchange,
            interval=request.interval,
            period=request.period
        )
        
        if data is None or data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {request.symbol}")
        
        # Detect candlestick patterns
        patterns = PatternRecognition.detect_candlestick_patterns(data)
        
        return {
            "symbol": request.symbol,
            "exchange": request.exchange,
            "period": request.period,
            "interval": request.interval,
            "candlestick_patterns": patterns,
            "detection_time": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/patterns/chart")
async def detect_chart_patterns(request: DataRequest):
    """Detect chart patterns for a stock."""
    try:
        # Fetch data
        zerodha_client = ZerodhaDataClient()
        if not zerodha_client.authenticate():
            raise HTTPException(status_code=401, detail="Zerodha authentication failed")
        
        data = await zerodha_client.get_historical_data(
            symbol=request.symbol,
            exchange=request.exchange,
            interval=request.interval,
            period=request.period
        )
        
        if data is None or data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {request.symbol}")
        
        # Detect chart patterns
        patterns = {
            'double_tops': PatternRecognition.detect_double_top(data['close']),
            'double_bottoms': PatternRecognition.detect_double_bottom(data['close']),
            'head_and_shoulders': PatternRecognition.detect_head_and_shoulders(data['close']),
            'triangles': PatternRecognition.detect_triangle(data['close'])
        }
        
        return {
            "symbol": request.symbol,
            "exchange": request.exchange,
            "period": request.period,
            "interval": request.interval,
            "chart_patterns": patterns,
            "detection_time": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/patterns/volume")
async def detect_volume_patterns(request: DataRequest):
    """Detect volume patterns for a stock."""
    try:
        # Fetch data
        zerodha_client = ZerodhaDataClient()
        if not zerodha_client.authenticate():
            raise HTTPException(status_code=401, detail="Zerodha authentication failed")
        
        data = await zerodha_client.get_historical_data(
            symbol=request.symbol,
            exchange=request.exchange,
            interval=request.interval,
            period=request.period
        )
        
        if data is None or data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {request.symbol}")
        
        # Detect volume patterns
        patterns = PatternRecognition.detect_volume_patterns(data)
        
        return {
            "symbol": request.symbol,
            "exchange": request.exchange,
            "period": request.period,
            "interval": request.interval,
            "volume_patterns": patterns,
            "detection_time": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# SECTOR ANALYSIS ENDPOINTS
# ============================================================================

@app.post("/sectors/info")
async def get_sector_info(request: SectorRequest):
    """Get sector information for a stock."""
    try:
        # Get sector information
        sector_info = {}
        
        # Try enhanced sector classifier first
        try:
            sector = enhanced_sector_classifier.get_stock_sector(request.symbol)
            if sector:
                sector_info['enhanced_sector'] = sector
                # Use the enhanced classifier's performance data helper
                sector_info['enhanced_sector_data'] = enhanced_sector_classifier.get_sector_performance_data(sector)
        except Exception:
            pass
        
        # Try regular sector classifier
        try:
            sector_classifier = SectorClassifier()
            sector = sector_classifier.get_stock_sector(request.symbol)
            if sector:
                sector_info['sector'] = sector
                # Build sector data from available classifier methods
                stocks = sector_classifier.get_sector_stocks(sector)
                sector_info['sector_data'] = {
                    "sector": sector,
                    "display_name": sector_classifier.get_sector_display_name(sector),
                    "primary_index": sector_classifier.get_primary_sector_index(sector),
                    "stocks": stocks,
                    "stock_count": len(stocks)
                }
        except Exception:
            pass
        
        # Get all available sectors
        try:
            sector_classifier = SectorClassifier()
            sector_info['all_sectors'] = sector_classifier.get_all_sectors()
        except Exception:
            sector_info['all_sectors'] = []
        
        return {
            "symbol": request.symbol,
            "sector_info": sector_info,
            "request_time": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/sectors/benchmarking")
async def get_sector_benchmarking(request: DataRequest):
    """Get sector benchmarking for a stock."""
    try:
        # Fetch data
        zerodha_client = ZerodhaDataClient()
        if not zerodha_client.authenticate():
            raise HTTPException(status_code=401, detail="Zerodha authentication failed")
        
        data = await zerodha_client.get_historical_data(
            symbol=request.symbol,
            exchange=request.exchange,
            interval=request.interval,
            period=request.period
        )
        
        if data is None or data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {request.symbol}")
        
        # Get sector benchmarking
        sector_benchmarking_provider = SectorBenchmarkingProvider()
        benchmarking = await sector_benchmarking_provider.get_comprehensive_benchmarking_async(
            request.symbol, data
        )
        
        return {
            "symbol": request.symbol,
            "exchange": request.exchange,
            "period": request.period,
            "interval": request.interval,
            "sector_benchmarking": benchmarking,
            "calculation_time": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sectors/rotation")
async def get_sector_rotation(period: str = "1M"):
    """Get sector rotation analysis."""
    try:
        sector_benchmarking_provider = SectorBenchmarkingProvider()
        rotation = await sector_benchmarking_provider.analyze_sector_rotation_async(period)
        
        return {
            "period": period,
            "sector_rotation": rotation,
            "calculation_time": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sectors/correlation")
async def get_sector_correlation(period: str = "3M"):
    """Get sector correlation matrix."""
    try:
        sector_benchmarking_provider = SectorBenchmarkingProvider()
        correlation = await sector_benchmarking_provider.generate_sector_correlation_matrix_async(period)
        
        return {
            "period": period,
            "sector_correlation": correlation,
            "calculation_time": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sectors/performance")
async def get_sector_performance(period: str = "1M"):
    """Get sector performance metrics."""
    try:
        sector_benchmarking_provider = SectorBenchmarkingProvider()
        performance = await sector_benchmarking_provider.get_sector_performance_async(period)
        
        return {
            "period": period,
            "sector_performance": performance,
            "calculation_time": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# ANALYSIS ENDPOINTS
# ============================================================================

@app.post("/analysis/full")
async def full_analysis(request: DataRequest):
    """Perform full analysis for a stock."""
    try:
        orchestrator = StockAnalysisOrchestrator()
        if not orchestrator.authenticate():
            raise HTTPException(status_code=401, detail="Zerodha authentication failed")
        
        # Perform analysis
        result = await orchestrator.enhanced_analyze_stock(
            symbol=request.symbol,
            exchange=request.exchange,
            period=request.period,
            interval=request.interval,
            sector=request.sector if request.sector else None
        )
        
        analysis_results, success_message, error_message = result
        
        if error_message:
            raise HTTPException(status_code=500, detail=error_message)
        
        return {
            "symbol": request.symbol,
            "exchange": request.exchange,
            "period": request.period,
            "interval": request.interval,
            "analysis_results": analysis_results,
            "success_message": success_message,
            "analysis_time": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analysis/enhanced")
async def enhanced_analysis(request: DataRequest):
    """Perform enhanced analysis with code execution."""
    try:
        orchestrator = StockAnalysisOrchestrator()
        if not orchestrator.authenticate():
            raise HTTPException(status_code=401, detail="Zerodha authentication failed")
        
        # Perform enhanced analysis
        result = await orchestrator.enhanced_analyze_stock(
            symbol=request.symbol,
            exchange=request.exchange,
            period=request.period,
            interval=request.interval,
            sector=request.sector if request.sector else None
        )
        
        analysis_results, success_message, error_message = result
        
        if error_message:
            raise HTTPException(status_code=500, detail=error_message)
        
        return {
            "symbol": request.symbol,
            "exchange": request.exchange,
            "period": request.period,
            "interval": request.interval,
            "enhanced_analysis_results": analysis_results,
            "success_message": success_message,
            "analysis_time": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analysis/risk")
async def risk_assessment(request: DataRequest):
    """Perform risk assessment for a stock."""
    try:
        # Fetch data
        zerodha_client = ZerodhaDataClient()
        if not zerodha_client.authenticate():
            raise HTTPException(status_code=401, detail="Zerodha authentication failed")
        
        data = await zerodha_client.get_historical_data(
            symbol=request.symbol,
            exchange=request.exchange,
            interval=request.interval,
            period=request.period
        )
        
        if data is None or data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {request.symbol}")
        
        # Calculate basic risk metrics using available functions
        returns = data['close'].pct_change().dropna()
        volatility = returns.std()
        sharpe_ratio = returns.mean() / returns.std() if returns.std() > 0 else 0
        
        risk_metrics = {
            "volatility": float(volatility),
            "sharpe_ratio": float(sharpe_ratio),
            "max_drawdown": float((data['close'] / data['close'].expanding().max() - 1).min()),
            "var_95": float(returns.quantile(0.05))
        }
        
        return {
            "symbol": request.symbol,
            "exchange": request.exchange,
            "period": request.period,
            "interval": request.interval,
            "risk_metrics": risk_metrics,
            "assessment_time": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analysis/backtest")
async def backtest_strategy(request: DataRequest):
    """Perform backtesting for a stock."""
    try:
        # Fetch data
        zerodha_client = ZerodhaDataClient()
        if not zerodha_client.authenticate():
            raise HTTPException(status_code=401, detail="Zerodha authentication failed")
        
        data = await zerodha_client.get_historical_data(
            symbol=request.symbol,
            exchange=request.exchange,
            interval=request.interval,
            period=request.period
        )
        
        if data is None or data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {request.symbol}")
        
        # Simple backtest simulation
        returns = data['close'].pct_change().dropna()
        cumulative_return = (1 + returns).prod() - 1
        
        backtest_results = {
            "total_return": float(cumulative_return),
            "annualized_return": float((1 + cumulative_return) ** (252 / len(returns)) - 1),
            "volatility": float(returns.std() * (252 ** 0.5)),
            "sharpe_ratio": float(returns.mean() / returns.std() * (252 ** 0.5) if returns.std() > 0 else 0),
            "max_drawdown": float((data['close'] / data['close'].expanding().max() - 1).min())
        }
        
        return {
            "symbol": request.symbol,
            "exchange": request.exchange,
            "period": request.period,
            "interval": request.interval,
            "backtest_results": backtest_results,
            "backtest_time": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Start the service endpoints server."""
    print("🚀 Starting Service Endpoints Server...")
    print("📍 Service: Individual component testing endpoints")
    print("🌐 Port: 8002")
    print("📊 Health: http://localhost:8002/health")
    print("📋 Status: http://localhost:8002/status")
    print("-" * 50)
    
    # Configuration
    host = os.getenv("SERVICE_ENDPOINTS_HOST", "0.0.0.0")
    port = int(os.getenv("SERVICE_ENDPOINTS_PORT", 8002))
    reload = os.getenv("SERVICE_ENDPOINTS_RELOAD", "false").lower() == "true"
    
    # Start the service
    uvicorn.run(
        "service_endpoints:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
        access_log=True
    )

if __name__ == "__main__":
    main()
