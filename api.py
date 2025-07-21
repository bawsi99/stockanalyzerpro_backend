# api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
import os
from agent_capabilities import StockAnalysisOrchestrator
import traceback
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import json
import numpy as np
import io
import base64
from pandas import Timestamp
import math
from fastapi.responses import JSONResponse

# Import sector components
from sector_benchmarking import sector_benchmarking_provider
from sector_classifier import sector_classifier
from enhanced_sector_classifier import enhanced_sector_classifier

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            if np.isinf(obj) or np.isnan(obj):
                return None
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, Timestamp):
            return obj.isoformat()
        return super().default(obj)

def make_json_serializable(obj):
    """Recursively convert objects to JSON serializable format."""
    if isinstance(obj, (str, int, type(None))):
        return obj
    elif isinstance(obj, bool):
        return bool(obj)  # Ensure it's a Python bool
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        if np.isinf(obj) or np.isnan(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)  # Convert NumPy boolean to Python boolean
    elif isinstance(obj, Timestamp):
        return obj.isoformat()
    else:
        return str(obj)

def convert_charts_to_base64(charts_dict: dict) -> dict:
    """Convert chart file paths to base64 encoded images."""
    converted_charts = {}
    
    for chart_name, chart_path in charts_dict.items():
        if isinstance(chart_path, str) and os.path.exists(chart_path):
            try:
                with open(chart_path, 'rb') as f:
                    img_data = f.read()
                    img_base64 = base64.b64encode(img_data).decode('utf-8')
                    converted_charts[chart_name] = {
                        'data': f"data:image/png;base64,{img_base64}",
                        'filename': os.path.basename(chart_path),
                        'type': 'image/png'
                    }
            except Exception as e:
                print(f"Error converting chart {chart_name}: {e}")
                converted_charts[chart_name] = {
                    'error': f"Failed to load chart: {str(e)}",
                    'filename': os.path.basename(chart_path) if isinstance(chart_path, str) else 'unknown'
                }
        else:
            converted_charts[chart_name] = {
                'error': 'Chart file not found',
                'path': chart_path
            }
    
    return converted_charts

def validate_analysis_results(results: dict) -> dict:
    """Validate and ensure all required fields are present in analysis results."""
    required_fields = {
        'ai_analysis': {},
        'indicators': {},
        'overlays': {},
        'indicator_summary_md': '',
        'chart_insights': '',
        'summary': {},
        'trading_guidance': {},
        'sector_benchmarking': {},
        'metadata': {}
    }
    
    validated_results = {}
    
    for field, default_value in required_fields.items():
        if field in results and results[field] is not None:
            validated_results[field] = results[field]
        else:
            validated_results[field] = default_value
            print(f"Warning: Missing or null field '{field}' in analysis results")
    
    return validated_results

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Add your frontend URL
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class AnalysisRequest(BaseModel):
    stock: str = Field(..., description="Stock symbol to analyze")
    exchange: str = Field(default="NSE", description="Stock exchange")
    period: int = Field(default=365, description="Analysis period in days")
    interval: str = Field(default="day", description="Data interval")
    output: Optional[str] = Field(default=None, description="Output directory")
    sector: Optional[str] = Field(default=None, description="Optional sector override")

class SectorAnalysisRequest(BaseModel):
    sector: str = Field(..., description="Sector to analyze")
    period: int = Field(default=365, description="Analysis period in days")

class SectorComparisonRequest(BaseModel):
    sectors: List[str] = Field(..., description="List of sectors to compare")
    period: int = Field(default=365, description="Analysis period in days")

@app.post("/analyze")
async def analyze(request: AnalysisRequest):
    output_dir = request.output or f"./output/{request.stock}"
    os.makedirs(output_dir, exist_ok=True)

    try:
        orchestrator = StockAnalysisOrchestrator()
        # Authenticate
        auth_success = orchestrator.authenticate()
        if not auth_success:
            raise HTTPException(status_code=401, detail="Failed to authenticate with Zerodha API")

        # Analyze stock with sector awareness
        results, success_message, error_message = await orchestrator.analyze_stock(
            symbol=request.stock,
            exchange=request.exchange,
            period=request.period,
            interval=request.interval,
            output_dir=output_dir,
            sector=request.sector  # Pass sector information
        )
        
        if error_message:
            raise HTTPException(status_code=500, detail=error_message)

        # Make all data JSON serializable
        serialized_results = make_json_serializable(results)

        # Clean, efficient response
        response = {
            "success": True,
            "stock_symbol": request.stock,
            "exchange": request.exchange,
            "analysis_period": f"{request.period} days",
            "interval": request.interval,
            "timestamp": pd.Timestamp.now().isoformat(),
            "message": success_message,
            "results": serialized_results
        }

        return JSONResponse(content=response)

    except Exception as e:
        traceback.print_exc()
        error_response = {
            "success": False,
            "error": str(e),
            "stock_symbol": request.stock,
            "exchange": request.exchange,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        raise HTTPException(status_code=500, detail=error_response)

@app.post("/sector/benchmark")
async def sector_benchmark(request: AnalysisRequest):
    """Get sector benchmarking for a specific stock."""
    try:
        # Get stock data
        orchestrator = StockAnalysisOrchestrator()
        if not orchestrator.authenticate():
            raise HTTPException(status_code=401, detail="Authentication failed")
        
        data = orchestrator.retrieve_stock_data(
            symbol=request.stock,
            exchange=request.exchange,
            period=request.period,
            interval=request.interval
        )
        
        if data is None:
            raise HTTPException(status_code=404, detail="Stock data not found")
        
        # Get comprehensive sector benchmarking
        benchmarking = sector_benchmarking_provider.get_comprehensive_benchmarking(request.stock, data)
        
        # Make JSON serializable
        serialized_benchmarking = make_json_serializable(benchmarking)
        
        response = {
            "success": True,
            "stock_symbol": request.stock,
            "sector_benchmarking": serialized_benchmarking,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
        return JSONResponse(content=response)
        
    except Exception as e:
        traceback.print_exc()
        error_response = {
            "success": False,
            "error": str(e),
            "stock_symbol": request.stock,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        raise HTTPException(status_code=500, detail=error_response)

@app.get("/sector/list")
async def get_sectors():
    """Get list of all available sectors."""
    try:
        sectors = sector_classifier.get_all_sectors()
        
        response = {
            "success": True,
            "sectors": sectors,
            "total_sectors": len(sectors),
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
        return JSONResponse(content=response)
        
    except Exception as e:
        traceback.print_exc()
        error_response = {
            "success": False,
            "error": str(e),
            "timestamp": pd.Timestamp.now().isoformat()
        }
        raise HTTPException(status_code=500, detail=error_response)

@app.get("/sector/{sector_name}/stocks")
async def get_sector_stocks(sector_name: str):
    """Get all stocks in a specific sector."""
    try:
        stocks = sector_classifier.get_sector_stocks(sector_name)
        sector_info = {
            "sector": sector_name,
            "display_name": sector_classifier.get_sector_display_name(sector_name),
            "primary_index": sector_classifier.get_primary_sector_index(sector_name),
            "stock_count": len(stocks)
        }
        
        response = {
            "success": True,
            "sector_info": sector_info,
            "stocks": stocks,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
        return JSONResponse(content=response)
        
    except Exception as e:
        traceback.print_exc()
        error_response = {
            "success": False,
            "error": str(e),
            "sector": sector_name,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        raise HTTPException(status_code=500, detail=error_response)

@app.get("/sector/{sector_name}/performance")
async def get_sector_performance(sector_name: str, period: int = 365):
    """Get sector performance data."""
    try:
        # Get sector index data
        sector_index = sector_classifier.get_primary_sector_index(sector_name)
        if not sector_index:
            raise HTTPException(status_code=404, detail=f"No index found for sector: {sector_name}")
        
        # Get sector data
        orchestrator = StockAnalysisOrchestrator()
        if not orchestrator.authenticate():
            raise HTTPException(status_code=401, detail="Authentication failed")
        
        sector_data = orchestrator.retrieve_stock_data(
            symbol=sector_index,
            exchange="NSE",
            period=period,
            interval="day"
        )
        
        if sector_data is None:
            raise HTTPException(status_code=404, detail="Sector data not found")
        
        # Calculate sector performance metrics
        sector_returns = sector_data['close'].pct_change().dropna()
        cumulative_return = (1 + sector_returns).prod() - 1
        volatility = sector_returns.std() * np.sqrt(252)
        
        # Get sector stocks
        sector_stocks = sector_classifier.get_sector_stocks(sector_name)
        
        performance_data = {
            "sector": sector_name,
            "sector_index": sector_index,
            "display_name": sector_classifier.get_sector_display_name(sector_name),
            "period_days": period,
            "cumulative_return": float(cumulative_return),
            "annualized_volatility": float(volatility),
            "stock_count": len(sector_stocks),
            "data_points": len(sector_data),
            "last_price": float(sector_data['close'].iloc[-1]),
            "last_date": sector_data.index[-1].isoformat()
        }
        
        response = {
            "success": True,
            "sector_performance": performance_data,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
        return JSONResponse(content=response)
        
    except Exception as e:
        traceback.print_exc()
        error_response = {
            "success": False,
            "error": str(e),
            "sector": sector_name,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        raise HTTPException(status_code=500, detail=error_response)

@app.post("/sector/compare")
async def compare_sectors(request: SectorComparisonRequest):
    """Compare multiple sectors."""
    try:
        comparison_data = {}
        
        for sector in request.sectors:
            try:
                # Get sector performance
                sector_index = sector_classifier.get_primary_sector_index(sector)
                if sector_index:
                    orchestrator = StockAnalysisOrchestrator()
                    if not orchestrator.authenticate():
                        raise HTTPException(status_code=401, detail="Authentication failed")
                    
                    sector_data = orchestrator.retrieve_stock_data(
                        symbol=sector_index,
                        exchange="NSE",
                        period=request.period,
                        interval="day"
                    )
                    
                    if sector_data is not None:
                        sector_returns = sector_data['close'].pct_change().dropna()
                        cumulative_return = (1 + sector_returns).prod() - 1
                        volatility = sector_returns.std() * np.sqrt(252)
                        
                        comparison_data[sector] = {
                            "sector": sector,
                            "display_name": sector_classifier.get_sector_display_name(sector),
                            "sector_index": sector_index,
                            "cumulative_return": float(cumulative_return),
                            "annualized_volatility": float(volatility),
                            "stock_count": len(sector_classifier.get_sector_stocks(sector)),
                            "last_price": float(sector_data['close'].iloc[-1])
                        }
                    else:
                        comparison_data[sector] = {
                            "sector": sector,
                            "error": "Data not available"
                        }
                else:
                    comparison_data[sector] = {
                        "sector": sector,
                        "error": "Index not found"
                    }
                    
            except Exception as e:
                comparison_data[sector] = {
                    "sector": sector,
                    "error": str(e)
                }
        
        response = {
            "success": True,
            "sector_comparison": comparison_data,
            "period_days": request.period,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
        return JSONResponse(content=response)
        
    except Exception as e:
        traceback.print_exc()
        error_response = {
            "success": False,
            "error": str(e),
            "sectors": request.sectors,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        raise HTTPException(status_code=500, detail=error_response)

@app.get("/stock/{symbol}/sector")
async def get_stock_sector(symbol: str):
    """Get sector information for a specific stock."""
    try:
        sector = sector_classifier.get_stock_sector(symbol)
        
        if sector:
            sector_info = {
                "stock_symbol": symbol,
                "sector": sector,
                "sector_name": sector_classifier.get_sector_display_name(sector),
                "sector_index": sector_classifier.get_primary_sector_index(sector),
                "sector_stocks": sector_classifier.get_sector_stocks(sector),
                "sector_stock_count": len(sector_classifier.get_sector_stocks(sector))
            }
        else:
            sector_info = {
                "stock_symbol": symbol,
                "sector": None,
                "sector_name": None,
                "sector_index": None,
                "sector_stocks": [],
                "sector_stock_count": 0,
                "note": "Stock not classified in any sector"
            }
        
        response = {
            "success": True,
            "sector_info": sector_info,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
        return JSONResponse(content=response)
        
    except Exception as e:
        traceback.print_exc()
        error_response = {
            "success": False,
            "error": str(e),
            "stock_symbol": symbol,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        raise HTTPException(status_code=500, detail=error_response)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": pd.Timestamp.now().isoformat()}

@app.get("/stock/{symbol}/info")
async def get_stock_info(symbol: str, exchange: str = "NSE"):
    """Get basic stock information."""
    try:
        orchestrator = StockAnalysisOrchestrator()
        
        if not orchestrator.authenticate():
            raise HTTPException(status_code=401, detail="Authentication failed")
        
        # Get current quote
        quote = orchestrator.data_client.get_quote(symbol, exchange)
        
        if quote:
            return {
                "success": True,
                "stock_symbol": symbol,
                "exchange": exchange,
                "quote": quote,
                "timestamp": pd.Timestamp.now().isoformat()
            }
        else:
            raise HTTPException(status_code=404, detail="Stock quote not found")
            
    except Exception as e:
        traceback.print_exc()
        error_response = {
            "success": False,
            "error": str(e),
            "stock_symbol": symbol,
            "exchange": exchange,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        raise HTTPException(status_code=500, detail=error_response)