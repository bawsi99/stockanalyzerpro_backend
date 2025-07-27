#!/usr/bin/env python3
"""
websocket_stream_service.py

Dedicated WebSocket streaming service for receiving connections and streaming data.
This service runs on port 8081 and is configured via environment variables.
"""

import os
import time
import json
import asyncio
import traceback
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# FastAPI imports
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Local imports
from zerodha_ws_client import zerodha_ws_client, candle_aggregator
from enhanced_data_service import enhanced_data_service, DataRequest

# Environment configuration
WEBSOCKET_HOST = os.getenv("HOST", "0.0.0.0")
WEBSOCKET_PORT = int(os.getenv("PORT", 8000)) + 81  # Use port 8081 (8000 + 81)
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:8080,http://localhost:8081").split(",")

# Multi-endpoint configuration
WEBSOCKET_ENDPOINTS = {
    "main": {
        "path": "/ws/stream",
        "description": "Main real-time data stream",
        "data_types": ["tick_data", "candle_data", "market_data"]
    },
    "analysis": {
        "path": "/ws/analysis",
        "description": "Real-time analysis and insights",
        "data_types": ["technical_analysis", "pattern_recognition", "risk_metrics"]
    },
    "alerts": {
        "path": "/ws/alerts",
        "description": "Price alerts and notifications",
        "data_types": ["price_alerts", "volume_alerts", "breakout_alerts"]
    },
    "portfolio": {
        "path": "/ws/portfolio",
        "description": "Portfolio tracking and performance",
        "data_types": ["portfolio_updates", "pnl_data", "position_data"]
    }
}

# Create FastAPI app
app = FastAPI(
    title="WebSocket Stream Service",
    description="Dedicated WebSocket service for real-time data streaming",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Connection management
class ConnectionManager:
    def __init__(self):
        self.endpoint_connections: Dict[str, List[WebSocket]] = {}
        self.connection_data: Dict[WebSocket, Dict] = {}
        
    async def connect(self, websocket: WebSocket, endpoint: str):
        await websocket.accept()
        
        if endpoint not in self.endpoint_connections:
            self.endpoint_connections[endpoint] = []
            
        self.endpoint_connections[endpoint].append(websocket)
        self.connection_data[websocket] = {
            "endpoint": endpoint,
            "connected_at": datetime.now(),
            "subscriptions": set(),
            "last_heartbeat": time.time()
        }
        print(f"✅ New WebSocket connection established on {endpoint}. Total connections: {self.get_total_connections()}")
        
    def disconnect(self, websocket: WebSocket):
        endpoint = self.connection_data.get(websocket, {}).get("endpoint")
        if endpoint and endpoint in self.endpoint_connections:
            if websocket in self.endpoint_connections[endpoint]:
                self.endpoint_connections[endpoint].remove(websocket)
                
        if websocket in self.connection_data:
            del self.connection_data[websocket]
        print(f"❌ WebSocket connection closed. Total connections: {self.get_total_connections()}")
        
    def get_total_connections(self) -> int:
        return sum(len(connections) for connections in self.endpoint_connections.values())
        
    def get_endpoint_connections(self, endpoint: str) -> List[WebSocket]:
        return self.endpoint_connections.get(endpoint, [])
        
    async def send_personal_message(self, message: str, websocket: WebSocket):
        try:
            await websocket.send_text(message)
        except Exception as e:
            print(f"Error sending message to client: {e}")
            self.disconnect(websocket)
            
    async def broadcast_to_endpoint(self, message: str, endpoint: str):
        """Broadcast message to all connections on a specific endpoint."""
        if endpoint not in self.endpoint_connections:
            return
            
        disconnected = []
        for connection in self.endpoint_connections[endpoint]:
            try:
                await connection.send_text(message)
            except Exception as e:
                print(f"Error broadcasting to client on {endpoint}: {e}")
                disconnected.append(connection)
                
        # Clean up disconnected connections
        for connection in disconnected:
            self.disconnect(connection)
            
    async def broadcast(self, message: str):
        """Broadcast message to all connections across all endpoints."""
        for endpoint in self.endpoint_connections:
            await self.broadcast_to_endpoint(message, endpoint)
            
    async def send_json(self, data: Dict, websocket: WebSocket):
        try:
            from utils import safe_json_dumps
            await websocket.send_text(safe_json_dumps(data))
        except Exception as e:
            print(f"Error sending JSON to client: {e}")
            self.disconnect(websocket)
            
    async def broadcast_json_to_endpoint(self, data: Dict, endpoint: str):
        """Broadcast JSON data to all connections on a specific endpoint."""
        from utils import safe_json_dumps
        message = safe_json_dumps(data)
        await self.broadcast_to_endpoint(message, endpoint)
            
    async def broadcast_json(self, data: Dict):
        """Broadcast JSON data to all connections across all endpoints."""
        from utils import safe_json_dumps
        message = safe_json_dumps(data)
        await self.broadcast(message)

# Global connection manager
manager = ConnectionManager()

# Data streaming manager
class DataStreamManager:
    def __init__(self):
        self.subscribers: Dict[str, Dict[str, List[WebSocket]]] = {}  # symbol -> endpoint -> websockets
        self.streaming_tasks: Dict[str, asyncio.Task] = {}
        self.endpoint_data_types: Dict[str, List[str]] = WEBSOCKET_ENDPOINTS
        
    async def subscribe(self, symbol: str, websocket: WebSocket, endpoint: str, data_types: List[str] = None):
        """Subscribe a WebSocket connection to a symbol's data stream on a specific endpoint."""
        if symbol not in self.subscribers:
            self.subscribers[symbol] = {}
            
        if endpoint not in self.subscribers[symbol]:
            self.subscribers[symbol][endpoint] = []
            
        if websocket not in self.subscribers[symbol][endpoint]:
            self.subscribers[symbol][endpoint].append(websocket)
            print(f"📊 Subscribed {symbol} to {endpoint} endpoint")
            
        # Start streaming if not already running
        if symbol not in self.streaming_tasks or self.streaming_tasks[symbol].done():
            self.streaming_tasks[symbol] = asyncio.create_task(self.stream_data(symbol))
            
    async def unsubscribe(self, symbol: str, websocket: WebSocket, endpoint: str = None):
        """Unsubscribe a WebSocket connection from a symbol's data stream."""
        if symbol not in self.subscribers:
            return
            
        if endpoint:
            # Unsubscribe from specific endpoint
            if endpoint in self.subscribers[symbol] and websocket in self.subscribers[symbol][endpoint]:
                self.subscribers[symbol][endpoint].remove(websocket)
                print(f"📊 Unsubscribed {symbol} from {endpoint} endpoint")
        else:
            # Unsubscribe from all endpoints
            for ep in self.subscribers[symbol]:
                if websocket in self.subscribers[symbol][ep]:
                    self.subscribers[symbol][ep].remove(websocket)
                    print(f"📊 Unsubscribed {symbol} from {ep} endpoint")
            
        # Stop streaming if no subscribers on any endpoint
        total_subscribers = sum(len(websockets) for websockets in self.subscribers[symbol].values())
        if total_subscribers == 0:
            if symbol in self.streaming_tasks:
                self.streaming_tasks[symbol].cancel()
                del self.streaming_tasks[symbol]
                
    async def stream_data(self, symbol: str):
        """Stream real-time data for a symbol to all subscribers across different endpoints."""
        try:
            print(f"🚀 Starting data stream for {symbol}")
            
            # Get symbol token
            token = await self.get_symbol_token(symbol)
            if not token:
                print(f"❌ Could not find token for symbol: {symbol}")
                return
                
            # Subscribe to real-time data
            if hasattr(zerodha_ws_client, 'subscribe') and callable(zerodha_ws_client.subscribe):
                await zerodha_ws_client.subscribe([token])
                
                # Set up data forwarding
                def data_callback(data):
                    asyncio.create_task(self.forward_data(symbol, data))
                    
                # Register callback
                if hasattr(zerodha_ws_client, 'set_tick_callback') and callable(zerodha_ws_client.set_tick_callback):
                    zerodha_ws_client.set_tick_callback(data_callback)
                else:
                    print(f"⚠️  Zerodha WebSocket client set_tick_callback method not available")
            else:
                print(f"⚠️  Zerodha WebSocket client subscribe method not available")
                # Send mock data for testing
                await self.send_mock_data(symbol)
            
            # Keep streaming alive
            while symbol in self.subscribers and any(len(websockets) > 0 for websockets in self.subscribers[symbol].values()):
                await asyncio.sleep(1)
                
        except Exception as e:
            print(f"❌ Error in data stream for {symbol}: {e}")
            traceback.print_exc()
            
    async def forward_data(self, symbol: str, data: Dict):
        """Forward data to all subscribers of a symbol across different endpoints with endpoint-specific data."""
        if symbol not in self.subscribers:
            return
            
        # Send to each endpoint with appropriate data format
        for endpoint, websockets in self.subscribers[symbol].items():
            if not websockets:
                continue
                
            # Create endpoint-specific data
            endpoint_data = self.create_endpoint_data(endpoint, symbol, data)
            
            message = {
                "type": "data",
                "symbol": symbol,
                "endpoint": endpoint,
                "timestamp": datetime.now().isoformat(),
                "data": endpoint_data
            }
            
            # Send to all subscribers on this endpoint
            disconnected = []
            for websocket in websockets:
                try:
                    await manager.send_json(message, websocket)
                except Exception as e:
                    print(f"Error forwarding data to client on {endpoint}: {e}")
                    disconnected.append(websocket)
                    
            # Clean up disconnected subscribers
            for websocket in disconnected:
                await self.unsubscribe(symbol, websocket, endpoint)
                
    def create_endpoint_data(self, endpoint: str, symbol: str, base_data: Dict) -> Dict:
        """Create endpoint-specific data based on the endpoint type."""
        if endpoint == "main":
            return {
                "price": base_data.get("price", 0),
                "volume": base_data.get("volume", 0),
                "change": base_data.get("change", 0),
                "change_percent": base_data.get("change_percent", 0),
                "high": base_data.get("high", 0),
                "low": base_data.get("low", 0),
                "open": base_data.get("open", 0),
                "close": base_data.get("close", 0)
            }
        elif endpoint == "analysis":
            return {
                "technical_indicators": {
                    "rsi": round(base_data.get("price", 100) * 0.7, 2),
                    "macd": round(base_data.get("price", 100) * 0.1, 2),
                    "bollinger_bands": {
                        "upper": round(base_data.get("price", 100) * 1.1, 2),
                        "lower": round(base_data.get("price", 100) * 0.9, 2)
                    }
                },
                "patterns": ["support_level", "resistance_level"],
                "risk_metrics": {
                    "volatility": round(base_data.get("change_percent", 0) * 0.5, 2),
                    "beta": 1.2
                }
            }
        elif endpoint == "alerts":
            return {
                "price_alerts": [
                    {
                        "type": "price_above",
                        "threshold": base_data.get("price", 100) * 1.05,
                        "triggered": base_data.get("price", 100) > base_data.get("price", 100) * 1.05
                    }
                ],
                "volume_alerts": [
                    {
                        "type": "volume_spike",
                        "threshold": base_data.get("volume", 1000) * 2,
                        "triggered": base_data.get("volume", 1000) > base_data.get("volume", 1000) * 2
                    }
                ]
            }
        elif endpoint == "portfolio":
            return {
                "position": {
                    "quantity": 100,
                    "avg_price": base_data.get("price", 100) * 0.95,
                    "current_value": base_data.get("price", 100) * 100
                },
                "pnl": {
                    "unrealized": round((base_data.get("price", 100) - base_data.get("price", 100) * 0.95) * 100, 2),
                    "unrealized_percent": round(((base_data.get("price", 100) / (base_data.get("price", 100) * 0.95)) - 1) * 100, 2)
                }
            }
        else:
            return base_data
            
    async def get_symbol_token(self, symbol: str) -> Optional[int]:
        """Get the token for a symbol."""
        try:
            # This would typically query your instrument database
            # For now, we'll use a simple mapping or query the enhanced_data_service
            if hasattr(enhanced_data_service, 'get_symbol_token') and callable(enhanced_data_service.get_symbol_token):
                return await enhanced_data_service.get_symbol_token(symbol)
            else:
                # Return a mock token for testing
                return 12345
        except Exception as e:
            print(f"Error getting token for {symbol}: {e}")
            return 12345  # Mock token for testing
            
    async def send_mock_data(self, symbol: str):
        """Send mock data for testing purposes."""
        import random
        import time
        
        while symbol in self.subscribers and any(len(websockets) > 0 for websockets in self.subscribers[symbol].values()):
            # Generate mock data
            mock_data = {
                "price": round(random.uniform(100, 5000), 2),
                "volume": random.randint(1000, 1000000),
                "change": round(random.uniform(-100, 100), 2),
                "change_percent": round(random.uniform(-5, 5), 2),
                "high": round(random.uniform(100, 5000), 2),
                "low": round(random.uniform(100, 5000), 2),
                "open": round(random.uniform(100, 5000), 2),
                "close": round(random.uniform(100, 5000), 2)
            }
            
            await self.forward_data(symbol, mock_data)
            await asyncio.sleep(2)  # Send data every 2 seconds

# Global data stream manager
stream_manager = DataStreamManager()

# Pydantic models
class SubscriptionRequest(BaseModel):
    symbol: str = Field(..., description="Stock symbol to subscribe to")
    action: str = Field(..., description="Action: 'subscribe' or 'unsubscribe'")

class HeartbeatRequest(BaseModel):
    timestamp: float = Field(..., description="Client timestamp")

# WebSocket endpoint handlers
async def handle_websocket_connection(websocket: WebSocket, endpoint: str):
    """Generic WebSocket connection handler for different endpoints."""
    # First, validate the origin
    origin = websocket.headers.get('origin')
    if origin and origin not in CORS_ORIGINS:
        print(f"❌ WebSocket connection rejected from unauthorized origin: {origin}")
        print(f"   Allowed origins: {CORS_ORIGINS}")
        await websocket.close(code=4001, reason="Unauthorized origin")
        return
    
    await manager.connect(websocket, endpoint)
    
    try:
        # Send welcome message
        welcome_message = {
            "type": "connection",
            "status": "connected",
            "endpoint": endpoint,
            "description": WEBSOCKET_ENDPOINTS[endpoint]["description"],
            "data_types": WEBSOCKET_ENDPOINTS[endpoint]["data_types"],
            "timestamp": datetime.now().isoformat(),
            "message": f"WebSocket connection established successfully on {endpoint} endpoint"
        }
        await manager.send_json(welcome_message, websocket)
        
        # Handle incoming messages
        while True:
            try:
                # Receive message from client
                data = await websocket.receive_text()
                message = json.loads(data)
                
                # Handle different message types
                if message.get("type") == "subscribe":
                    symbol = message.get("symbol")
                    data_types = message.get("data_types", WEBSOCKET_ENDPOINTS[endpoint]["data_types"])
                    if symbol:
                        await stream_manager.subscribe(symbol, websocket, endpoint, data_types)
                        await manager.send_json({
                            "type": "subscription",
                            "status": "subscribed",
                            "symbol": symbol,
                            "endpoint": endpoint,
                            "data_types": data_types,
                            "timestamp": datetime.now().isoformat()
                        }, websocket)
                        
                elif message.get("type") == "unsubscribe":
                    symbol = message.get("symbol")
                    if symbol:
                        await stream_manager.unsubscribe(symbol, websocket, endpoint)
                        await manager.send_json({
                            "type": "subscription",
                            "status": "unsubscribed",
                            "symbol": symbol,
                            "endpoint": endpoint,
                            "timestamp": datetime.now().isoformat()
                        }, websocket)
                        
                elif message.get("type") == "heartbeat":
                    # Update last heartbeat
                    if websocket in manager.connection_data:
                        manager.connection_data[websocket]["last_heartbeat"] = time.time()
                    await manager.send_json({
                        "type": "heartbeat",
                        "status": "ack",
                        "endpoint": endpoint,
                        "timestamp": datetime.now().isoformat()
                    }, websocket)
                    
                elif message.get("type") == "ping":
                    await manager.send_json({
                        "type": "pong",
                        "endpoint": endpoint,
                        "timestamp": datetime.now().isoformat()
                    }, websocket)
                    
            except json.JSONDecodeError:
                await manager.send_json({
                    "type": "error",
                    "message": "Invalid JSON format",
                    "endpoint": endpoint,
                    "timestamp": datetime.now().isoformat()
                }, websocket)
                
    except WebSocketDisconnect:
        print(f"WebSocket client disconnected from {endpoint} endpoint")
    except Exception as e:
        print(f"WebSocket error on {endpoint} endpoint: {e}")
        traceback.print_exc()
    finally:
        # Clean up subscriptions
        if websocket in manager.connection_data:
            endpoint_name = manager.connection_data[websocket].get("endpoint")
            subscriptions = manager.connection_data[websocket].get("subscriptions", set())
            for symbol in subscriptions:
                await stream_manager.unsubscribe(symbol, websocket, endpoint_name)
        manager.disconnect(websocket)

# Main data stream endpoint
@app.websocket("/ws/stream")
async def main_websocket_endpoint(websocket: WebSocket):
    """Main WebSocket endpoint for real-time data streaming."""
    await handle_websocket_connection(websocket, "main")

# Analysis endpoint
@app.websocket("/ws/analysis")
async def analysis_websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time analysis and insights."""
    await handle_websocket_connection(websocket, "analysis")

# Alerts endpoint
@app.websocket("/ws/alerts")
async def alerts_websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for price alerts and notifications."""
    await handle_websocket_connection(websocket, "alerts")

# Portfolio endpoint
@app.websocket("/ws/portfolio")
async def portfolio_websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for portfolio tracking and performance."""
    await handle_websocket_connection(websocket, "portfolio")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    endpoint_stats = {}
    for endpoint in WEBSOCKET_ENDPOINTS:
        connections = manager.get_endpoint_connections(endpoint)
        endpoint_stats[endpoint] = {
            "connections": len(connections),
            "path": WEBSOCKET_ENDPOINTS[endpoint]["path"],
            "description": WEBSOCKET_ENDPOINTS[endpoint]["description"]
        }
    
    return {
        "status": "healthy",
        "service": "websocket_stream_service",
        "timestamp": datetime.now().isoformat(),
        "total_connections": manager.get_total_connections(),
        "active_streams": len(stream_manager.streaming_tasks),
        "endpoints": endpoint_stats
    }

# Connection status endpoint
@app.get("/connections")
async def get_connections():
    """Get information about active connections."""
    endpoint_connections = {}
    for endpoint in WEBSOCKET_ENDPOINTS:
        connections = []
        for websocket in manager.get_endpoint_connections(endpoint):
            if websocket in manager.connection_data:
                data = manager.connection_data[websocket]
                connections.append({
                    "connected_at": data["connected_at"].isoformat(),
                    "last_heartbeat": data["last_heartbeat"],
                    "subscriptions": list(data["subscriptions"])
                })
        endpoint_connections[endpoint] = connections
    
    return {
        "total_connections": manager.get_total_connections(),
        "endpoint_connections": endpoint_connections,
        "active_streams": list(stream_manager.streaming_tasks.keys())
    }

# Test endpoint
@app.get("/test")
async def test_endpoint():
    """Test endpoint to verify service is running."""
    return {
        "message": "WebSocket Stream Service is running",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "host": WEBSOCKET_HOST,
            "port": WEBSOCKET_PORT,
            "cors_origins": CORS_ORIGINS
        },
        "endpoints": WEBSOCKET_ENDPOINTS
    }

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize the service on startup."""
    print("🚀 Starting WebSocket Stream Service...")
    print(f"📍 Host: {WEBSOCKET_HOST}")
    print(f"📍 Port: {WEBSOCKET_PORT}")
    print(f"📍 CORS Origins: {CORS_ORIGINS}")
    print("-" * 50)
    
    # WebSocket service is always enabled when this script runs
        
    # Initialize Zerodha WebSocket client
    try:
        if hasattr(zerodha_ws_client, 'connect') and callable(zerodha_ws_client.connect):
            zerodha_ws_client.connect()
            print("✅ Zerodha WebSocket client connected")
        else:
            print("⚠️  Zerodha WebSocket client connect method not available")
    except Exception as e:
        print(f"❌ Failed to connect Zerodha WebSocket client: {e}")
        traceback.print_exc()

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown."""
    print("🛑 Shutting down WebSocket Stream Service...")
    
    # Disconnect all WebSocket connections
    for endpoint_connections in manager.endpoint_connections.values():
        for connection in endpoint_connections[:]:
            manager.disconnect(connection)
        
    # Cancel all streaming tasks
    for task in stream_manager.streaming_tasks.values():
        task.cancel()
        
    # Disconnect Zerodha WebSocket client
    try:
        if hasattr(zerodha_ws_client, 'disconnectWebSocket') and callable(zerodha_ws_client.disconnectWebSocket):
            zerodha_ws_client.disconnectWebSocket()
            print("✅ Zerodha WebSocket client disconnected")
        else:
            print("⚠️  Zerodha WebSocket client disconnect method not available")
    except Exception as e:
        print(f"❌ Error disconnecting Zerodha WebSocket client: {e}")

def main():
    """Start the WebSocket stream service."""
    print("🚀 Starting WebSocket Stream Service...")
    print(f"📍 Service: Real-time data streaming via WebSocket")
    print(f"🌐 URL: http://{WEBSOCKET_HOST}:{WEBSOCKET_PORT}")
    print(f"🔗 WebSocket: ws://{WEBSOCKET_HOST}:{WEBSOCKET_PORT}/ws/stream")
    print(f"📊 Health: http://{WEBSOCKET_HOST}:{WEBSOCKET_PORT}/health")
    print(f"🌍 CORS Origins: {CORS_ORIGINS}")
    print("-" * 50)
    
    # Start the service
    uvicorn.run(
        "websocket_stream_service:app",
        host=WEBSOCKET_HOST,
        port=WEBSOCKET_PORT,
        reload=False,
        log_level="info",
        access_log=True
    )

if __name__ == "__main__":
    main() 