# Multi-Endpoint WebSocket Stream Service

A dedicated WebSocket service for real-time data streaming with multiple endpoints, running on port 8081. Each endpoint provides different types of data and functionality.

## Configuration

The service uses existing environment variables from the `.env` file:

- **HOST**: Server host (default: 0.0.0.0)
- **PORT**: Base port (default: 8000) - WebSocket service runs on PORT + 81 (8081)
- **CORS_ORIGINS**: Comma-separated list of allowed frontend URLs for WebSocket connections

The WebSocket service automatically uses the CORS_ORIGINS configuration to determine which frontend URLs can establish WebSocket connections.

## Starting the Service

### Option 1: Start individually
```bash
cd backend
python start_websocket_service.py
```

### Option 2: Start with all services
```bash
cd backend
python run_services.py
```

## Service Endpoints

### WebSocket Endpoints

#### 1. Main Data Stream
- **URL**: `ws://localhost:8081/ws/stream`
- **Purpose**: Real-time market data streaming
- **Data Types**: tick_data, candle_data, market_data

#### 2. Analysis Stream
- **URL**: `ws://localhost:8081/ws/analysis`
- **Purpose**: Real-time technical analysis and insights
- **Data Types**: technical_analysis, pattern_recognition, risk_metrics

#### 3. Alerts Stream
- **URL**: `ws://localhost:8081/ws/alerts`
- **Purpose**: Price alerts and notifications
- **Data Types**: price_alerts, volume_alerts, breakout_alerts

#### 4. Portfolio Stream
- **URL**: `ws://localhost:8081/ws/portfolio`
- **Purpose**: Portfolio tracking and performance
- **Data Types**: portfolio_updates, pnl_data, position_data

### HTTP Endpoints
- **Health Check**: `GET http://localhost:8081/health`
- **Connections**: `GET http://localhost:8081/connections`
- **Test**: `GET http://localhost:8081/test`

## WebSocket Protocol

### Connection
Connect to `ws://localhost:8081/ws/stream` to establish a WebSocket connection.

### Welcome Message
Upon connection, you'll receive a welcome message:
```json
{
  "type": "connection",
  "status": "connected",
  "timestamp": "2025-01-27T10:30:00.000Z",
  "message": "WebSocket connection established successfully"
}
```

### Message Types

#### 1. Subscribe to Symbol
**Send:**
```json
{
  "type": "subscribe",
  "symbol": "RELIANCE"
}
```

**Receive:**
```json
{
  "type": "subscription",
  "status": "subscribed",
  "symbol": "RELIANCE",
  "timestamp": "2025-01-27T10:30:00.000Z"
}
```

#### 2. Unsubscribe from Symbol
**Send:**
```json
{
  "type": "unsubscribe",
  "symbol": "RELIANCE"
}
```

**Receive:**
```json
{
  "type": "subscription",
  "status": "unsubscribed",
  "symbol": "RELIANCE",
  "timestamp": "2025-01-27T10:30:00.000Z"
}
```

#### 3. Heartbeat
**Send:**
```json
{
  "type": "heartbeat",
  "timestamp": 1706352600.123
}
```

**Receive:**
```json
{
  "type": "heartbeat",
  "status": "ack",
  "timestamp": "2025-01-27T10:30:00.000Z"
}
```

#### 4. Ping/Pong
**Send:**
```json
{
  "type": "ping",
  "timestamp": 1706352600.123
}
```

**Receive:**
```json
{
  "type": "pong",
  "timestamp": "2025-01-27T10:30:00.000Z"
}
```

#### 5. Real-time Data
**Receive:**
```json
{
  "type": "data",
  "symbol": "RELIANCE",
  "timestamp": "2025-01-27T10:30:00.000Z",
  "data": {
    "price": 2500.50,
    "volume": 1000000,
    "change": 25.50,
    "change_percent": 1.03
  }
}
```

#### 6. Error Messages
**Receive:**
```json
{
  "type": "error",
  "message": "Invalid JSON format",
  "timestamp": "2025-01-27T10:30:00.000Z"
}
```

## Testing

Use the provided test clients to verify the service:

### Single Endpoint Testing
```bash
cd backend
python test_websocket_client.py
```

### Multi-Endpoint Testing
```bash
cd backend
python test_multi_endpoint_client.py
```

## Integration with Frontend

### JavaScript Examples

#### Main Data Stream
```javascript
const ws = new WebSocket('ws://localhost:8081/ws/stream');

ws.onopen = function() {
    console.log('Connected to main data stream');
    
    // Subscribe to a symbol
    ws.send(JSON.stringify({
        type: 'subscribe',
        symbol: 'RELIANCE'
    }));
};

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    if (data.type === 'data') {
        // Handle real-time market data
        updateChart(data.data);
    }
};
```

#### Analysis Stream
```javascript
const analysisWs = new WebSocket('ws://localhost:8081/ws/analysis');

analysisWs.onopen = function() {
    console.log('Connected to analysis stream');
    
    analysisWs.send(JSON.stringify({
        type: 'subscribe',
        symbol: 'RELIANCE'
    }));
};

analysisWs.onmessage = function(event) {
    const data = JSON.parse(event.data);
    if (data.type === 'data') {
        // Handle technical analysis data
        updateIndicators(data.data.technical_indicators);
        updatePatterns(data.data.patterns);
    }
};
```

#### Alerts Stream
```javascript
const alertsWs = new WebSocket('ws://localhost:8081/ws/alerts');

alertsWs.onopen = function() {
    console.log('Connected to alerts stream');
    
    alertsWs.send(JSON.stringify({
        type: 'subscribe',
        symbol: 'RELIANCE'
    }));
};

alertsWs.onmessage = function(event) {
    const data = JSON.parse(event.data);
    if (data.type === 'data') {
        // Handle alerts
        data.data.price_alerts.forEach(alert => {
            if (alert.triggered) {
                showNotification(`Price alert: ${alert.type}`);
            }
        });
    }
};
```

#### Portfolio Stream
```javascript
const portfolioWs = new WebSocket('ws://localhost:8081/ws/portfolio');

portfolioWs.onopen = function() {
    console.log('Connected to portfolio stream');
    
    portfolioWs.send(JSON.stringify({
        type: 'subscribe',
        symbol: 'RELIANCE'
    }));
};

portfolioWs.onmessage = function(event) {
    const data = JSON.parse(event.data);
    if (data.type === 'data') {
        // Handle portfolio updates
        updatePortfolioValue(data.data.position.current_value);
        updatePnL(data.data.pnl);
    }
};
```

### React Hook Example
```javascript
import { useEffect, useRef, useState } from 'react';

function useWebSocket(url) {
    const ws = useRef(null);
    const [isConnected, setIsConnected] = useState(false);
    const [data, setData] = useState(null);

    useEffect(() => {
        ws.current = new WebSocket(url);

        ws.current.onopen = () => {
            setIsConnected(true);
        };

        ws.current.onmessage = (event) => {
            const message = JSON.parse(event.data);
            if (message.type === 'data') {
                setData(message.data);
            }
        };

        ws.current.onclose = () => {
            setIsConnected(false);
        };

        return () => {
            if (ws.current) {
                ws.current.close();
            }
        };
    }, [url]);

    const subscribe = (symbol) => {
        if (ws.current && isConnected) {
            ws.current.send(JSON.stringify({
                type: 'subscribe',
                symbol: symbol
            }));
        }
    };

    return { isConnected, data, subscribe };
}
```

## Architecture

The WebSocket service integrates with:

1. **Zerodha WebSocket Client**: For real-time market data
2. **Enhanced Data Service**: For symbol token mapping
3. **Connection Manager**: Manages WebSocket connections
4. **Data Stream Manager**: Handles data streaming and subscriptions

## Monitoring

- **Health Check**: Monitor service status via `/health` endpoint
- **Connections**: View active connections via `/connections` endpoint
- **Logs**: Check console output for connection and streaming logs

## Troubleshooting

1. **Connection Refused**: Ensure the service is running on port 8081
2. **No Data**: Check if the symbol exists and has valid token mapping
3. **Disconnections**: Implement automatic reconnection logic in your client
4. **High Memory Usage**: Monitor connection count and implement connection limits

## Security

- The service respects CORS configuration from environment variables
- No authentication is required by default (can be added if needed)
- Consider implementing rate limiting for production use 