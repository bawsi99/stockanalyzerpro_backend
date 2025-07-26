# Zerodha API Setup Guide

## Overview
This application can work in two modes:
1. **Historical Data Only** (default) - Works without Zerodha credentials
2. **Live Data Streaming** - Requires Zerodha API credentials

## Quick Start (Historical Data Only)
If you only need historical data, no setup is required. The application will work immediately with cached data.

## Live Data Setup (Optional)

### Step 1: Get Zerodha API Credentials
1. Go to [Zerodha Developer Console](https://developers.kite.trade/)
2. Create a new application
3. Note down your API Key and API Secret

### Step 2: Generate Access Token
1. Use the provided `update_request_token.py` script
2. Follow the authentication flow
3. Your access token will be automatically saved

### Step 3: Create .env File
Create a `.env` file in the backend directory with:

```env
# Zerodha API Configuration
ZERODHA_API_KEY=your_api_key_here
ZERODHA_API_SECRET=your_api_secret_here
ZERODHA_ACCESS_TOKEN=your_access_token_here
ZERODHA_REQUEST_TOKEN=your_request_token_here

# JWT Configuration
JWT_SECRET=your-secret-key-change-in-production

# API Keys (comma-separated for multiple keys)
API_KEYS=your_api_key_1,your_api_key_2

# Authentication
REQUIRE_AUTH=true

# Redis Configuration (optional)
REDIS_URL=redis://localhost:6379/0
```

### Step 4: Restart the Application
After setting up the credentials, restart the backend service.

## Troubleshooting

### Authentication Errors (403)
- Check if your access token has expired
- Regenerate your access token using the authentication flow
- Ensure your API key and secret are correct

### WebSocket Connection Issues
- Verify your internet connection
- Check if Zerodha services are available
- Ensure your account has the necessary permissions

### Historical Data Still Works
Even if live data fails, historical data will continue to work via REST API endpoints.

## Features Available

### Without Zerodha Credentials
- ✅ Historical data retrieval
- ✅ Technical analysis
- ✅ Pattern recognition
- ✅ Risk assessment
- ✅ Chart visualization

### With Zerodha Credentials
- ✅ All above features
- ✅ Live price updates
- ✅ Real-time chart updates
- ✅ Live tick data
- ✅ Market depth information

## API Endpoints

### Historical Data (Always Available)
- `GET /stock/{symbol}/history` - Get historical data
- `POST /data/optimized` - Get optimized data
- `GET /stock/{symbol}/info` - Get stock information

### Live Data (Requires Credentials)
- `WebSocket /ws/stream` - Live data streaming
- `GET /ws/health` - WebSocket health check
- `GET /ws/connections` - Connection statistics

## Security Notes
- Never commit your `.env` file to version control
- Use strong JWT secrets in production
- Regularly rotate your access tokens
- Monitor your API usage limits 