# 🔐 Zerodha API Setup Guide

## Current Status
❌ **Authentication Failed** - Tokens are expired

## Quick Fix Steps

### 1. Generate New Request Token
1. **Log in to Zerodha Kite** (https://kite.zerodha.com)
2. **Go to API section**:
   - Click on your profile (top right)
   - Go to "API" section
   - Or directly visit: https://kite.zerodha.com/connect/api

### 2. Generate Request Token
1. Click **"Generate Request Token"**
2. Copy the generated token (it will look like: `abc123def456...`)
3. **Update your .env file**:
   ```bash
   ZERODHA_REQUEST_TOKEN=your_new_request_token_here
   ```

### 3. Run Authentication Test
```bash
python test_zerodha_auth.py
```

This will automatically:
- Generate a new access token
- Update your .env file
- Test the connection

### 4. Restart Backend Server
After successful authentication:
```bash
# Stop current server (Ctrl+C)
# Then restart:
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

## Troubleshooting

### If Request Token Generation Fails:
1. **Check API Permissions**:
   - Ensure your API key has WebSocket permissions
   - Enable "Market Data" permissions

2. **Check Account Status**:
   - Ensure your Zerodha account is active
   - Check if there are any restrictions

3. **Market Hours**:
   - WebSocket connections work best during market hours
   - Current market hours: 9:15 AM - 3:30 PM (Mon-Fri)

### Common Error Messages:
- **"Token is invalid or has expired"**: Generate new request token
- **"Incorrect api_key or access_token"**: Update both tokens
- **"403 Forbidden"**: Check API permissions and account status

## Manual Token Update

If automatic update fails, manually update your `.env` file:

```env
ZERODHA_API_KEY=qlxn60ikpx1z7huj
ZERODHA_API_SECRET=2qggz6oh5q687pi9zyg65306g2sknmbk
ZERODHA_REQUEST_TOKEN=your_new_request_token_here
ZERODHA_ACCESS_TOKEN=will_be_auto_generated
```

## Testing Connection

After updating tokens, test the connection:

```bash
# Test authentication
python test_zerodha_auth.py

# Test WebSocket connection
python test_websocket_connection.py

# Test full system
python setup_live_charts.py
```

## Expected Success Output

When working correctly, you should see:
```
✅ Authentication successful!
   User: Your Name
   Email: your.email@example.com
   Broker: ZERODHA
```

## Security Notes

⚠️ **Important Security Reminders**:
- Never commit your `.env` file to version control
- Keep your API secret secure
- Request tokens expire quickly - generate new ones as needed
- Access tokens are automatically refreshed by the system

## Support

If you continue to have issues:
1. Check Zerodha's API documentation
2. Verify your account has API access enabled
3. Contact Zerodha support if needed 