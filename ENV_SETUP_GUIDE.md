# Environment Configuration Guide

## 🔐 JWT Authentication Setup

Create a `.env` file in the `backend/` directory with the following configuration:

### Required Environment Variables

```bash
# ===== JWT AUTHENTICATION CONFIGURATION =====
# IMPORTANT: Change this to a strong, unique secret key in production
JWT_SECRET=your-super-secret-jwt-key-change-this-in-production
REQUIRE_AUTH=true

# ===== API KEYS CONFIGURATION =====
# Comma-separated list of valid API keys for authentication
API_KEYS=test-api-key-1,test-api-key-2,production-api-key

# ===== ZERODHA API CONFIGURATION =====
# Add your Zerodha API credentials here for live data
ZERODHA_API_KEY=your-zerodha-api-key
ZERODHA_API_SECRET=your-zerodha-api-secret
ZERODHA_ACCESS_TOKEN=your-zerodha-access-token

# ===== DATABASE CONFIGURATION =====
# Supabase configuration (if using)
SUPABASE_URL=your-supabase-url
SUPABASE_KEY=your-supabase-anon-key

# ===== ENVIRONMENT CONFIGURATION =====
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=INFO

# ===== SERVER CONFIGURATION =====
HOST=0.0.0.0
PORT=8000
CORS_ORIGINS=http://localhost:3000,http://localhost:8080,http://127.0.0.1:3000,http://127.0.0.1:8080

# ===== WEBSOCKET CONFIGURATION =====
WS_HEARTBEAT_INTERVAL=30
WS_MAX_CONNECTIONS=100
WS_RECONNECT_ATTEMPTS=5

# ===== CACHE CONFIGURATION =====
CACHE_ENABLED=true
CACHE_TTL=300
CACHE_MAX_SIZE=1000

# ===== SECURITY CONFIGURATION =====
# Rate limiting
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=60

# Session configuration
SESSION_SECRET=your-session-secret-key
SESSION_TTL=3600
```

## 🚀 Quick Setup Steps

### 1. Create the .env file
```bash
cd backend/
touch .env
```

### 2. Add the configuration
Copy the above configuration into your `.env` file and replace the placeholder values:

#### For Testing (Minimal Setup):
```bash
JWT_SECRET=test-jwt-secret-key-12345
REQUIRE_AUTH=true
API_KEYS=test-key-1,test-key-2
ENVIRONMENT=development
DEBUG=true
```

#### For Production:
```bash
JWT_SECRET=your-very-long-random-secret-key-here
REQUIRE_AUTH=true
API_KEYS=your-production-api-key
ZERODHA_API_KEY=your-actual-zerodha-key
ZERODHA_API_SECRET=your-actual-zerodha-secret
ENVIRONMENT=production
DEBUG=false
```

## 🔑 Key Configuration Details

### JWT_SECRET
- **Purpose**: Used to sign and verify JWT tokens for WebSocket authentication
- **Format**: Any string (recommend 32+ characters for production)
- **Example**: `JWT_SECRET=my-super-secret-jwt-key-for-websocket-auth-2024`

### API_KEYS
- **Purpose**: Comma-separated list of valid API keys for authentication
- **Format**: `key1,key2,key3`
- **Example**: `API_KEYS=test-key-123,prod-key-456,admin-key-789`

### REQUIRE_AUTH
- **Purpose**: Enable/disable authentication requirement
- **Values**: `true` or `false`
- **Default**: `true` (recommended for production)

### ZERODHA_API_* (Optional)
- **Purpose**: For live stock data from Zerodha
- **Required**: Only if you want real market data
- **Get from**: Zerodha developer console

## 🧪 Testing Configuration

### 1. Test JWT Token Creation
```bash
curl -X POST "http://localhost:8000/auth/token" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "test_user"}'
```

### 2. Test Token Verification
```bash
curl "http://localhost:8000/auth/verify?token=YOUR_TOKEN_HERE"
```

### 3. Test WebSocket Connection
The frontend should now be able to connect to WebSocket without 403 errors.

## 🔒 Security Best Practices

1. **Never commit .env files** to version control
2. **Use strong, random JWT secrets** in production
3. **Rotate API keys** regularly
4. **Limit CORS origins** to your actual domains
5. **Use HTTPS** in production
6. **Set DEBUG=false** in production

## 🚨 Troubleshooting

### WebSocket 403 Errors
- Check that `JWT_SECRET` is set
- Verify `REQUIRE_AUTH=true`
- Ensure frontend is sending valid tokens

### Authentication Failures
- Verify JWT token format
- Check token expiration
- Ensure API keys are correctly formatted

### Live Data Issues
- Verify Zerodha API credentials
- Check API rate limits
- Ensure market hours for live data

## 📝 Example .env for Development

```bash
# Development configuration
JWT_SECRET=dev-jwt-secret-key-2024
REQUIRE_AUTH=true
API_KEYS=dev-api-key-1,dev-api-key-2
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=DEBUG
HOST=0.0.0.0
PORT=8000
CORS_ORIGINS=http://localhost:3000,http://localhost:8080
```

After setting up the `.env` file, restart your backend server for the changes to take effect. 