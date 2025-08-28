#!/usr/bin/env python3
"""
Startup script that overrides CORS origins to include port 8080
and manually starts the Zerodha WebSocket client
"""

import os
import sys
import asyncio

# Override CORS_ORIGINS to include port 8080
os.environ['CORS_ORIGINS'] = 'http://localhost:3000,http://localhost:8080,http://localhost:5173,http://127.0.0.1:3000,http://127.0.0.1:8080,http://127.0.0.1:5173,https://stock-analyzer-pro.vercel.app,https://stock-analyzer-pro-git-prototype-aaryan-manawats-projects.vercel.app,https://stock-analyzer-pro-git-prototype-aaryan-manawats-projects.vercel.app'

print("🔧 CORS_ORIGINS overridden to include port 8080")
print(f"   New CORS_ORIGINS: {os.environ['CORS_ORIGINS']}")

# Import and run the consolidated service
from consolidated_service import app
import uvicorn

async def start_zerodha_ws_client():
    """Manually start the Zerodha WebSocket client."""
    try:
        print("🚀 Manually starting Zerodha WebSocket client...")
        
        # Import the data service startup function
        from data_service import startup_event
        
        # Create a mock event loop context if needed
        try:
            # Try to get the running event loop
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # Create a new event loop if none is running
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Run the startup event
        await startup_event()
        print("✅ Zerodha WebSocket client started successfully!")
        
    except Exception as e:
        print(f"❌ Error starting Zerodha WebSocket client: {e}")
        print("📊 Historical data will still be available")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    print(f"🚀 Starting Consolidated Service on {host}:{port}")
    print(f"📊 Data Service mounted at /data")
    print(f"🔍 Analysis Service mounted at /analysis")
    print(f"🌐 CORS Origins: {os.environ['CORS_ORIGINS']}")
    
    # Start the Zerodha WebSocket client in the background
    async def main():
        # Start the WebSocket client
        await start_zerodha_ws_client()
        
        # Start the FastAPI server
        config = uvicorn.Config(
            "start_with_cors_fix:app",
            host=host,
            port=port,
            reload=False,
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()
    
    # Run the main function
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("👋 Shutting down...")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
