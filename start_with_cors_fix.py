#!/usr/bin/env python3
"""
Startup script that overrides CORS origins to include port 8080
"""

import os
import sys

# Override CORS_ORIGINS to include port 8080
os.environ['CORS_ORIGINS'] = 'http://localhost:3000,http://localhost:8080,http://localhost:5173,http://127.0.0.1:3000,http://127.0.0.1:8080,http://127.0.0.1:5173,https://stock-analyzer-pro.vercel.app,https://stock-analyzer-pro-git-prototype-aaryan-manawats-projects.vercel.app,https://stock-analyzer-pro-git-prototype-aaryan-manawats-projects.vercel.app'

print("🔧 CORS_ORIGINS overridden to include port 8080")
print(f"   New CORS_ORIGINS: {os.environ['CORS_ORIGINS']}")

# Import and run the consolidated service
from consolidated_service import app
import uvicorn

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    print(f"🚀 Starting Consolidated Service on {host}:{port}")
    print(f"📊 Data Service mounted at /data")
    print(f"🔍 Analysis Service mounted at /analysis")
    print(f"🌐 CORS Origins: {os.environ['CORS_ORIGINS']}")
    
    uvicorn.run(
        "start_with_cors_fix:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )
