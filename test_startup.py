#!/usr/bin/env python3
"""
Test script to verify the startup sequence and event loop fix.
"""

import asyncio
import uvicorn
from api import app

async def test_startup():
    """Test the startup sequence"""
    print("Testing startup sequence...")
    
    # Simulate the startup event
    await app.router.startup()
    
    print("Startup test completed successfully!")

if __name__ == "__main__":
    print("Starting test server...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info") 