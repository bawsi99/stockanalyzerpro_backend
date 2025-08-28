#!/usr/bin/env python3
"""
Check and fix CORS origins configuration.
Run this to debug CORS issues.
"""

import os

def check_cors_origins():
    """Check current CORS origins configuration."""
    print("🔍 Checking CORS Origins Configuration")
    print("=" * 50)
    
    # Check environment variable
    cors_env = os.getenv("CORS_ORIGINS", "")
    print(f"📋 CORS_ORIGINS environment variable:")
    print(f"   Raw value: '{cors_env}'")
    
    if cors_env:
        # Parse and show what we're getting
        origins = [origin.strip() for origin in cors_env.split(",") if origin.strip()]
        print(f"   Parsed origins: {origins}")
        
        # Check for malformed entries
        malformed = []
        valid = []
        for origin in origins:
            if origin.startswith("CORS_ORIGINS=") or "=" in origin:
                malformed.append(origin)
            elif origin.startswith(("http://", "https://")):
                valid.append(origin)
            else:
                malformed.append(origin)
        
        print(f"   ✅ Valid origins: {valid}")
        print(f"   ❌ Malformed origins: {malformed}")
        
        if malformed:
            print("\n🔧 Fixing malformed CORS origins...")
            # Create clean CORS origins
            clean_origins = [
                "http://localhost:3000",
                "http://localhost:8080", 
                "http://localhost:5173",
                "http://127.0.0.1:3000",
                "http://127.0.0.1:8080",
                "http://127.0.0.1:5173",
                "https://stock-analyzer-pro.vercel.app",
                "https://stock-analyzer-pro-git-prototype-aaryan-manawats-projects.vercel.app",
                "https://stock-analyzer-cl9o3tivx-aaryan-manawats-projects.vercel.app",
                "https://stockanalyzer-pro.vercel.app"
            ]
            
            # Add any valid origins from environment
            for origin in valid:
                if origin not in clean_origins:
                    clean_origins.append(origin)
            
            print(f"   🎯 Clean CORS origins: {clean_origins}")
            print(f"\n💡 Set this environment variable:")
            print(f"   CORS_ORIGINS={','.join(clean_origins)}")
            
    else:
        print("   ❌ CORS_ORIGINS not set")
        print("\n💡 Set this environment variable:")
        print("   CORS_ORIGINS=http://localhost:3000,http://localhost:5173,https://stock-analyzer-pro.vercel.app")
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    check_cors_origins()
