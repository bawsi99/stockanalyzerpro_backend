#!/usr/bin/env python3
"""
Minimal test to check if text extraction is working properly after the fix
"""

import os
import sys
from pathlib import Path

# Add the project root to path
current_dir = Path(__file__).parent
project_root = current_dir.parent  # Go up to 3.0
sys.path.insert(0, str(project_root))

# Set a test API key if none exists
if not os.getenv('GEMINI_API_KEY') and not os.getenv('GEMINI_API_KEY1'):
    print("⚠️  No API key found. Please set GEMINI_API_KEY or GEMINI_API_KEY1")
    sys.exit(1)

# Now test the import
try:
    from backend.llm import get_llm_client
    print("✅ Import successful")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Test creating a client
try:
    client = get_llm_client("volume_anomaly_agent")
    print(f"✅ Client created: {client.get_provider_info()}")
except Exception as e:
    print(f"❌ Client creation failed: {e}")
    import traceback
    traceback.print_exc()