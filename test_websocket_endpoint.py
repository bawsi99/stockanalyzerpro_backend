#!/usr/bin/env python3
"""
test_websocket_endpoint.py

Test to verify the WebSocket endpoint is properly registered.
"""

import sys
import os

# Add the backend directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_websocket_endpoint():
    """Test if the WebSocket endpoint is properly registered."""
    try:
        # Import the consolidated service
        from consolidated_service import app
        
        print("✅ Consolidated service imported successfully")
        
        # Check if the WebSocket endpoint is registered
        routes = app.routes
        websocket_routes = [route for route in routes if hasattr(route, 'path') and '/ws/stream' in str(route.path)]
        
        if websocket_routes:
            print(f"✅ WebSocket endpoint found: {websocket_routes}")
        else:
            print("❌ WebSocket endpoint not found in routes")
            print("Available routes:")
            for route in routes:
                if hasattr(route, 'path'):
                    print(f"  - {route.path}")
        
        # Check if data_websocket is imported
        try:
            from consolidated_service import data_websocket
            print("✅ data_websocket imported successfully")
        except ImportError as e:
            print(f"❌ Failed to import data_websocket: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing WebSocket endpoint: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_websocket_endpoint()
