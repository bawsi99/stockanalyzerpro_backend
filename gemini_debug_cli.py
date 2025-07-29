#!/usr/bin/env python3
"""
Command-line interface for managing Gemini debugging configuration.
"""

import sys
import os
from pathlib import Path

# Add the backend directory to the Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

from gemini.debug_config import (
    enable_gemini_debug, 
    disable_gemini_debug, 
    set_gemini_log_level, 
    show_gemini_debug_status,
    print_env_help
)

def main():
    """Main CLI function"""
    if len(sys.argv) < 2:
        print_usage()
        return
    
    command = sys.argv[1].lower()
    
    if command == "enable":
        enable_gemini_debug()
    elif command == "disable":
        disable_gemini_debug()
    elif command == "status":
        show_gemini_debug_status()
    elif command == "level":
        if len(sys.argv) < 3:
            print("❌ Please specify log level: DEBUG, INFO, WARNING, or ERROR")
            return
        level = sys.argv[2].upper()
        try:
            set_gemini_log_level(level)
        except ValueError as e:
            print(f"❌ {e}")
    elif command == "env":
        print_env_help()
    elif command == "test":
        run_test()
    elif command == "help":
        print_usage()
    else:
        print(f"❌ Unknown command: {command}")
        print_usage()

def print_usage():
    """Print usage information"""
    print("🔧 Gemini Debug CLI")
    print("=" * 30)
    print("Usage: python gemini_debug_cli.py <command>")
    print()
    print("Commands:")
    print("  enable    - Enable Gemini debugging")
    print("  disable   - Disable Gemini debugging")
    print("  status    - Show current debug status")
    print("  level <level> - Set log level (DEBUG/INFO/WARNING/ERROR)")
    print("  env       - Show environment variables")
    print("  test      - Run a test to verify debugging")
    print("  help      - Show this help message")
    print()
    print("Examples:")
    print("  python gemini_debug_cli.py enable")
    print("  python gemini_debug_cli.py level DEBUG")
    print("  python gemini_debug_cli.py status")

def run_test():
    """Run a simple test to verify debugging is working"""
    print("🧪 Running Gemini Debug Test")
    print("=" * 30)
    
    # Check if API key is available
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY not set")
        print("Please set your API key first:")
        print("export GEMINI_API_KEY='your-api-key-here'")
        return
    
    print("✅ API key found")
    
    try:
        # Import and run the test
        from test_gemini_debug import main as test_main
        test_main()
    except ImportError as e:
        print(f"❌ Could not import test module: {e}")
        print("Make sure you're running this from the backend directory")
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    main() 