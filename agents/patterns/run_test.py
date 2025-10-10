#!/usr/bin/env python3
"""
Pattern Analysis Test Runner

This is a simple wrapper script to run the pattern analysis test with proper imports.
Run this from the backend/agents/patterns directory.
"""

import os
import sys
import subprocess

# Add backend to path
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, backend_dir)

def main():
    """Run the test from the backend directory with proper Python path"""
    
    # Change to backend directory
    original_cwd = os.getcwd()
    os.chdir(backend_dir)
    
    try:
        # Build the command
        python_path = sys.executable
        test_script = os.path.join("agents", "patterns", "test_pattern_analysis_direct.py")
        
        # Pass through all arguments
        cmd = [python_path, test_script] + sys.argv[1:]
        
        # Run the test
        result = subprocess.run(cmd, cwd=backend_dir)
        sys.exit(result.returncode)
        
    finally:
        # Restore original working directory
        os.chdir(original_cwd)

if __name__ == "__main__":
    main()