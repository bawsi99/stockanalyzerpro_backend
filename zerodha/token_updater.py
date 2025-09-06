#!/usr/bin/env python3
"""
Simple script to update the ZERODHA_REQUEST_TOKEN in .env file.
"""

import os
import sys

def update_request_token():
    """Update the request token in .env file."""
    
    print("🔑 Zerodha Request Token Updater")
    print("=" * 40)
    
    # Get current token
    current_token = os.getenv('ZERODHA_REQUEST_TOKEN', 'NOT SET')
    print(f"Current Request Token: {current_token[:10] if current_token != 'NOT SET' else 'NOT SET'}...")
    
    # Get new token from user
    print("\n📝 Enter your new request token from Zerodha Kite:")
    print("   (Get it from: https://kite.zerodha.com/connect/api)")
    new_token = input("New Request Token: ").strip()
    
    if not new_token:
        print("❌ No token provided!")
        return False
    
    # Update .env file
    env_file = '../config/.env'
    
    try:
        # Read current .env file
        with open(env_file, 'r') as f:
            lines = f.readlines()
        
        # Update request token
        updated = False
        for i, line in enumerate(lines):
            if line.startswith('ZERODHA_REQUEST_TOKEN='):
                lines[i] = f'ZERODHA_REQUEST_TOKEN={new_token}\n'
                updated = True
                break
        
        # If not found, add it
        if not updated:
            lines.append(f'ZERODHA_REQUEST_TOKEN={new_token}\n')
        
        # Write back to file
        with open(env_file, 'w') as f:
            f.writelines(lines)
        
        print(f"✅ Updated .env file with new request token!")
        print(f"   New Token: {new_token[:10]}...")
        
        # Test authentication
        print(f"\n🔄 Testing authentication with new token...")
        os.system('python test_zerodha_auth.py')
        
        return True
        
    except Exception as e:
        print(f"❌ Error updating .env file: {e}")
        return False

if __name__ == "__main__":
    success = update_request_token()
    sys.exit(0 if success else 1) 