#!/usr/bin/env python3
"""
Test Zerodha authentication and generate new access token if needed.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_zerodha_auth():
    """Test Zerodha authentication and generate new access token."""
    from kiteconnect import KiteConnect
    
    api_key = os.getenv('ZERODHA_API_KEY')
    api_secret = os.getenv('ZERODHA_API_SECRET')
    access_token = os.getenv('ZERODHA_ACCESS_TOKEN')
    
    print("🔐 Testing Zerodha Authentication")
    print("=" * 40)
    print(f"API Key: {api_key[:10]}..." if api_key else "❌ API Key not set")
    print(f"API Secret: {'✅ Set' if api_secret else '❌ Not set'}")
    print(f"Access Token: {access_token[:10]}..." if access_token else "❌ Access Token not set")
    
    if not api_key or not api_secret:
        print("\n❌ Missing API credentials!")
        print("Please set ZERODHA_API_KEY and ZERODHA_API_SECRET in your .env file")
        return False
    
    # Initialize KiteConnect
    kite = KiteConnect(api_key=api_key)
    
    try:
        # Test with current access token
        if access_token:
            print(f"\n🔄 Testing current access token...")
            kite.set_access_token(access_token)
            
            try:
                # Try to get user profile
                profile = kite.profile()
                print(f"✅ Authentication successful!")
                print(f"   User: {profile.get('user_name', 'Unknown')}")
                print(f"   Email: {profile.get('email', 'Unknown')}")
                print(f"   Broker: {profile.get('broker', 'Unknown')}")
                return True
            except Exception as e:
                print(f"❌ Current access token failed: {e}")
                print("   This usually means the token has expired")
        
        # Try to generate new access token using request token
        request_token = os.getenv('ZERODHA_REQUEST_TOKEN')
        if request_token:
            print(f"\n🔄 Generating new access token using request token...")
            try:
                data = kite.generate_session(request_token, api_secret=api_secret)
                new_access_token = data["access_token"]
                print(f"✅ New access token generated!")
                print(f"   New Token: {new_access_token[:10]}...")
                
                # Update .env file
                update_env_file(new_access_token)
                print(f"✅ Updated .env file with new access token")
                return True
                
            except Exception as e:
                print(f"❌ Failed to generate new access token: {e}")
                print("   Request token might be expired or invalid")
        
        print(f"\n❌ No valid authentication method available")
        print("   Please:")
        print("   1. Log in to Zerodha Kite")
        print("   2. Go to API section")
        print("   3. Generate a new request token")
        print("   4. Update ZERODHA_REQUEST_TOKEN in your .env file")
        return False
        
    except Exception as e:
        print(f"❌ Authentication test failed: {e}")
        return False

def update_env_file(new_access_token):
    """Update the .env file with new access token."""
    env_file = '.env'
    
    # Read current .env file
    with open(env_file, 'r') as f:
        lines = f.readlines()
    
    # Update access token
    updated = False
    for i, line in enumerate(lines):
        if line.startswith('ZERODHA_ACCESS_TOKEN='):
            lines[i] = f'ZERODHA_ACCESS_TOKEN={new_access_token}\n'
            updated = True
            break
    
    # Write back to file
    with open(env_file, 'w') as f:
        f.writelines(lines)
    
    if not updated:
        print("Warning: Could not update .env file automatically")

def check_market_hours():
    """Check if market is currently open."""
    import datetime
    
    now = datetime.datetime.now()
    current_time = now.time()
    
    # NSE market hours: 9:15 AM to 3:30 PM (Monday to Friday)
    market_start = datetime.time(9, 15)
    market_end = datetime.time(15, 30)
    
    is_weekday = now.weekday() < 5  # Monday = 0, Friday = 4
    is_market_hours = market_start <= current_time <= market_end
    
    print(f"\n📅 Market Hours Check:")
    print(f"   Current Time: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Day of Week: {now.strftime('%A')}")
    print(f"   Market Hours: 9:15 AM - 3:30 PM (Mon-Fri)")
    print(f"   Is Weekday: {'✅ Yes' if is_weekday else '❌ No'}")
    print(f"   Is Market Hours: {'✅ Yes' if is_market_hours else '❌ No'}")
    print(f"   Market Status: {'🟢 Open' if (is_weekday and is_market_hours) else '🔴 Closed'}")
    
    return is_weekday and is_market_hours

if __name__ == "__main__":
    print("🔧 Zerodha Authentication Test")
    print("=" * 50)
    
    # Check market hours first
    market_open = check_market_hours()
    
    # Test authentication
    auth_success = test_zerodha_auth()
    
    print("\n" + "=" * 50)
    if auth_success:
        print("🎉 Authentication successful!")
        if not market_open:
            print("⚠️  Note: Market is currently closed")
            print("   WebSocket connections may not work outside market hours")
    else:
        print("❌ Authentication failed!")
        print("   Please check your credentials and try again")
    
    sys.exit(0 if auth_success else 1) 