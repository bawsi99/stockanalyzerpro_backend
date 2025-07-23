#!/usr/bin/env python3
"""
JWT Secret Generator
Generates a secure random JWT secret key for environment configuration.
"""

import secrets
import string
import os

def generate_jwt_secret(length: int = 64) -> str:
    """Generate a secure random JWT secret key."""
    # Use a combination of letters, digits, and special characters
    characters = string.ascii_letters + string.digits + "!@#$%^&*()_+-=[]{}|;:,.<>?"
    
    # Generate random secret
    secret = ''.join(secrets.choice(characters) for _ in range(length))
    return secret

def main():
    """Main function to generate and display JWT secret."""
    print("🔐 JWT Secret Generator")
    print("=" * 50)
    
    # Generate secret
    jwt_secret = generate_jwt_secret()
    
    print(f"Generated JWT Secret ({len(jwt_secret)} characters):")
    print(f"JWT_SECRET={jwt_secret}")
    print()
    
    # Check if .env file exists
    env_file = ".env"
    if os.path.exists(env_file):
        print(f"⚠️  Warning: {env_file} already exists!")
        print("Please manually add the JWT_SECRET to your existing .env file.")
    else:
        print(f"📝 Creating {env_file} file with JWT secret...")
        try:
            with open(env_file, 'w') as f:
                f.write(f"# JWT Authentication Configuration\n")
                f.write(f"JWT_SECRET={jwt_secret}\n")
                f.write(f"REQUIRE_AUTH=true\n")
                f.write(f"\n")
                f.write(f"# API Keys Configuration\n")
                f.write(f"API_KEYS=test-api-key-1,test-api-key-2\n")
                f.write(f"\n")
                f.write(f"# Environment Configuration\n")
                f.write(f"ENVIRONMENT=development\n")
                f.write(f"DEBUG=true\n")
                f.write(f"LOG_LEVEL=INFO\n")
                f.write(f"\n")
                f.write(f"# Server Configuration\n")
                f.write(f"HOST=0.0.0.0\n")
                f.write(f"PORT=8000\n")
                f.write(f"CORS_ORIGINS=http://localhost:3000,http://localhost:8080\n")
            
            print(f"✅ {env_file} created successfully!")
            print("🔧 You can now start your backend server.")
        except Exception as e:
            print(f"❌ Error creating {env_file}: {e}")
            print("Please manually create the .env file and add the JWT_SECRET.")
    
    print()
    print("🔒 Security Notes:")
    print("- Keep this secret secure and never share it")
    print("- Use different secrets for development and production")
    print("- Consider using environment variables in production")
    print("- Rotate secrets regularly for better security")

if __name__ == "__main__":
    main() 