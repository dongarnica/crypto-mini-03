#!/usr/bin/env python3
"""
Test Alpaca Client with Real Credentials

This script tests the Alpaca client with the provided API credentials.
"""

import os
import sys

# Add the current directory to Python path
sys.path.append('/workspaces/crypto-mini-03')

# Load environment variables from .env file
def load_env_file():
    """Load environment variables from .env file."""
    env_path = '/workspaces/crypto-mini-03/.env'
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value
        print("✅ Environment variables loaded from .env file")
    else:
        print("❌ .env file not found")

def test_alpaca_connection():
    """Test the Alpaca client with real credentials."""
    try:
        from alpaca.alpaca_client import AlpacaCryptoClient
        
        # Initialize client with paper trading
        print("🔄 Initializing Alpaca client...")
        client = AlpacaCryptoClient(paper=True)
        
        print("✅ Client initialized successfully")
        print(f"   - Paper trading: {client.paper}")
        print(f"   - Base URL: {client.base_url}")
        
        # Test account access
        print("\n🔄 Testing account access...")
        account = client.get_account()
        
        print("✅ Account information retrieved:")
        print(f"   - Account ID: {account.get('id', 'N/A')}")
        print(f"   - Status: {account.get('status', 'N/A')}")
        print(f"   - Crypto Status: {account.get('crypto_status', 'N/A')}")
        print(f"   - Equity: ${float(account.get('equity', 0)):,.2f}")
        print(f"   - Cash: ${float(account.get('cash', 0)):,.2f}")
        print(f"   - Buying Power: ${float(account.get('buying_power', 0)):,.2f}")
        
        # Test crypto eligibility
        print("\n🔄 Checking crypto eligibility...")
        is_eligible = client.check_crypto_eligibility()
        
        if is_eligible:
            print("✅ Account is eligible for crypto trading")
            
            # Test getting crypto assets
            print("\n🔄 Getting available crypto assets...")
            assets = client.get_crypto_assets()
            print(f"✅ Found {len(assets)} tradable crypto assets")
            
            # Show first few assets
            if assets:
                print("   Available crypto assets:")
                for asset in assets[:5]:
                    print(f"   - {asset.get('symbol', 'N/A')}: {asset.get('name', 'N/A')}")
            
            # Test getting current price
            print("\n🔄 Getting current BTC price...")
            btc_price = client.get_current_price('BTC/USD')
            if btc_price:
                print(f"✅ BTC/USD: ${btc_price:,.2f}")
            else:
                print("⚠️  Could not retrieve BTC price")
            
            # Test portfolio
            print("\n🔄 Getting portfolio...")
            portfolio = client.get_portfolio()
            if portfolio:
                positions = portfolio.get('positions', [])
                print(f"✅ Portfolio retrieved: {len(positions)} crypto positions")
            
        else:
            print("⚠️  Account not eligible for crypto trading")
            print("   This might be expected for new accounts or if crypto agreement not signed")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing Alpaca connection: {e}")
        return False

def main():
    """Main test function."""
    print("🚀 ALPACA CLIENT REAL CONNECTION TEST")
    print("="*50)
    
    # Load environment variables
    load_env_file()
    
    # Check if credentials are available
    api_key = os.getenv('ALPACA_API_KEY')
    secret_key = os.getenv('ALPACA_SECRET_KEY')
    
    if not api_key or not secret_key:
        print("❌ Alpaca API credentials not found in environment")
        print("   Make sure ALPACA_API_KEY and ALPACA_SECRET_KEY are set in .env file")
        return
    
    print(f"✅ Found API credentials")
    print(f"   - API Key: {api_key[:8]}...")
    print(f"   - Secret Key: {secret_key[:8]}...")
    
    # Test the connection
    success = test_alpaca_connection()
    
    if success:
        print("\n🎉 All tests completed successfully!")
        print("\n📚 Next steps:")
        print("   1. Sign crypto agreement in Alpaca dashboard if needed")
        print("   2. Fund your paper trading account")
        print("   3. Start placing test orders")
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")

if __name__ == "__main__":
    main()
