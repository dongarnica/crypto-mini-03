#!/usr/bin/env python3
"""
Test script for the Alpaca Crypto Client

This script demonstrates how to use the Alpaca Crypto Client
and tests basic functionality without requiring API credentials.
"""

import os
import sys
from datetime import datetime

# Add the current directory to Python path
sys.path.append('/workspaces/crypto-mini-03')

def test_import():
    """Test that the Alpaca client can be imported."""
    try:
        from alpaca.alpaca_client import AlpacaCryptoClient
        print("✅ Successfully imported AlpacaCryptoClient")
        return True
    except ImportError as e:
        print(f"❌ Failed to import AlpacaCryptoClient: {e}")
        return False

def test_initialization_without_credentials():
    """Test initialization behavior when no credentials are provided."""
    try:
        from alpaca.alpaca_client import AlpacaCryptoClient
        
        # Clear environment variables for this test
        old_key = os.environ.pop('ALPACA_API_KEY', None)
        old_secret = os.environ.pop('ALPACA_SECRET_KEY', None)
        
        try:
            client = AlpacaCryptoClient()
            print("❌ Client initialized without credentials (this shouldn't happen)")
            return False
        except ValueError as e:
            print(f"✅ Correctly raised ValueError when no credentials provided: {e}")
            return True
        finally:
            # Restore environment variables if they existed
            if old_key:
                os.environ['ALPACA_API_KEY'] = old_key
            if old_secret:
                os.environ['ALPACA_SECRET_KEY'] = old_secret
                
    except Exception as e:
        print(f"❌ Unexpected error during initialization test: {e}")
        return False

def test_initialization_with_dummy_credentials():
    """Test initialization with dummy credentials (won't connect but should initialize)."""
    try:
        from alpaca.alpaca_client import AlpacaCryptoClient
        
        client = AlpacaCryptoClient(
            api_key='dummy_key',
            secret_key='dummy_secret',
            paper=True
        )
        print("✅ Successfully initialized client with dummy credentials")
        print(f"   - Paper trading: {client.paper}")
        print(f"   - Base URL: {client.base_url}")
        print(f"   - Supported crypto count: {len(client.SUPPORTED_CRYPTO)}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to initialize client with dummy credentials: {e}")
        return False

def test_class_methods():
    """Test that all expected methods exist on the client class."""
    try:
        from alpaca.alpaca_client import AlpacaCryptoClient
        
        expected_methods = [
            'get_account',
            'check_crypto_eligibility', 
            'get_crypto_assets',
            'get_asset_info',
            'get_portfolio',
            'get_current_price',
            'place_market_order',
            'place_limit_order',
            'place_stop_limit_order',
            'get_orders',
            'get_order',
            'cancel_order',
            'cancel_all_orders',
            'get_crypto_bars',
            'calculate_order_size',
            'get_position',
            'close_position',
            'print_portfolio_summary'
        ]
        
        missing_methods = []
        for method_name in expected_methods:
            if not hasattr(AlpacaCryptoClient, method_name):
                missing_methods.append(method_name)
        
        if missing_methods:
            print(f"❌ Missing methods: {missing_methods}")
            return False
        else:
            print(f"✅ All {len(expected_methods)} expected methods found")
            return True
            
    except Exception as e:
        print(f"❌ Error checking class methods: {e}")
        return False

def show_example_usage():
    """Show example usage of the client."""
    print("\n" + "="*60)
    print("📚 EXAMPLE USAGE")
    print("="*60)
    
    example_code = '''
# 1. Set environment variables (required for real usage):
export ALPACA_API_KEY="your_api_key_here"
export ALPACA_SECRET_KEY="your_secret_key_here"

# 2. Initialize the client:
from alpaca.alpaca_client import AlpacaCryptoClient

client = AlpacaCryptoClient(paper=True)  # Use paper trading

# 3. Check crypto eligibility:
if client.check_crypto_eligibility():
    print("Account ready for crypto trading!")

# 4. Get available crypto assets:
assets = client.get_crypto_assets()
print(f"Available crypto assets: {len(assets)}")

# 5. Get current portfolio:
client.print_portfolio_summary()

# 6. Get current price:
btc_price = client.get_current_price('BTC/USD')
print(f"BTC/USD: ${btc_price:,.2f}")

# 7. Place a market order:
order = client.place_market_order(
    symbol='BTC/USD',
    notional=100.0,  # $100 worth
    side='buy'
)

# 8. Place a limit order:
order = client.place_limit_order(
    symbol='BTC/USD',
    limit_price=50000.0,
    qty=0.001,
    side='buy'
)

# 9. Get historical data:
bars = client.get_crypto_bars(
    symbol='BTC/USD',
    timeframe='1Hour',
    limit=100
)
'''
    
    print(example_code)

def main():
    """Run all tests."""
    print("🧪 TESTING ALPACA CRYPTO CLIENT")
    print("="*50)
    
    tests = [
        ("Import Test", test_import),
        ("No Credentials Test", test_initialization_without_credentials),
        ("Dummy Credentials Test", test_initialization_with_dummy_credentials),
        ("Class Methods Test", test_class_methods)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 {test_name}:")
        if test_func():
            passed += 1
    
    print(f"\n📊 TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The Alpaca client is ready to use.")
        show_example_usage()
    else:
        print("⚠️  Some tests failed. Check the errors above.")

if __name__ == "__main__":
    main()
