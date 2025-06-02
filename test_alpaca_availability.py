#!/usr/bin/env python3
"""
Test script to check what crypto symbols are available in Alpaca
"""

import os
import sys
from datetime import datetime

# Add directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'alpaca'))

from alpaca_client import AlpacaCryptoClient
from dotenv import load_dotenv
load_dotenv()

def test_alpaca_crypto_availability():
    """Test what crypto symbols are available in Alpaca"""
    print("🔍 Testing Alpaca Crypto Symbol Availability")
    print("=" * 60)
    
    # Initialize Alpaca client
    client = AlpacaCryptoClient(paper=True)
    
    # Check crypto eligibility
    print("1️⃣ Checking crypto trading eligibility...")
    is_eligible = client.check_crypto_eligibility()
    print(f"   Crypto eligible: {is_eligible}")
    
    # Get all available crypto assets
    print("\n2️⃣ Getting available crypto assets...")
    crypto_assets = client.get_crypto_assets()
    
    if crypto_assets:
        print(f"   Found {len(crypto_assets)} tradable crypto assets:")
        for asset in crypto_assets:
            symbol = asset.get('symbol', 'N/A')
            name = asset.get('name', 'N/A')
            tradable = asset.get('tradable', False)
            print(f"   - {symbol}: {name} (Tradable: {tradable})")
    else:
        print("   No crypto assets found")
    
    # Test the supported crypto list from the client
    print("\n3️⃣ Client's supported crypto list:")
    for symbol in client.SUPPORTED_CRYPTO:
        print(f"   - {symbol}")
    
    # Test specific symbols
    print("\n4️⃣ Testing specific crypto symbols...")
    test_symbols = ['BTC/USD', 'ETH/USD', 'ADA/USD', 'BTCUSD', 'ETHUSD', 'ADAUSD']
    
    for symbol in test_symbols:
        print(f"\n   Testing {symbol}:")
        try:
            asset_info = client.get_asset_info(symbol)
            if asset_info:
                print(f"   ✅ Asset found: {asset_info.get('name', 'N/A')} - Tradable: {asset_info.get('tradable', False)}")
            else:
                print(f"   ❌ Asset not found")
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")

if __name__ == "__main__":
    test_alpaca_crypto_availability()
