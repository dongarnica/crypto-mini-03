#!/usr/bin/env python3
"""
Test script to verify Alpaca symbol conversion is working correctly
"""

import os
import sys
from datetime import datetime

# Add directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'alpaca'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'trading'))

from alpaca_client import AlpacaCryptoClient
from config.symbol_manager import symbol_manager
from dotenv import load_dotenv
load_dotenv()

def convert_symbol_format(binance_symbol):
    """Convert Binance symbol format to Alpaca format using SymbolManager"""
    return symbol_manager.binance_to_alpaca_format(binance_symbol)

def test_alpaca_symbol_conversion():
    """Test Alpaca symbol conversion and price retrieval"""
    print("🔍 Testing Alpaca Symbol Conversion and Price Retrieval")
    print("=" * 60)
    
    # Initialize Alpaca client
    api_key = os.getenv('ALPACA_API_KEY')
    secret_key = os.getenv('ALPACA_SECRET_KEY')
    base_url = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
    
    client = AlpacaCryptoClient(api_key=api_key, secret_key=secret_key, base_url=base_url)
    
    # Test symbols in Binance format
    binance_symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT']
    
    print(f"📊 Testing {len(binance_symbols)} symbol conversions...")
    
    for i, binance_symbol in enumerate(binance_symbols, 1):
        print(f"\n[{i}] Testing {binance_symbol}:")
        
        # Convert symbol format
        alpaca_symbol = convert_symbol_format(binance_symbol)
        print(f"   🔄 Converted: {binance_symbol} → {alpaca_symbol}")
        
        try:
            # Test getting current price from Alpaca
            print(f"   📈 Getting current price for {alpaca_symbol}...")
            current_price = client.get_current_price(alpaca_symbol)
            print(f"   ✅ Alpaca price: ${current_price:.2f}")
            
        except Exception as e:
            print(f"   ❌ Error getting price for {alpaca_symbol}: {str(e)}")
            
    print("\n🎯 Testing complete!")

if __name__ == "__main__":
    test_alpaca_symbol_conversion()
