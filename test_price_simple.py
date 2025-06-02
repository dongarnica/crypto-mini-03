#!/usr/bin/env python3
"""
Simple test script to get current prices from Binance
"""

import os
import sys
from datetime import datetime

# Add the binance directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'binance'))
from binance_client import BinanceUSClient

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

def test_current_prices():
    """Test getting current prices from Binance."""
    print("🔍 Testing Binance Current Price Retrieval")
    print("=" * 50)
    
    # Initialize client
    api_key = os.getenv('BINANCE_API_KEY')
    api_secret = os.getenv('BINANCE_API_SECRET')
    client = BinanceUSClient(api_key=api_key, api_secret=api_secret)
    
    # Test symbols (start with just a few)
    test_symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT']
    
    print(f"📊 Testing {len(test_symbols)} symbols...")
    
    for i, symbol in enumerate(test_symbols, 1):
        print(f"\n[{i}] Testing {symbol}:")
        
        try:
            # Get current price
            print("   Getting current price...")
            current_price = client.get_price(symbol)
            print(f"   ✅ Current price: ${current_price['price']:.2f}")
            
            # Get average price
            print("   Getting average price...")
            avg_price = client.get_avg_price(symbol)
            print(f"   ✅ Average price: ${avg_price['price']:.2f}")
            
            # Get 24hr ticker
            print("   Getting 24hr ticker...")
            ticker_24hr = client.get_24hr_ticker(symbol)
            print(f"   ✅ 24h change: {ticker_24hr['priceChangePercent']:+.2f}%")
            print(f"   ✅ Volume: {ticker_24hr['volume']:,.0f}")
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
    
    print(f"\n🎉 Price test completed!")

if __name__ == "__main__":
    test_current_prices()
