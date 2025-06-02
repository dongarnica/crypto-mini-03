#!/usr/bin/env python3
"""
Test Binance Price Fallback System
==================================

Test script to verify that the Binance price fallback system
works correctly when Alpaca prices are unavailable.
"""

import sys
sys.path.append('/workspaces/crypto-mini-03')

from trading.strategy_engine_refactored import TradingStrategyEngine
from trading.models import TradingConfig
from binance.binance_client import BinanceUSClient

def test_price_fallback():
    """Test the price fallback mechanism."""
    print("🧪 Testing Binance Price Fallback System")
    print("=" * 50)
    
    # Create a minimal config
    config = TradingConfig(
        paper_trading=True,
        log_level="INFO"
    )
    
    # Create engine instance
    engine = TradingStrategyEngine(config)
    
    # Initialize just the Binance client
    try:
        engine.binance_client = BinanceUSClient()
        print("✅ Binance client initialized")
    except Exception as e:
        print(f"❌ Failed to initialize Binance client: {e}")
        return
    
    # Test symbols to check
    test_symbols = [
        ('BTC/USD', 'BTCUSDT'),
        ('ETH/USD', 'ETHUSDT'),
        ('ADA/USD', 'ADAUSDT'),
        ('DOT/USD', 'DOTUSDT'),
        ('LINK/USD', 'LINKUSDT')
    ]
    
    print("\n💰 Testing Price Retrieval:")
    print("-" * 50)
    
    for alpaca_symbol, binance_symbol in test_symbols:
        try:
            # Test our fallback method (this will skip Alpaca since trading_client is None)
            price = engine.get_current_price_with_fallback(alpaca_symbol, binance_symbol)
            
            if price > 0:
                print(f"✅ {binance_symbol:10} -> ${price:>10,.2f}")
            else:
                print(f"❌ {binance_symbol:10} -> No price data")
                
        except Exception as e:
            print(f"❌ {binance_symbol:10} -> Error: {e}")
    
    print("\n🔍 Testing Direct Binance Calls:")
    print("-" * 50)
    
    for _, binance_symbol in test_symbols:
        try:
            # Test direct Binance API calls
            price_data = engine.binance_client.get_price(binance_symbol)
            avg_data = engine.binance_client.get_avg_price(binance_symbol)
            
            if price_data and 'price' in price_data:
                current_price = float(price_data['price'])
                avg_price = float(avg_data['price'])
                print(f"✅ {binance_symbol:10} -> Current: ${current_price:>10,.2f} | Avg: ${avg_price:>10,.2f}")
            else:
                print(f"❌ {binance_symbol:10} -> No data returned")
                
        except Exception as e:
            print(f"❌ {binance_symbol:10} -> Error: {e}")
    
    print("\n✅ Price fallback test completed!")

if __name__ == "__main__":
    test_price_fallback()
