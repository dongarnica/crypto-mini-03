#!/usr/bin/env python3
"""
Final test to verify the symbol format fix is working correctly
"""

import os
import sys
from datetime import datetime

# Add directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'alpaca'))

from alpaca_client import AlpacaCryptoClient
from dotenv import load_dotenv
load_dotenv()

def convert_symbol_format(binance_symbol):
    """Convert Binance symbol format to Alpaca format"""
    # Remove USDT suffix and add slash
    if binance_symbol.endswith('USDT'):
        base = binance_symbol[:-4]  # Remove 'USDT'
        return f"{base}/USD"
    return binance_symbol

def test_final_symbol_fix():
    """Test the final symbol format fix with supported symbols only"""
    print("🎯 Final Test: Symbol Format Fix Verification")
    print("=" * 60)
    
    # Initialize Alpaca client
    client = AlpacaCryptoClient(paper=True)
    
    # Test symbols - using only supported symbols (no ADA)
    binance_symbols = ['BTCUSDT', 'ETHUSDT', 'DOTUSDT']
    
    print(f"📊 Testing {len(binance_symbols)} supported symbol conversions...")
    
    success_count = 0
    total_count = len(binance_symbols)
    
    for i, binance_symbol in enumerate(binance_symbols, 1):
        print(f"\n[{i}] Testing {binance_symbol}:")
        
        # Convert symbol format
        alpaca_symbol = convert_symbol_format(binance_symbol)
        print(f"   🔄 Converted: {binance_symbol} → {alpaca_symbol}")
        
        try:
            # Test getting current price from Alpaca
            print(f"   📈 Getting current price for {alpaca_symbol}...")
            current_price = client.get_current_price(alpaca_symbol)
            
            if current_price:
                print(f"   ✅ Success! Price: ${current_price:.2f}")
                success_count += 1
            else:
                print(f"   ❌ Failed: No price returned")
                
        except Exception as e:
            print(f"   ❌ Failed: {str(e)}")
    
    print(f"\n🎯 Test Results:")
    print(f"   Success Rate: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")
    
    if success_count == total_count:
        print("   🎉 ALL TESTS PASSED! Symbol format fix is working correctly.")
    else:
        print("   ⚠️ Some tests failed. Further investigation needed.")

if __name__ == "__main__":
    test_final_symbol_fix()
