#!/usr/bin/env python3
"""
Manual BTC Trade Script
=======================

Places a manual $10 market buy order for BTC on Alpaca.

Usage: python manual_btc_trade.py
"""

import os
import sys
from dotenv import load_dotenv

# Add the current directory to Python path
sys.path.append('/workspaces/crypto-mini-03')

from alpaca.alpaca_client import AlpacaCryptoClient

def main():
    """Execute a manual BTC trade."""
    print("🚀 Manual BTC Trade Script")
    print("=" * 40)
    
    # Load environment variables
    load_dotenv('/workspaces/crypto-mini-03/.env')
    
    try:
        # Initialize Alpaca client
        print("📡 Connecting to Alpaca...")
        client = AlpacaCryptoClient()
        
        # Check account status
        account = client.get_account()
        if not account:
            print("❌ Failed to connect to Alpaca account")
            return
            
        print(f"✅ Connected to account: {account.get('account_number', 'Unknown')}")
        print(f"💰 Buying power: ${float(account.get('buying_power', 0)):,.2f}")
        
        # Get current BTC price
        print("\n📈 Getting current BTC price...")
        btc_symbol = 'BTC/USD'
        current_price = client.get_current_price(btc_symbol)
        
        if current_price:
            print(f"💰 Current BTC price: ${current_price:,.2f}")
            
            # Calculate how much BTC we can buy with $10
            btc_amount = 10.0 / current_price
            print(f"📊 $10 will buy approximately {btc_amount:.8f} BTC")
        else:
            print("⚠️  Could not get current BTC price from Alpaca")
            print("🔄 Proceeding with notional order (Alpaca will calculate amount)")
        
        # Confirm trade
        print(f"\n🎯 Ready to place trade:")
        print(f"   Symbol: {btc_symbol}")
        print(f"   Side: BUY")
        print(f"   Amount: $10.00")
        print(f"   Order Type: MARKET")
        
        response = input("\n❓ Proceed with trade? (y/N): ").strip().lower()
        
        if response != 'y':
            print("❌ Trade cancelled by user")
            return
        
        # Place the trade
        print("\n🔄 Placing market buy order...")
        order = client.place_market_order(
            symbol=btc_symbol,
            notional=10.0,  # $10 worth of BTC
            side='buy'
        )
        
        if order:
            print("✅ Order placed successfully!")
            print(f"📋 Order Details:")
            print(f"   Order ID: {order.get('id')}")
            print(f"   Symbol: {order.get('symbol')}")
            print(f"   Side: {order.get('side').upper()}")
            print(f"   Status: {order.get('status')}")
            print(f"   Notional: ${order.get('notional', 'N/A')}")
            print(f"   Quantity: {order.get('qty', 'N/A')} BTC")
            
            # Wait a moment and check order status
            import time
            print("\n⏳ Checking order status...")
            time.sleep(2)
            
            updated_order = client.get_order(order['id'])
            if updated_order:
                print(f"📊 Updated Status: {updated_order.get('status')}")
                if updated_order.get('filled_qty'):
                    print(f"✅ Filled: {updated_order.get('filled_qty')} BTC")
                if updated_order.get('filled_avg_price'):
                    print(f"💰 Average Fill Price: ${float(updated_order.get('filled_avg_price')):,.2f}")
        else:
            print("❌ Failed to place order")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
