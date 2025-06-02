#!/usr/bin/env python3
"""
Alpaca Test Trade Script

This script places a test trade for $10 worth of BTC on Alpaca paper trading.
"""

import os
import sys
import time
from datetime import datetime

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

def main():
    """Place a test trade for $10 worth of BTC."""
    print("🚀 ALPACA BTC TEST TRADE")
    print("="*40)
    print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("💰 Order: BUY $10 worth of BTC/USD")
    print("🧪 Environment: Paper Trading")
    print("="*40)
    
    # Load environment variables
    load_env_file()
    
    try:
        from alpaca.alpaca_client import AlpacaCryptoClient
        
        # Initialize client
        print("\n1️⃣ Initializing Alpaca client...")
        client = AlpacaCryptoClient(paper=True)
        print("✅ Client initialized successfully")
        
        # Check account status
        print("\n2️⃣ Checking account status...")
        account = client.get_account()
        print(f"✅ Account Status: {account.get('status')}")
        print(f"✅ Crypto Status: {account.get('crypto_status')}")
        print(f"💰 Available Cash: ${float(account.get('cash', 0)):,.2f}")
        print(f"💳 Buying Power: ${float(account.get('buying_power', 0)):,.2f}")
        
        # Check if we have enough buying power
        buying_power = float(account.get('buying_power', 0))
        if buying_power < 10:
            print(f"❌ Insufficient buying power: ${buying_power:.2f} < $10.00")
            return
        
        # Get current BTC price for reference (even if the API method has issues)
        print("\n3️⃣ Getting current market info...")
        try:
            # Try to get BTC price
            btc_price = client.get_current_price('BTC/USD')
            if btc_price:
                print(f"📈 BTC/USD Current Price: ${btc_price:,.2f}")
                btc_quantity = 10.0 / btc_price
                print(f"📊 $10 will buy approximately: {btc_quantity:.8f} BTC")
            else:
                print("⚠️  Could not retrieve current BTC price (API method issue)")
                print("📊 $10 will buy approximately 0.0001-0.0002 BTC (estimated)")
        except Exception as e:
            print(f"⚠️  Price retrieval failed: {e}")
            print("📊 Proceeding with $10 notional order anyway...")
        
        # Show current portfolio before trade
        print("\n4️⃣ Current portfolio before trade...")
        try:
            portfolio = client.get_portfolio()
            positions = portfolio.get('positions', [])
            btc_position = None
            
            for pos in positions:
                if pos['symbol'] == 'BTC/USD':
                    btc_position = pos
                    break
            
            if btc_position:
                print(f"📊 Current BTC Position: {btc_position['qty']:.8f} BTC")
                print(f"💰 Market Value: ${btc_position['market_value']:.2f}")
            else:
                print("📊 No current BTC position")
                
        except Exception as e:
            print(f"⚠️  Could not retrieve current positions: {e}")
        
        # Place the order
        print("\n5️⃣ Placing BTC buy order...")
        print("🔄 Submitting market order for $10 worth of BTC/USD...")
        
        order = client.place_market_order(
            symbol='BTC/USD',
            side='buy',
            notional=10.0  # $10 worth
        )
        
        if order:
            print("✅ ORDER PLACED SUCCESSFULLY!")
            print("="*40)
            print(f"📋 Order ID: {order.get('id')}")
            print(f"📊 Symbol: {order.get('symbol')}")
            print(f"🔄 Side: {order.get('side').upper()}")
            print(f"📈 Order Type: {order.get('order_type').upper()}")
            print(f"💰 Notional Amount: ${order.get('notional', 'N/A')}")
            print(f"⏰ Time in Force: {order.get('time_in_force').upper()}")
            print(f"📅 Status: {order.get('status')}")
            print(f"🕐 Created: {order.get('created_at')}")
            print("="*40)
            
            # Wait a moment for order processing
            print("\n6️⃣ Waiting for order processing...")
            time.sleep(3)
            
            # Check order status
            order_id = order.get('id')
            updated_order = client.get_order(order_id)
            
            if updated_order:
                print("📊 Updated Order Status:")
                print(f"   Status: {updated_order.get('status')}")
                print(f"   Filled Qty: {updated_order.get('filled_qty', 'N/A')}")
                print(f"   Filled Avg Price: ${float(updated_order.get('filled_avg_price', 0)):,.2f}" if updated_order.get('filled_avg_price') else "   Filled Avg Price: N/A")
            
            # Show updated portfolio
            print("\n7️⃣ Portfolio after trade...")
            try:
                portfolio = client.get_portfolio()
                positions = portfolio.get('positions', [])
                btc_position = None
                
                for pos in positions:
                    if pos['symbol'] == 'BTC/USD':
                        btc_position = pos
                        break
                
                if btc_position:
                    print(f"✅ Updated BTC Position: {btc_position['qty']:.8f} BTC")
                    print(f"💰 Market Value: ${btc_position['market_value']:.2f}")
                    print(f"📊 Unrealized P&L: ${btc_position['unrealized_pl']:.2f}")
                else:
                    print("⚠️  BTC position not found in portfolio yet (may need time to settle)")
                    
            except Exception as e:
                print(f"⚠️  Could not retrieve updated positions: {e}")
            
            print("\n🎉 TEST TRADE COMPLETED SUCCESSFULLY!")
            print("💡 This was a paper trade - no real money was spent")
            
        else:
            print("❌ ORDER FAILED!")
            print("Check the logs above for error details")
            
    except Exception as e:
        print(f"❌ Error during trade execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
