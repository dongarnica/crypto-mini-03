#!/usr/bin/env python3
"""
Alpaca Crypto Trading Client
============================

A comprehensive client for trading cryptocurrencies on the Alpaca exchange.
Supports all crypto trading features including market/limit/stop-limit orders,
account management, and real-time data.

Features:
- Crypto account management and status checking
- Market, Limit, and Stop Limit orders
- Fractional trading support
- Real-time crypto data and price feeds
- Portfolio management and position tracking
- Comprehensive error handling and logging

Requirements:
- alpaca-trade-api>=3.0.0
- requests>=2.28.0

Author: Crypto Trading Client
Date: June 1, 2025
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple
import requests
import time
from decimal import Decimal

try:
    import alpaca_trade_api as tradeapi
    from alpaca_trade_api.rest import REST, TimeFrame
    from alpaca_trade_api.common import URL
except ImportError:
    print("alpaca-trade-api not installed. Installing...")
    os.system("pip install alpaca-trade-api")
    import alpaca_trade_api as tradeapi
    from alpaca_trade_api.rest import REST, TimeFrame
    from alpaca_trade_api.common import URL


class AlpacaCryptoClient:
    """
    Alpaca Crypto Trading Client with comprehensive crypto trading support.
    """
    
    def __init__(self, api_key: str = None, secret_key: str = None, 
                 base_url: str = None, paper: bool = True):
        """
        Initialize the Alpaca Crypto Client.
        
        Args:
            api_key: Alpaca API key
            secret_key: Alpaca secret key  
            base_url: Base URL for API (paper or live)
            paper: Whether to use paper trading (default: True)
        """
        # Get credentials from environment if not provided
        self.api_key = api_key or os.getenv('ALPACA_API_KEY')
        self.secret_key = secret_key or os.getenv('ALPACA_SECRET_KEY')
        
        if not self.api_key or not self.secret_key:
            raise ValueError("API credentials required. Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables.")
        
        # Set base URL
        if base_url:
            self.base_url = base_url
        elif paper:
            self.base_url = "https://paper-api.alpaca.markets"
        else:
            self.base_url = "https://api.alpaca.markets"
        
        self.paper = paper
        
        # Initialize REST API client
        self.api = REST(
            key_id=self.api_key,
            secret_key=self.secret_key,
            base_url=self.base_url,
            api_version='v2'
        )
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        # Trading limits
        self.MAX_ORDER_NOTIONAL = 200000  # $200k per order limit
        
        # Supported cryptocurrencies (common ones)
        self.SUPPORTED_CRYPTO = [
            'BTC/USD', 'ETH/USD', 'ADA/USD', 'DOT/USD', 'LINK/USD',
            'LTC/USD', 'BCH/USD', 'XLM/USD', 'UNI/USD', 'SOL/USD',
            'MATIC/USD', 'AVAX/USD', 'USDC/USD', 'USDT/USD'
        ]
        
        self.logger.info(f"Alpaca Crypto Client initialized ({'Paper' if paper else 'Live'} trading)")
    
    def get_account(self) -> Dict:
        """
        Get account information including crypto status.
        
        Returns:
            Dict: Account information with crypto_status
        """
        try:
            account = self.api.get_account()
            account_dict = account._raw
            
            self.logger.info(f"Account status: {account_dict.get('status')}")
            self.logger.info(f"Crypto status: {account_dict.get('crypto_status', 'Not available')}")
            
            return account_dict
            
        except Exception as e:
            self.logger.error(f"Error getting account info: {e}")
            raise
    
    def check_crypto_eligibility(self) -> bool:
        """
        Check if account is eligible for crypto trading.
        
        Returns:
            bool: True if crypto trading is active
        """
        try:
            account = self.get_account()
            crypto_status = account.get('crypto_status')
            
            if crypto_status == 'ACTIVE':
                self.logger.info("✅ Account is enabled for crypto trading")
                return True
            elif crypto_status == 'INACTIVE':
                self.logger.warning("⚠️ Account not enabled for crypto trading")
                return False
            elif crypto_status == 'SUBMISSION_FAILED':
                self.logger.error("❌ Crypto account submission failed")
                return False
            else:
                self.logger.warning(f"Unknown crypto status: {crypto_status}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error checking crypto eligibility: {e}")
            return False
    
    def get_crypto_assets(self) -> List[Dict]:
        """
        Get all tradable cryptocurrency assets.
        
        Returns:
            List[Dict]: List of crypto assets
        """
        try:
            assets = self.api.list_assets(status='active', asset_class='crypto')
            crypto_assets = []
            
            for asset in assets:
                asset_dict = asset._raw
                if asset_dict.get('tradable', False):
                    crypto_assets.append(asset_dict)
            
            self.logger.info(f"Found {len(crypto_assets)} tradable crypto assets")
            return crypto_assets
            
        except Exception as e:
            self.logger.error(f"Error getting crypto assets: {e}")
            return []
    
    def get_asset_info(self, symbol: str) -> Optional[Dict]:
        """
        Get detailed information about a specific crypto asset.
        
        Args:
            symbol: Crypto symbol (e.g., 'BTC/USD')
            
        Returns:
            Optional[Dict]: Asset information or None
        """
        try:
            asset = self.api.get_asset(symbol)
            asset_dict = asset._raw
            
            self.logger.info(f"Asset {symbol}: {asset_dict.get('name')} - Tradable: {asset_dict.get('tradable')}")
            return asset_dict
            
        except Exception as e:
            self.logger.error(f"Error getting asset info for {symbol}: {e}")
            return None
    
    def get_portfolio(self) -> Dict:
        """
        Get current portfolio including crypto positions.
        
        Returns:
            Dict: Portfolio information
        """
        try:
            account = self.get_account()
            positions = self.api.list_positions()
            
            portfolio = {
                'account': {
                    'equity': float(account.get('equity', 0)),
                    'cash': float(account.get('cash', 0)),
                    'buying_power': float(account.get('buying_power', 0)),
                    'non_marginable_buying_power': float(account.get('non_marginable_buying_power', 0))
                },
                'positions': []
            }
            
            for position in positions:
                pos_dict = position._raw
                if pos_dict.get('asset_class') == 'crypto':
                    portfolio['positions'].append({
                        'symbol': pos_dict.get('symbol'),
                        'qty': float(pos_dict.get('qty', 0)),
                        'market_value': float(pos_dict.get('market_value', 0)),
                        'unrealized_pl': float(pos_dict.get('unrealized_pl', 0)),
                        'unrealized_plpc': float(pos_dict.get('unrealized_plpc', 0)),
                        'avg_entry_price': float(pos_dict.get('avg_entry_price', 0))
                    })
            
            self.logger.info(f"Portfolio: {len(portfolio['positions'])} crypto positions")
            return portfolio
            
        except Exception as e:
            self.logger.error(f"Error getting portfolio: {e}")
            return {}
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Get current price for a crypto symbol.
        
        Args:
            symbol: Crypto symbol (e.g., 'BTC/USD')
            
        Returns:
            Optional[float]: Current price or None
        """
        try:
            latest_trade = self.api.get_latest_crypto_trade(symbol)
            if latest_trade:
                price = float(latest_trade.price)
                self.logger.info(f"{symbol} current price: ${price:,.2f}")
                return price
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting current price for {symbol}: {e}")
            return None
    
    def place_market_order(self, symbol: str, qty: float = None, notional: float = None, 
                          side: str = 'buy', commission: float = None) -> Optional[Dict]:
        """
        Place a market order for crypto.
        
        Args:
            symbol: Crypto symbol (e.g., 'BTC/USD')
            qty: Quantity of crypto to trade
            notional: Notional amount in quote currency
            side: 'buy' or 'sell'
            commission: Optional commission amount
            
        Returns:
            Optional[Dict]: Order information or None
        """
        try:
            # Validate order size
            if notional and notional > self.MAX_ORDER_NOTIONAL:
                raise ValueError(f"Order notional ${notional:,.2f} exceeds limit of ${self.MAX_ORDER_NOTIONAL:,.2f}")
            
            # Prepare order parameters
            order_params = {
                'symbol': symbol,
                'side': side,
                'type': 'market',
                'time_in_force': 'ioc'  # Immediate or Cancel for crypto
            }
            
            if qty is not None:
                order_params['qty'] = str(qty)
            elif notional is not None:
                order_params['notional'] = str(notional)
            else:
                raise ValueError("Either qty or notional must be specified")
            
            if commission is not None:
                order_params['commission'] = str(commission)
            
            # Place order
            order = self.api.submit_order(**order_params)
            order_dict = order._raw
            
            self.logger.info(f"Market order placed: {side} {symbol} - Order ID: {order_dict.get('id')}")
            return order_dict
            
        except Exception as e:
            self.logger.error(f"Error placing market order: {e}")
            return None
    
    def place_limit_order(self, symbol: str, limit_price: float, qty: float = None, 
                         notional: float = None, side: str = 'buy', 
                         time_in_force: str = 'gtc', commission: float = None) -> Optional[Dict]:
        """
        Place a limit order for crypto.
        
        Args:
            symbol: Crypto symbol (e.g., 'BTC/USD')
            limit_price: Limit price for the order
            qty: Quantity of crypto to trade
            notional: Notional amount in quote currency
            side: 'buy' or 'sell'
            time_in_force: 'gtc' or 'ioc'
            commission: Optional commission amount
            
        Returns:
            Optional[Dict]: Order information or None
        """
        try:
            # Validate order size
            if notional and notional > self.MAX_ORDER_NOTIONAL:
                raise ValueError(f"Order notional ${notional:,.2f} exceeds limit of ${self.MAX_ORDER_NOTIONAL:,.2f}")
            
            # Prepare order parameters
            order_params = {
                'symbol': symbol,
                'side': side,
                'type': 'limit',
                'limit_price': str(limit_price),
                'time_in_force': time_in_force
            }
            
            if qty is not None:
                order_params['qty'] = str(qty)
            elif notional is not None:
                order_params['notional'] = str(notional)
            else:
                raise ValueError("Either qty or notional must be specified")
            
            if commission is not None:
                order_params['commission'] = str(commission)
            
            # Place order
            order = self.api.submit_order(**order_params)
            order_dict = order._raw
            
            self.logger.info(f"Limit order placed: {side} {symbol} @ ${limit_price:.2f} - Order ID: {order_dict.get('id')}")
            return order_dict
            
        except Exception as e:
            self.logger.error(f"Error placing limit order: {e}")
            return None
    
    def place_stop_limit_order(self, symbol: str, stop_price: float, limit_price: float,
                              qty: float = None, notional: float = None, side: str = 'sell',
                              time_in_force: str = 'gtc', commission: float = None) -> Optional[Dict]:
        """
        Place a stop-limit order for crypto.
        
        Args:
            symbol: Crypto symbol (e.g., 'BTC/USD')
            stop_price: Stop price to trigger the order
            limit_price: Limit price for execution
            qty: Quantity of crypto to trade
            notional: Notional amount in quote currency
            side: 'buy' or 'sell'
            time_in_force: 'gtc' or 'ioc'
            commission: Optional commission amount
            
        Returns:
            Optional[Dict]: Order information or None
        """
        try:
            # Validate order size
            if notional and notional > self.MAX_ORDER_NOTIONAL:
                raise ValueError(f"Order notional ${notional:,.2f} exceeds limit of ${self.MAX_ORDER_NOTIONAL:,.2f}")
            
            # Prepare order parameters
            order_params = {
                'symbol': symbol,
                'side': side,
                'type': 'stop_limit',
                'stop_price': str(stop_price),
                'limit_price': str(limit_price),
                'time_in_force': time_in_force
            }
            
            if qty is not None:
                order_params['qty'] = str(qty)
            elif notional is not None:
                order_params['notional'] = str(notional)
            else:
                raise ValueError("Either qty or notional must be specified")
            
            if commission is not None:
                order_params['commission'] = str(commission)
            
            # Place order
            order = self.api.submit_order(**order_params)
            order_dict = order._raw
            
            self.logger.info(f"Stop-limit order placed: {side} {symbol} stop@${stop_price:.2f} limit@${limit_price:.2f} - Order ID: {order_dict.get('id')}")
            return order_dict
            
        except Exception as e:
            self.logger.error(f"Error placing stop-limit order: {e}")
            return None
    
    def get_orders(self, status: str = 'all', limit: int = 50, symbols: List[str] = None) -> List[Dict]:
        """
        Get orders with optional filtering.
        
        Args:
            status: Order status filter ('all', 'open', 'closed')
            limit: Maximum number of orders to return
            symbols: List of symbols to filter by
            
        Returns:
            List[Dict]: List of orders
        """
        try:
            orders = self.api.list_orders(
                status=status,
                limit=limit,
                symbols=','.join(symbols) if symbols else None
            )
            
            order_list = []
            for order in orders:
                order_dict = order._raw
                # Filter for crypto orders only
                if order_dict.get('asset_class') == 'crypto':
                    order_list.append(order_dict)
            
            self.logger.info(f"Retrieved {len(order_list)} crypto orders")
            return order_list
            
        except Exception as e:
            self.logger.error(f"Error getting orders: {e}")
            return []
    
    def get_order(self, order_id: str) -> Optional[Dict]:
        """
        Get a specific order by ID.
        
        Args:
            order_id: Order ID
            
        Returns:
            Optional[Dict]: Order information or None
        """
        try:
            order = self.api.get_order(order_id)
            order_dict = order._raw
            
            self.logger.info(f"Order {order_id}: {order_dict.get('status')}")
            return order_dict
            
        except Exception as e:
            self.logger.error(f"Error getting order {order_id}: {e}")
            return None
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            bool: True if successful
        """
        try:
            self.api.cancel_order(order_id)
            self.logger.info(f"Order {order_id} canceled successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error canceling order {order_id}: {e}")
            return False
    
    def cancel_all_orders(self) -> bool:
        """
        Cancel all open orders.
        
        Returns:
            bool: True if successful
        """
        try:
            self.api.cancel_all_orders()
            self.logger.info("All orders canceled successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error canceling all orders: {e}")
            return False
    
    def get_crypto_bars(self, symbol: str, timeframe: str = '1Day', 
                       start: datetime = None, end: datetime = None, 
                       limit: int = 1000) -> List[Dict]:
        """
        Get historical crypto price bars.
        
        Args:
            symbol: Crypto symbol (e.g., 'BTC/USD')
            timeframe: Bar timeframe ('1Min', '5Min', '15Min', '1Hour', '1Day')
            start: Start datetime
            end: End datetime
            limit: Maximum number of bars
            
        Returns:
            List[Dict]: List of price bars
        """
        try:
            # Map timeframe string to TimeFrame enum
            timeframe_map = {
                '1Min': TimeFrame.Minute,
                '5Min': TimeFrame(5, 'Min'),
                '15Min': TimeFrame(15, 'Min'),
                '1Hour': TimeFrame.Hour,
                '1Day': TimeFrame.Day
            }
            
            tf = timeframe_map.get(timeframe, TimeFrame.Day)
            
            # Default to last 30 days if no dates provided
            if not end:
                end = datetime.now()
            if not start:
                start = end - timedelta(days=30)
            
            bars = self.api.get_crypto_bars(
                symbol,
                tf,
                start=start.isoformat(),
                end=end.isoformat(),
                limit=limit
            )
            
            bar_list = []
            for bar in bars:
                bar_dict = bar._raw
                bar_list.append(bar_dict)
            
            self.logger.info(f"Retrieved {len(bar_list)} bars for {symbol}")
            return bar_list
            
        except Exception as e:
            self.logger.error(f"Error getting crypto bars for {symbol}: {e}")
            return []
    
    def calculate_order_size(self, symbol: str, notional_amount: float) -> Optional[float]:
        """
        Calculate the quantity for a given notional amount.
        
        Args:
            symbol: Crypto symbol
            notional_amount: Amount in quote currency
            
        Returns:
            Optional[float]: Calculated quantity
        """
        try:
            current_price = self.get_current_price(symbol)
            if current_price:
                quantity = notional_amount / current_price
                self.logger.info(f"${notional_amount:.2f} of {symbol} = {quantity:.9f} units")
                return quantity
            return None
            
        except Exception as e:
            self.logger.error(f"Error calculating order size: {e}")
            return None
    
    def get_position(self, symbol: str) -> Optional[Dict]:
        """
        Get position for a specific crypto symbol.
        
        Args:
            symbol: Crypto symbol
            
        Returns:
            Optional[Dict]: Position information or None
        """
        try:
            position = self.api.get_position(symbol)
            position_dict = position._raw
            
            self.logger.info(f"Position {symbol}: {position_dict.get('qty')} units")
            return position_dict
            
        except Exception as e:
            # Position might not exist
            if "position does not exist" in str(e).lower():
                return None
            self.logger.error(f"Error getting position for {symbol}: {e}")
            return None
    
    def close_position(self, symbol: str, qty: float = None, percentage: float = None) -> Optional[Dict]:
        """
        Close a crypto position.
        
        Args:
            symbol: Crypto symbol
            qty: Specific quantity to close
            percentage: Percentage of position to close (0.0 to 1.0)
            
        Returns:
            Optional[Dict]: Order information or None
        """
        try:
            position = self.get_position(symbol)
            if not position:
                self.logger.warning(f"No position found for {symbol}")
                return None
            
            current_qty = float(position.get('qty', 0))
            if current_qty == 0:
                self.logger.warning(f"No quantity to close for {symbol}")
                return None
            
            # Calculate quantity to close
            if percentage is not None:
                close_qty = abs(current_qty) * percentage
            elif qty is not None:
                close_qty = min(qty, abs(current_qty))
            else:
                close_qty = abs(current_qty)  # Close entire position
            
            # Determine side (opposite of current position)
            side = 'sell' if current_qty > 0 else 'buy'
            
            # Place market order to close position
            order = self.place_market_order(symbol, qty=close_qty, side=side)
            
            if order:
                self.logger.info(f"Position close order placed for {symbol}: {close_qty} units")
            
            return order
            
        except Exception as e:
            self.logger.error(f"Error closing position for {symbol}: {e}")
            return None
    
    def print_portfolio_summary(self) -> None:
        """Print a formatted portfolio summary."""
        try:
            portfolio = self.get_portfolio()
            
            print("\n" + "="*60)
            print("📊 ALPACA CRYPTO PORTFOLIO SUMMARY")
            print("="*60)
            
            account = portfolio['account']
            print(f"💰 Account Equity: ${account['equity']:,.2f}")
            print(f"💵 Cash: ${account['cash']:,.2f}")
            print(f"🔋 Buying Power: ${account['buying_power']:,.2f}")
            print(f"🔒 Non-Marginable BP: ${account['non_marginable_buying_power']:,.2f}")
            
            positions = portfolio['positions']
            if positions:
                print(f"\n🏆 CRYPTO POSITIONS ({len(positions)})")
                print("-" * 60)
                
                total_market_value = 0
                total_unrealized_pl = 0
                
                for pos in positions:
                    symbol = pos['symbol']
                    qty = pos['qty']
                    market_value = pos['market_value']
                    unrealized_pl = pos['unrealized_pl']
                    unrealized_plpc = pos['unrealized_plpc']
                    avg_entry = pos['avg_entry_price']
                    
                    total_market_value += market_value
                    total_unrealized_pl += unrealized_pl
                    
                    pl_color = "🟢" if unrealized_pl >= 0 else "🔴"
                    
                    print(f"{symbol:>12}: {qty:>12.6f} units")
                    print(f"{'':>12}  Market Value: ${market_value:>10,.2f}")
                    print(f"{'':>12}  Avg Entry: ${avg_entry:>12,.2f}")
                    print(f"{'':>12}  P&L: {pl_color} ${unrealized_pl:>10,.2f} ({unrealized_plpc:>6.1%})")
                    print("-" * 40)
                
                print(f"\n📈 TOTAL POSITIONS VALUE: ${total_market_value:,.2f}")
                print(f"📊 TOTAL UNREALIZED P&L: ${total_unrealized_pl:,.2f}")
                
            else:
                print("\n📭 No crypto positions found")
            
            print("="*60)
            
        except Exception as e:
            self.logger.error(f"Error printing portfolio summary: {e}")


def main():
    """Example usage of the Alpaca Crypto Client."""
    print("🚀 Alpaca Crypto Trading Client Demo")
    print("=" * 50)
    
    try:
        # Initialize client (paper trading by default)
        client = AlpacaCryptoClient(paper=True)
        
        # Check crypto eligibility
        print("\n1️⃣ Checking crypto trading eligibility...")
        is_eligible = client.check_crypto_eligibility()
        
        if not is_eligible:
            print("❌ Account not eligible for crypto trading")
            return
        
        # Get available crypto assets
        print("\n2️⃣ Getting available crypto assets...")
        crypto_assets = client.get_crypto_assets()
        print(f"Found {len(crypto_assets)} tradable crypto assets")
        
        # Show first few assets
        for asset in crypto_assets[:5]:
            print(f"  - {asset['symbol']}: {asset['name']}")
        
        # Get current portfolio
        print("\n3️⃣ Portfolio overview...")
        client.print_portfolio_summary()
        
        # Get current BTC price
        print("\n4️⃣ Getting current prices...")
        btc_price = client.get_current_price('BTC/USD')
        eth_price = client.get_current_price('ETH/USD')
        
        if btc_price:
            print(f"BTC/USD: ${btc_price:,.2f}")
        if eth_price:
            print(f"ETH/USD: ${eth_price:,.2f}")
        
        # Example orders (commented out to avoid accidental execution)
        print("\n5️⃣ Example order functions (not executed)...")
        print("# Buy $100 worth of BTC")
        print("# client.place_market_order('BTC/USD', notional=100, side='buy')")
        print("# ")
        print("# Buy 0.001 BTC at $50,000")
        print("# client.place_limit_order('BTC/USD', limit_price=50000, qty=0.001, side='buy')")
        print("# ")
        print("# Sell 0.001 BTC with stop at $45,000, limit at $44,500")
        print("# client.place_stop_limit_order('BTC/USD', stop_price=45000, limit_price=44500, qty=0.001, side='sell')")
        
        # Get recent orders
        print("\n6️⃣ Recent orders...")
        recent_orders = client.get_orders(limit=5)
        if recent_orders:
            for order in recent_orders:
                print(f"  Order: {order['side']} {order['symbol']} - Status: {order['status']}")
        else:
            print("  No recent crypto orders found")
        
        print("\n✅ Demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")


if __name__ == "__main__":
    main()