import requests
import time
import hmac
import hashlib
import json
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import pandas as pd
from urllib.parse import urlencode
import logging

class BinanceUSClient:
    """
    Binance US API client for cryptocurrency algorithmic trading data access.
    
    Handles retrieval of candlestick data, order books, and trade history
    with proper error handling and rate limiting.
    """
    
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None, 
                 testnet: bool = False):
        """
        Initialize the Binance US client.
        
        Args:
            api_key: Binance US API key (optional for public endpoints)
            api_secret: Binance US API secret (optional for public endpoints)
            testnet: Whether to use testnet (currently not available for Binance US)
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "https://api.binance.us"
        
        # Rate limiting
        self.last_request_time = 0
        self.request_interval = 0.1  # 100ms between requests
        self.weight_used = 0
        self.weight_reset_time = time.time() + 60
        self.max_weight = 1200  # Binance US limit per minute
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def _rate_limit(self, weight: int = 1):
        """Implement rate limiting to avoid API limits."""
        current_time = time.time()
        
        # Reset weight counter if minute has passed
        if current_time > self.weight_reset_time:
            self.weight_used = 0
            self.weight_reset_time = current_time + 60
        
        # Check if we're approaching weight limit
        if self.weight_used + weight > self.max_weight:
            sleep_time = self.weight_reset_time - current_time
            if sleep_time > 0:
                self.logger.warning(f"Rate limit approaching, sleeping for {sleep_time:.2f}s")
                time.sleep(sleep_time)
                self.weight_used = 0
                self.weight_reset_time = time.time() + 60
        
        # Ensure minimum time between requests
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.request_interval:
            time.sleep(self.request_interval - time_since_last)
        
        self.weight_used += weight
        self.last_request_time = time.time()
    
    def _create_signature(self, params: Dict) -> str:
        """Create HMAC SHA256 signature for authenticated requests."""
        if not self.api_secret:
            raise ValueError("API secret required for authenticated requests")
        
        query_string = urlencode(params)
        return hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    def _make_request(self, endpoint: str, params: Dict = None, 
                     authenticated: bool = False, weight: int = 1) -> Dict:
        """Make HTTP request to Binance US API with error handling."""
        self._rate_limit(weight)
        
        params = params or {}
        url = f"{self.base_url}{endpoint}"
        headers = {}
        
        if authenticated:
            if not self.api_key:
                raise ValueError("API key required for authenticated requests")
            
            params['timestamp'] = int(time.time() * 1000)
            params['signature'] = self._create_signature(params)
            headers['X-MBX-APIKEY'] = self.api_key
        
        try:
            response = requests.get(url, params=params, headers=headers, timeout=30)
            response.raise_for_status()
            
            # Update rate limit info from headers
            if 'x-mbx-used-weight-1m' in response.headers:
                self.weight_used = int(response.headers['x-mbx-used-weight-1m'])
            
            return response.json()
            
        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', 60))
                self.logger.error(f"Rate limit exceeded. Retrying after {retry_after}s")
                time.sleep(retry_after)
                return self._make_request(endpoint, params, authenticated, weight)
            else:
                error_msg = f"HTTP {response.status_code}: {response.text}"
                self.logger.error(error_msg)
                raise Exception(error_msg)
        
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request failed: {str(e)}")
            raise
    
    def get_server_time(self) -> Dict:
        """Get server time."""
        return self._make_request("/api/v3/time")
    
    def get_exchange_info(self) -> Dict:
        """Get exchange trading rules and symbol information."""
        return self._make_request("/api/v3/exchangeInfo", weight=10)
    
    def get_candlestick_data(self, symbol: str, interval: str, 
                           start_time: Optional[Union[str, datetime]] = None,
                           end_time: Optional[Union[str, datetime]] = None,
                           limit: int = 500) -> pd.DataFrame:
        """
        Get candlestick/kline data.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            interval: Kline interval ('1m', '5m', '15m', '1h', '4h', '1d', etc.)
            start_time: Start time (datetime object or timestamp)
            end_time: End time (datetime object or timestamp)
            limit: Number of klines (max 1000)
            
        Returns:
            DataFrame with OHLCV data
        """
        params = {
            'symbol': symbol.upper(),
            'interval': interval,
            'limit': min(limit, 1000)
        }
        
        if start_time:
            if isinstance(start_time, datetime):
                params['startTime'] = int(start_time.timestamp() * 1000)
            else:
                params['startTime'] = start_time
                
        if end_time:
            if isinstance(end_time, datetime):
                params['endTime'] = int(end_time.timestamp() * 1000)
            else:
                params['endTime'] = end_time
        
        data = self._make_request("/api/v3/klines", params, weight=1)
        
        # Convert to DataFrame
        df = pd.DataFrame(data, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # Convert data types
        numeric_columns = ['open', 'high', 'low', 'close', 'volume',
                          'quote_asset_volume', 'taker_buy_base_asset_volume',
                          'taker_buy_quote_asset_volume']
        
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col])
        
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        df['number_of_trades'] = df['number_of_trades'].astype(int)
        
        # Drop unnecessary column
        df = df.drop('ignore', axis=1)
        
        return df
    
    def get_order_book(self, symbol: str, limit: int = 100) -> Dict:
        """
        Get order book depth.
        
        Args:
            symbol: Trading pair symbol
            limit: Number of entries (5, 10, 20, 50, 100, 500, 1000, 5000)
            
        Returns:
            Dictionary with bids and asks
        """
        valid_limits = [5, 10, 20, 50, 100, 500, 1000, 5000]
        if limit not in valid_limits:
            limit = min(valid_limits, key=lambda x: abs(x - limit))
        
        params = {
            'symbol': symbol.upper(),
            'limit': limit
        }
        
        weight = 1 if limit <= 100 else 5 if limit <= 500 else 10
        data = self._make_request("/api/v3/depth", params, weight=weight)
        
        # Convert string prices/quantities to float
        data['bids'] = [[float(price), float(qty)] for price, qty in data['bids']]
        data['asks'] = [[float(price), float(qty)] for price, qty in data['asks']]
        
        return data
    
    def get_recent_trades(self, symbol: str, limit: int = 500) -> List[Dict]:
        """
        Get recent trades.
        
        Args:
            symbol: Trading pair symbol
            limit: Number of trades (max 1000)
            
        Returns:
            List of recent trades
        """
        params = {
            'symbol': symbol.upper(),
            'limit': min(limit, 1000)
        }
        
        data = self._make_request("/api/v3/trades", params, weight=1)
        
        # Convert data types
        for trade in data:
            trade['price'] = float(trade['price'])
            trade['qty'] = float(trade['qty'])
            trade['time'] = pd.to_datetime(trade['time'], unit='ms')
        
        return data
    
    def get_historical_trades(self, symbol: str, limit: int = 500, 
                            from_id: Optional[int] = None) -> List[Dict]:
        """
        Get historical trades (requires API key).
        
        Args:
            symbol: Trading pair symbol
            limit: Number of trades (max 1000)
            from_id: Trade ID to start from
            
        Returns:
            List of historical trades
        """
        params = {
            'symbol': symbol.upper(),
            'limit': min(limit, 1000)
        }
        
        if from_id:
            params['fromId'] = from_id
        
        data = self._make_request("/api/v3/historicalTrades", params, 
                                authenticated=True, weight=5)
        
        # Convert data types
        for trade in data:
            trade['price'] = float(trade['price'])
            trade['qty'] = float(trade['qty'])
            trade['time'] = pd.to_datetime(trade['time'], unit='ms')
        
        return data
    
    def get_24hr_ticker(self, symbol: Optional[str] = None) -> Union[Dict, List[Dict]]:
        """
        Get 24hr ticker price change statistics.
        
        Args:
            symbol: Trading pair symbol (if None, returns all symbols)
            
        Returns:
            Ticker data for symbol or all symbols
        """
        params = {}
        if symbol:
            params['symbol'] = symbol.upper()
            weight = 1
        else:
            weight = 40
        
        data = self._make_request("/api/v3/ticker/24hr", params, weight=weight)
        
        # Convert numeric fields
        def convert_ticker(ticker):
            numeric_fields = ['priceChange', 'priceChangePercent', 'weightedAvgPrice',
                            'prevClosePrice', 'lastPrice', 'lastQty', 'bidPrice',
                            'bidQty', 'askPrice', 'askQty', 'openPrice', 'highPrice',
                            'lowPrice', 'volume', 'quoteVolume']
            
            for field in numeric_fields:
                if field in ticker:
                    ticker[field] = float(ticker[field])
            
            ticker['openTime'] = pd.to_datetime(ticker['openTime'], unit='ms')
            ticker['closeTime'] = pd.to_datetime(ticker['closeTime'], unit='ms')
            ticker['count'] = int(ticker['count'])
            
            return ticker
        
        if isinstance(data, list):
            return [convert_ticker(ticker) for ticker in data]
        else:
            return convert_ticker(data)
    
    def get_price(self, symbol: Optional[str] = None) -> Union[Dict, List[Dict]]:
        """
        Get latest price for symbol(s).
        
        Args:
            symbol: Trading pair symbol (if None, returns all symbols)
            
        Returns:
            Price data for symbol or all symbols
        """
        params = {}
        if symbol:
            params['symbol'] = symbol.upper()
            weight = 1
        else:
            weight = 2
        
        data = self._make_request("/api/v3/ticker/price", params, weight=weight)
        
        # Convert price to float
        if isinstance(data, list):
            for item in data:
                item['price'] = float(item['price'])
        else:
            data['price'] = float(data['price'])
        
        return data
    
    def get_avg_price(self, symbol: str) -> Dict:
        """
        Get current average price for a symbol.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Average price data
        """
        params = {'symbol': symbol.upper()}
        data = self._make_request("/api/v3/avgPrice", params, weight=1)
        data['price'] = float(data['price'])
        return data

