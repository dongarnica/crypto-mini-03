# Example usage and utility functions
from binance_client import BinanceUSClient
from typing import List


def create_trading_client(api_key: str = None, api_secret: str = None) -> BinanceUSClient:
    """Factory function to create a configured Binance US client."""
    return BinanceUSClient(api_key=api_key, api_secret=api_secret)

def get_popular_symbols() -> List[str]:
    """Get list of popular trading symbols."""
    return [
        'BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT',
        'LINKUSDT', 'LTCUSDT', 'BCHUSDT', 'XLMUSDT',
        'UNIUSDT', 'SOLUSDT', 'MATICUSDT', 'AVAXUSDT'
    ]

if __name__ == "__main__":
    # Example usage
    client = create_trading_client()
    
    # Get server time
    print("Server time:", client.get_server_time())
    
    # Get BTCUSDT candlestick data
    btc_data = client.get_candlestick_data('BTCUSDT', '1h', limit=100)
    print(f"\nBTC 1h data shape: {btc_data.shape}")
    print(btc_data.head())
    
    # Get order book
    order_book = client.get_order_book('BTCUSDT', limit=10)
    print(f"\nOrder book - Top bid: ${order_book['bids'][0][0]}, Top ask: ${order_book['asks'][0][0]}")
    
    # Get recent trades
    trades = client.get_recent_trades('BTCUSDT', limit=5)
    print(f"\nRecent trades count: {len(trades)}")
    
    # Get 24hr ticker
    ticker = client.get_24hr_ticker('BTCUSDT')
    print(f"\nBTC 24hr change: {ticker['priceChangePercent']:.2f}%")