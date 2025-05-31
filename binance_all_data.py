#!/usr/bin/env python3
"""
Comprehensive Binance Data Retrieval Script
============================================

This script fetches and displays ALL available data from Binance US API:
- Server time and exchange information
- Market data for popular cryptocurrencies
- Order books and recent trades
- Price tickers and statistics
- Historical data and analytics

Author: Binance Data Explorer
"""

import os
import sys
from datetime import datetime, timedelta
from typing import List, Dict
import pandas as pd
import json

# Add the binance directory to the path to import the client
sys.path.append(os.path.join(os.path.dirname(__file__), 'binance'))
from binance_client import BinanceUSClient

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

class BinanceDataExplorer:
    """Comprehensive Binance data explorer and analyzer."""
    
    def __init__(self):
        """Initialize the Binance client with API credentials."""
        api_key = os.getenv('BINANCE_API_KEY')
        api_secret = os.getenv('BINANCE_API_SECRET')
        
        self.client = BinanceUSClient(api_key=api_key, api_secret=api_secret)
        # Load popular symbols from .env or use default list
        symbols_env = os.getenv('BINANCE_SYMBOLS')
        if symbols_env:
            self.popular_symbols = [s.strip().upper() for s in symbols_env.split(',') if s.strip()]
        else:
            self.popular_symbols = [
            'BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT',
            'LINKUSDT', 'LTCUSDT', 'BCHUSDT', 'XLMUSDT',
            'UNIUSDT', 'SOLUSDT', 'MATICUSDT', 'AVAXUSDT'
            ]
        
    def print_header(self, title: str, width: int = 80):
        """Print a formatted header."""
        print("\n" + "=" * width)
        print(f"{title:^{width}}")
        print("=" * width)
    
    def print_section(self, title: str, width: int = 60):
        """Print a formatted section header."""
        print(f"\n{'-' * width}")
        print(f"{title}")
        print(f"{'-' * width}")
    
    def get_server_info(self):
        """Get and display server time and exchange information."""
        self.print_header("🌐 BINANCE SERVER & EXCHANGE INFORMATION")
        
        # Server time
        print("\n📅 Server Time:")
        server_time = self.client.get_server_time()
        timestamp = server_time['serverTime']
        readable_time = datetime.fromtimestamp(timestamp / 1000)
        print(f"   Timestamp: {timestamp}")
        print(f"   Readable: {readable_time}")
        
        # Exchange info
        print("\n🏦 Exchange Information:")
        exchange_info = self.client.get_exchange_info()
        print(f"   Timezone: {exchange_info['timezone']}")
        print(f"   Server Time: {exchange_info['serverTime']}")
        print(f"   Rate Limits: {len(exchange_info['rateLimits'])} types")
        print(f"   Exchange Filters: {len(exchange_info['exchangeFilters'])}")
        print(f"   Total Symbols: {len(exchange_info['symbols'])}")
        
        # Show first few symbols
        print("\n💰 Available Trading Pairs (first 10):")
        for i, symbol in enumerate(exchange_info['symbols'][:10]):
            status = "✅" if symbol['status'] == 'TRADING' else "❌"
            print(f"   {i+1:2d}. {symbol['symbol']:12} {status} {symbol['baseAsset']}/{symbol['quoteAsset']}")
        
        return exchange_info
    
    def get_market_overview(self):
        """Get comprehensive market overview."""
        self.print_header("📊 MARKET OVERVIEW & STATISTICS")
        
        # All 24hr tickers
        print("\n📈 24-Hour Market Statistics:")
        all_tickers = self.client.get_24hr_ticker()
        
        # Sort by volume and show top performers
        sorted_tickers = sorted(all_tickers, key=lambda x: float(x['volume']), reverse=True)
        
        print(f"\n🔥 Top 10 by Volume:")
        print(f"{'Rank':<4} {'Symbol':<12} {'Price':<12} {'Change %':<10} {'Volume':<15}")
        print("-" * 65)
        
        for i, ticker in enumerate(sorted_tickers[:10]):
            change_emoji = "📈" if float(ticker['priceChangePercent']) > 0 else "📉"
            print(f"{i+1:<4} {ticker['symbol']:<12} ${float(ticker['lastPrice']):>9.2f} "
                  f"{change_emoji}{float(ticker['priceChangePercent']):>6.2f}% "
                  f"{float(ticker['volume']):>13,.0f}")
        
        # Price changes
        print(f"\n🚀 Biggest Gainers:")
        gainers = sorted(all_tickers, key=lambda x: float(x['priceChangePercent']), reverse=True)[:5]
        for ticker in gainers:
            print(f"   {ticker['symbol']:<12} +{float(ticker['priceChangePercent']):>6.2f}% "
                  f"(${float(ticker['lastPrice']):>8.2f})")
        
        print(f"\n📉 Biggest Losers:")
        losers = sorted(all_tickers, key=lambda x: float(x['priceChangePercent']))[:5]
        for ticker in losers:
            print(f"   {ticker['symbol']:<12} {float(ticker['priceChangePercent']):>7.2f}% "
                  f"(${float(ticker['lastPrice']):>8.2f})")
        
        return all_tickers
    
    def get_detailed_symbol_data(self, symbol: str):
        """Get comprehensive data for a specific symbol."""
        self.print_section(f"📋 DETAILED DATA FOR {symbol}")
        
        try:
            # Current price and average price
            current_price = self.client.get_price(symbol)
            avg_price = self.client.get_avg_price(symbol)
            ticker_24hr = self.client.get_24hr_ticker(symbol)
            
            print(f"\n💵 Price Information:")
            print(f"   Current Price: ${float(current_price['price']):,.2f}")
            print(f"   Average Price: ${float(avg_price['price']):,.2f}")
            print(f"   24h Change: {float(ticker_24hr['priceChangePercent']):+.2f}%")
            print(f"   24h High: ${float(ticker_24hr['highPrice']):,.2f}")
            print(f"   24h Low: ${float(ticker_24hr['lowPrice']):,.2f}")
            print(f"   24h Volume: {float(ticker_24hr['volume']):,.0f}")
            
            # Order book
            order_book = self.client.get_order_book(symbol, limit=10)
            print(f"\n📚 Order Book (Top 5):")
            print(f"   {'Bids (Buy)':<20} {'Asks (Sell)':<20}")
            print(f"   {'Price':>10} {'Qty':>8}   {'Price':>10} {'Qty':>8}")
            print(f"   {'-' * 40}")
            
            for i in range(min(5, len(order_book['bids']))):
                bid_price, bid_qty = order_book['bids'][i]
                ask_price, ask_qty = order_book['asks'][i]
                print(f"   ${bid_price:>9.2f} {bid_qty:>7.3f}   ${ask_price:>9.2f} {ask_qty:>7.3f}")
            
            spread = float(order_book['asks'][0][0]) - float(order_book['bids'][0][0])
            spread_pct = (spread / float(order_book['bids'][0][0])) * 100
            print(f"   Spread: ${spread:.2f} ({spread_pct:.3f}%)")
            
            # Recent trades
            recent_trades = self.client.get_recent_trades(symbol, limit=5)
            print(f"\n🔄 Recent Trades:")
            print(f"   {'Time':<12} {'Price':<12} {'Quantity':<12} {'Side'}")
            print(f"   {'-' * 48}")
            
            for trade in recent_trades[:5]:
                side = "BUY" if trade['isBuyerMaker'] else "SELL"
                trade_time = trade['time'].strftime('%H:%M:%S')
                print(f"   {trade_time:<12} ${trade['price']:<10.2f} {trade['qty']:<11.4f} {side}")
            
            # Candlestick data (1 hour intervals, last 24 hours)
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=24)
            klines = self.client.get_candlestick_data(symbol, '1h', 
                                                    start_time=start_time, 
                                                    end_time=end_time)
            
            print(f"\n📊 24-Hour Candlestick Data (1h intervals):")
            print(f"   Data points: {len(klines)}")
            print(f"   Time range: {klines['open_time'].iloc[0]} to {klines['open_time'].iloc[-1]}")
            print(f"   OHLC Summary:")
            print(f"     Open:  ${klines['open'].iloc[0]:.2f}")
            print(f"     High:  ${klines['high'].max():.2f}")
            print(f"     Low:   ${klines['low'].min():.2f}")
            print(f"     Close: ${klines['close'].iloc[-1]:.2f}")
            print(f"     Avg Volume: {klines['volume'].mean():,.0f}")
            
        except Exception as e:
            print(f"   ❌ Error retrieving data for {symbol}: {str(e)}")
    
    def get_historical_analysis(self):
        """Perform historical data analysis."""
        self.print_header("📈 HISTORICAL DATA ANALYSIS")
        
        symbol = 'BTCUSDT'  # Focus on Bitcoin for historical analysis
        
        # Get different timeframes
        timeframes = ['1h', '4h', '1d']
        
        for timeframe in timeframes:
            self.print_section(f"Bitcoin ({symbol}) - {timeframe} Analysis")
            
            try:
                # Get last 100 candles
                df = self.client.get_candlestick_data(symbol, timeframe, limit=100)
                
                print(f"\n📊 Statistics for last 100 {timeframe} candles:")
                print(f"   Period: {df['open_time'].iloc[0]} to {df['open_time'].iloc[-1]}")
                print(f"   Current Price: ${df['close'].iloc[-1]:,.2f}")
                print(f"   Period High: ${df['high'].max():,.2f}")
                print(f"   Period Low: ${df['low'].min():,.2f}")
                print(f"   Price Change: {((df['close'].iloc[-1] / df['open'].iloc[0]) - 1) * 100:+.2f}%")
                print(f"   Average Volume: {df['volume'].mean():,.0f}")
                print(f"   Max Volume: {df['volume'].max():,.0f}")
                
                # Simple technical indicators
                df['sma_20'] = df['close'].rolling(window=20).mean()
                df['sma_50'] = df['close'].rolling(window=50).mean()
                current_price = df['close'].iloc[-1]
                sma_20 = df['sma_20'].iloc[-1]
                sma_50 = df['sma_50'].iloc[-1]
                
                print(f"\n📉 Simple Moving Averages:")
                print(f"   SMA 20: ${sma_20:,.2f}")
                print(f"   SMA 50: ${sma_50:,.2f}")
                
                if current_price > sma_20 > sma_50:
                    print(f"   📈 Bullish trend (Price > SMA20 > SMA50)")
                elif current_price < sma_20 < sma_50:
                    print(f"   📉 Bearish trend (Price < SMA20 < SMA50)")
                else:
                    print(f"   🔄 Mixed signals")
                
            except Exception as e:
                print(f"   ❌ Error analyzing {timeframe} data: {str(e)}")
    
    def export_data_summary(self):
        """Export a summary of all data to files."""
        self.print_header("💾 DATA EXPORT SUMMARY")
        
        try:
            # Create exports directory
            os.makedirs('binance_exports', exist_ok=True)
            
            # Export market overview
            all_tickers = self.client.get_24hr_ticker()
            ticker_df = pd.DataFrame(all_tickers)
            ticker_file = 'binance_exports/market_overview.csv'
            ticker_df.to_csv(ticker_file, index=False)
            print(f"✅ Market overview exported to: {ticker_file}")
            
            # Export popular symbols data
            for symbol in self.popular_symbols[:3]:  # Limit to 3 to avoid rate limits
                try:
                    df = self.client.get_candlestick_data(symbol, '1h', limit=100)
                    symbol_file = f'binance_exports/{symbol}_1h_data.csv'
                    df.to_csv(symbol_file, index=False)
                    print(f"✅ {symbol} hourly data exported to: {symbol_file}")
                except Exception as e:
                    print(f"❌ Failed to export {symbol}: {str(e)}")
            
            print(f"\n📁 All exports saved to: ./binance_exports/")
            
        except Exception as e:
            print(f"❌ Export error: {str(e)}")
    
    def run_complete_analysis(self):
        """Run the complete Binance data analysis."""
        print("🚀 Starting Comprehensive Binance Data Analysis...")
        print(f"📅 Analysis Time: {datetime.now()}")
        
        try:
            # 1. Server and exchange info
            self.get_server_info()
            
            # 2. Market overview
            self.get_market_overview()
            
            # 3. Detailed data for popular symbols
            self.print_header("🔍 DETAILED SYMBOL ANALYSIS")
            for symbol in self.popular_symbols[:5]:  # Analyze first 5 to manage rate limits
                self.get_detailed_symbol_data(symbol)
            
            # 4. Historical analysis
            self.get_historical_analysis()
            
            # 5. Export data
            self.export_data_summary()
            
            self.print_header("✨ ANALYSIS COMPLETE")
            print("🎉 All available Binance data has been retrieved and analyzed!")
            print("📊 Check the 'binance_exports' folder for exported data files.")
            
        except Exception as e:
            print(f"\n❌ Analysis failed: {str(e)}")
            print("🔧 Please check your API credentials and internet connection.")

def main():
    """Main execution function."""
    print("🌟 Binance Comprehensive Data Explorer")
    print("=" * 50)
    
    # Check for dependencies
    try:
        explorer = BinanceDataExplorer()
        explorer.run_complete_analysis()
    except ImportError as e:
        print(f"❌ Missing dependency: {str(e)}")
        print("📦 Please install required packages:")
        print("   pip install pandas python-dotenv")
    except Exception as e:
        print(f"❌ Initialization failed: {str(e)}")
        print("🔧 Please check your .env file and API credentials.")

if __name__ == "__main__":
    main()
