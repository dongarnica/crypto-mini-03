#!/usr/bin/env python3
"""
Historical Market Data Export Script
====================================

This script exports 1 year of hourly market data history for specified cryptocurrency symbols.
It handles Binance API rate limits by chunking requests and provides comprehensive error handling.

Features:
- Exports 1 year of hourly OHLCV data
- Automatic request chunking to handle API limits
- Progress tracking and resumable downloads
- Data validation and quality checks
- Multiple export formats (CSV, JSON, Parquet)
- Comprehensive logging and error handling

Usage:
    python export_historical_data.py
    python export_historical_data.py --symbols BTCUSDT,ETHUSDT --format csv
    python export_historical_data.py --symbols BTCUSDT --start-date 2023-01-01 --end-date 2024-01-01

Author: Crypto Data Exporter
Date: May 31, 2025
"""

import os
import sys
import argparse
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import pandas as pd
import time
import logging
from pathlib import Path

# Add the binance directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'binance'))
from binance_client import BinanceUSClient

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

class HistoricalDataExporter:
    """
    Historical cryptocurrency data exporter with comprehensive features.
    """
    
    def __init__(self, output_dir: str = "historical_exports"):
        """
        Initialize the historical data exporter.
        
        Args:
            output_dir: Directory to save exported data
        """
        # Initialize Binance client
        api_key = os.getenv('BINANCE_API_KEY')
        api_secret = os.getenv('BINANCE_API_SECRET')
        self.client = BinanceUSClient(api_key=api_key, api_secret=api_secret)
        
        # Setup output directory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Default symbols for export
        self.default_symbols = [
            'BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT',
            'LINKUSDT', 'LTCUSDT', 'BCHUSDT', 'XLMUSDT',
            'UNIUSDT', 'SOLUSDT', 'MATICUSDT', 'AVAXUSDT'
        ]
        
        # API limits configuration
        self.max_klines_per_request = 1000  # Binance limit
        self.ms_per_hour = 60 * 60 * 1000  # Milliseconds in an hour
        self.request_delay = 0.1  # Delay between requests to avoid rate limits
        
    def setup_logging(self):
        """Setup comprehensive logging."""
        log_file = self.output_dir / 'export_log.log'
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        
        # Setup logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Prevent duplicate logs
        self.logger.propagate = False
    
    def print_header(self, title: str):
        """Print a formatted header."""
        print("\n" + "="*80)
        print(f" {title}")
        print("="*80)
    
    def print_progress(self, current: int, total: int, symbol: str, start_time: datetime):
        """Print progress information."""
        percentage = (current / total) * 100
        elapsed = datetime.now() - start_time
        if current > 0:
            eta = elapsed * (total - current) / current
            eta_str = str(eta).split('.')[0]  # Remove microseconds
        else:
            eta_str = "calculating..."
        
        print(f"\r{symbol}: {current}/{total} chunks ({percentage:.1f}%) - "
              f"Elapsed: {str(elapsed).split('.')[0]} - ETA: {eta_str}", end="")
    
    def validate_symbol(self, symbol: str) -> bool:
        """
        Validate if a symbol exists on Binance.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            True if symbol exists, False otherwise
        """
        try:
            exchange_info = self.client.get_exchange_info()
            symbols = [s['symbol'] for s in exchange_info['symbols']]
            return symbol.upper() in symbols
        except Exception as e:
            self.logger.warning(f"Could not validate symbol {symbol}: {e}")
            return True  # Assume valid if we can't check
    
    def calculate_date_chunks(self, start_date: datetime, end_date: datetime) -> List[tuple]:
        """
        Calculate date chunks for API requests.
        
        Since Binance allows max 1000 klines per request, and we want hourly data,
        we need to chunk requests to cover the full date range.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            List of (start_time, end_time) tuples
        """
        chunks = []
        current_start = start_date
        
        # Each chunk covers 1000 hours (max klines per request)
        chunk_duration = timedelta(hours=1000)
        
        while current_start < end_date:
            current_end = min(current_start + chunk_duration, end_date)
            chunks.append((current_start, current_end))
            current_start = current_end
            
        return chunks
    
    def fetch_symbol_data(self, symbol: str, start_date: datetime, end_date: datetime,
                         show_progress: bool = True) -> pd.DataFrame:
        """
        Fetch historical data for a single symbol.
        
        Args:
            symbol: Trading pair symbol
            start_date: Start date for data
            end_date: End date for data
            show_progress: Whether to show progress
            
        Returns:
            DataFrame with historical OHLCV data
        """
        self.logger.info(f"Starting data collection for {symbol}")
        
        # Calculate chunks
        chunks = self.calculate_date_chunks(start_date, end_date)
        total_chunks = len(chunks)
        
        if show_progress:
            print(f"\n{symbol}: Fetching {total_chunks} chunks of data...")
        
        all_data = []
        start_time = datetime.now()
        
        for i, (chunk_start, chunk_end) in enumerate(chunks):
            try:
                # Show progress
                if show_progress:
                    self.print_progress(i, total_chunks, symbol, start_time)
                
                # Fetch data chunk
                chunk_data = self.client.get_candlestick_data(
                    symbol=symbol,
                    interval='1h',
                    start_time=chunk_start,
                    end_time=chunk_end,
                    limit=1000
                )
                
                if not chunk_data.empty:
                    all_data.append(chunk_data)
                
                # Rate limiting delay
                time.sleep(self.request_delay)
                
            except Exception as e:
                self.logger.error(f"Error fetching chunk {i+1}/{total_chunks} for {symbol}: {e}")
                # Continue with next chunk
                continue
        
        if show_progress:
            self.print_progress(total_chunks, total_chunks, symbol, start_time)
            print()  # New line after progress
        
        # Combine all data
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            # Remove duplicates and sort by time
            combined_df = combined_df.drop_duplicates(subset=['open_time']).sort_values('open_time')
            combined_df = combined_df.reset_index(drop=True)
            
            self.logger.info(f"Collected {len(combined_df)} records for {symbol}")
            return combined_df
        else:
            self.logger.warning(f"No data collected for {symbol}")
            return pd.DataFrame()
    
    def validate_data(self, df: pd.DataFrame, symbol: str) -> Dict:
        """
        Validate the quality of collected data.
        
        Args:
            df: DataFrame with market data
            symbol: Symbol name
            
        Returns:
            Dictionary with validation results
        """
        if df.empty:
            return {'valid': False, 'issues': ['No data collected']}
        
        issues = []
        
        # Check for missing values
        missing_counts = df.isnull().sum()
        if missing_counts.sum() > 0:
            issues.append(f"Missing values detected: {missing_counts.to_dict()}")
        
        # Check for negative prices
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if (df[col] <= 0).any():
                issues.append(f"Negative or zero values in {col}")
        
        # Check for invalid OHLC relationships
        invalid_ohlc = df[(df['high'] < df['low']) | 
                         (df['high'] < df['open']) | 
                         (df['high'] < df['close']) |
                         (df['low'] > df['open']) | 
                         (df['low'] > df['close'])]
        
        if not invalid_ohlc.empty:
            issues.append(f"Invalid OHLC relationships in {len(invalid_ohlc)} rows")
        
        # Check time continuity
        df_sorted = df.sort_values('open_time')
        time_diffs = df_sorted['open_time'].diff()
        expected_diff = pd.Timedelta(hours=1)
        gaps = time_diffs[time_diffs > expected_diff * 1.1]  # Allow 10% tolerance
        
        if not gaps.empty:
            issues.append(f"Time gaps detected: {len(gaps)} instances")
        
        validation_result = {
            'valid': len(issues) == 0,
            'issues': issues,
            'total_records': len(df),
            'date_range': (df['open_time'].min(), df['open_time'].max()),
            'price_range': (df['close'].min(), df['close'].max())
        }
        
        return validation_result
    
    def save_data(self, df: pd.DataFrame, symbol: str, format: str = 'csv') -> str:
        """
        Save data in the specified format.
        
        Args:
            df: DataFrame to save
            symbol: Symbol name
            format: Export format ('csv', 'json', 'parquet')
            
        Returns:
            Path to saved file
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if format.lower() == 'csv':
            filename = f"{symbol}_1year_hourly_{timestamp}.csv"
            filepath = self.output_dir / filename
            df.to_csv(filepath, index=False)
            
        elif format.lower() == 'json':
            filename = f"{symbol}_1year_hourly_{timestamp}.json"
            filepath = self.output_dir / filename
            df.to_json(filepath, orient='records', date_format='iso')
            
        elif format.lower() == 'parquet':
            filename = f"{symbol}_1year_hourly_{timestamp}.parquet"
            filepath = self.output_dir / filename
            df.to_parquet(filepath, index=False)
            
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        return str(filepath)
    
    def export_symbol_data(self, symbol: str, start_date: datetime, end_date: datetime,
                          format: str = 'csv') -> Dict:
        """
        Export historical data for a single symbol.
        
        Args:
            symbol: Trading pair symbol
            start_date: Start date
            end_date: End date
            format: Export format
            
        Returns:
            Dictionary with export results
        """
        symbol = symbol.upper()
        
        # Validate symbol
        if not self.validate_symbol(symbol):
            return {'success': False, 'error': f'Invalid symbol: {symbol}'}
        
        try:
            # Fetch data
            df = self.fetch_symbol_data(symbol, start_date, end_date)
            
            if df.empty:
                return {'success': False, 'error': f'No data collected for {symbol}'}
            
            # Validate data
            validation = self.validate_data(df, symbol)
            
            # Save data
            filepath = self.save_data(df, symbol, format)
            
            # Prepare result
            result = {
                'success': True,
                'symbol': symbol,
                'records_count': len(df),
                'date_range': (df['open_time'].min().isoformat(), 
                              df['open_time'].max().isoformat()),
                'file_path': filepath,
                'file_size_mb': os.path.getsize(filepath) / (1024 * 1024),
                'validation': validation
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to export data for {symbol}: {e}")
            return {'success': False, 'error': str(e)}
    
    def export_multiple_symbols(self, symbols: List[str], start_date: datetime, 
                               end_date: datetime, format: str = 'csv') -> Dict:
        """
        Export historical data for multiple symbols.
        
        Args:
            symbols: List of trading pair symbols
            start_date: Start date
            end_date: End date
            format: Export format
            
        Returns:
            Dictionary with export results
        """
        self.print_header(f"Historical Data Export - {len(symbols)} Symbols")
        
        print(f"Date Range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        print(f"Interval: 1 hour")
        print(f"Export Format: {format.upper()}")
        print(f"Output Directory: {self.output_dir}")
        print(f"Symbols: {', '.join(symbols)}")
        
        results = {
            'start_time': datetime.now().isoformat(),
            'parameters': {
                'symbols': symbols,
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat(),
                'format': format,
                'output_dir': str(self.output_dir)
            },
            'exports': {}
        }
        
        total_symbols = len(symbols)
        successful_exports = 0
        failed_exports = 0
        
        for i, symbol in enumerate(symbols, 1):
            print(f"\n[{i}/{total_symbols}] Processing {symbol}...")
            
            result = self.export_symbol_data(symbol, start_date, end_date, format)
            results['exports'][symbol] = result
            
            if result['success']:
                successful_exports += 1
                print(f"✅ {symbol}: {result['records_count']:,} records saved to {Path(result['file_path']).name}")
                if result['validation']['issues']:
                    print(f"⚠️  Data quality issues: {result['validation']['issues']}")
            else:
                failed_exports += 1
                print(f"❌ {symbol}: {result['error']}")
        
        # Summary
        results['end_time'] = datetime.now().isoformat()
        results['summary'] = {
            'total_symbols': total_symbols,
            'successful_exports': successful_exports,
            'failed_exports': failed_exports,
            'success_rate': f"{(successful_exports/total_symbols)*100:.1f}%"
        }
        
        # Save export report
        report_file = self.output_dir / f"export_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.print_header("Export Summary")
        print(f"✅ Successful: {successful_exports}/{total_symbols}")
        print(f"❌ Failed: {failed_exports}/{total_symbols}")
        print(f"📊 Success Rate: {results['summary']['success_rate']}")
        print(f"📁 Export Report: {report_file}")
        print(f"📁 Output Directory: {self.output_dir}")
        
        return results

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Export 1 year of hourly cryptocurrency market data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python export_historical_data.py
  python export_historical_data.py --symbols BTCUSDT,ETHUSDT
  python export_historical_data.py --symbols BTCUSDT --format json
  python export_historical_data.py --start-date 2023-01-01 --end-date 2024-01-01
        """
    )
    
    parser.add_argument(
        '--symbols',
        type=str,
        help='Comma-separated list of symbols (default: popular coins)'
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date in YYYY-MM-DD format (default: 1 year ago)'
    )
    
    parser.add_argument(
        '--end-date',
        type=str,
        help='End date in YYYY-MM-DD format (default: today)'
    )
    
    parser.add_argument(
        '--format',
        choices=['csv', 'json', 'parquet'],
        default='csv',
        help='Export format (default: csv)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='historical_exports',
        help='Output directory (default: historical_exports)'
    )
    
    return parser.parse_args()

def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Initialize exporter
    exporter = HistoricalDataExporter(output_dir=args.output_dir)
    
    # Parse symbols
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(',')]
    else:
        symbols = exporter.default_symbols
    
    # Parse dates
    if args.start_date:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    else:
        start_date = datetime.now() - timedelta(days=365)  # 1 year ago
    
    if args.end_date:
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
    else:
        end_date = datetime.now()
    
    # Validate date range
    if start_date >= end_date:
        print("❌ Error: Start date must be before end date")
        return
    
    if (end_date - start_date).days > 400:
        print("⚠️  Warning: Date range exceeds 400 days. This may take a long time.")
        response = input("Continue? (y/N): ")
        if response.lower() != 'y':
            print("Export cancelled.")
            return
    
    # Run export
    try:
        results = exporter.export_multiple_symbols(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            format=args.format
        )
        
        if results['summary']['successful_exports'] > 0:
            print(f"\n🎉 Export completed successfully!")
        else:
            print(f"\n💥 Export failed for all symbols. Check logs for details.")
            
    except KeyboardInterrupt:
        print(f"\n⏹️  Export interrupted by user")
    except Exception as e:
        print(f"\n💥 Export failed: {e}")
        exporter.logger.error(f"Export failed: {e}")

if __name__ == "__main__":
    main()
