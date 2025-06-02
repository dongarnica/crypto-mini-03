#!/usr/bin/env python3
"""
Simple test for file discovery functionality.
"""

import os
import sys
import glob
import re
from datetime import datetime

def test_file_discovery():
    """Test the file discovery logic independently."""
    print("🔍 TESTING FILE DISCOVERY LOGIC")
    print("=" * 50)
    
    historical_exports_dir = "/workspaces/crypto-mini-03/historical_exports"
    symbol = "BTCUSDT"
    
    print(f"Directory: {historical_exports_dir}")
    print(f"Symbol: {symbol}")
    
    # Pattern: {SYMBOL}_1year_hourly_{YYYYMMDD_HHMMSS}.csv
    pattern = os.path.join(historical_exports_dir, f"{symbol}_1year_hourly_*.csv")
    matching_files = glob.glob(pattern)
    
    print(f"\nPattern: {pattern}")
    print(f"Found {len(matching_files)} files:")
    
    for file_path in matching_files:
        filename = os.path.basename(file_path)
        print(f"  {filename}")
        
        # Extract timestamp
        match = re.search(r'(\d{8}_\d{6})\.csv$', filename)
        if match:
            timestamp_str = match.group(1)
            try:
                timestamp = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                print(f"    -> Timestamp: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            except ValueError as e:
                print(f"    -> Error parsing timestamp: {e}")
    
    if matching_files:
        # Sort by timestamp and get latest
        file_timestamps = []
        for file_path in matching_files:
            filename = os.path.basename(file_path)
            match = re.search(r'(\d{8}_\d{6})\.csv$', filename)
            if match:
                timestamp_str = match.group(1)
                try:
                    timestamp = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                    file_timestamps.append((timestamp, file_path))
                except ValueError:
                    continue
        
        if file_timestamps:
            file_timestamps.sort(key=lambda x: x[0], reverse=True)
            latest_file = file_timestamps[0][1]
            latest_timestamp = file_timestamps[0][0]
            
            print(f"\n✅ Latest file: {os.path.basename(latest_file)}")
            print(f"✅ Latest timestamp: {latest_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            return latest_file
    
    print("❌ No valid files found")
    return None

def test_all_symbols():
    """Test discovery for all available symbols."""
    print("\n🔍 TESTING ALL AVAILABLE SYMBOLS")
    print("=" * 50)
    
    historical_exports_dir = "/workspaces/crypto-mini-03/historical_exports"
    pattern = os.path.join(historical_exports_dir, "*_1year_hourly_*.csv")
    all_files = glob.glob(pattern)
    
    print(f"Found {len(all_files)} total files")
    
    symbols = set()
    for file_path in all_files:
        filename = os.path.basename(file_path)
        symbol_match = re.match(r'^([A-Z]+USDT)_', filename)
        if symbol_match:
            symbols.add(symbol_match.group(1))
    
    symbols_list = sorted(list(symbols))
    print(f"Found {len(symbols_list)} unique symbols:")
    
    for symbol in symbols_list:
        print(f"  {symbol}")
        
        # Find latest for this symbol
        symbol_pattern = os.path.join(historical_exports_dir, f"{symbol}_1year_hourly_*.csv")
        symbol_files = glob.glob(symbol_pattern)
        
        if symbol_files:
            file_timestamps = []
            for file_path in symbol_files:
                filename = os.path.basename(file_path)
                match = re.search(r'(\d{8}_\d{6})\.csv$', filename)
                if match:
                    timestamp_str = match.group(1)
                    try:
                        timestamp = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                        file_timestamps.append((timestamp, file_path))
                    except ValueError:
                        continue
            
            if file_timestamps:
                file_timestamps.sort(key=lambda x: x[0], reverse=True)
                latest_file = file_timestamps[0][1]
                latest_timestamp = file_timestamps[0][0]
                
                filename = os.path.basename(latest_file)
                print(f"    Latest: {filename} ({latest_timestamp.strftime('%Y-%m-%d %H:%M:%S')})")
    
    return symbols_list

def main():
    """Main test function."""
    print("🧪 FILE DISCOVERY FUNCTIONALITY TEST")
    print("=" * 60)
    
    # Test single symbol
    latest_file = test_file_discovery()
    
    # Test all symbols
    symbols = test_all_symbols()
    
    print(f"\n📊 SUMMARY")
    print("=" * 30)
    print(f"✅ Single symbol test: {'PASSED' if latest_file else 'FAILED'}")
    print(f"✅ Available symbols: {len(symbols) if symbols else 0}")
    
    if latest_file and symbols:
        print(f"\n🎯 File discovery is working correctly!")
        print(f"• Found latest file for BTCUSDT")
        print(f"• Discovered {len(symbols)} available symbols")
        print(f"• Ready for ML pipeline integration")
    else:
        print(f"\n❌ File discovery needs debugging")

if __name__ == "__main__":
    main()