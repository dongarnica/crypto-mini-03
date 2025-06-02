#!/usr/bin/env python3
"""
Test script for the updated ML pipeline with automatic file discovery.
This demonstrates how to use the pipeline without hardcoding file paths.
"""

import sys
import os

# Add the ml directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'ml'))

from ml_pipeline_improved import ImprovedCryptoLSTMPipeline

def test_single_symbol():
    """Test the pipeline with a single symbol using automatic file discovery."""
    print("🧪 TESTING SINGLE SYMBOL WITH AUTO FILE DISCOVERY")
    print("=" * 60)
    
    # Test with BTCUSDT
    symbol = 'BTCUSDT'
    
    pipeline = ImprovedCryptoLSTMPipeline(
        symbol=symbol,
        confidence_threshold=0.3,
        lookback_period=24,
        prediction_horizon=3,
        use_binary_classification=False
    )
    
    print(f"Testing symbol: {symbol}")
    print(f"Configuration:")
    print(f"  - Confidence threshold: {pipeline.confidence_threshold:.0%}")
    print(f"  - Prediction horizon: {pipeline.prediction_horizon} hours")
    print(f"  - Classification type: {'Binary' if pipeline.use_binary_classification else '3-Class'}")
    
    try:
        # Test file discovery
        latest_file = pipeline.find_latest_historical_file(symbol)
        print(f"\n✅ Found latest file: {os.path.basename(latest_file)}")
        
        # Test data loading
        data = pipeline.load_data_from_symbol(symbol)
        print(f"✅ Loaded {len(data)} data points")
        print(f"   Date range: {data['open_time'].min()} to {data['open_time'].max()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_available_symbols():
    """Test discovery of all available symbols."""
    print("\n🔍 TESTING AVAILABLE SYMBOLS DISCOVERY")
    print("=" * 60)
    
    pipeline = ImprovedCryptoLSTMPipeline()
    
    try:
        symbols = pipeline.get_available_symbols()
        print(f"✅ Found {len(symbols)} symbols")
        
        # Show latest file for each symbol
        print(f"\nLatest files for each symbol:")
        for symbol in symbols[:5]:  # Show first 5
            try:
                latest_file = pipeline.find_latest_historical_file(symbol)
                filename = os.path.basename(latest_file)
                print(f"  {symbol}: {filename}")
            except Exception as e:
                print(f"  {symbol}: Error - {e}")
        
        if len(symbols) > 5:
            print(f"  ... and {len(symbols) - 5} more symbols")
        
        return symbols
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return []

def test_multiple_symbols():
    """Test analysis of multiple symbols."""
    print("\n🔄 TESTING MULTIPLE SYMBOLS ANALYSIS")
    print("=" * 60)
    
    # Create pipeline for multi-symbol analysis
    pipeline = ImprovedCryptoLSTMPipeline(
        confidence_threshold=0.35,
        lookback_period=24,
        prediction_horizon=3,
        use_binary_classification=True  # Use binary for faster testing
    )
    
    try:
        # Get available symbols
        symbols = pipeline.get_available_symbols()
        
        if not symbols:
            print("❌ No symbols found")
            return False
        
        # Test with first 2 symbols for speed
        test_symbols = symbols[:2]
        print(f"Testing with symbols: {test_symbols}")
        
        # Run multi-symbol analysis
        results = pipeline.analyze_multiple_symbols(test_symbols, max_symbols=2)
        
        print(f"\n📊 RESULTS SUMMARY:")
        for symbol, result in results.items():
            if result.get('success'):
                print(f"✅ {symbol}: Success")
            else:
                print(f"❌ {symbol}: Failed - {result.get('error', 'Unknown error')}")
        
        return results
        
    except Exception as e:
        print(f"❌ Error in multi-symbol analysis: {e}")
        return {}

def main():
    """Main test function."""
    print("🚀 TESTING UPDATED ML PIPELINE WITH AUTO FILE DISCOVERY")
    print("=" * 70)
    print("FEATURES BEING TESTED:")
    print("✅ Automatic latest file discovery per symbol")
    print("✅ No hardcoded file paths")
    print("✅ Multi-symbol support")
    print("✅ Available symbols discovery")
    print("=" * 70)
    
    # Test 1: Single symbol with auto discovery
    success1 = test_single_symbol()
    
    # Test 2: Available symbols discovery
    symbols = test_available_symbols()
    
    # Test 3: Multiple symbols (if we have symbols and first test passed)
    if success1 and symbols:
        results = test_multiple_symbols()
    else:
        print("\nSkipping multi-symbol test due to previous failures")
        results = {}
    
    # Final summary
    print(f"\n" + "=" * 70)
    print("🏁 TEST SUMMARY")
    print("=" * 70)
    
    if success1:
        print("✅ Single symbol auto-discovery: PASSED")
    else:
        print("❌ Single symbol auto-discovery: FAILED")
    
    if symbols:
        print(f"✅ Symbol discovery: PASSED ({len(symbols)} symbols found)")
    else:
        print("❌ Symbol discovery: FAILED")
    
    if results:
        successful_multi = sum(1 for r in results.values() if r.get('success'))
        total_multi = len(results)
        print(f"✅ Multi-symbol analysis: {successful_multi}/{total_multi} successful")
    else:
        print("❌ Multi-symbol analysis: SKIPPED or FAILED")
    
    print(f"\n🎯 The ML pipeline now supports:")
    print("• Automatic discovery of latest historical files")
    print("• No need for hardcoded file paths")
    print("• Easy switching between different cryptocurrencies")
    print("• Batch analysis of multiple symbols")
    
    print(f"\n💡 Example usage:")
    print("```python")
    print("# Simple usage - automatically finds latest BTCUSDT file")
    print("pipeline = ImprovedCryptoLSTMPipeline(symbol='BTCUSDT')")
    print("result = pipeline.run_complete_analysis()")
    print("")
    print("# Multi-symbol analysis")
    print("pipeline = ImprovedCryptoLSTMPipeline()")
    print("results = pipeline.analyze_multiple_symbols(['BTCUSDT', 'ETHUSDT'])")
    print("```")

if __name__ == "__main__":
    main()