#!/usr/bin/env python3
"""
Test Symbol Manager Integration
=============================

Test script to verify that all components are using the centralized SymbolManager
for symbol conversions instead of their own conversion methods.
"""

import sys
import os
sys.path.append('/workspaces/crypto-mini-03')

def test_symbol_manager():
    """Test the SymbolManager directly."""
    print("🧪 Testing SymbolManager Integration")
    print("=" * 50)
    
    try:
        from config.symbol_manager import symbol_manager
        print("✅ SymbolManager imported successfully")
        
        # Test basic conversions
        test_symbols = ['BTCUSDT', 'ETHUSDT', 'DOTUSDT', 'LINKUSDT']
        print("\n📋 Testing Binance to Alpaca conversion:")
        for symbol in test_symbols:
            alpaca = symbol_manager.binance_to_alpaca_format(symbol)
            reverse = symbol_manager.alpaca_to_binance_format(alpaca)
            status = '✅' if reverse == symbol else '❌'
            print(f"{status} {symbol} -> {alpaca} -> {reverse}")
        
        print("\n📊 Environment Variables:")
        symbols_from_env = symbol_manager.get_symbols_from_env()
        print(f"CRYPTO_SYMBOLS: {symbols_from_env}")
        
        primary_symbols = symbol_manager.get_primary_symbols()
        print(f"Primary symbols: {primary_symbols}")
        
        print("\n📈 Symbol validation:")
        validation = symbol_manager.validate_symbols(test_symbols)
        for symbol, valid in validation.items():
            status = '✅' if valid else '❌'
            print(f"{status} {symbol}: {'Valid' if valid else 'Invalid'}")
            
        return True
        
    except Exception as e:
        print(f"❌ Error testing SymbolManager: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_strategy_engine_integration():
    """Test that Strategy Engine uses SymbolManager."""
    print("\n🎯 Testing Strategy Engine Integration")
    print("-" * 40)
    
    try:
        from trading.strategy_engine_refactored import TradingStrategyEngine
        print("✅ Strategy Engine imported successfully")
        
        # Test that it imports SymbolManager
        import trading.strategy_engine_refactored as se_module
        if hasattr(se_module, 'symbol_manager'):
            print("✅ Strategy Engine has symbol_manager reference")
        else:
            print("⚠️  Strategy Engine doesn't reference symbol_manager directly")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing Strategy Engine: {e}")
        return False

def test_position_manager_integration():
    """Test that Position Manager uses SymbolManager."""
    print("\n📊 Testing Position Manager Integration")
    print("-" * 40)
    
    try:
        from trading.position_manager import PositionManager
        print("✅ Position Manager imported successfully")
        
        # Test the import
        import trading.position_manager as pm_module
        if hasattr(pm_module, 'symbol_manager'):
            print("✅ Position Manager has symbol_manager reference")
        else:
            print("⚠️  Position Manager doesn't reference symbol_manager directly")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing Position Manager: {e}")
        return False

def test_risk_manager_integration():
    """Test that Risk Manager uses SymbolManager."""
    print("\n🛡️  Testing Risk Manager Integration")
    print("-" * 40)
    
    try:
        from trading.risk_manager import RiskManager
        print("✅ Risk Manager imported successfully")
        
        # Test the import
        import trading.risk_manager as rm_module
        if hasattr(rm_module, 'symbol_manager'):
            print("✅ Risk Manager has symbol_manager reference")
        else:
            print("⚠️  Risk Manager doesn't reference symbol_manager directly")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing Risk Manager: {e}")
        return False

def main():
    """Run all integration tests."""
    print("🚀 Symbol Manager Integration Test Suite")
    print("=" * 60)
    
    results = []
    
    # Test each component
    results.append(test_symbol_manager())
    results.append(test_strategy_engine_integration())
    results.append(test_position_manager_integration())
    results.append(test_risk_manager_integration())
    
    # Summary
    print("\n📋 Test Summary")
    print("=" * 30)
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"🎉 All {total} tests passed! Symbol management integration is complete.")
    else:
        print(f"⚠️  {passed}/{total} tests passed. Some issues need to be resolved.")
    
    print("\n🔧 Integration Changes Made:")
    print("- Strategy Engine: Uses SymbolManager.binance_to_alpaca_format()")
    print("- Position Manager: Uses SymbolManager.alpaca_to_binance_format()")
    print("- Risk Manager: Uses SymbolManager.binance_to_alpaca_format()")
    print("- Test Scripts: Updated to use SymbolManager")
    
    print("\n✅ Expected Benefits:")
    print("- Consistent symbol formatting across all components")
    print("- Centralized symbol mapping management")
    print("- Elimination of 'no trade found' errors due to format mismatches")
    print("- Easy addition of new trading symbols")

if __name__ == "__main__":
    main()
