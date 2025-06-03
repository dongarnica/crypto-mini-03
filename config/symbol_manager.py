#!/usr/bin/env python3
"""
Centralized Symbol Management
============================

Manages all cryptocurrency symbol mappings and conversions between 
different exchange formats (Binance, Alpaca) using environment variables.

Author: Crypto Trading Strategy Engine
Date: June 3, 2025
"""

import os
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class SymbolMapping:
    """Symbol mapping between different exchange formats."""
    binance: str      # BTCUSDT
    alpaca: str       # BTC/USD
    base: str         # BTC
    quote: str        # USD
    display: str      # Bitcoin
    

class SymbolManager:
    """Centralized symbol management for all exchanges."""
    
    def __init__(self):
        """Initialize symbol manager."""
        self.logger = logging.getLogger('SymbolManager')
        
        # Standard symbol mappings
        self.mappings: Dict[str, SymbolMapping] = {
            'BTCUSDT': SymbolMapping('BTCUSDT', 'BTC/USD', 'BTC', 'USD', 'Bitcoin'),
            'ETHUSDT': SymbolMapping('ETHUSDT', 'ETH/USD', 'ETH', 'USD', 'Ethereum'),
            'DOTUSDT': SymbolMapping('DOTUSDT', 'DOT/USD', 'DOT', 'USD', 'Polkadot'),
            'LINKUSDT': SymbolMapping('LINKUSDT', 'LINK/USD', 'LINK', 'USD', 'Chainlink'),
            'LTCUSDT': SymbolMapping('LTCUSDT', 'LTC/USD', 'LTC', 'USD', 'Litecoin'),
            'BCHUSDT': SymbolMapping('BCHUSDT', 'BCH/USD', 'BCH', 'USD', 'Bitcoin Cash'),
            'UNIUSDT': SymbolMapping('UNIUSDT', 'UNI/USD', 'UNI', 'USD', 'Uniswap'),
            'SOLUSDT': SymbolMapping('SOLUSDT', 'SOL/USD', 'SOL', 'USD', 'Solana'),
            'AVAXUSDT': SymbolMapping('AVAXUSDT', 'AVAX/USD', 'AVAX', 'USD', 'Avalanche'),
            'ADAUSDT': SymbolMapping('ADAUSDT', 'ADA/USD', 'ADA', 'USD', 'Cardano'),
            'MATICUSDT': SymbolMapping('MATICUSDT', 'MATIC/USD', 'MATIC', 'USD', 'Polygon'),
            'XLMUSDT': SymbolMapping('XLMUSDT', 'XLM/USD', 'XLM', 'USD', 'Stellar'),
        }
        
        # Create reverse mappings for quick lookup
        self.alpaca_to_binance = {mapping.alpaca: mapping.binance for mapping in self.mappings.values()}
        self.binance_to_alpaca = {mapping.binance: mapping.alpaca for mapping in self.mappings.values()}
        
    def get_symbols_from_env(self, env_var: str = 'CRYPTO_SYMBOLS') -> List[str]:
        """
        Get trading symbols from environment variable.
        
        Args:
            env_var: Environment variable name (default: CRYPTO_SYMBOLS)
            
        Returns:
            List of symbols in Binance format
        """
        symbols_str = os.getenv(env_var, '')
        if not symbols_str:
            self.logger.warning(f"No symbols found in {env_var}, using defaults")
            return ['BTCUSDT', 'ETHUSDT', 'DOTUSDT']
            
        symbols = [s.strip() for s in symbols_str.split(',') if s.strip()]
        self.logger.info(f"Loaded {len(symbols)} symbols from {env_var}: {symbols}")
        return symbols
        
    def get_primary_symbols(self) -> Dict[str, str]:
        """Get primary trading symbols from environment."""
        return {
            'primary': os.getenv('PRIMARY_SYMBOL', 'BTCUSDT'),
            'secondary': os.getenv('SECONDARY_SYMBOL', 'ETHUSDT'),
            'tertiary': os.getenv('TERTIARY_SYMBOL', 'DOTUSDT')
        }
        
    def get_symbol_groups(self) -> Dict[str, List[str]]:
        """Get symbol groups from environment."""
        return {
            'major': self.get_symbols_from_env('MAJOR_SYMBOLS'),
            'defi': self.get_symbols_from_env('DEFI_SYMBOLS'),
            'all': self.get_symbols_from_env('CRYPTO_SYMBOLS')
        }
        
    def binance_to_alpaca_format(self, binance_symbol: str) -> str:
        """
        Convert Binance symbol to Alpaca format.
        
        Args:
            binance_symbol: Symbol in Binance format (e.g., 'BTCUSDT')
            
        Returns:
            Symbol in Alpaca format (e.g., 'BTC/USD')
        """
        if binance_symbol in self.binance_to_alpaca:
            return self.binance_to_alpaca[binance_symbol]
            
        # Fallback conversion for unmapped symbols
        if binance_symbol.endswith('USDT'):
            base = binance_symbol[:-4]
            alpaca_format = f"{base}/USD"
            self.logger.warning(f"Using fallback conversion: {binance_symbol} -> {alpaca_format}")
            return alpaca_format
        elif binance_symbol.endswith('USD') and '/' not in binance_symbol:
            base = binance_symbol[:-3]
            alpaca_format = f"{base}/USD"
            self.logger.warning(f"Using fallback conversion: {binance_symbol} -> {alpaca_format}")
            return alpaca_format
        else:
            self.logger.warning(f"Unknown symbol format: {binance_symbol}")
            return binance_symbol
            
    def alpaca_to_binance_format(self, alpaca_symbol: str) -> str:
        """
        Convert Alpaca symbol to Binance format.
        
        Args:
            alpaca_symbol: Symbol in Alpaca format (e.g., 'BTC/USD')
            
        Returns:
            Symbol in Binance format (e.g., 'BTCUSDT')
        """
        if alpaca_symbol in self.alpaca_to_binance:
            return self.alpaca_to_binance[alpaca_symbol]
            
        # Fallback conversion for unmapped symbols
        if '/' in alpaca_symbol:
            base, quote = alpaca_symbol.split('/')
            if quote.upper() == 'USD':
                binance_format = f"{base}USDT"
                self.logger.warning(f"Using fallback conversion: {alpaca_symbol} -> {binance_format}")
                return binance_format
                
        self.logger.warning(f"Unknown symbol format: {alpaca_symbol}")
        return alpaca_symbol
        
    def convert_symbols_to_alpaca(self, binance_symbols: List[str]) -> List[str]:
        """Convert list of Binance symbols to Alpaca format."""
        return [self.binance_to_alpaca_format(symbol) for symbol in binance_symbols]
        
    def convert_symbols_to_binance(self, alpaca_symbols: List[str]) -> List[str]:
        """Convert list of Alpaca symbols to Binance format."""
        return [self.alpaca_to_binance_format(symbol) for symbol in alpaca_symbols]
        
    def get_symbol_info(self, symbol: str) -> Optional[SymbolMapping]:
        """
        Get complete symbol information.
        
        Args:
            symbol: Symbol in any format (Binance or Alpaca)
            
        Returns:
            SymbolMapping object or None if not found
        """
        # Try direct lookup (Binance format)
        if symbol in self.mappings:
            return self.mappings[symbol]
            
        # Try reverse lookup (Alpaca format)
        binance_symbol = self.alpaca_to_binance_format(symbol)
        if binance_symbol in self.mappings:
            return self.mappings[binance_symbol]
            
        return None
        
    def validate_symbols(self, symbols: List[str]) -> Dict[str, bool]:
        """
        Validate if symbols are supported.
        
        Args:
            symbols: List of symbols to validate
            
        Returns:
            Dictionary mapping symbol to validation status
        """
        results = {}
        for symbol in symbols:
            results[symbol] = self.get_symbol_info(symbol) is not None
            
        return results
        
    def get_supported_symbols(self) -> List[str]:
        """Get list of all supported symbols in Binance format."""
        return list(self.mappings.keys())
        
    def print_symbol_mappings(self):
        """Print all symbol mappings for debugging."""
        print("📊 Symbol Mappings:")
        print("-" * 60)
        for binance_symbol, mapping in self.mappings.items():
            print(f"{mapping.binance:10} -> {mapping.alpaca:10} ({mapping.display})")


# Global symbol manager instance
symbol_manager = SymbolManager()

# Convenience functions for backward compatibility
def get_trading_symbols() -> List[str]:
    """Get trading symbols from environment."""
    return symbol_manager.get_symbols_from_env()

def convert_to_alpaca(binance_symbol: str) -> str:
    """Convert Binance symbol to Alpaca format."""
    return symbol_manager.binance_to_alpaca_format(binance_symbol)

def convert_to_binance(alpaca_symbol: str) -> str:
    """Convert Alpaca symbol to Binance format."""
    return symbol_manager.alpaca_to_binance_format(alpaca_symbol)


if __name__ == "__main__":
    # Test the symbol manager
    print("🧪 Testing Symbol Manager")
    print("=" * 40)
    
    sm = SymbolManager()
    
    # Test environment loading
    symbols = sm.get_symbols_from_env()
    print(f"Symbols from env: {symbols}")
    
    # Test conversions
    test_conversions = [
        ('BTCUSDT', 'BTC/USD'),
        ('ETHUSDT', 'ETH/USD'),
        ('DOTUSDT', 'DOT/USD')
    ]
    
    print("\n🔄 Testing Conversions:")
    for binance, expected_alpaca in test_conversions:
        alpaca = sm.binance_to_alpaca_format(binance)
        reverse = sm.alpaca_to_binance_format(alpaca)
        status = "✅" if alpaca == expected_alpaca and reverse == binance else "❌"
        print(f"{status} {binance} <-> {alpaca} <-> {reverse}")
    
    # Print all mappings
    print()
    sm.print_symbol_mappings()
