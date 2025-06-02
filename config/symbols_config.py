"""
Centralized cryptocurrency symbols configuration.
Loads all symbols from .env file and provides them to the entire application.
"""

import os
from typing import List, Dict, Set

# Load environment variables from .env file
def load_env_variables():
    """Load environment variables from .env file manually."""
    env_vars = {}
    try:
        with open('.env', 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key.strip()] = value.strip()
    except FileNotFoundError:
        print("Warning: .env file not found")
    return env_vars

# Load environment variables
ENV_VARS = load_env_variables()

class CryptoSymbolsConfig:
    """Centralized configuration for cryptocurrency symbols."""
    
    def __init__(self):
        """Initialize and load all symbols from environment variables."""
        self._load_symbols_from_env()
        self._validate_symbols()
    
    def _load_symbols_from_env(self):
        """Load all symbol lists from environment variables."""
        # Main crypto symbols list
        crypto_symbols = ENV_VARS.get('CRYPTO_SYMBOLS', '')
        self.crypto_symbols = [s.strip() for s in crypto_symbols.split(',') if s.strip()]
        
        # Primary trading pairs
        self.primary_symbol = ENV_VARS.get('PRIMARY_SYMBOL', 'BTCUSDT')
        self.secondary_symbol = ENV_VARS.get('SECONDARY_SYMBOL', 'ETHUSDT')
        self.tertiary_symbol = ENV_VARS.get('TERTIARY_SYMBOL', 'ADAUSDT')
        
        # Additional symbol categories
        defi_symbols = ENV_VARS.get('DEFI_SYMBOLS', '')
        self.defi_symbols = [s.strip() for s in defi_symbols.split(',') if s.strip()]
        
        altcoin_symbols = ENV_VARS.get('ALTCOIN_SYMBOLS', '')
        self.altcoin_symbols = [s.strip() for s in altcoin_symbols.split(',') if s.strip()]
        
        # Create consolidated lists
        self._create_consolidated_lists()
    
    def _create_consolidated_lists(self):
        """Create consolidated symbol lists from all sources."""
        # Combine all symbols into a master set to avoid duplicates
        all_symbols_set = set()
        
        # Add from main crypto symbols
        all_symbols_set.update(self.crypto_symbols)
        
        # Add primary symbols
        all_symbols_set.add(self.primary_symbol)
        all_symbols_set.add(self.secondary_symbol)
        all_symbols_set.add(self.tertiary_symbol)
        
        # Add category symbols
        all_symbols_set.update(self.defi_symbols)
        all_symbols_set.update(self.altcoin_symbols)
        
        # Convert to sorted list for consistency
        self.all_symbols = sorted(list(all_symbols_set))
        
        # Create priority-ordered lists
        self.primary_symbols = [self.primary_symbol, self.secondary_symbol, self.tertiary_symbol]
        
        # High-priority symbols (most liquid and popular)
        self.high_priority_symbols = [
            'BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT', 'LINKUSDT'
        ]
        
        # Medium-priority symbols
        self.medium_priority_symbols = [
            'LTCUSDT', 'BCHUSDT', 'AVAXUSDT', 'MATICUSDT', 'UNIUSDT'
        ]
        
        # All trading symbols (filter only those we have in our master list)
        self.trading_symbols = [s for s in self.all_symbols if s in self.all_symbols]
    
    def _validate_symbols(self):
        """Validate that all symbols follow expected format."""
        valid_symbols = []
        
        for symbol in self.all_symbols:
            # Basic validation: should end with USDT and have reasonable length
            if symbol.endswith('USDT') and len(symbol) >= 6 and len(symbol) <= 12:
                valid_symbols.append(symbol)
            else:
                print(f"⚠️  Warning: Invalid symbol format: {symbol}")
        
        self.all_symbols = valid_symbols
        
        # Update other lists to only include valid symbols
        self.crypto_symbols = [s for s in self.crypto_symbols if s in valid_symbols]
        self.defi_symbols = [s for s in self.defi_symbols if s in valid_symbols]
        self.altcoin_symbols = [s for s in self.altcoin_symbols if s in valid_symbols]
        self.high_priority_symbols = [s for s in self.high_priority_symbols if s in valid_symbols]
        self.medium_priority_symbols = [s for s in self.medium_priority_symbols if s in valid_symbols]
    
    def get_symbols_by_category(self, category: str) -> List[str]:
        """
        Get symbols by category.
        
        Args:
            category: Category name ('all', 'crypto', 'defi', 'altcoin', 'primary', 'high_priority', 'medium_priority', 'trading')
            
        Returns:
            List of symbols for the specified category
        """
        category = category.lower()
        
        if category == 'all':
            return self.all_symbols.copy()
        elif category == 'crypto':
            return self.crypto_symbols.copy()
        elif category == 'defi':
            return self.defi_symbols.copy()
        elif category == 'altcoin':
            return self.altcoin_symbols.copy()
        elif category == 'primary':
            return self.primary_symbols.copy()
        elif category == 'high_priority':
            return self.high_priority_symbols.copy()
        elif category == 'medium_priority':
            return self.medium_priority_symbols.copy()
        elif category == 'trading':
            return self.trading_symbols.copy()
        else:
            raise ValueError(f"Unknown category: {category}")
    
    def get_symbols_for_trading(self, max_symbols: int = None, priority: str = 'high') -> List[str]:
        """
        Get symbols optimized for trading.
        
        Args:
            max_symbols: Maximum number of symbols to return
            priority: Priority level ('high', 'medium', 'all')
            
        Returns:
            List of symbols suitable for trading
        """
        if priority == 'high':
            symbols = self.high_priority_symbols.copy()
        elif priority == 'medium':
            symbols = self.high_priority_symbols + self.medium_priority_symbols
        else:
            symbols = self.all_symbols.copy()
        
        # Remove duplicates while preserving order
        seen = set()
        unique_symbols = []
        for symbol in symbols:
            if symbol not in seen:
                seen.add(symbol)
                unique_symbols.append(symbol)
        
        if max_symbols:
            return unique_symbols[:max_symbols]
        
        return unique_symbols
    
    def is_valid_symbol(self, symbol: str) -> bool:
        """Check if a symbol is in our valid symbols list."""
        return symbol in self.all_symbols
    
    def get_symbol_info(self) -> Dict:
        """Get comprehensive information about all symbols."""
        return {
            'total_symbols': len(self.all_symbols),
            'all_symbols': self.all_symbols,
            'primary_symbols': self.primary_symbols,
            'high_priority_symbols': self.high_priority_symbols,
            'medium_priority_symbols': self.medium_priority_symbols,
            'defi_symbols': self.defi_symbols,
            'altcoin_symbols': self.altcoin_symbols,
            'crypto_symbols': self.crypto_symbols,
            'categories': {
                'primary': len(self.primary_symbols),
                'high_priority': len(self.high_priority_symbols),
                'medium_priority': len(self.medium_priority_symbols),
                'defi': len(self.defi_symbols),
                'altcoin': len(self.altcoin_symbols),
                'total': len(self.all_symbols)
            }
        }
    
    def print_summary(self):
        """Print a summary of all configured symbols."""
        print("\n" + "="*80)
        print("🪙 CRYPTOCURRENCY SYMBOLS CONFIGURATION")
        print("="*80)
        
        info = self.get_symbol_info()
        
        print(f"📊 Total Symbols: {info['total_symbols']}")
        print(f"🥇 Primary Symbols: {len(self.primary_symbols)}")
        print(f"⭐ High Priority: {len(self.high_priority_symbols)}")
        print(f"📈 Medium Priority: {len(self.medium_priority_symbols)}")
        print(f"🔗 DeFi Symbols: {len(self.defi_symbols)}")
        print(f"🪙 Altcoin Symbols: {len(self.altcoin_symbols)}")
        
        print(f"\n🎯 PRIMARY TRADING SYMBOLS:")
        for i, symbol in enumerate(self.primary_symbols, 1):
            print(f"  {i}. {symbol}")
        
        print(f"\n⭐ HIGH PRIORITY SYMBOLS:")
        for symbol in self.high_priority_symbols:
            print(f"  • {symbol}")
        
        print(f"\n📊 ALL AVAILABLE SYMBOLS:")
        # Print in rows of 6 for better formatting
        symbols_per_row = 6
        for i in range(0, len(self.all_symbols), symbols_per_row):
            row_symbols = self.all_symbols[i:i+symbols_per_row]
            print(f"  {' | '.join(f'{s:>10}' for s in row_symbols)}")
        
        print("\n" + "="*80)


# Create global instance
symbols_config = CryptoSymbolsConfig()

# Convenience functions for easy access
def get_all_symbols() -> List[str]:
    """Get all available cryptocurrency symbols."""
    return symbols_config.all_symbols.copy()

def get_trading_symbols(max_symbols: int = None, priority: str = 'high') -> List[str]:
    """Get symbols optimized for trading."""
    return symbols_config.get_symbols_for_trading(max_symbols, priority)

def get_primary_symbols() -> List[str]:
    """Get primary trading symbols."""
    return symbols_config.primary_symbols.copy()

def get_high_priority_symbols() -> List[str]:
    """Get high priority symbols."""
    return symbols_config.high_priority_symbols.copy()

def get_symbols_by_category(category: str) -> List[str]:
    """Get symbols by category."""
    return symbols_config.get_symbols_by_category(category)

def is_valid_symbol(symbol: str) -> bool:
    """Check if a symbol is valid."""
    return symbols_config.is_valid_symbol(symbol)

def print_symbols_summary():
    """Print symbols configuration summary."""
    symbols_config.print_summary()


if __name__ == "__main__":
    # Demo usage
    print_symbols_summary()
    
    print(f"\n🎯 Recommended trading symbols (top 5): {get_trading_symbols(5)}")
    print(f"📈 All high priority symbols: {get_high_priority_symbols()}")
    print(f"🔗 DeFi symbols: {get_symbols_by_category('defi')}")
