#!/usr/bin/env python3
"""
Main Trading Strategy Engine (Refactored)
=========================================

Orchestrates all trading components including ML predictions, 
risk management, position tracking, and trade execution.

Author: Crypto Trading Strategy Engine
Date: June 2, 2025
"""

import os
import sys
import logging
import time
import signal
import traceback
import argparse
from datetime import datetime
from typing import List

# Add the parent directory to Python path for imports
sys.path.append('/workspaces/crypto-mini-03')

# Import our existing modules
from alpaca.alpaca_client import AlpacaCryptoClient
from binance.binance_client import BinanceUSClient
from dotenv import load_dotenv

# Import our refactored components
from trading.models import TradingConfig, TradingSignal
from trading.position_manager import PositionManager
from trading.risk_manager import RiskManager
from trading.trade_executor import TradeExecutor
from trading.ml_engine import MLEngine
from trading.portfolio_manager import PortfolioManager
from trading.enhanced_output import EnhancedOutputDisplay

# Load environment variables
load_dotenv('/workspaces/crypto-mini-03/.env')

# Import centralized symbols configuration
try:
    from config.symbols_config import get_trading_symbols, get_all_symbols, is_valid_symbol
    SYMBOLS_CONFIG_AVAILABLE = True
except ImportError:
    SYMBOLS_CONFIG_AVAILABLE = False
    print("⚠️  Warning: Centralized symbols configuration not available, using fallback")

# Import centralized symbol manager
from config.symbol_manager import symbol_manager


class TradingStrategyEngine:
    """
    Main trading strategy engine that orchestrates all components
    for automated crypto trading with ML predictions.
    """
    
    def __init__(self, config: TradingConfig):
        """
        Initialize the trading strategy engine.
        
        Args:
            config: Trading configuration
        """
        self.config = config
        self.running = False
        
        # Initialize enhanced display timer (5 minutes = 300 seconds)
        self.last_enhanced_display_time = 0
        self.enhanced_display_interval = 300  # 5 minutes
        
        # Setup logging
        self.setup_logging()
        
        # Initialize components
        self.trading_client = None
        self.binance_client = None
        self.position_manager = None
        self.risk_manager = None
        self.trade_executor = None
        self.ml_engine = None
        self.portfolio_manager = None
        self.enhanced_display = None
        
        self.logger.info("🚀 Trading Strategy Engine initialized")
        self.logger.info(f"Configuration: {self.config.__dict__}")
    
    def setup_logging(self) -> None:
        """Setup logging configuration."""
        log_level = getattr(logging, self.config.log_level.upper())
        
        # Determine log directory based on environment
        if os.path.exists('/app'):
            # Running in Docker container
            log_dir = '/app/trading/logs'
        else:
            # Running in development environment
            log_dir = '/workspaces/crypto-mini-03/trading/logs'
        
        # Create logs directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Setup logger
        self.logger = logging.getLogger('TradingEngine')
        self.logger.setLevel(log_level)
        
        # Remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler
        if self.config.save_trades:
            file_handler = logging.FileHandler(
                f'{log_dir}/trading_{datetime.now().strftime("%Y%m%d")}.log'
            )
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
    
    def validate_symbols_with_data(self, symbols: List[str]) -> List[str]:
        """
        Validate symbols against available historical data files.
        
        Args:
            symbols: List of symbols to validate
            
        Returns:
            List of valid symbols that have historical data
        """
        valid_symbols = []
        # Determine historical exports directory based on environment
        if os.path.exists('/app'):
            # Running in Docker container
            historical_exports_dir = '/app/historical_exports'
        else:
            # Running in development environment
            historical_exports_dir = '/workspaces/crypto-mini-03/historical_exports'
        
        try:
            if not os.path.exists(historical_exports_dir):
                self.logger.warning(f"Historical exports directory not found: {historical_exports_dir}")
                return symbols  # Return all symbols if directory doesn't exist
            
            # Get list of available data files
            available_files = os.listdir(historical_exports_dir)
            available_symbols = set()
            
            for file in available_files:
                if file.endswith('.csv') and 'USDT' in file:
                    # Extract symbol from filename
                    symbol = file.split('_')[0]
                    available_symbols.add(symbol)
            
            self.logger.info(f"📁 Found historical data for {len(available_symbols)} symbols: {sorted(available_symbols)}")
            
            # Validate requested symbols
            for symbol in symbols:
                if symbol in available_symbols:
                    valid_symbols.append(symbol)
                    self.logger.info(f"✅ {symbol}: Historical data available")
                else:
                    self.logger.warning(f"❌ {symbol}: No historical data found")
            
            if len(valid_symbols) == 0:
                self.logger.error("No valid symbols found with historical data!")
                return symbols[:3]  # Return first 3 symbols as emergency fallback
            
            self.logger.info(f"🎯 Using {len(valid_symbols)} validated symbols: {valid_symbols}")
            return valid_symbols
            
        except Exception as e:
            self.logger.error(f"Error validating symbols: {e}")
            return symbols  # Return original symbols on error
    
    def initialize_components(self, symbols: List[str]) -> None:
        """
        Initialize all trading components.
        
        Args:
            symbols: List of crypto symbols to trade
        """
        self.logger.info("🔧 Initializing trading components...")
        
        # Convert symbols to Alpaca format
        converted_symbols = [self._convert_symbol_format(symbol) for symbol in symbols]
        self.logger.info(f"🔄 Symbol conversion: {symbols} -> {converted_symbols}")
        
        # Initialize Alpaca trading client
        try:
            self.trading_client = AlpacaCryptoClient(paper=self.config.paper_trading)
            
            # Check crypto eligibility
            if not self.trading_client.check_crypto_eligibility():
                raise ValueError("Account not eligible for crypto trading")
            
            self.logger.info("✅ Alpaca trading client initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Alpaca trading client: {e}")
            self.logger.warning("⚠️  Continuing with limited functionality...")
        
        # Initialize Binance client for price data fallback
        try:
            self.binance_client = BinanceUSClient()
            self.logger.info("✅ Binance client initialized for price data")
        except Exception as e:
            self.logger.error(f"Failed to initialize Binance client: {e}")
            self.logger.warning("⚠️  No price data fallback available")
        
        # Initialize component managers (with converted symbols)
        self.position_manager = PositionManager(self.config, self.trading_client)
        self.portfolio_manager = PortfolioManager(self.trading_client)
        self.trade_executor = TradeExecutor(self.config, self.trading_client, self.position_manager)
        self.risk_manager = RiskManager(self.config, self.position_manager, self.trading_client, self.trade_executor)
        self.ml_engine = MLEngine(self.config)
        
        # CRITICAL FIX: Link trade executor to risk manager for proper stop loss execution
        self.risk_manager.set_trade_executor(self.trade_executor)
        
        # Initialize enhanced output display
        self.enhanced_display = EnhancedOutputDisplay(self.logger)
        
        # Initialize ML pipelines with original symbols (for historical data) - 3-CLASS ONLY
        self.ml_engine.initialize_pipelines(symbols)
        
        # Store both original and converted symbols for reference
        self.original_symbols = symbols
        self.converted_symbols = converted_symbols
        
        self.logger.info("🎯 CRITICAL: Using 3-class models exclusively - binary models abandoned")
        self.logger.info("📊 3-class models provide: Buy, Hold, Sell signals with higher profitability")
        
        # Load existing positions and update portfolio info
        self.position_manager.load_existing_positions()
        self.portfolio_manager.update_portfolio_info()
        
        # Start performance tracking
        self.portfolio_manager.start_tracking()
        
        self.logger.info("✅ All components initialized successfully")
    
    def process_trading_signals(self, symbols: List[str]) -> None:
        """
        Main trading logic - process signals for all symbols.
        
        Args:
            symbols: List of symbols to process (original format)
        """
        try:
            self.logger.info(f"📡 Processing trading signals for {len(symbols)} symbols...")
            
            # Collect all ML predictions for enhanced display
            ml_predictions = []
            
            for i, symbol in enumerate(symbols):
                try:
                    # Get ML prediction using original symbol format
                    ml_prediction = self.ml_engine.get_prediction(symbol)
                    
                    if not ml_prediction:
                        continue
                    
                    # Add symbol to prediction for display
                    ml_prediction['symbol'] = symbol
                    ml_predictions.append(ml_prediction)
                    
                    signal_str = ml_prediction.get('signal', 'HOLD')
                    confidence = ml_prediction.get('confidence', 0.0)
                    current_price = ml_prediction.get('current_price', 0.0)
                    
                    # Convert signal string to enum
                    signal = self._convert_signal_string(signal_str)
                    
                    # Get converted symbol for trading
                    trading_symbol = self.converted_symbols[i] if i < len(self.converted_symbols) else self._convert_symbol_format(symbol)
                    
                    # Always use Binance for current price (more reliable)
                    binance_price = self.get_current_price_with_fallback(trading_symbol, symbol)
                    if binance_price > 0:
                        current_price = binance_price
                        ml_prediction['current_price'] = current_price  # Update for display
                        self.logger.debug(f"💰 Using Binance price for {symbol}: ${current_price:,.2f}")
                    elif current_price <= 0:
                        self.logger.warning(f"❌ No price data available for {symbol}, skipping")
                        continue
                    
                    # Check if signal meets confidence threshold
                    if signal != TradingSignal.HOLD and confidence >= self.config.min_confidence:
                        
                        # Check if we can trade (using trading symbol)
                        if self.trade_executor.can_trade(trading_symbol, signal):
                            
                            # Calculate position size
                            position_size = self.risk_manager.calculate_position_size(
                                symbol=trading_symbol,
                                confidence=confidence,
                                current_price=current_price,
                                portfolio_value=self.portfolio_manager.get_portfolio_value(),
                                available_cash=self.portfolio_manager.get_available_cash()
                            )
                            
                            # Check risk management approval
                            if self.risk_manager.should_allow_trade(
                                symbol=trading_symbol,
                                signal=signal,
                                position_size=position_size,
                                portfolio_value=self.portfolio_manager.get_portfolio_value()
                            ):
                                # Execute trade (using trading symbol)
                                trade_record = self.trade_executor.execute_trade(
                                    symbol=trading_symbol,
                                    signal=signal,
                                    confidence=confidence,
                                    current_price=current_price,
                                    position_size=position_size,
                                    ml_prediction=ml_prediction
                                )
                                
                                if trade_record:
                                    self.logger.info(f"✅ Trade executed: {trade_record.action} {trading_symbol} (from {symbol})")
                                    # Update performance metrics
                                    self.portfolio_manager.update_performance_metrics(trade_record)
                    
                except Exception as e:
                    self.logger.error(f"Error processing {symbol}: {e}")
                    continue
            
            # Display enhanced output with signal summary and predictions
            current_time = time.time()
            time_since_last_display = current_time - self.last_enhanced_display_time
            
            # Show enhanced display every 5 minutes when there are trade updates
            if (self.enhanced_display and 
                (ml_predictions or time_since_last_display >= self.enhanced_display_interval)):
                
                if not ml_predictions:
                    # Create dummy predictions for display if none exist
                    ml_predictions = []
                    for symbol in symbols[:3]:  # Show status for first 3 symbols
                        dummy_prediction = {
                            'symbol': symbol,
                            'signal': 'HOLD',
                            'confidence': 0.5,
                            'current_price': self.get_current_price_with_fallback(
                                self.converted_symbols[symbols.index(symbol)] if symbol in symbols else symbol, 
                                symbol
                            ),
                            'predicted_return': 0.0,
                            'timestamp': datetime.now().isoformat()
                        }
                        ml_predictions.append(dummy_prediction)
                
                self.enhanced_display.display_full_status(ml_predictions)
                self.last_enhanced_display_time = current_time
                self.logger.info(f"📊 Enhanced display shown at {datetime.now().strftime('%H:%M:%S')}")
            elif time_since_last_display >= self.enhanced_display_interval:
                self.logger.info(f"⏰ Enhanced display interval reached ({self.enhanced_display_interval}s), but no enhanced display available")
            
            # Update position prices with current Binance data before risk checks
            self.update_position_prices_from_binance()
            
            # Check risk management for all positions
            self.risk_manager.check_risk_management(
                self.portfolio_manager.get_portfolio_value()
            )
            
            # Update portfolio information
            self.portfolio_manager.update_portfolio_info()
            
        except Exception as e:
            self.logger.error(f"Error in trading signal processing: {e}")
    
    def start_trading(self, symbols: List[str], update_interval: int = 300) -> None:
        """
        Start the trading engine.
        
        Args:
            symbols: List of crypto symbols to trade
            update_interval: Update interval in seconds (default: 5 minutes)
        """
        self.logger.info("🚀 Starting Crypto Trading Strategy Engine...")
        self.running = True
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        try:
            # Initialize components
            self.initialize_components(symbols)
            
            # Print initial status
            self.portfolio_manager.print_status(self.position_manager)
            
            cycle_count = 0
            last_hour = datetime.now().hour
            
            while self.running:
                try:
                    loop_start = time.time()
                    
                    # Process trading signals
                    self.process_trading_signals(symbols)
                    
                    # Print status every 10 cycles (about 50 minutes with 5-min intervals)
                    cycle_count += 1
                    if cycle_count % 10 == 0:
                        self.portfolio_manager.print_status(self.position_manager)
                    
                    # Reset daily counters at midnight
                    current_hour = datetime.now().hour
                    if last_hour == 23 and current_hour == 0:
                        self.trade_executor.reset_daily_counters()
                    last_hour = current_hour
                    
                    # Calculate sleep time
                    loop_duration = time.time() - loop_start
                    sleep_time = max(0, update_interval - loop_duration)
                    
                    # Display countdown timer if enhanced display is available
                    if sleep_time > 0 and self.enhanced_display:
                        self.enhanced_display.display_countdown(int(sleep_time))
                    elif sleep_time > 0:
                        time.sleep(sleep_time)
                    
                except KeyboardInterrupt:
                    self.logger.info("Received interrupt signal, shutting down...")
                    break
                except Exception as e:
                    self.logger.error(f"Error in main trading loop: {e}")
                    self.logger.error(traceback.format_exc())
                    time.sleep(60)  # Wait 1 minute before retrying
        
        except Exception as e:
            self.logger.error(f"Fatal error in trading engine: {e}")
            self.logger.error(traceback.format_exc())
        
        finally:
            self.stop_trading()
    
    def stop_trading(self) -> None:
        """Stop the trading engine gracefully."""
        self.logger.info("🛑 Stopping trading engine...")
        self.running = False
        
        # Final status
        if self.portfolio_manager and self.position_manager:
            self.portfolio_manager.print_status(self.position_manager)
        
        self.logger.info("✅ Trading engine stopped")
    
    def _signal_handler(self, signum, frame) -> None:
        """Handle system signals for graceful shutdown."""
        self.logger.info(f"Received signal {signum}, stopping trading...")
        self.running = False
    
    def _convert_symbol_format(self, symbol: str) -> str:
        """Convert symbol format from Binance to Alpaca format using SymbolManager."""
        return symbol_manager.binance_to_alpaca_format(symbol)
    
    def _convert_signal_string(self, signal_str: str) -> TradingSignal:
        """Convert signal string to TradingSignal enum."""
        signal_str_upper = signal_str.upper()
        
        if signal_str_upper in ['BUY', 'UP']:
            return TradingSignal.BUY
        elif signal_str_upper in ['SELL', 'DOWN']:
            return TradingSignal.SELL
        else:
            return TradingSignal.HOLD
    
    def get_current_price_with_fallback(self, symbol: str, original_symbol: str = None) -> float:
        """
        Get current price using Binance as primary source with improved symbol handling.
        
        Args:
            symbol: Symbol in Alpaca format (e.g., 'BTC/USD') - for compatibility
            original_symbol: Symbol in Binance format (e.g., 'BTCUSDT') - preferred
            
        Returns:
            Current price as float, or 0.0 if failed
        """
        try:
            # Determine the Binance symbol to use
            binance_symbol = original_symbol if original_symbol else symbol_manager.alpaca_to_binance_format(symbol)
            
            # Validate Binance symbol format
            if not self._validate_binance_symbol_format(binance_symbol):
                self.logger.error(f"❌ Invalid Binance symbol format: {binance_symbol}")
                return 0.0
            
            # Use Binance as primary price source
            if self.binance_client:
                return self._get_binance_price_direct(binance_symbol)
            
            self.logger.error(f"❌ No Binance client available for {binance_symbol}")
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Error getting price for {original_symbol or symbol}: {e}")
            return 0.0
    
    def update_position_prices_from_binance(self) -> None:
        """Update all position prices using Binance price data."""
        try:
            positions = self.position_manager.get_all_positions()
            
            for symbol, position in positions.items():
                try:
                    # CRITICAL FIX: symbol from position manager is already in Binance format (BTCUSDT)
                    # Validate the symbol format before making API calls
                    if not self._validate_binance_symbol_format(symbol):
                        self.logger.error(f"❌ Invalid Binance symbol format: {symbol}")
                        continue
                    
                    # Get current price directly from Binance using Binance format
                    binance_price = self._get_binance_price_direct(symbol)
                    
                    if binance_price > 0:
                        # Update position with current Binance price
                        self.position_manager.update_position_price(symbol, binance_price)
                        self.logger.debug(f"💰 Updated {symbol} position price: ${binance_price:,.2f}")
                    else:
                        self.logger.warning(f"❌ Failed to get Binance price for position {symbol}")
                        
                except Exception as e:
                    self.logger.error(f"Error updating price for position {symbol}: {e}")
                    continue
                    
        except Exception as e:
            self.logger.error(f"Error updating position prices from Binance: {e}")
    
    def force_enhanced_display(self, symbols: List[str]) -> None:
        """
        Force display of enhanced output for testing purposes.
        
        Args:
            symbols: List of symbols to display status for
        """
        if not self.enhanced_display:
            self.logger.warning("Enhanced display not initialized")
            return
        
        try:
            # Create current predictions for display
            ml_predictions = []
            
            for symbol in symbols[:5]:  # Show status for up to 5 symbols
                try:
                    # Get current price
                    trading_symbol = self._convert_symbol_format(symbol)
                    current_price = self.get_current_price_with_fallback(trading_symbol, symbol)
                    
                    # Try to get real ML prediction
                    ml_prediction = None
                    if self.ml_engine:
                        ml_prediction = self.ml_engine.get_prediction(symbol)
                    
                    if ml_prediction:
                        ml_prediction['symbol'] = symbol
                        if current_price > 0:
                            ml_prediction['current_price'] = current_price
                        ml_predictions.append(ml_prediction)
                    else:
                        # Create dummy prediction for display
                        dummy_prediction = {
                            'symbol': symbol,
                            'signal': 'HOLD',
                            'confidence': 0.5,
                            'current_price': current_price if current_price > 0 else 0.0,
                            'predicted_return': 0.0,
                            'timestamp': datetime.now().isoformat()
                        }
                        ml_predictions.append(dummy_prediction)
                        
                except Exception as e:
                    self.logger.error(f"Error creating prediction for {symbol}: {e}")
                    continue
            
            if ml_predictions:
                self.enhanced_display.display_full_status(ml_predictions)
                self.last_enhanced_display_time = time.time()
                self.logger.info(f"🎯 Enhanced display forced at {datetime.now().strftime('%H:%M:%S')}")
            else:
                self.logger.warning("No predictions available for enhanced display")
                
        except Exception as e:
            self.logger.error(f"Error forcing enhanced display: {e}")
    
    def _validate_binance_symbol_format(self, symbol: str) -> bool:
        """
        Validate that symbol is in proper Binance format.
        
        Args:
            symbol: Symbol to validate
            
        Returns:
            True if valid Binance format
        """
        try:
            if not symbol or not isinstance(symbol, str):
                return False
            
            # Check if it's a known mapped symbol
            if symbol in symbol_manager.mappings:
                return True
            
            # Check if it follows Binance USDT pattern
            if symbol.endswith('USDT') and len(symbol) > 4:
                base = symbol[:-4]
                return base.isalpha() and base.isupper()
            
            # Check for other quote currencies if needed
            for quote in ['USD', 'BTC', 'ETH', 'BNB']:
                if symbol.endswith(quote) and len(symbol) > len(quote):
                    base = symbol[:-len(quote)]
                    if base.isalpha() and base.isupper():
                        return True
            
            return False
            
        except Exception as e:
            self.logger.warning(f"Error validating symbol format {symbol}: {e}")
            return False
    
    def _get_binance_price_direct(self, binance_symbol: str) -> float:
        """
        Get price directly from Binance with multiple fallback strategies.
        
        Args:
            binance_symbol: Symbol in Binance format (e.g., 'BTCUSDT')
            
        Returns:
            Current price as float, or 0.0 if failed
        """
        try:
            if not self.binance_client:
                self.logger.error("❌ Binance client not available")
                return 0.0
            
            # Strategy 1: Get current price ticker
            try:
                price_data = self.binance_client.get_price(binance_symbol)
                if price_data and 'price' in price_data:
                    price = float(price_data['price'])
                    if price > 0:
                        self.logger.debug(f"💰 Binance ticker price for {binance_symbol}: ${price:,.2f}")
                        return price
            except Exception as e:
                self.logger.warning(f"Binance ticker price failed for {binance_symbol}: {e}")
            
            # Strategy 2: Get 24h average price
            try:
                avg_data = self.binance_client.get_avg_price(binance_symbol)
                if avg_data and 'price' in avg_data:
                    price = float(avg_data['price'])
                    if price > 0:
                        self.logger.info(f"💰 Binance avg price for {binance_symbol}: ${price:,.2f}")
                        return price
            except Exception as e:
                self.logger.warning(f"Binance avg price failed for {binance_symbol}: {e}")
            
            # Strategy 3: Try with alternative symbol formats (if available)
            alternative_symbols = self._get_alternative_symbol_formats(binance_symbol)
            for alt_symbol in alternative_symbols:
                try:
                    price_data = self.binance_client.get_price(alt_symbol)
                    if price_data and 'price' in price_data:
                        price = float(price_data['price'])
                        if price > 0:
                            self.logger.info(f"💰 Binance price (alt format) for {alt_symbol}: ${price:,.2f}")
                            return price
                except Exception:
                    continue
            
            self.logger.error(f"❌ All price strategies failed for {binance_symbol}")
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Error getting Binance price for {binance_symbol}: {e}")
            return 0.0
    
    def _get_alternative_symbol_formats(self, binance_symbol: str) -> List[str]:
        """
        Get alternative symbol formats to try if main symbol fails.
        
        Args:
            binance_symbol: Original Binance symbol
            
        Returns:
            List of alternative symbol formats to try
        """
        alternatives = []
        
        try:
            # If symbol ends with USDT, try USD
            if binance_symbol.endswith('USDT'):
                base = binance_symbol[:-4]
                alternatives.append(f"{base}USD")
            
            # If symbol ends with USD, try USDT
            elif binance_symbol.endswith('USD') and not binance_symbol.endswith('USDT'):
                base = binance_symbol[:-3]
                alternatives.append(f"{base}USDT")
                alternatives.append(f"{base}BUSD")  # Also try BUSD
            
        except Exception as e:
            self.logger.debug(f"Error generating alternatives for {binance_symbol}: {e}")
        
        return alternatives


def main():
    """
    Example usage of the refactored trading strategy engine.
    
    Command-line options:
    --test-display              Test the enhanced display and exit (useful for debugging)
    --symbols SYMBOL1 SYMBOL2   Override symbols to trade (e.g., BTCUSDT ETHUSDT)
    --max-symbols N             Maximum number of symbols to trade (default: 8)
    
    Examples:
    python strategy_engine_refactored.py --test-display
    python strategy_engine_refactored.py --symbols BTCUSDT ETHUSDT ADAUSDT
    python strategy_engine_refactored.py --max-symbols 5
    python strategy_engine_refactored.py --test-display --symbols BTCUSDT ETHUSDT
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Crypto Trading Strategy Engine (Refactored)')
    parser.add_argument('--test-display', action='store_true', 
                       help='Test the enhanced display and exit')
    parser.add_argument('--symbols', nargs='+', 
                       help='Override symbols to trade (e.g., BTCUSDT ETHUSDT)')
    parser.add_argument('--max-symbols', type=int, default=15,
                       help='Maximum number of symbols to trade (default: 15)')
    
    args = parser.parse_args()
    
    print("🚀 Crypto Trading Strategy Engine (Refactored)")
    print("="*50)
    
    # Configuration - UPDATED FOR COMPREHENSIVE CRYPTO TRADING
    config = TradingConfig(
        max_position_size=0.15,  # 15% max per position (balanced for 15 symbols)
        max_portfolio_risk=0.95,  # 95% max portfolio risk (up from 35% for dedicated crypto trading)
        min_confidence=0.20,     # 20% minimum confidence (lowered for more trades)
        paper_trading=True,      # Start with paper trading
        max_trades_per_day=25,   # Max 25 trades per day (increased for more symbols)
        log_level="DEBUG",       # Enable detailed logging
        use_binary_classification=False  # CRITICAL: 3-class models only
    )
    
    # Determine symbols to use
    if args.symbols:
        symbols = args.symbols
        print(f"📊 Using command-line symbols: {symbols}")
    else:
        # Import symbols from centralized configuration - USE ALL CONFIGURED CRYPTOS
        try:
            from config.symbols_config import get_trading_symbols
            # Get more symbols for comprehensive crypto trading (increased from 8 to 15)
            symbols = get_trading_symbols(max_symbols=15, priority='high')
            print(f"📊 Using centralized symbol configuration: {symbols}")
        except ImportError:
            # Fallback to comprehensive crypto list if config module not available
            symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'DOGEUSDT', 'XRPUSDT', 
                      'AVAXUSDT', 'LINKUSDT', 'DOTUSDT', 'LTCUSDT', 'BCHUSDT',
                      'UNIUSDT', 'AAVEUSDT', 'SUSHIUSDT', 'YFIUSDT', 'SHIBUSDT']
            print(f"⚠️  Using fallback comprehensive crypto symbols: {symbols}")
    
    try:
        # Create trading engine
        engine = TradingStrategyEngine(config)
        
        # Validate symbols against available historical data
        validated_symbols = engine.validate_symbols_with_data(symbols)
        print(f"🎯 Validated symbols with historical data: {validated_symbols}")
        
        if args.test_display:
            # Test enhanced display mode
            print("\n🎯 Testing Enhanced Display...")
            print("Initializing components for display test...")
            
            # Initialize only the components needed for display testing
            engine.initialize_components(validated_symbols)
            
            print("Forcing enhanced display...")
            engine.force_enhanced_display(validated_symbols)
            
            print("\n✅ Enhanced display test completed!")
            print("You can now run without --test-display to start normal trading.")
            return
        
        print("Starting trading engine...")
        print("Press Ctrl+C to stop")
        
        # Start trading (5-minute intervals) with validated symbols
        engine.start_trading(validated_symbols, update_interval=300)
        
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
