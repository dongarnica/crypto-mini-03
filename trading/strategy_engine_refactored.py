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
        self.risk_manager = RiskManager(self.config, self.position_manager, self.trading_client)
        self.ml_engine = MLEngine(self.config)
        
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
            if ml_predictions and self.enhanced_display:
                self.enhanced_display.display_full_status(ml_predictions)
            
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
        Get current price using Binance as primary source.
        
        Args:
            symbol: Symbol in Alpaca format (e.g., 'BTC/USD') - kept for compatibility
            original_symbol: Symbol in Binance format (e.g., 'BTCUSDT')
            
        Returns:
            Current price as float, or 0.0 if failed
        """
        try:
            # Use Binance as primary price source
            if self.binance_client and original_symbol:
                try:
                    price_data = self.binance_client.get_price(original_symbol)
                    if price_data and 'price' in price_data:
                        price = float(price_data['price'])
                        self.logger.debug(f"💰 Binance price for {original_symbol}: ${price:,.2f}")
                        return price
                except Exception as e:
                    self.logger.warning(f"Binance price failed for {original_symbol}: {e}")
            
            # Fallback to average price from Binance
            if self.binance_client and original_symbol:
                try:
                    avg_data = self.binance_client.get_avg_price(original_symbol)
                    if avg_data and 'price' in avg_data:
                        price = float(avg_data['price'])
                        self.logger.info(f"💰 Binance avg price for {original_symbol}: ${price:,.2f}")
                        return price
                except Exception as e:
                    self.logger.warning(f"Binance avg price failed for {original_symbol}: {e}")
            
            self.logger.error(f"❌ No price data available for {original_symbol}")
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Error getting price for {original_symbol}: {e}")
            return 0.0
    
    def update_position_prices_from_binance(self) -> None:
        """Update all position prices using Binance price data."""
        try:
            positions = self.position_manager.get_all_positions()
            
            for symbol, position in positions.items():
                try:
                    # Get current price from Binance
                    binance_price = self.get_current_price_with_fallback(None, symbol)
                    
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

def main():
    """Example usage of the refactored trading strategy engine."""
    print("🚀 Crypto Trading Strategy Engine (Refactored)")
    print("="*50)
    
    # Configuration
    config = TradingConfig(
        max_position_size=0.05,  # 5% max per position
        min_confidence=0.35,     # 35% minimum confidence (optimized for 3-class)
        paper_trading=True,      # Start with paper trading
        max_trades_per_day=20,   # Max 20 trades per day
        log_level="DEBUG",       # Enable detailed logging
        use_binary_classification=False  # CRITICAL: 3-class models only
    )
    
    # Import symbols from centralized configuration
    try:
        from config.symbols_config import get_trading_symbols
        symbols = get_trading_symbols(max_symbols=8, priority='high')
        print(f"📊 Using centralized symbol configuration: {symbols}")
    except ImportError:
        # Fallback to hardcoded symbols if config module not available
        symbols = ['BTCUSDT', 'ETHUSDT', 'DOTUSDT', 'LINKUSDT', 'ADAUSDT']
        print(f"⚠️  Using fallback symbols: {symbols}")
    
    try:
        # Create and start trading engine
        engine = TradingStrategyEngine(config)
        
        # Validate symbols against available historical data
        validated_symbols = engine.validate_symbols_with_data(symbols)
        print(f"🎯 Validated symbols with historical data: {validated_symbols}")
        
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
