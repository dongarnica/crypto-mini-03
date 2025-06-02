#!/usr/bin/env python3
"""
Comprehensive Trading Strategy Engine
===================================

Integrates machine learning predictions with automated trading execution
through the Alpaca crypto trading platform. Includes risk management,
position sizing, and real-time monitoring.

Features:
- ML-driven trading signal generation
- Automated trade execution via Alpaca
- Advanced risk management and position sizing
- Real-time monitoring and logging
- Portfolio rebalancing
- Strategy performance tracking

Author: Crypto Trading Strategy Engine
Date: June 1, 2025
"""

import os
import sys
import json
import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
from enum import Enum
import signal
import traceback

# Add the parent directory to Python path for imports
sys.path.append('/workspaces/crypto-mini-03')

# Import our existing modules
from ml.ml_pipeline_improved import ImprovedCryptoLSTMPipeline
from alpaca.alpaca_client import AlpacaCryptoClient


class TradingSignal(Enum):
    """Trading signal types."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    CLOSE_LONG = "CLOSE_LONG"
    CLOSE_SHORT = "CLOSE_SHORT"
    WAIT = "WAIT"


class PositionType(Enum):
    """Position types."""
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"


@dataclass
class TradingConfig:
    """Configuration for trading strategy."""
    # Risk Management
    max_position_size: float = 0.05  # 5% max per position
    max_portfolio_risk: float = 0.20  # 20% max total risk
    stop_loss_pct: float = 0.02  # 2% stop loss
    take_profit_pct: float = 0.04  # 4% take profit
    
    # Signal Thresholds
    min_confidence: float = 0.35  # Minimum ML confidence
    high_confidence: float = 0.60  # High confidence threshold
    
    # Position Sizing
    base_position_size: float = 0.02  # 2% base position
    confidence_multiplier: float = 2.0  # Scale position by confidence
    volatility_adjustment: bool = True  # Adjust for volatility
    
    # Trading Controls
    max_trades_per_day: int = 10
    min_time_between_trades: int = 300  # 5 minutes
    paper_trading: bool = True  # Start with paper trading
    
    # ML Model Settings
    prediction_horizon: int = 4  # 4 hours
    lookback_period: int = 24  # 24 hours
    use_binary_classification: bool = False
    
    # Monitoring
    log_level: str = "INFO"
    save_trades: bool = True
    real_time_monitoring: bool = True


@dataclass
class Position:
    """Represents a trading position."""
    symbol: str
    position_type: PositionType
    entry_price: float
    quantity: float
    entry_time: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    current_price: Optional[float] = None
    unrealized_pnl: float = 0.0
    confidence: float = 0.0
    ml_signal: Optional[str] = None
    
    def update_current_price(self, price: float):
        """Update current price and calculate PnL."""
        self.current_price = price
        if self.position_type == PositionType.LONG:
            self.unrealized_pnl = (price - self.entry_price) * self.quantity
        elif self.position_type == PositionType.SHORT:
            self.unrealized_pnl = (self.entry_price - price) * self.quantity
    
    def get_pnl_percentage(self) -> float:
        """Get PnL as percentage."""
        if self.current_price and self.entry_price:
            if self.position_type == PositionType.LONG:
                return (self.current_price - self.entry_price) / self.entry_price
            elif self.position_type == PositionType.SHORT:
                return (self.entry_price - self.current_price) / self.entry_price
        return 0.0


@dataclass
class TradeRecord:
    """Record of executed trade."""
    timestamp: datetime
    symbol: str
    action: str
    quantity: float
    price: float
    confidence: float
    ml_signal: str
    order_id: Optional[str] = None
    commission: float = 0.0
    pnl: float = 0.0
    reason: str = ""


class TradingStrategyEngine:
    """
    Main trading strategy engine that integrates ML predictions with
    automated trading execution and risk management.
    """
    
    def __init__(self, config: TradingConfig):
        """
        Initialize the trading strategy engine.
        
        Args:
            config: Trading configuration
        """
        self.config = config
        self.running = False
        self.positions: Dict[str, Position] = {}
        self.trade_history: List[TradeRecord] = []
        self.daily_trade_count = 0
        self.last_trade_time = {}
        
        # Setup logging
        self.setup_logging()
        
        # Initialize ML pipelines and trading client
        self.ml_pipelines = {}
        self.trading_client = None
        self.data_cache = {}
        
        # Performance tracking
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'start_time': None,
            'last_update': None
        }
        
        # Risk management
        self.portfolio_value = 0.0
        self.available_cash = 0.0
        self.portfolio_risk = 0.0
        
        self.logger.info("Trading Strategy Engine initialized")
        self.logger.info(f"Configuration: {asdict(self.config)}")
    
    def setup_logging(self):
        """Setup logging configuration."""
        log_level = getattr(logging, self.config.log_level.upper())
        
        # Create logs directory
        os.makedirs('/workspaces/crypto-mini-03/trading/logs', exist_ok=True)
        
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
                f'/workspaces/crypto-mini-03/trading/logs/trading_{datetime.now().strftime("%Y%m%d")}.log'
            )
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
    
    def initialize_components(self, symbols: List[str]):
        """
        Initialize ML pipelines and trading client.
        
        Args:
            symbols: List of crypto symbols to trade
        """
        self.logger.info("Initializing trading components...")
        
        # Initialize Alpaca trading client
        try:
            self.trading_client = AlpacaCryptoClient(paper=self.config.paper_trading)
            
            # Check crypto eligibility
            if not self.trading_client.check_crypto_eligibility():
                raise ValueError("Account not eligible for crypto trading")
            
            self.logger.info("✅ Alpaca trading client initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize trading client: {e}")
            raise
        
        # Initialize ML pipelines for each symbol
        for symbol in symbols:
            try:
                # Create ML pipeline
                pipeline = ImprovedCryptoLSTMPipeline(
                    symbol=symbol,
                    lookback_period=self.config.lookback_period,
                    prediction_horizon=self.config.prediction_horizon,
                    confidence_threshold=self.config.min_confidence,
                    use_binary_classification=self.config.use_binary_classification
                )
                
                # Try to load existing model
                model_path = f'/workspaces/crypto-mini-03/ml_results/models/{symbol}_lstm_model.keras'
                if os.path.exists(model_path):
                    # Load the trained model (we'll implement this)
                    self.logger.info(f"Loading existing model for {symbol}")
                    # pipeline.load_model(model_path)  # Will implement this method
                else:
                    self.logger.warning(f"No trained model found for {symbol}")
                
                self.ml_pipelines[symbol] = pipeline
                self.logger.info(f"✅ ML pipeline initialized for {symbol}")
                
            except Exception as e:
                self.logger.error(f"Failed to initialize ML pipeline for {symbol}: {e}")
                # Continue with other symbols
        
        # Update portfolio information
        self.update_portfolio_info()
        
        self.logger.info(f"Initialized {len(self.ml_pipelines)} ML pipelines")
    
    def update_portfolio_info(self):
        """Update current portfolio information."""
        try:
            portfolio = self.trading_client.get_portfolio()
            account = portfolio.get('account', {})
            
            self.portfolio_value = account.get('equity', 0.0)
            self.available_cash = account.get('cash', 0.0)
            
            # Calculate current portfolio risk
            total_position_value = 0.0
            for position in self.positions.values():
                if position.current_price:
                    position_value = position.quantity * position.current_price
                    total_position_value += abs(position_value)
            
            self.portfolio_risk = total_position_value / self.portfolio_value if self.portfolio_value > 0 else 0.0
            
            self.logger.debug(f"Portfolio Value: ${self.portfolio_value:.2f}, "
                            f"Available Cash: ${self.available_cash:.2f}, "
                            f"Portfolio Risk: {self.portfolio_risk:.1%}")
            
        except Exception as e:
            self.logger.error(f"Error updating portfolio info: {e}")
    
    def get_ml_prediction(self, symbol: str) -> Optional[Dict]:
        """
        Get ML prediction for a symbol.
        
        Args:
            symbol: Crypto symbol
            
        Returns:
            Dict with prediction information or None
        """
        if symbol not in self.ml_pipelines:
            self.logger.warning(f"No ML pipeline for {symbol}")
            return None
        
        try:
            pipeline = self.ml_pipelines[symbol]
            
            # Get recent data (in a real implementation, we'd fetch this from data source)
            # For now, we'll simulate this
            prediction = pipeline.predict_with_lower_threshold()
            
            self.logger.debug(f"ML Prediction for {symbol}: {prediction.get('signal')} "
                            f"(confidence: {prediction.get('confidence', 0):.1%})")
            
            return prediction
            
        except Exception as e:
            self.logger.error(f"Error getting ML prediction for {symbol}: {e}")
            return None
    
    def calculate_position_size(self, symbol: str, confidence: float, current_price: float) -> float:
        """
        Calculate position size based on risk management rules.
        
        Args:
            symbol: Crypto symbol
            confidence: ML confidence level
            current_price: Current market price
            
        Returns:
            Position size in quote currency (USD)
        """
        try:
            # Base position size as percentage of portfolio
            base_size = self.portfolio_value * self.config.base_position_size
            
            # Adjust for confidence
            confidence_adjusted = base_size * (1 + (confidence - 0.5) * self.config.confidence_multiplier)
            
            # Apply maximum position size limit
            max_position_value = self.portfolio_value * self.config.max_position_size
            position_size = min(confidence_adjusted, max_position_value)
            
            # Ensure we don't exceed available cash
            position_size = min(position_size, self.available_cash * 0.95)  # Leave 5% buffer
            
            # Ensure we don't exceed portfolio risk limits
            if self.portfolio_risk + (position_size / self.portfolio_value) > self.config.max_portfolio_risk:
                remaining_risk = self.config.max_portfolio_risk - self.portfolio_risk
                position_size = min(position_size, remaining_risk * self.portfolio_value)
            
            # Minimum position size check
            min_size = 10.0  # $10 minimum
            if position_size < min_size:
                self.logger.warning(f"Calculated position size too small: ${position_size:.2f}")
                return 0.0
            
            self.logger.info(f"Position size for {symbol}: ${position_size:.2f} "
                           f"(confidence: {confidence:.1%}, price: ${current_price:.2f})")
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 0.0
    
    def should_trade(self, symbol: str, signal: TradingSignal) -> bool:
        """
        Check if we should execute a trade based on various conditions.
        
        Args:
            symbol: Crypto symbol
            signal: Trading signal
            
        Returns:
            True if we should trade
        """
        # Check daily trade limit
        if self.daily_trade_count >= self.config.max_trades_per_day:
            self.logger.warning("Daily trade limit reached")
            return False
        
        # Check minimum time between trades
        if symbol in self.last_trade_time:
            time_since_last = (datetime.now() - self.last_trade_time[symbol]).total_seconds()
            if time_since_last < self.config.min_time_between_trades:
                self.logger.debug(f"Too soon since last trade for {symbol}")
                return False
        
        # Check portfolio risk limits
        if self.portfolio_risk >= self.config.max_portfolio_risk and signal in [TradingSignal.BUY]:
            self.logger.warning("Portfolio risk limit reached")
            return False
        
        # Don't trade if already have max position in this symbol
        if symbol in self.positions:
            current_position_value = abs(self.positions[symbol].quantity * 
                                       (self.positions[symbol].current_price or 0))
            max_position_value = self.portfolio_value * self.config.max_position_size
            
            if current_position_value >= max_position_value * 0.9:  # 90% of max
                self.logger.debug(f"Already at max position size for {symbol}")
                return False
        
        return True
    
    def execute_trade(self, symbol: str, signal: TradingSignal, 
                     confidence: float, current_price: float, 
                     ml_prediction: Dict) -> Optional[TradeRecord]:
        """
        Execute a trade based on the signal.
        
        Args:
            symbol: Crypto symbol
            signal: Trading signal
            confidence: ML confidence
            current_price: Current market price
            ml_prediction: Full ML prediction data
            
        Returns:
            TradeRecord if trade was executed
        """
        try:
            trade_record = None
            
            if signal == TradingSignal.BUY:
                trade_record = self._execute_buy_order(symbol, confidence, current_price, ml_prediction)
            
            elif signal == TradingSignal.SELL:
                trade_record = self._execute_sell_order(symbol, confidence, current_price, ml_prediction)
            
            elif signal in [TradingSignal.CLOSE_LONG, TradingSignal.CLOSE_SHORT]:
                trade_record = self._execute_close_order(symbol, signal, current_price, ml_prediction)
            
            # Update tracking
            if trade_record:
                self.trade_history.append(trade_record)
                self.daily_trade_count += 1
                self.last_trade_time[symbol] = datetime.now()
                
                # Update performance metrics
                self.performance_metrics['total_trades'] += 1
                self.performance_metrics['last_update'] = datetime.now()
                
                if self.config.save_trades:
                    self._save_trade_record(trade_record)
            
            return trade_record
            
        except Exception as e:
            self.logger.error(f"Error executing trade: {e}")
            return None
    
    def _execute_buy_order(self, symbol: str, confidence: float, 
                          current_price: float, ml_prediction: Dict) -> Optional[TradeRecord]:
        """Execute a buy order."""
        try:
            # Calculate position size
            position_size = self.calculate_position_size(symbol, confidence, current_price)
            
            if position_size <= 0:
                return None
            
            # Convert symbol format for Alpaca (e.g., BTCUSDT -> BTC/USD)
            alpaca_symbol = self._convert_symbol_format(symbol)
            
            # Place market order
            order = self.trading_client.place_market_order(
                symbol=alpaca_symbol,
                notional=position_size,
                side='buy'
            )
            
            if order:
                # Create position record
                quantity = position_size / current_price  # Approximate
                
                # Calculate stop loss and take profit
                stop_loss = current_price * (1 - self.config.stop_loss_pct)
                take_profit = current_price * (1 + self.config.take_profit_pct)
                
                position = Position(
                    symbol=symbol,
                    position_type=PositionType.LONG,
                    entry_price=current_price,
                    quantity=quantity,
                    entry_time=datetime.now(),
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    current_price=current_price,
                    confidence=confidence,
                    ml_signal=ml_prediction.get('signal')
                )
                
                self.positions[symbol] = position
                
                # Create trade record
                trade_record = TradeRecord(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    action='BUY',
                    quantity=quantity,
                    price=current_price,
                    confidence=confidence,
                    ml_signal=ml_prediction.get('signal'),
                    order_id=order.get('id'),
                    reason=f"ML Signal: {ml_prediction.get('recommendation')}"
                )
                
                self.logger.info(f"✅ BUY ORDER EXECUTED: {symbol} ${position_size:.2f} @ ${current_price:.2f} "
                               f"(confidence: {confidence:.1%})")
                
                return trade_record
            
        except Exception as e:
            self.logger.error(f"Error executing buy order for {symbol}: {e}")
        
        return None
    
    def _execute_sell_order(self, symbol: str, confidence: float, 
                           current_price: float, ml_prediction: Dict) -> Optional[TradeRecord]:
        """Execute a sell order (short selling not implemented in this version)."""
        # For now, we'll just close long positions when we get sell signals
        if symbol in self.positions and self.positions[symbol].position_type == PositionType.LONG:
            return self._execute_close_order(symbol, TradingSignal.CLOSE_LONG, current_price, ml_prediction)
        
        self.logger.debug(f"SELL signal for {symbol} but no long position to close")
        return None
    
    def _execute_close_order(self, symbol: str, signal: TradingSignal, 
                            current_price: float, ml_prediction: Dict) -> Optional[TradeRecord]:
        """Execute a close order."""
        if symbol not in self.positions:
            return None
        
        try:
            position = self.positions[symbol]
            alpaca_symbol = self._convert_symbol_format(symbol)
            
            # Close the position
            order = self.trading_client.close_position(alpaca_symbol, percentage=1.0)
            
            if order:
                # Calculate PnL
                pnl = position.unrealized_pnl
                
                # Create trade record
                trade_record = TradeRecord(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    action='CLOSE',
                    quantity=position.quantity,
                    price=current_price,
                    confidence=ml_prediction.get('confidence', 0),
                    ml_signal=ml_prediction.get('signal'),
                    order_id=order.get('id'),
                    pnl=pnl,
                    reason=f"Close signal: {ml_prediction.get('recommendation')}"
                )
                
                # Update performance metrics
                if pnl > 0:
                    self.performance_metrics['winning_trades'] += 1
                else:
                    self.performance_metrics['losing_trades'] += 1
                
                self.performance_metrics['total_pnl'] += pnl
                
                # Remove position
                del self.positions[symbol]
                
                self.logger.info(f"✅ POSITION CLOSED: {symbol} P&L: ${pnl:.2f}")
                
                return trade_record
            
        except Exception as e:
            self.logger.error(f"Error closing position for {symbol}: {e}")
        
        return None
    
    def _convert_symbol_format(self, symbol: str) -> str:
        """Convert symbol format from Binance to Alpaca format."""
        # Convert BTCUSDT -> BTC/USD
        if symbol.endswith('USDT'):
            base = symbol[:-4]
            return f"{base}/USD"
        elif symbol.endswith('USD'):
            base = symbol[:-3]
            return f"{base}/USD"
        else:
            # Assume it's already in correct format
            return symbol
    
    def check_risk_management(self):
        """Check and enforce risk management rules."""
        try:
            current_time = datetime.now()
            
            for symbol, position in list(self.positions.items()):
                # Get current price
                alpaca_symbol = self._convert_symbol_format(symbol)
                current_price = self.trading_client.get_current_price(alpaca_symbol)
                
                if current_price:
                    position.update_current_price(current_price)
                    
                    # Check stop loss
                    if position.stop_loss and current_price <= position.stop_loss:
                        self.logger.warning(f"Stop loss triggered for {symbol} @ ${current_price:.2f}")
                        ml_prediction = {'signal': 'STOP_LOSS', 'confidence': 1.0, 'recommendation': 'Stop loss triggered'}
                        self._execute_close_order(symbol, TradingSignal.CLOSE_LONG, current_price, ml_prediction)
                    
                    # Check take profit
                    elif position.take_profit and current_price >= position.take_profit:
                        self.logger.info(f"Take profit triggered for {symbol} @ ${current_price:.2f}")
                        ml_prediction = {'signal': 'TAKE_PROFIT', 'confidence': 1.0, 'recommendation': 'Take profit triggered'}
                        self._execute_close_order(symbol, TradingSignal.CLOSE_LONG, current_price, ml_prediction)
            
            # Update portfolio info
            self.update_portfolio_info()
            
        except Exception as e:
            self.logger.error(f"Error in risk management check: {e}")
    
    def process_trading_signals(self, symbols: List[str]):
        """
        Main trading logic loop - process signals for all symbols.
        
        Args:
            symbols: List of symbols to process
        """
        try:
            self.logger.info(f"Processing trading signals for {len(symbols)} symbols...")
            
            for symbol in symbols:
                try:
                    # Get ML prediction
                    ml_prediction = self.get_ml_prediction(symbol)
                    
                    if not ml_prediction:
                        continue
                    
                    signal_str = ml_prediction.get('signal', 'HOLD')
                    confidence = ml_prediction.get('confidence', 0.0)
                    current_price = ml_prediction.get('current_price', 0.0)
                    
                    # Convert signal string to enum
                    signal = None
                    if signal_str in ['Buy', 'UP', 'Up']:
                        signal = TradingSignal.BUY
                    elif signal_str in ['Sell', 'DOWN', 'Down']:
                        signal = TradingSignal.SELL
                    else:
                        signal = TradingSignal.HOLD
                    
                    # Check if we should trade
                    if signal != TradingSignal.HOLD and confidence >= self.config.min_confidence:
                        if self.should_trade(symbol, signal):
                            trade_record = self.execute_trade(symbol, signal, confidence, current_price, ml_prediction)
                            
                            if trade_record:
                                self.logger.info(f"Trade executed: {trade_record.action} {symbol}")
                    
                except Exception as e:
                    self.logger.error(f"Error processing {symbol}: {e}")
                    continue
            
            # Check risk management for all positions
            self.check_risk_management()
            
        except Exception as e:
            self.logger.error(f"Error in trading signal processing: {e}")
    
    def _save_trade_record(self, trade: TradeRecord):
        """Save trade record to file."""
        try:
            trades_file = f'/workspaces/crypto-mini-03/trading/logs/trades_{datetime.now().strftime("%Y%m%d")}.json'
            
            # Load existing trades
            trades = []
            if os.path.exists(trades_file):
                with open(trades_file, 'r') as f:
                    trades = json.load(f)
            
            # Add new trade
            trade_dict = asdict(trade)
            trade_dict['timestamp'] = trade.timestamp.isoformat()
            trades.append(trade_dict)
            
            # Save back
            with open(trades_file, 'w') as f:
                json.dump(trades, f, indent=2)
            
        except Exception as e:
            self.logger.error(f"Error saving trade record: {e}")
    
    def get_performance_summary(self) -> Dict:
        """Get current performance summary."""
        try:
            total_trades = self.performance_metrics['total_trades']
            winning_trades = self.performance_metrics['winning_trades']
            losing_trades = self.performance_metrics['losing_trades']
            
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # Calculate portfolio return
            start_value = 10000.0  # Assume $10k starting portfolio
            current_value = self.portfolio_value
            total_return = (current_value - start_value) / start_value if start_value > 0 else 0
            
            return {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'total_pnl': self.performance_metrics['total_pnl'],
                'portfolio_value': self.portfolio_value,
                'total_return': total_return,
                'active_positions': len(self.positions),
                'portfolio_risk': self.portfolio_risk,
                'available_cash': self.available_cash
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating performance summary: {e}")
            return {}
    
    def print_status(self):
        """Print current trading status."""
        try:
            print("\n" + "="*80)
            print("🤖 CRYPTO TRADING STRATEGY ENGINE STATUS")
            print("="*80)
            
            # Performance summary
            perf = self.get_performance_summary()
            print(f"💰 Portfolio Value: ${perf.get('portfolio_value', 0):,.2f}")
            print(f"💵 Available Cash: ${perf.get('available_cash', 0):,.2f}")
            print(f"📊 Total Return: {perf.get('total_return', 0):+.1%}")
            print(f"🎯 Portfolio Risk: {perf.get('portfolio_risk', 0):.1%}")
            print(f"📈 Total Trades: {perf.get('total_trades', 0)}")
            print(f"✅ Win Rate: {perf.get('win_rate', 0):.1%}")
            print(f"💲 Total P&L: ${perf.get('total_pnl', 0):+,.2f}")
            
            # Active positions
            print(f"\n🏆 ACTIVE POSITIONS ({len(self.positions)})")
            print("-" * 60)
            
            if self.positions:
                for symbol, position in self.positions.items():
                    pnl_pct = position.get_pnl_percentage() * 100
                    pnl_color = "🟢" if position.unrealized_pnl >= 0 else "🔴"
                    
                    print(f"{symbol:>12}: {position.quantity:>10.6f} @ ${position.entry_price:>8.2f}")
                    print(f"{'':>12}  Current: ${position.current_price or 0:>8.2f}")
                    print(f"{'':>12}  P&L: {pnl_color} ${position.unrealized_pnl:>8.2f} ({pnl_pct:+.1f}%)")
                    print("-" * 40)
            else:
                print("No active positions")
            
            print("\n" + "="*80)
            
        except Exception as e:
            self.logger.error(f"Error printing status: {e}")
    
    def start_trading(self, symbols: List[str], update_interval: int = 300):
        """
        Start the trading engine.
        
        Args:
            symbols: List of crypto symbols to trade
            update_interval: Update interval in seconds (default: 5 minutes)
        """
        self.logger.info("🚀 Starting Crypto Trading Strategy Engine...")
        self.running = True
        self.performance_metrics['start_time'] = datetime.now()
        
        # Reset daily counters
        self.daily_trade_count = 0
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        try:
            # Initialize components
            self.initialize_components(symbols)
            
            # Print initial status
            self.print_status()
            
            while self.running:
                try:
                    loop_start = time.time()
                    
                    # Process trading signals
                    self.process_trading_signals(symbols)
                    
                    # Print status every 10 cycles (about 50 minutes with 5-min intervals)
                    if hasattr(self, '_cycle_count'):
                        self._cycle_count += 1
                    else:
                        self._cycle_count = 1
                    
                    if self._cycle_count % 10 == 0:
                        self.print_status()
                    
                    # Reset daily counters at midnight
                    current_hour = datetime.now().hour
                    if hasattr(self, '_last_hour'):
                        if self._last_hour == 23 and current_hour == 0:
                            self.daily_trade_count = 0
                            self.logger.info("Reset daily trade counter")
                    self._last_hour = current_hour
                    
                    # Calculate sleep time
                    loop_duration = time.time() - loop_start
                    sleep_time = max(0, update_interval - loop_duration)
                    
                    if sleep_time > 0:
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
    
    def stop_trading(self):
        """Stop the trading engine gracefully."""
        self.logger.info("🛑 Stopping trading engine...")
        self.running = False
        
        # Close all positions if requested
        if self.positions:
            self.logger.info(f"Closing {len(self.positions)} open positions...")
            for symbol in list(self.positions.keys()):
                try:
                    alpaca_symbol = self._convert_symbol_format(symbol)
                    current_price = self.trading_client.get_current_price(alpaca_symbol)
                    if current_price:
                        ml_prediction = {'signal': 'SHUTDOWN', 'confidence': 1.0, 'recommendation': 'System shutdown'}
                        self._execute_close_order(symbol, TradingSignal.CLOSE_LONG, current_price, ml_prediction)
                except Exception as e:
                    self.logger.error(f"Error closing position {symbol}: {e}")
        
        # Final status
        self.print_status()
        
        self.logger.info("✅ Trading engine stopped")
    
    def _signal_handler(self, signum, frame):
        """Handle system signals for graceful shutdown."""
        self.logger.info(f"Received signal {signum}, initiating shutdown...")
        self.running = False


def main():
    """Example usage of the trading strategy engine."""
    print("🚀 Crypto Trading Strategy Engine")
    print("="*50)
    
    # Configuration
    config = TradingConfig(
        max_position_size=0.05,  # 5% max per position
        min_confidence=0.35,     # 35% minimum confidence
        paper_trading=True,      # Start with paper trading
        max_trades_per_day=20,   # Max 20 trades per day
        log_level="INFO"
    )
    
    # Symbols to trade
    symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'LINKUSDT']
    
    try:
        # Create and start trading engine
        engine = TradingStrategyEngine(config)
        
        print("Starting trading engine...")
        print("Press Ctrl+C to stop")
        
        # Start trading (5-minute intervals)
        engine.start_trading(symbols, update_interval=300)
        
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
