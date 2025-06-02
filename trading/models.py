#!/usr/bin/env python3
"""
Trading Models and Configuration
===============================

Core data models, enums, and configuration classes for the trading strategy engine.

Author: Crypto Trading Strategy Engine
Date: June 2, 2025
"""

from datetime import datetime
from typing import Optional
from dataclasses import dataclass
from enum import Enum


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
    dynamic_stop_loss: bool = True  # Enable dynamic stop-loss based on volatility
    trailing_stop_loss: bool = True  # Enable trailing stop-loss
    
    # Signal Thresholds
    min_confidence: float = 0.30  # Minimum ML confidence
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
    log_level: str = "DEBUG"
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
    highest_price: Optional[float] = None
    lowest_price: Optional[float] = None
    
    def __post_init__(self):
        """Initialize tracking prices after creation."""
        if self.highest_price is None:
            self.highest_price = self.entry_price
        if self.lowest_price is None:
            self.lowest_price = self.entry_price
    
    def update_current_price(self, price: float):
        """Update current price and calculate PnL."""
        self.current_price = price
        
        # Update price tracking for trailing stops
        if self.highest_price is None or price > self.highest_price:
            self.highest_price = price
        if self.lowest_price is None or price < self.lowest_price:
            self.lowest_price = price
        
        # Calculate PnL
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
