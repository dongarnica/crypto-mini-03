"""
Trading Strategy Engine Package
===============================

A modular cryptocurrency trading strategy engine with separated concerns:

Components:
- models: Data models, enums, and configuration classes
- position_manager: Position tracking and broker synchronization
- risk_manager: Risk controls and stop-loss management
- trade_executor: Order placement and execution
- ml_engine: ML pipeline management and predictions
- portfolio_manager: Portfolio tracking and performance metrics
- strategy_engine_refactored: Main orchestrating engine

Author: Crypto Trading Strategy Engine
Date: June 2, 2025
"""

from .models import (
    TradingSignal,
    PositionType,
    TradingConfig,
    Position,
    TradeRecord
)

from .position_manager import PositionManager
from .risk_manager import RiskManager
from .trade_executor import TradeExecutor
from .ml_engine import MLEngine
from .portfolio_manager import PortfolioManager
from .strategy_engine_refactored import TradingStrategyEngine

__version__ = "2.0.0"
__author__ = "Crypto Trading Strategy Engine"

__all__ = [
    # Data models
    "TradingSignal",
    "PositionType", 
    "TradingConfig",
    "Position",
    "TradeRecord",
    
    # Core components
    "PositionManager",
    "RiskManager",
    "TradeExecutor",
    "MLEngine",
    "PortfolioManager",
    "TradingStrategyEngine"
]
