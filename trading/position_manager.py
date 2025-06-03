#!/usr/bin/env python3
"""
Position Manager
===============

Handles position tracking, synchronization with broker, and position-related operations.

Author: Crypto Trading Strategy Engine
Date: June 2, 2025
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional

from trading.models import Position, PositionType, TradingConfig
from config.symbol_manager import symbol_manager


class PositionManager:
    """Manages trading positions and synchronization with broker."""
    
    def __init__(self, config: TradingConfig, trading_client):
        """
        Initialize position manager.
        
        Args:
            config: Trading configuration
            trading_client: Alpaca trading client
        """
        self.config = config
        self.trading_client = trading_client
        self.positions: Dict[str, Position] = {}
        self.logger = logging.getLogger('PositionManager')
    
    def add_position(self, position: Position) -> None:
        """Add a new position to tracking."""
        self.positions[position.symbol] = position
        self.logger.info(f"Added position: {position.symbol} ({position.position_type.value}) "
                        f"{position.quantity:.6f} @ ${position.entry_price:.2f}")
    
    def remove_position(self, symbol: str) -> Optional[Position]:
        """Remove a position from tracking."""
        if symbol in self.positions:
            position = self.positions.pop(symbol)
            self.logger.info(f"Removed position: {symbol}")
            return position
        return None
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a symbol."""
        return self.positions.get(symbol)
    
    def has_position(self, symbol: str) -> bool:
        """Check if we have a position in this symbol."""
        return symbol in self.positions
    
    def get_all_positions(self) -> Dict[str, Position]:
        """Get all current positions."""
        return self.positions.copy()
    
    def get_position_count(self) -> int:
        """Get number of active positions."""
        return len(self.positions)
    
    def update_position_price(self, symbol: str, current_price: float) -> bool:
        """
        Update current price for a position.
        
        Args:
            symbol: Symbol to update
            current_price: New current price
            
        Returns:
            True if position was updated
        """
        if symbol in self.positions:
            self.positions[symbol].update_current_price(current_price)
            return True
        return False
    
    def get_total_position_value(self) -> float:
        """Calculate total value of all positions."""
        total_value = 0.0
        for position in self.positions.values():
            if position.current_price:
                position_value = position.quantity * position.current_price
                total_value += abs(position_value)
        return total_value
    
    def get_total_unrealized_pnl(self) -> float:
        """Calculate total unrealized PnL across all positions."""
        total_pnl = 0.0
        for position in self.positions.values():
            total_pnl += position.unrealized_pnl
        return total_pnl
    
    def load_existing_positions(self) -> None:
        """
        Load existing positions from Alpaca and sync with position manager.
        """
        try:
            self.logger.info("🔍 Loading existing positions from broker...")
            
            # Get all positions from Alpaca
            alpaca_positions = self.trading_client.get_all_positions()
            
            if not alpaca_positions:
                self.logger.info("No existing positions found")
                return
            
            loaded_count = 0
            for alpaca_pos in alpaca_positions:
                try:
                    position = self._convert_alpaca_position(alpaca_pos)
                    if position:
                        self.add_position(position)
                        loaded_count += 1
                        
                except Exception as pos_error:
                    self.logger.error(f"Error loading position {alpaca_pos}: {pos_error}")
                    continue
            
            self.logger.info(f"✅ Successfully loaded {loaded_count} existing positions")
            
        except Exception as e:
            self.logger.error(f"Error loading existing positions: {e}")
    
    def sync_positions_with_broker(self) -> None:
        """
        Sync tracked positions with current broker positions.
        This helps detect positions closed outside the system.
        """
        try:
            self.logger.debug("🔄 Syncing positions with broker...")
            
            # Get current broker positions
            alpaca_positions = self.trading_client.get_all_positions()
            broker_symbols = set()
            
            if alpaca_positions:
                for pos in alpaca_positions:
                    alpaca_symbol = pos.get('symbol', '')
                    symbol = self._convert_alpaca_symbol_to_binance(alpaca_symbol)
                    broker_symbols.add(symbol)
                    
                    # Update existing position with current data
                    if symbol in self.positions:
                        current_price = float(pos.get('current_price', 0))
                        unrealized_pl = float(pos.get('unrealized_pl', 0))
                        
                        self.positions[symbol].current_price = current_price
                        self.positions[symbol].unrealized_pnl = unrealized_pl
            
            # Remove positions that no longer exist in broker
            our_symbols = set(self.positions.keys())
            closed_symbols = our_symbols - broker_symbols
            
            for symbol in closed_symbols:
                self.logger.info(f"🔄 Position {symbol} was closed outside system")
                self.remove_position(symbol)
            
        except Exception as e:
            self.logger.debug(f"Error syncing positions: {e}")
    
    def _convert_alpaca_position(self, alpaca_pos: dict) -> Optional[Position]:
        """Convert Alpaca position data to our Position object."""
        try:
            # Extract position details
            alpaca_symbol = alpaca_pos.get('symbol', '')
            qty = float(alpaca_pos.get('qty', 0))
            avg_entry_price = float(alpaca_pos.get('avg_entry_price', 0))
            current_price = float(alpaca_pos.get('current_price', 0))
            unrealized_pl = float(alpaca_pos.get('unrealized_pl', 0))
            side = alpaca_pos.get('side', 'long')
            
            # Convert symbol format
            symbol = self._convert_alpaca_symbol_to_binance(alpaca_symbol)
            
            # Determine position type
            position_type = PositionType.LONG if side == 'long' else PositionType.SHORT
            
            # Create position object
            position = Position(
                symbol=symbol,
                position_type=position_type,
                entry_price=avg_entry_price,
                quantity=abs(qty),
                entry_time=datetime.now(),  # We don't have exact entry time from Alpaca
                current_price=current_price,
                unrealized_pnl=unrealized_pl,
                confidence=0.5,  # Default confidence for existing positions
                ml_signal="EXISTING"
            )
            
            # Calculate stop loss and take profit based on current settings
            if position_type == PositionType.LONG:
                position.stop_loss = current_price * (1 - self.config.stop_loss_pct)
                position.take_profit = current_price * (1 + self.config.take_profit_pct)
            else:
                position.stop_loss = current_price * (1 + self.config.stop_loss_pct)
                position.take_profit = current_price * (1 - self.config.take_profit_pct)
            
            self.logger.info(f"✅ Converted position: {symbol} ({side}) "
                           f"{qty:+.6f} @ ${avg_entry_price:.2f} "
                           f"(Current: ${current_price:.2f}, P&L: ${unrealized_pl:+.2f})")
            
            return position
            
        except Exception as e:
            self.logger.error(f"Error converting Alpaca position: {e}")
            return None
    
    def _convert_alpaca_symbol_to_binance(self, alpaca_symbol: str) -> str:
        """Convert Alpaca symbol format to Binance format using SymbolManager."""
        return symbol_manager.alpaca_to_binance_format(alpaca_symbol)
