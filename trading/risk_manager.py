#!/usr/bin/env python3
"""
Risk Manager
============

Handles risk management logic including stop-losses, take-profits, 
position sizing limits, and portfolio risk controls.

Author: Crypto Trading Strategy Engine
Date: June 2, 2025
"""

import logging
import traceback
from datetime import datetime
from typing import Dict, Optional

import pandas as pd
import numpy as np

from trading.models import Position, PositionType, TradingConfig, TradingSignal


class RiskManager:
    """Manages risk controls and position monitoring."""
    
    def __init__(self, config: TradingConfig, position_manager, trading_client):
        """
        Initialize risk manager.
        
        Args:
            config: Trading configuration
            position_manager: Position manager instance
            trading_client: Trading client for price data and order execution
        """
        self.config = config
        self.position_manager = position_manager
        self.trading_client = trading_client
        self.logger = logging.getLogger('RiskManager')
    
    def check_risk_management(self, portfolio_value: float) -> None:
        """
        Check and enforce risk management rules including dynamic and trailing stop-loss.
        
        Args:
            portfolio_value: Current portfolio value
        """
        try:
            self.logger.debug("🛡️ Checking risk management rules...")
            
            # Sync positions with broker first
            self.position_manager.sync_positions_with_broker()
            
            positions = self.position_manager.get_all_positions()
            
            for symbol, position in list(positions.items()):
                # Get current price with error handling
                alpaca_symbol = self._convert_symbol_format(symbol)
                try:
                    current_price = self.trading_client.get_current_price(alpaca_symbol)
                except Exception as price_error:
                    self.logger.warning(f"Failed to get current price for {symbol}: {price_error}")
                    continue
                
                if current_price:
                    # Update position price
                    self.position_manager.update_position_price(symbol, current_price)
                    
                    # Update dynamic stop-loss if enabled
                    if self.config.dynamic_stop_loss:
                        self._update_dynamic_stop_loss(symbol, position, current_price)
                    
                    # Update trailing stop-loss if enabled
                    if self.config.trailing_stop_loss:
                        self._update_trailing_stop_loss(symbol, position, current_price)
                    
                    # Check stop loss
                    if position.stop_loss and current_price <= position.stop_loss:
                        self.logger.warning(f"⚠️ Stop loss triggered for {symbol} @ ${current_price:.2f} "
                                          f"(SL: ${position.stop_loss:.2f})")
                        self._trigger_stop_loss(symbol, current_price)
                    
                    # Check take profit
                    elif position.take_profit and current_price >= position.take_profit:
                        self.logger.info(f"🎯 Take profit triggered for {symbol} @ ${current_price:.2f} "
                                       f"(TP: ${position.take_profit:.2f})")
                        self._trigger_take_profit(symbol, current_price)
            
        except Exception as e:
            self.logger.error(f"Error in risk management check: {e}")
            traceback.print_exc()
    
    def should_allow_trade(self, symbol: str, signal: TradingSignal, 
                          position_size: float, portfolio_value: float) -> bool:
        """
        Check if a trade should be allowed based on risk management rules.
        
        Args:
            symbol: Trading symbol
            signal: Trading signal
            position_size: Proposed position size
            portfolio_value: Current portfolio value
            
        Returns:
            True if trade should be allowed
        """
        # Check portfolio risk limits
        current_position_value = self.position_manager.get_total_position_value()
        portfolio_risk = current_position_value / portfolio_value if portfolio_value > 0 else 0.0
        
        if signal == TradingSignal.BUY:
            # Check if adding this position would exceed portfolio risk
            new_portfolio_risk = (current_position_value + position_size) / portfolio_value
            if new_portfolio_risk > self.config.max_portfolio_risk:
                self.logger.warning(f"Trade would exceed portfolio risk limit: "
                                  f"{new_portfolio_risk:.1%} > {self.config.max_portfolio_risk:.1%}")
                return False
        
        # Check if already at max position size for this symbol
        if self.position_manager.has_position(symbol):
            position = self.position_manager.get_position(symbol)
            current_position_value = abs(position.quantity * (position.current_price or 0))
            max_position_value = portfolio_value * self.config.max_position_size
            
            if current_position_value >= max_position_value * 0.9:  # 90% of max
                self.logger.debug(f"Already at max position size for {symbol}")
                return False
        
        return True
    
    def calculate_position_size(self, symbol: str, confidence: float, 
                              current_price: float, portfolio_value: float, 
                              available_cash: float) -> float:
        """
        Calculate position size based on risk management rules.
        
        Args:
            symbol: Trading symbol
            confidence: ML confidence level
            current_price: Current market price
            portfolio_value: Current portfolio value
            available_cash: Available cash
            
        Returns:
            Position size in quote currency (USD)
        """
        try:
            # Base position size as percentage of portfolio
            base_size = portfolio_value * self.config.base_position_size
            
            # Adjust for confidence
            confidence_adjusted = base_size * (1 + (confidence - 0.5) * self.config.confidence_multiplier)
            
            # Apply volatility adjustment if enabled
            if self.config.volatility_adjustment:
                volatility_factor = self._calculate_volatility_adjustment(symbol)
                confidence_adjusted *= volatility_factor
                self.logger.debug(f"Applied volatility adjustment for {symbol}: {volatility_factor:.3f}")
            
            # Apply maximum position size limit
            max_position_value = portfolio_value * self.config.max_position_size
            position_size = min(confidence_adjusted, max_position_value)
            
            # Ensure we don't exceed available cash
            position_size = min(position_size, available_cash * 0.95)  # Leave 5% buffer
            
            # Ensure we don't exceed portfolio risk limits
            current_position_value = self.position_manager.get_total_position_value()
            portfolio_risk = current_position_value / portfolio_value if portfolio_value > 0 else 0.0
            
            if portfolio_risk + (position_size / portfolio_value) > self.config.max_portfolio_risk:
                remaining_risk = self.config.max_portfolio_risk - portfolio_risk
                position_size = min(position_size, remaining_risk * portfolio_value)
            
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
    
    def _update_dynamic_stop_loss(self, symbol: str, position: Position, current_price: float) -> None:
        """Update dynamic stop-loss based on current volatility."""
        try:
            # Calculate volatility-adjusted stop-loss
            volatility_factor = self._calculate_volatility_adjustment(symbol)
            adjusted_stop_loss_pct = self.config.stop_loss_pct * volatility_factor
            
            if position.position_type == PositionType.LONG:
                new_stop_loss = current_price * (1 - adjusted_stop_loss_pct)
                # Only move stop-loss up (tighten), never down (loosen)
                if not position.stop_loss or new_stop_loss > position.stop_loss:
                    old_stop_loss = position.stop_loss
                    position.stop_loss = new_stop_loss
                    if old_stop_loss:
                        self.logger.debug(f"Dynamic stop-loss updated for {symbol}: "
                                        f"${old_stop_loss:.2f} -> ${new_stop_loss:.2f} "
                                        f"(volatility factor: {volatility_factor:.3f})")
            else:  # SHORT position
                new_stop_loss = current_price * (1 + adjusted_stop_loss_pct)
                # Only move stop-loss down (tighten), never up (loosen)
                if not position.stop_loss or new_stop_loss < position.stop_loss:
                    old_stop_loss = position.stop_loss
                    position.stop_loss = new_stop_loss
                    if old_stop_loss:
                        self.logger.debug(f"Dynamic stop-loss updated for {symbol}: "
                                        f"${old_stop_loss:.2f} -> ${new_stop_loss:.2f} "
                                        f"(volatility factor: {volatility_factor:.3f})")
                        
        except Exception as e:
            self.logger.warning(f"Error updating dynamic stop-loss for {symbol}: {e}")
    
    def _update_trailing_stop_loss(self, symbol: str, position: Position, current_price: float) -> None:
        """Update trailing stop-loss based on favorable price movement."""
        try:
            if position.position_type == PositionType.LONG:
                # Track highest price achieved
                if position.highest_price is None or current_price > position.highest_price:
                    position.highest_price = current_price
                
                # Calculate trailing stop based on highest price achieved
                trailing_stop_pct = self.config.stop_loss_pct
                new_trailing_stop = position.highest_price * (1 - trailing_stop_pct)
                
                # Only move stop-loss up (tighten), never down
                if not position.stop_loss or new_trailing_stop > position.stop_loss:
                    old_stop_loss = position.stop_loss
                    position.stop_loss = new_trailing_stop
                    if old_stop_loss and abs(new_trailing_stop - old_stop_loss) > 0.01:
                        self.logger.debug(f"Trailing stop-loss updated for {symbol}: "
                                        f"${old_stop_loss:.2f} -> ${new_trailing_stop:.2f} "
                                        f"(highest: ${position.highest_price:.2f})")
                        
            else:  # SHORT position
                # Track lowest price achieved
                if position.lowest_price is None or current_price < position.lowest_price:
                    position.lowest_price = current_price
                
                # Calculate trailing stop based on lowest price achieved
                trailing_stop_pct = self.config.stop_loss_pct
                new_trailing_stop = position.lowest_price * (1 + trailing_stop_pct)
                
                # Only move stop-loss down (tighten), never up
                if not position.stop_loss or new_trailing_stop < position.stop_loss:
                    old_stop_loss = position.stop_loss
                    position.stop_loss = new_trailing_stop
                    if old_stop_loss and abs(new_trailing_stop - old_stop_loss) > 0.01:
                        self.logger.debug(f"Trailing stop-loss updated for {symbol}: "
                                        f"${old_stop_loss:.2f} -> ${new_trailing_stop:.2f} "
                                        f"(lowest: ${position.lowest_price:.2f})")
                        
        except Exception as e:
            self.logger.warning(f"Error updating trailing stop-loss for {symbol}: {e}")
    
    def _calculate_volatility_adjustment(self, symbol: str) -> float:
        """Calculate volatility adjustment factor for position sizing."""
        try:
            # This would typically use recent price data to calculate volatility
            # For now, return a default factor
            # TODO: Implement actual volatility calculation using price history
            
            # Placeholder volatility calculation
            # In practice, you'd get recent price data and calculate standard deviation
            typical_vol = 0.04  # 4% daily volatility baseline
            
            # For now, return 1.0 (no adjustment)
            # In a real implementation, you'd calculate:
            # vol_ratio = actual_volatility / typical_vol
            # return max(0.5, min(1.5, 1.0 / (1.0 + vol_ratio * 0.5)))
            
            return 1.0
            
        except Exception as e:
            self.logger.warning(f"Error calculating volatility adjustment for {symbol}: {e}")
            return 1.0
    
    def _trigger_stop_loss(self, symbol: str, current_price: float) -> None:
        """Trigger stop loss by closing the position."""
        try:
            # This would typically integrate with the trade executor
            # For now, just log the action
            self.logger.warning(f"🛑 STOP LOSS TRIGGERED: {symbol} @ ${current_price:.2f}")
            # TODO: Integrate with trade executor to actually close the position
            
        except Exception as e:
            self.logger.error(f"Error triggering stop loss for {symbol}: {e}")
    
    def _trigger_take_profit(self, symbol: str, current_price: float) -> None:
        """Trigger take profit by closing the position."""
        try:
            # This would typically integrate with the trade executor
            # For now, just log the action
            self.logger.info(f"🎯 TAKE PROFIT TRIGGERED: {symbol} @ ${current_price:.2f}")
            # TODO: Integrate with trade executor to actually close the position
            
        except Exception as e:
            self.logger.error(f"Error triggering take profit for {symbol}: {e}")
    
    def _convert_symbol_format(self, symbol: str) -> str:
        """Convert symbol format from Binance to Alpaca format."""
        try:
            if symbol.endswith('USDT'):
                base = symbol[:-4]
                return f"{base}/USD"
            return symbol
        except Exception:
            return symbol
