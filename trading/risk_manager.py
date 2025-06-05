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
from config.symbol_manager import symbol_manager


class RiskManager:
    """Manages risk controls and position monitoring."""
    
    def __init__(self, config: TradingConfig, position_manager, trading_client, trade_executor=None):
        """
        Initialize risk manager.
        
        Args:
            config: Trading configuration
            position_manager: Position manager instance
            trading_client: Trading client for price data and order execution
            trade_executor: Trade executor instance (optional, can be set later)
        """
        self.config = config
        self.position_manager = position_manager
        self.trading_client = trading_client
        self.trade_executor = trade_executor  # Allow setting later to avoid circular imports
        self.logger = logging.getLogger('RiskManager')
    
    def set_trade_executor(self, trade_executor) -> None:
        """Set the trade executor (to avoid circular imports)."""
        self.trade_executor = trade_executor
        self.logger.info("🔗 Trade executor linked to risk manager")
    
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
                # Skip price updates - prices should be updated from main trading loop via Binance
                # The strategy engine will call update_position_price with Binance prices
                current_price = position.current_price
                if not current_price or current_price <= 0:
                    self.logger.debug(f"No current price available for {symbol}, skipping risk checks")
                    continue
                
                # Update position price is handled by main trading loop
                
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
        # CRITICAL: Reject negative or zero position sizes immediately
        if position_size <= 0:
            self.logger.warning(f"Trade rejected: invalid position size ${position_size:.2f}")
            return False
        
        # Check portfolio risk limits
        current_position_value = self.position_manager.get_total_position_value()
        portfolio_risk = current_position_value / portfolio_value if portfolio_value > 0 else 0.0
        
        # Enhanced debug logging
        self.logger.debug(f"Risk check for {symbol} {signal.value}:")
        self.logger.debug(f"  Current portfolio risk: {portfolio_risk:.1%}")
        self.logger.debug(f"  Max portfolio risk: {self.config.max_portfolio_risk:.1%}")
        self.logger.debug(f"  Proposed position size: ${position_size:.2f}")
        
        if signal == TradingSignal.BUY:
            # Check if adding this position would exceed portfolio risk
            new_portfolio_risk = (current_position_value + position_size) / portfolio_value
            if new_portfolio_risk > self.config.max_portfolio_risk:
                self.logger.warning(f"Trade would exceed portfolio risk limit: "
                                  f"{new_portfolio_risk:.1%} > {self.config.max_portfolio_risk:.1%}")
                return False
        elif signal in [TradingSignal.SELL, TradingSignal.CLOSE_LONG, TradingSignal.CLOSE_SHORT]:
            # Always allow sell trades as they reduce risk
            self.logger.debug(f"Allowing {signal.value} trade to reduce portfolio risk")
            return True
        
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
            
            # Debug logging for risk calculation
            current_position_value = self.position_manager.get_total_position_value()
            portfolio_risk = current_position_value / portfolio_value if portfolio_value > 0 else 0.0
            
            self.logger.debug(f"Position sizing debug for {symbol}:")
            self.logger.debug(f"  Portfolio value: ${portfolio_value:,.2f}")
            self.logger.debug(f"  Available cash: ${available_cash:,.2f}")
            self.logger.debug(f"  Current position value: ${current_position_value:,.2f}")
            self.logger.debug(f"  Current portfolio risk: {portfolio_risk:.1%}")
            self.logger.debug(f"  Max portfolio risk: {self.config.max_portfolio_risk:.1%}")
            self.logger.debug(f"  Base position size: ${base_size:.2f}")
            self.logger.debug(f"  Confidence adjusted: ${confidence_adjusted:.2f}")
            self.logger.debug(f"  Position size before risk check: ${position_size:.2f}")
            
            # Ensure we don't exceed portfolio risk limits
            current_position_value = self.position_manager.get_total_position_value()
            portfolio_risk = current_position_value / portfolio_value if portfolio_value > 0 else 0.0
            
            # CRITICAL FIX: Handle case where portfolio risk already exceeds maximum
            if portfolio_risk >= self.config.max_portfolio_risk:
                self.logger.warning(f"🚨 Portfolio risk {portfolio_risk:.1%} already exceeds maximum {self.config.max_portfolio_risk:.1%}")
                self.logger.warning(f"🛑 Blocking new trades until risk is reduced")
                return 0.0
            
            # Check if adding this position would exceed portfolio risk
            proposed_risk = portfolio_risk + (position_size / portfolio_value)
            if proposed_risk > self.config.max_portfolio_risk:
                remaining_risk = self.config.max_portfolio_risk - portfolio_risk
                # Ensure remaining_risk is positive before calculating position size
                if remaining_risk > 0:
                    max_allowed_size = remaining_risk * portfolio_value
                    position_size = min(position_size, max_allowed_size)
                    self.logger.info(f"⚠️ Position size reduced to stay within risk limits: ${position_size:.2f}")
                else:
                    self.logger.warning(f"🚨 No remaining risk capacity: portfolio_risk={portfolio_risk:.1%}, max={self.config.max_portfolio_risk:.1%}")
                    return 0.0
            
            # CRITICAL SAFETY CHECK: Ensure position size is never negative
            if position_size < 0:
                self.logger.error(f"CRITICAL: Position size calculation resulted in negative value: ${position_size:.2f}")
                return 0.0
            
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
            self.logger.warning(f"🛑 STOP LOSS TRIGGERED: {symbol} @ ${current_price:.2f}")
            
            # CRITICAL FIX: Actually execute the stop loss trade
            if self.trade_executor and self.position_manager.has_position(symbol):
                position = self.position_manager.get_position(symbol)
                
                # Create a mock ML prediction for the close order
                mock_prediction = {
                    'signal': 'CLOSE',
                    'confidence': 1.0,  # High confidence for risk management action
                    'recommendation': f'Stop loss triggered @ ${current_price:.2f}',
                    'current_price': current_price
                }
                
                # Execute close order
                close_signal = TradingSignal.CLOSE_LONG if position.position_type == PositionType.LONG else TradingSignal.CLOSE_SHORT
                trade_record = self.trade_executor.execute_trade(
                    symbol=symbol,
                    signal=close_signal,
                    confidence=1.0,
                    current_price=current_price,
                    position_size=0,  # Not needed for close
                    ml_prediction=mock_prediction
                )
                
                if trade_record:
                    self.logger.info(f"✅ Stop loss executed for {symbol}")
                else:
                    self.logger.error(f"❌ Failed to execute stop loss for {symbol}")
            else:
                self.logger.warning(f"⚠️ Cannot execute stop loss - trade executor not available or no position for {symbol}")
            
        except Exception as e:
            self.logger.error(f"Error triggering stop loss for {symbol}: {e}")
    
    def _trigger_take_profit(self, symbol: str, current_price: float) -> None:
        """Trigger take profit by closing the position."""
        try:
            self.logger.info(f"🎯 TAKE PROFIT TRIGGERED: {symbol} @ ${current_price:.2f}")
            
            # CRITICAL FIX: Actually execute the take profit trade
            if self.trade_executor and self.position_manager.has_position(symbol):
                position = self.position_manager.get_position(symbol)
                
                # Create a mock ML prediction for the close order
                mock_prediction = {
                    'signal': 'CLOSE',
                    'confidence': 1.0,  # High confidence for risk management action
                    'recommendation': f'Take profit triggered @ ${current_price:.2f}',
                    'current_price': current_price
                }
                
                # Execute close order
                close_signal = TradingSignal.CLOSE_LONG if position.position_type == PositionType.LONG else TradingSignal.CLOSE_SHORT
                trade_record = self.trade_executor.execute_trade(
                    symbol=symbol,
                    signal=close_signal,
                    confidence=1.0,
                    current_price=current_price,
                    position_size=0,  # Not needed for close
                    ml_prediction=mock_prediction
                )
                
                if trade_record:
                    self.logger.info(f"✅ Take profit executed for {symbol}")
                else:
                    self.logger.error(f"❌ Failed to execute take profit for {symbol}")
            else:
                self.logger.warning(f"⚠️ Cannot execute take profit - trade executor not available or no position for {symbol}")
            
        except Exception as e:
            self.logger.error(f"Error triggering take profit for {symbol}: {e}")
    
    def _convert_symbol_format(self, symbol: str) -> str:
        """Convert symbol format from Binance to Alpaca format using SymbolManager."""
        return symbol_manager.binance_to_alpaca_format(symbol)
