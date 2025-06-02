#!/usr/bin/env python3
"""
Trade Executor
==============

Handles trade execution logic including order placement, 
order management, and trade record keeping.

Author: Crypto Trading Strategy Engine
Date: June 2, 2025
"""

import logging
import time
from datetime import datetime
from typing import Dict, List, Optional

from trading.models import Position, PositionType, TradingConfig, TradingSignal, TradeRecord


class TradeExecutor:
    """Handles trade execution and order management."""
    
    def __init__(self, config: TradingConfig, trading_client, position_manager):
        """
        Initialize trade executor.
        
        Args:
            config: Trading configuration
            trading_client: Alpaca trading client
            position_manager: Position manager instance
        """
        self.config = config
        self.trading_client = trading_client
        self.position_manager = position_manager
        self.logger = logging.getLogger('TradeExecutor')
        
        # Trade tracking
        self.trade_history: List[TradeRecord] = []
        self.daily_trade_count = 0
        self.last_trade_time: Dict[str, datetime] = {}
    
    def can_trade(self, symbol: str, signal: TradingSignal) -> bool:
        """
        Check if we can execute a trade based on trading controls.
        
        Args:
            symbol: Trading symbol
            signal: Trading signal
            
        Returns:
            True if we can trade
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
        
        return True
    
    def execute_trade(self, symbol: str, signal: TradingSignal, 
                     confidence: float, current_price: float, 
                     position_size: float, ml_prediction: Dict) -> Optional[TradeRecord]:
        """
        Execute a trade based on the signal.
        
        Args:
            symbol: Crypto symbol
            signal: Trading signal
            confidence: ML confidence
            current_price: Current market price
            position_size: Calculated position size
            ml_prediction: Full ML prediction data
            
        Returns:
            TradeRecord if trade was executed
        """
        try:
            trade_record = None
            
            if signal == TradingSignal.BUY:
                trade_record = self._execute_buy_order(
                    symbol, confidence, current_price, position_size, ml_prediction
                )
            
            elif signal == TradingSignal.SELL:
                trade_record = self._execute_sell_order(
                    symbol, confidence, current_price, ml_prediction
                )
            
            elif signal in [TradingSignal.CLOSE_LONG, TradingSignal.CLOSE_SHORT]:
                trade_record = self._execute_close_order(
                    symbol, signal, current_price, ml_prediction
                )
            
            # Update tracking if trade was executed
            if trade_record:
                self._update_trade_tracking(trade_record)
            
            return trade_record
            
        except Exception as e:
            self.logger.error(f"Error executing trade: {e}")
            return None
    
    def _execute_buy_order(self, symbol: str, confidence: float, 
                          current_price: float, position_size: float,
                          ml_prediction: Dict) -> Optional[TradeRecord]:
        """Execute a buy order."""
        try:
            self.logger.info(f"🔵 Executing BUY order for {symbol}: "
                           f"${position_size:.2f} @ ${current_price:.2f}")
            
            if position_size <= 0:
                self.logger.warning(f"Invalid position size for {symbol}: ${position_size:.2f}")
                return None
            
            # Convert symbol format for Alpaca
            alpaca_symbol = self._convert_symbol_format(symbol)
            
            # Place market order with retry logic
            order = self._place_order_with_retry(
                symbol=alpaca_symbol,
                side='buy',
                notional=position_size,
                order_type='market'
            )
            
            if order:
                order_id = order.get('id')
                order_status = order.get('status', 'unknown')
                
                if order_status in ['new', 'accepted', 'pending_new']:
                    # Calculate approximate quantity
                    quantity = position_size / current_price
                    
                    # Calculate stop loss and take profit
                    stop_loss = current_price * (1 - self.config.stop_loss_pct)
                    take_profit = current_price * (1 + self.config.take_profit_pct)
                    
                    # Create position record
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
                    
                    # Add to position manager
                    self.position_manager.add_position(position)
                    
                    # Create trade record
                    trade_record = TradeRecord(
                        timestamp=datetime.now(),
                        symbol=symbol,
                        action='BUY',
                        quantity=quantity,
                        price=current_price,
                        confidence=confidence,
                        ml_signal=ml_prediction.get('signal'),
                        order_id=order_id,
                        reason=f"ML Signal: {ml_prediction.get('recommendation')}"
                    )
                    
                    self.logger.info(f"✅ BUY order placed: {symbol} - Order ID: {order_id}")
                    return trade_record
                
                else:
                    self.logger.error(f"Order failed for {symbol}: {order_status}")
                    return None
            
            else:
                self.logger.error(f"Failed to place order for {symbol}")
                return None
            
        except Exception as e:
            self.logger.error(f"Error executing buy order for {symbol}: {e}")
            return None
    
    def _execute_sell_order(self, symbol: str, confidence: float, 
                           current_price: float, ml_prediction: Dict) -> Optional[TradeRecord]:
        """Execute a sell order (close long position if exists)."""
        # For now, we'll just close long positions when we get sell signals
        if self.position_manager.has_position(symbol):
            position = self.position_manager.get_position(symbol)
            if position.position_type == PositionType.LONG:
                return self._execute_close_order(
                    symbol, TradingSignal.CLOSE_LONG, current_price, ml_prediction
                )
        
        self.logger.debug(f"SELL signal for {symbol} but no long position to close")
        return None
    
    def _execute_close_order(self, symbol: str, signal: TradingSignal, 
                            current_price: float, ml_prediction: Dict) -> Optional[TradeRecord]:
        """Execute a close order."""
        try:
            if not self.position_manager.has_position(symbol):
                self.logger.warning(f"No position to close for {symbol}")
                return None
            
            position = self.position_manager.get_position(symbol)
            
            self.logger.info(f"🔴 Executing CLOSE order for {symbol}: "
                           f"{position.quantity:.6f} @ ${current_price:.2f}")
            
            # Convert symbol format for Alpaca
            alpaca_symbol = self._convert_symbol_format(symbol)
            
            # Close the position
            order = self._place_order_with_retry(
                symbol=alpaca_symbol,
                side='sell',
                qty=position.quantity,
                order_type='market'
            )
            
            if order:
                order_id = order.get('id')
                order_status = order.get('status', 'unknown')
                
                if order_status in ['new', 'accepted', 'pending_new', 'filled', 'partially_filled']:
                    # Calculate PnL
                    pnl = position.unrealized_pnl or (
                        (current_price - position.entry_price) * position.quantity
                    )
                    
                    # Create trade record
                    trade_record = TradeRecord(
                        timestamp=datetime.now(),
                        symbol=symbol,
                        action='CLOSE',
                        quantity=position.quantity,
                        price=current_price,
                        confidence=ml_prediction.get('confidence', 0),
                        ml_signal=ml_prediction.get('signal'),
                        order_id=order_id,
                        pnl=pnl,
                        reason=f"Close signal: {ml_prediction.get('recommendation')}"
                    )
                    
                    # Remove position from tracking
                    self.position_manager.remove_position(symbol)
                    
                    self.logger.info(f"✅ CLOSE order placed: {symbol} - Order ID: {order_id}, "
                                   f"P&L: ${pnl:+.2f}")
                    return trade_record
                
                else:
                    self.logger.error(f"Close order failed for {symbol}: {order_status}")
                    return None
            
            else:
                self.logger.error(f"Failed to place close order for {symbol}")
                return None
            
        except Exception as e:
            self.logger.error(f"Error executing close order for {symbol}: {e}")
            return None
    
    def _place_order_with_retry(self, symbol: str, side: str, 
                               order_type: str = 'market', 
                               qty: Optional[float] = None,
                               notional: Optional[float] = None,
                               max_retries: int = 3) -> Optional[dict]:
        """Place order with retry logic."""
        for attempt in range(max_retries):
            try:
                if notional:
                    order = self.trading_client.place_crypto_order(
                        symbol=symbol,
                        side=side,
                        order_type=order_type,
                        time_in_force='gtc',
                        notional=notional
                    )
                else:
                    order = self.trading_client.place_crypto_order(
                        symbol=symbol,
                        side=side,
                        order_type=order_type,
                        time_in_force='gtc',
                        qty=qty
                    )
                
                if order:
                    return order
                    
            except Exception as api_error:
                self.logger.warning(f"API error (attempt {attempt + 1}/{max_retries}) "
                                  f"for {symbol}: {api_error}")
                if attempt == max_retries - 1:
                    raise  # Re-raise on final attempt
                time.sleep(1 * (attempt + 1))  # Exponential backoff
        
        return None
    
    def _update_trade_tracking(self, trade_record: TradeRecord) -> None:
        """Update trade tracking information."""
        self.trade_history.append(trade_record)
        self.daily_trade_count += 1
        self.last_trade_time[trade_record.symbol] = datetime.now()
        
        # Save trade record if configured
        if self.config.save_trades:
            self._save_trade_record(trade_record)
    
    def _save_trade_record(self, trade: TradeRecord) -> None:
        """Save trade record to file."""
        try:
            import json
            import os
            
            # Create trades directory
            trades_dir = '/workspaces/crypto-mini-03/trading/trades'
            os.makedirs(trades_dir, exist_ok=True)
            
            # Save to JSON file
            trade_file = f"{trades_dir}/trades_{datetime.now().strftime('%Y%m%d')}.json"
            
            trade_data = {
                'timestamp': trade.timestamp.isoformat(),
                'symbol': trade.symbol,
                'action': trade.action,
                'quantity': trade.quantity,
                'price': trade.price,
                'confidence': trade.confidence,
                'ml_signal': trade.ml_signal,
                'order_id': trade.order_id,
                'commission': trade.commission,
                'pnl': trade.pnl,
                'reason': trade.reason
            }
            
            # Append to file
            if os.path.exists(trade_file):
                with open(trade_file, 'r') as f:
                    trades = json.load(f)
            else:
                trades = []
            
            trades.append(trade_data)
            
            with open(trade_file, 'w') as f:
                json.dump(trades, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving trade record: {e}")
    
    def _convert_symbol_format(self, symbol: str) -> str:
        """Convert symbol format from Binance to Alpaca format."""
        try:
            if symbol.endswith('USDT'):
                base = symbol[:-4]
                return f"{base}/USD"
            return symbol
        except Exception:
            return symbol
    
    def get_trade_history(self) -> List[TradeRecord]:
        """Get trade history."""
        return self.trade_history.copy()
    
    def get_daily_trade_count(self) -> int:
        """Get current daily trade count."""
        return self.daily_trade_count
    
    def reset_daily_counters(self) -> None:
        """Reset daily trade counters."""
        self.daily_trade_count = 0
        self.logger.info("Reset daily trade counter")
