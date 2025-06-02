#!/usr/bin/env python3
"""
Portfolio Manager
=================

Handles portfolio tracking, performance metrics, and account information.

Author: Crypto Trading Strategy Engine
Date: June 2, 2025
"""

import logging
from datetime import datetime
from typing import Dict, List


class PortfolioManager:
    """Manages portfolio information and performance tracking."""
    
    def __init__(self, trading_client):
        """
        Initialize portfolio manager.
        
        Args:
            trading_client: Alpaca trading client
        """
        self.trading_client = trading_client
        self.logger = logging.getLogger('PortfolioManager')
        
        # Portfolio information
        self.portfolio_value = 0.0
        self.available_cash = 0.0
        self.portfolio_risk = 0.0
        
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
    
    def update_portfolio_info(self) -> None:
        """Update current portfolio information from broker."""
        try:
            portfolio = self.trading_client.get_portfolio()
            account = portfolio.get('account', {})
            
            self.portfolio_value = float(account.get('equity', 0.0))
            self.available_cash = float(account.get('cash', 0.0))
            
            self.logger.debug(f"Portfolio Value: ${self.portfolio_value:.2f}, "
                            f"Available Cash: ${self.available_cash:.2f}")
            
        except Exception as e:
            self.logger.error(f"Error updating portfolio info: {e}")
    
    def calculate_portfolio_risk(self, total_position_value: float) -> float:
        """
        Calculate current portfolio risk percentage.
        
        Args:
            total_position_value: Total value of all positions
            
        Returns:
            Portfolio risk as percentage (0.0 to 1.0)
        """
        if self.portfolio_value > 0:
            self.portfolio_risk = total_position_value / self.portfolio_value
        else:
            self.portfolio_risk = 0.0
        
        return self.portfolio_risk
    
    def update_performance_metrics(self, trade_record) -> None:
        """
        Update performance metrics with a new trade.
        
        Args:
            trade_record: TradeRecord object
        """
        try:
            self.performance_metrics['total_trades'] += 1
            self.performance_metrics['last_update'] = datetime.now()
            
            # Update PnL tracking
            if trade_record.pnl != 0:  # Only count closed positions
                self.performance_metrics['total_pnl'] += trade_record.pnl
                
                if trade_record.pnl > 0:
                    self.performance_metrics['winning_trades'] += 1
                else:
                    self.performance_metrics['losing_trades'] += 1
            
            self.logger.debug(f"Updated performance metrics: "
                            f"Total trades: {self.performance_metrics['total_trades']}, "
                            f"Total PnL: ${self.performance_metrics['total_pnl']:.2f}")
            
        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {e}")
    
    def get_performance_summary(self, position_manager) -> Dict:
        """
        Get comprehensive performance summary.
        
        Args:
            position_manager: Position manager instance
            
        Returns:
            Dict with performance metrics
        """
        try:
            total_trades = self.performance_metrics['total_trades']
            winning_trades = self.performance_metrics['winning_trades']
            losing_trades = self.performance_metrics['losing_trades']
            
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # Calculate portfolio return
            start_value = 10000.0  # Assume $10k starting portfolio
            current_value = self.portfolio_value
            total_return = (current_value - start_value) / start_value if start_value > 0 else 0
            
            # Get position information
            total_position_value = position_manager.get_total_position_value()
            portfolio_risk = self.calculate_portfolio_risk(total_position_value)
            
            return {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'total_pnl': self.performance_metrics['total_pnl'],
                'portfolio_value': self.portfolio_value,
                'total_return': total_return,
                'active_positions': position_manager.get_position_count(),
                'portfolio_risk': portfolio_risk,
                'available_cash': self.available_cash,
                'total_position_value': total_position_value
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating performance summary: {e}")
            return {}
    
    def print_status(self, position_manager) -> None:
        """
        Print current portfolio and trading status.
        
        Args:
            position_manager: Position manager instance
        """
        try:
            print("\n" + "="*80)
            print("🤖 CRYPTO TRADING STRATEGY ENGINE STATUS")
            print("="*80)
            
            # Performance summary
            perf = self.get_performance_summary(position_manager)
            print(f"💰 Portfolio Value: ${perf.get('portfolio_value', 0):,.2f}")
            print(f"💵 Available Cash: ${perf.get('available_cash', 0):,.2f}")
            print(f"📊 Total Return: {perf.get('total_return', 0):+.1%}")
            print(f"🎯 Portfolio Risk: {perf.get('portfolio_risk', 0):.1%}")
            print(f"📈 Total Trades: {perf.get('total_trades', 0)}")
            print(f"✅ Win Rate: {perf.get('win_rate', 0):.1%}")
            print(f"💲 Total P&L: ${perf.get('total_pnl', 0):+,.2f}")
            
            # Active positions
            positions = position_manager.get_all_positions()
            print(f"\n🏆 ACTIVE POSITIONS ({len(positions)})")
            print("-" * 60)
            
            if positions:
                for symbol, position in positions.items():
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
    
    def start_tracking(self) -> None:
        """Start performance tracking."""
        self.performance_metrics['start_time'] = datetime.now()
        self.logger.info("📊 Started performance tracking")
    
    def get_portfolio_value(self) -> float:
        """Get current portfolio value."""
        return self.portfolio_value
    
    def get_available_cash(self) -> float:
        """Get available cash."""
        return self.available_cash
    
    def get_portfolio_risk(self) -> float:
        """Get current portfolio risk percentage."""
        return self.portfolio_risk
