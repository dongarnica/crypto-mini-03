#!/usr/bin/env python3
"""
Enhanced Output Display for Trading Strategy Engine
===================================================

Provides comprehensive output including signal summaries, model statistics,
and countdown displays for the trading strategy engine.

Author: Crypto Trading Strategy Engine
Date: June 3, 2025
"""

import time
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class SignalSummary:
    """Summary of ML trading signals across all symbols."""
    total_symbols: int = 0
    buy_signals: int = 0
    sell_signals: int = 0
    hold_signals: int = 0
    high_confidence_signals: int = 0
    average_confidence: float = 0.0
    tradeable_signals: int = 0
    
    
@dataclass
class ModelStats:
    """Model training statistics."""
    symbol: str
    model_type: str
    accuracy: float
    total_return: float
    executed_trades: int
    features_count: int
    last_trained: str
    

class EnhancedOutputDisplay:
    """Enhanced output display for trading strategy engine."""
    
    def __init__(self, logger):
        """Initialize enhanced output display."""
        self.logger = logger
        self.ml_results_file = "/workspaces/crypto-mini-03/ml_results/enhanced_training_results_20250602_234110.json"
        self.signal_history = []
        self.model_stats_cache = {}
        self.last_update = None
        
    def load_model_statistics(self) -> Dict[str, ModelStats]:
        """Load model training statistics from results file."""
        try:
            if not os.path.exists(self.ml_results_file):
                self.logger.warning("No ML results file found")
                return {}
            
            with open(self.ml_results_file, 'r') as f:
                results = json.load(f)
            
            model_stats = {}
            
            for symbol, data in results.items():
                if symbol == 'timestamp':
                    continue
                    
                # Get 3-class model stats (preferred)
                three_class = data.get('3_class', {})
                if three_class:
                    # Get best backtest results
                    backtest_25 = three_class.get('backtest_25', {})
                    backtest_30 = three_class.get('backtest_30', {})
                    
                    # Use the better performing backtest
                    backtest = backtest_30 if backtest_30.get('total_return', -1) > backtest_25.get('total_return', -1) else backtest_25
                    
                    model_stats[symbol] = ModelStats(
                        symbol=symbol,
                        model_type="3-class",
                        accuracy=0.0,  # Not directly available in this format
                        total_return=backtest.get('total_return', 0.0),
                        executed_trades=backtest.get('executed_trades', 0),
                        features_count=three_class.get('features_count', 0),
                        last_trained=data.get('timestamp', 'Unknown')
                    )
            
            self.model_stats_cache = model_stats
            return model_stats
            
        except Exception as e:
            self.logger.error(f"Error loading model statistics: {e}")
            return {}
    
    def collect_signal_summary(self, predictions: List[Dict]) -> SignalSummary:
        """Collect and summarize trading signals."""
        summary = SignalSummary()
        
        if not predictions:
            return summary
        
        summary.total_symbols = len(predictions)
        confidences = []
        
        for pred in predictions:
            signal = pred.get('signal', 'HOLD')
            confidence = pred.get('confidence', 0.0)
            tradeable = pred.get('tradeable', False)
            high_confidence = pred.get('high_confidence', False)
            
            confidences.append(confidence)
            
            if signal == 'BUY':
                summary.buy_signals += 1
            elif signal == 'SELL':
                summary.sell_signals += 1
            else:
                summary.hold_signals += 1
            
            if high_confidence:
                summary.high_confidence_signals += 1
            
            if tradeable:
                summary.tradeable_signals += 1
        
        if confidences:
            summary.average_confidence = sum(confidences) / len(confidences)
        
        # Store in history for trends
        self.signal_history.append({
            'timestamp': datetime.now(),
            'summary': summary
        })
        
        # Keep only last 10 entries
        if len(self.signal_history) > 10:
            self.signal_history = self.signal_history[-10:]
        
        return summary
    
    def format_countdown(self, seconds_remaining: int) -> str:
        """Format countdown timer as MM:SS."""
        if seconds_remaining <= 0:
            return "00:00"
        
        minutes = seconds_remaining // 60
        seconds = seconds_remaining % 60
        return f"{minutes:02d}:{seconds:02d}"
    
    def print_enhanced_status(self, symbols: List[str], predictions: List[Dict], 
                            cycle_count: int, next_cycle_seconds: int = 300):
        """Print comprehensive trading status with enhanced information."""
        
        # Header
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print("\n" + "="*80)
        print(f"🚀 CRYPTO TRADING STRATEGY ENGINE - CYCLE #{cycle_count}")
        print(f"⏰ {current_time}")
        print("="*80)
        
        # Signal Summary
        signal_summary = self.collect_signal_summary(predictions)
        self.print_signal_summary(signal_summary)
        
        # Detailed Predictions
        self.print_detailed_predictions(predictions)
        
        # Model Statistics
        if not self.model_stats_cache:
            self.load_model_statistics()
        self.print_model_statistics()
        
        # Countdown Timer
        self.print_countdown_section(next_cycle_seconds)
        
        print("="*80)
        self.last_update = datetime.now()
    
    def print_signal_summary(self, summary: SignalSummary):
        """Print signal summary section."""
        print(f"\n📊 SIGNAL SUMMARY")
        print("-" * 40)
        print(f"🎯 Total Symbols Analyzed: {summary.total_symbols}")
        print(f"📈 BUY Signals:           {summary.buy_signals}")
        print(f"📉 SELL Signals:          {summary.sell_signals}")
        print(f"⏸️  HOLD Signals:          {summary.hold_signals}")
        print(f"⚡ High Confidence:       {summary.high_confidence_signals}")
        print(f"✅ Tradeable Signals:     {summary.tradeable_signals}")
        print(f"🎲 Average Confidence:    {summary.average_confidence:.1%}")
        
        # Signal distribution
        if summary.total_symbols > 0:
            buy_pct = (summary.buy_signals / summary.total_symbols) * 100
            sell_pct = (summary.sell_signals / summary.total_symbols) * 100
            hold_pct = (summary.hold_signals / summary.total_symbols) * 100
            print(f"📊 Distribution: BUY {buy_pct:.0f}% | SELL {sell_pct:.0f}% | HOLD {hold_pct:.0f}%")
    
    def print_detailed_predictions(self, predictions: List[Dict]):
        """Print detailed prediction information."""
        if not predictions:
            return
            
        print(f"\n🔮 DETAILED PREDICTIONS")
        print("-" * 70)
        print(f"{'Symbol':<10} {'Signal':<5} {'Conf':<6} {'Price':<12} {'Recommendation'}")
        print("-" * 70)
        
        for pred in predictions:
            symbol = pred.get('symbol', 'UNKNOWN')
            signal = pred.get('signal', 'HOLD')
            confidence = pred.get('confidence', 0.0)
            price = pred.get('current_price', 0.0)
            recommendation = pred.get('recommendation', 'No recommendation')
            
            # Color coding based on signal
            signal_emoji = "📈" if signal == 'BUY' else "📉" if signal == 'SELL' else "⏸️"
            
            # Truncate recommendation if too long
            if len(recommendation) > 35:
                recommendation = recommendation[:32] + "..."
            
            print(f"{symbol:<10} {signal_emoji}{signal:<4} {confidence:<5.1%} ${price:<11,.2f} {recommendation}")
    
    def print_model_statistics(self):
        """Print model training statistics."""
        if not self.model_stats_cache:
            print(f"\n🤖 MODEL STATISTICS")
            print("-" * 40)
            print("No model statistics available")
            return
        
        print(f"\n🤖 MODEL STATISTICS")
        print("-" * 80)
        print(f"{'Symbol':<10} {'Type':<8} {'Return':<8} {'Trades':<7} {'Features':<9} {'Trained'}")
        print("-" * 80)
        
        for symbol, stats in self.model_stats_cache.items():
            return_str = f"{stats.total_return:.1%}" if stats.total_return != 0 else "0.0%"
            trained_date = stats.last_trained[:10] if len(stats.last_trained) > 10 else stats.last_trained
            
            print(f"{symbol:<10} {stats.model_type:<8} {return_str:<8} {stats.executed_trades:<7} "
                  f"{stats.features_count:<9} {trained_date}")
        
        # Summary statistics
        total_symbols = len(self.model_stats_cache)
        profitable_models = sum(1 for stats in self.model_stats_cache.values() if stats.total_return > 0)
        avg_return = sum(stats.total_return for stats in self.model_stats_cache.values()) / total_symbols if total_symbols > 0 else 0
        total_trades = sum(stats.executed_trades for stats in self.model_stats_cache.values())
        
        print("-" * 80)
        print(f"📈 Profitable Models: {profitable_models}/{total_symbols} ({profitable_models/total_symbols*100:.0f}%)")
        print(f"📊 Average Return: {avg_return:.1%}")
        print(f"🔄 Total Backtest Trades: {total_trades}")
    
    def print_countdown_section(self, next_cycle_seconds: int):
        """Print countdown section."""
        print(f"\n⏱️  NEXT TRADING CYCLE")
        print("-" * 40)
        
        # Calculate next cycle time
        next_cycle_time = datetime.now() + timedelta(seconds=next_cycle_seconds)
        countdown = self.format_countdown(next_cycle_seconds)
        
        print(f"⏰ Next Update: {next_cycle_time.strftime('%H:%M:%S')}")
        print(f"⏳ Countdown: {countdown}")
        print(f"🔄 Interval: {next_cycle_seconds//60} minutes")
        
        # Add some trading tips or status
        tips = [
            "💡 Tip: 3-class models provide Buy/Hold/Sell signals for better profitability",
            "💡 Tip: High confidence signals (>55%) are more reliable for trading",
            "💡 Tip: Monitor market volatility during low confidence periods",
            "💡 Tip: Diversification across symbols helps reduce overall risk",
            "💡 Tip: Stop-loss and take-profit levels are automatically managed"
        ]
        
        import random
        tip = random.choice(tips)
        print(f"\n{tip}")
    
    def print_compact_status(self, signal_summary: SignalSummary, countdown_seconds: int):
        """Print compact status update between full cycles."""
        current_time = datetime.now().strftime("%H:%M:%S")
        countdown = self.format_countdown(countdown_seconds)
        
        print(f"\n⚡ {current_time} | BUY:{signal_summary.buy_signals} SELL:{signal_summary.sell_signals} "
              f"HOLD:{signal_summary.hold_signals} | Conf:{signal_summary.average_confidence:.1%} | "
              f"Next: {countdown}")
    
    def start_countdown_display(self, total_seconds: int, update_interval: int = 30):
        """Display live countdown with updates."""
        start_time = time.time()
        last_update = 0
        
        while True:
            elapsed = time.time() - start_time
            remaining = max(0, total_seconds - int(elapsed))
            
            if remaining <= 0:
                break
            
            # Update every interval
            if int(elapsed) >= last_update + update_interval:
                countdown = self.format_countdown(remaining)
                current_time = datetime.now().strftime("%H:%M:%S")
                print(f"\r⏳ {current_time} | Next cycle in: {countdown} | Press Ctrl+C to stop", end="", flush=True)
                last_update = int(elapsed)
            
            time.sleep(1)
    
    def display_countdown(self, total_seconds: int):
        """Display countdown timer between trading cycles."""
        try:
            start_time = time.time()
            
            while True:
                elapsed = time.time() - start_time
                remaining = max(0, total_seconds - int(elapsed))
                
                if remaining <= 0:
                    break
                
                countdown = self.format_countdown(remaining)
                current_time = datetime.now().strftime("%H:%M:%S")
                
                # Clear line and display countdown
                print(f"\r⏳ {current_time} | Next trading cycle in: {countdown} | Press Ctrl+C to stop", end="", flush=True)
                
                time.sleep(1)
            
            # Clear the countdown line
            print("\r" + " " * 80 + "\r", end="", flush=True)
            
        except KeyboardInterrupt:
            # User interrupted countdown
            print("\r" + " " * 80 + "\r", end="", flush=True)
            print("⚠️  Countdown interrupted, proceeding to next cycle...")
            return
    
    def display_full_status(self, ml_predictions: List[Dict]):
        """Display comprehensive trading status with all enhancements."""
        try:
            # Print header
            print("\n" + "="*80)
            print(f"📊 CRYPTO TRADING STRATEGY ENGINE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("="*80)
            
            # Generate signal summary
            signal_summary = self.collect_signal_summary(ml_predictions)
            
            # Print signal summary
            self.print_signal_summary(signal_summary)
            
            # Print detailed predictions
            self.print_detailed_predictions(ml_predictions)
            
            # Load and print model statistics
            self.load_model_statistics()
            self.print_model_statistics()
            
            print("="*80 + "\n")
            
        except Exception as e:
            self.logger.error(f"Error displaying full status: {e}")
