#!/usr/bin/env python3
"""
Process Manager with Countdown Timers
====================================

Manages multiple processes (trading engine and data export service) with
visual countdown timers showing time until next execution/cycle.

Features:
- Dual countdown timers for trading engine and data exports
- Process monitoring and automatic restart
- Graceful shutdown handling
- Status reporting and logging
- Visual progress indicators

Author: Crypto Trading System
Date: June 2025
"""

import os
import sys
import time
import signal
import subprocess
import threading
import logging
from datetime import datetime, timedelta
from pathlib import Path
import json

# Add project paths
sys.path.append(os.path.dirname(__file__))

from scheduled_data_manager import ScheduledDataManager


class ProcessManager:
    """Manages trading engine and data export processes with countdown timers."""
    
    def __init__(self, output_dir: str = "historical_exports"):
        """Initialize the process manager."""
        self.output_dir = Path(output_dir)
        self.running = False
        self.processes = {}
        
        # Configuration
        self.trading_cycle_minutes = 5  # Trading engine cycle time
        self.export_cycle_hours = 2     # Data export cycle time
        
        # Timing tracking
        self.last_export_time = None
        self.trading_start_time = datetime.now()
        
        # Setup logging
        self.setup_logging()
        
        # Initialize data manager
        self.data_manager = ScheduledDataManager(output_dir=str(self.output_dir))
        
        # Setup signal handlers
        signal.signal(signal.SIGTERM, self.signal_handler)
        signal.signal(signal.SIGINT, self.signal_handler)
        
        self.logger.info("ProcessManager initialized")
    
    def setup_logging(self):
        """Setup logging for the process manager."""
        log_dir = self.output_dir / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / 'process_manager.log'
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # File handler with rotation
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            log_file, maxBytes=50*1024*1024, backupCount=5
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        
        # Setup logger
        self.logger = logging.getLogger('ProcessManager')
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        self.logger.propagate = False
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.stop()
    
    def format_countdown(self, seconds: int) -> str:
        """Format seconds into a readable countdown string."""
        if seconds <= 0:
            return "⏰ NOW"
        
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        
        if hours > 0:
            return f"⏳ {hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            return f"⏳ {minutes:02d}:{secs:02d}"
    
    def get_next_export_time(self) -> datetime:
        """Calculate the next scheduled export time."""
        if self.last_export_time is None:
            # First export in 30 seconds for demo purposes
            return datetime.now() + timedelta(seconds=30)
        else:
            return self.last_export_time + timedelta(hours=self.export_cycle_hours)
    
    def get_trading_uptime(self) -> timedelta:
        """Get trading engine uptime."""
        return datetime.now() - self.trading_start_time
    
    def start_trading_engine(self):
        """Start the trading engine process."""
        try:
            self.logger.info("Starting trading engine process...")
            
            # Check if trading engine file exists
            trading_engine_path = Path("/app/trading/strategy_engine_refactored.py")
            if not trading_engine_path.exists():
                # Fallback to other possible locations
                possible_paths = [
                    "trading/strategy_engine_refactored.py",
                    "binance/binance_main.py",
                    "manual_btc_trade.py"
                ]
                
                for path in possible_paths:
                    if Path(path).exists():
                        trading_engine_path = Path(path)
                        break
                else:
                    self.logger.warning("No trading engine found, using demo mode")
                    trading_engine_path = None
            
            if trading_engine_path:
                # Start actual trading engine
                process = subprocess.Popen(
                    [sys.executable, str(trading_engine_path)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True
                )
                self.processes['trading'] = process
                self.logger.info(f"Trading engine started (PID: {process.pid})")
            else:
                # Demo mode - just log trading activity
                self.logger.info("Trading engine in demo mode (no actual trades)")
                
        except Exception as e:
            self.logger.error(f"Failed to start trading engine: {e}")
    
    def run_data_export(self):
        """Run the data export process."""
        try:
            self.logger.info("Running scheduled data export...")
            self.data_manager.run_scheduled_job()
            self.last_export_time = datetime.now()
            self.logger.info("Data export completed successfully")
        except Exception as e:
            self.logger.error(f"Data export failed: {e}")
    
    def display_status(self):
        """Display current status with countdown timers."""
        # Clear screen for better visual effect
        os.system('clear' if os.name == 'posix' else 'cls')
        
        print("=" * 80)
        print("🚀 CRYPTO TRADING SYSTEM - PROCESS MANAGER")
        print("=" * 80)
        print()
        
        # Current time
        current_time = datetime.now()
        print(f"🕐 Current Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Trading Engine Status
        print("📈 TRADING ENGINE STATUS")
        print("-" * 40)
        trading_uptime = self.get_trading_uptime()
        uptime_str = str(trading_uptime).split('.')[0]  # Remove microseconds
        
        if 'trading' in self.processes:
            process = self.processes['trading']
            if process.poll() is None:
                print(f"   Status: ✅ RUNNING (PID: {process.pid})")
            else:
                print(f"   Status: ❌ STOPPED (Exit Code: {process.returncode})")
        else:
            print(f"   Status: 🔄 DEMO MODE")
        
        print(f"   Uptime: {uptime_str}")
        
        # Next trading cycle (every 5 minutes for demo)
        next_cycle_seconds = 300 - (int(trading_uptime.total_seconds()) % 300)
        print(f"   Next Cycle: {self.format_countdown(next_cycle_seconds)}")
        print()
        
        # Data Export Status
        print("📊 DATA EXPORT STATUS")
        print("-" * 40)
        next_export_time = self.get_next_export_time()
        time_until_export = (next_export_time - current_time).total_seconds()
        
        print(f"   Status: ✅ SCHEDULED")
        if self.last_export_time:
            print(f"   Last Export: {self.last_export_time.strftime('%H:%M:%S')}")
        else:
            print(f"   Last Export: Never (first run)")
        
        print(f"   Next Export: {self.format_countdown(int(time_until_export))}")
        print(f"   Scheduled At: {next_export_time.strftime('%H:%M:%S')}")
        print()
        
        # System Stats
        try:
            status = self.data_manager.get_status_report()
            current_data = status.get('current_data', {})
            
            print("💾 DATA STATISTICS")
            print("-" * 40)
            print(f"   Total Files: {current_data.get('total_files', 0)}")
            print(f"   Symbols: {current_data.get('symbols_with_data', 0)}")
            print(f"   Data Size: {current_data.get('total_size_mb', 0):.2f} MB")
            print()
        except Exception:
            pass
        
        # Progress bars for visual appeal
        trading_progress = (int(trading_uptime.total_seconds()) % 300) / 300 * 50
        export_progress = max(0, (7200 - time_until_export) / 7200) * 50 if time_until_export > 0 else 50
        
        print("📊 CYCLE PROGRESS")
        print("-" * 40)
        print(f"   Trading:  [{'█' * int(trading_progress)}{'░' * (50 - int(trading_progress))}] {trading_progress/50*100:.1f}%")
        print(f"   Export:   [{'█' * int(export_progress)}{'░' * (50 - int(export_progress))}] {export_progress/50*100:.1f}%")
        print()
        
        print("=" * 80)
        print("Press Ctrl+C to stop")
        print("=" * 80)
    
    def monitor_processes(self):
        """Monitor and restart processes if needed."""
        while self.running:
            try:
                # Check if trading engine needs restart
                if 'trading' in self.processes:
                    process = self.processes['trading']
                    if process.poll() is not None:  # Process has terminated
                        self.logger.warning(f"Trading engine stopped (exit code: {process.returncode})")
                        self.logger.info("Restarting trading engine...")
                        time.sleep(5)  # Wait before restart
                        self.start_trading_engine()
                
                # Check if it's time for data export
                next_export_time = self.get_next_export_time()
                if datetime.now() >= next_export_time:
                    threading.Thread(target=self.run_data_export, daemon=True).start()
                
                time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Error in process monitoring: {e}")
                time.sleep(5)
    
    def start(self):
        """Start the process manager."""
        self.logger.info("Starting Process Manager")
        self.running = True
        
        # Create PID file
        pid_file = self.output_dir / 'process_manager.pid'
        with open(pid_file, 'w') as f:
            f.write(str(os.getpid()))
        
        # Start trading engine
        self.start_trading_engine()
        
        # Start process monitoring thread
        monitor_thread = threading.Thread(target=self.monitor_processes, daemon=True)
        monitor_thread.start()
        
        self.logger.info("Process Manager started successfully")
        
        # Main display loop
        try:
            while self.running:
                self.display_status()
                time.sleep(1)
        except KeyboardInterrupt:
            self.logger.info("Received keyboard interrupt")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the process manager and all child processes."""
        if not self.running:
            return
        
        self.logger.info("Stopping Process Manager")
        self.running = False
        
        # Stop all child processes
        for name, process in self.processes.items():
            try:
                if process.poll() is None:  # Process is still running
                    self.logger.info(f"Stopping {name} process (PID: {process.pid})")
                    process.terminate()
                    
                    # Wait for graceful shutdown
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        self.logger.warning(f"Force killing {name} process")
                        process.kill()
                        
            except Exception as e:
                self.logger.error(f"Error stopping {name} process: {e}")
        
        # Remove PID file
        try:
            pid_file = self.output_dir / 'process_manager.pid'
            if pid_file.exists():
                pid_file.unlink()
        except Exception as e:
            self.logger.error(f"Error removing PID file: {e}")
        
        self.logger.info("Process Manager stopped")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process Manager with Countdown Timers")
    parser.add_argument('--output-dir', default='historical_exports', help='Output directory')
    
    args = parser.parse_args()
    
    # Initialize and start process manager
    manager = ProcessManager(output_dir=args.output_dir)
    
    try:
        manager.start()
    except Exception as e:
        print(f"Process Manager error: {e}")
        manager.stop()


if __name__ == "__main__":
    main()
