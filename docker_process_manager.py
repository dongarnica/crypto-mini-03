#!/usr/bin/env python3
"""
Docker Process Manager with Enhanced Output
==========================================

Manages all trading system components in a Docker container with enhanced output
and countdown displays for data export, ML training, and trading engine.

Features:
- Unified process management for Docker environment
- Enhanced countdown timers for all processes
- Real-time status monitoring
- Automatic process restart
- Beautiful terminal output with progress bars
- Integrated logging

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
import queue
from typing import Dict, Optional, List

# Add project paths
sys.path.append(os.path.dirname(__file__))

from scheduled_data_manager import ScheduledDataManager
from trading.enhanced_output import EnhancedOutputDisplay


class DockerProcessManager:
    """Enhanced process manager for Docker container deployment."""
    
    def __init__(self, output_dir: str = "historical_exports"):
        """Initialize the Docker process manager."""
        self.output_dir = Path(output_dir)
        self.running = False
        self.processes = {}
        self.process_queues = {}
        
        # Configuration
        self.trading_cycle_minutes = 5      # Trading engine cycle time
        self.export_cycle_hours = 2         # Data export cycle time
        self.ml_retrain_cycle_hours = 6     # ML retraining cycle time
        
        # Timing tracking
        self.start_time = datetime.now()
        self.last_export_time = None
        self.last_ml_retrain_time = None
        self.trading_start_time = datetime.now()
        
        # Enhanced display
        self.setup_logging()
        self.enhanced_display = EnhancedOutputDisplay(self.logger)
        
        # Initialize data manager
        self.data_manager = ScheduledDataManager(output_dir=str(self.output_dir))
        
        # Setup signal handlers
        signal.signal(signal.SIGTERM, self.signal_handler)
        signal.signal(signal.SIGINT, self.signal_handler)
        
        self.logger.info("🐳 Docker Process Manager initialized")
    
    def setup_logging(self):
        """Setup comprehensive logging for Docker environment."""
        log_dir = self.output_dir / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / 'docker_process_manager.log'
        
        # Create formatter with enhanced output
        formatter = logging.Formatter(
            '%(asctime)s - 🐳 [DOCKER] %(name)s - %(levelname)s - %(message)s',
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
        self.logger = logging.getLogger('DockerProcessManager')
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        self.logger.propagate = False
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        self.logger.info(f"🛑 Received signal {signum}, initiating graceful shutdown...")
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
            # First export in 30 seconds for immediate demonstration
            return datetime.now() + timedelta(seconds=30)
        else:
            return self.last_export_time + timedelta(hours=self.export_cycle_hours)
    
    def get_next_ml_retrain_time(self) -> datetime:
        """Calculate the next ML retraining time."""
        if self.last_ml_retrain_time is None:
            # First retrain in 2 minutes for demonstration
            return datetime.now() + timedelta(minutes=2)
        else:
            return self.last_ml_retrain_time + timedelta(hours=self.ml_retrain_cycle_hours)
    
    def get_process_uptime(self, process_name: str) -> timedelta:
        """Get process uptime."""
        if process_name == 'trading':
            return datetime.now() - self.trading_start_time
        else:
            return datetime.now() - self.start_time
    
    def start_trading_engine(self):
        """Start the trading engine with enhanced monitoring."""
        try:
            self.logger.info("🚀 Starting enhanced trading engine...")
            
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
                    self.logger.warning("⚠️ No trading engine found, using enhanced demo mode")
                    self.start_demo_trading_engine()
                    return
            
            # Start actual trading engine with enhanced output
            env = os.environ.copy()
            env['ENHANCED_OUTPUT'] = 'true'
            env['DOCKER_MODE'] = 'true'
            
            process = subprocess.Popen(
                [sys.executable, str(trading_engine_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                env=env
            )
            
            self.processes['trading'] = process
            self.process_queues['trading'] = queue.Queue()
            
            # Start output monitoring thread
            threading.Thread(
                target=self.monitor_process_output,
                args=('trading', process),
                daemon=True
            ).start()
            
            self.logger.info(f"✅ Trading engine started (PID: {process.pid})")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to start trading engine: {e}")
            self.start_demo_trading_engine()
    
    def start_demo_trading_engine(self):
        """Start enhanced demo trading engine."""
        self.logger.info("🎭 Starting enhanced demo trading engine...")
        self.processes['trading'] = 'demo'
        self.trading_start_time = datetime.now()
    
    def monitor_process_output(self, process_name: str, process: subprocess.Popen):
        """Monitor process output and log important messages."""
        try:
            while process.poll() is None:
                line = process.stdout.readline()
                if line:
                    # Filter and log important messages
                    if any(keyword in line.lower() for keyword in ['error', 'warning', 'signal', 'trade']):
                        self.logger.info(f"[{process_name.upper()}] {line.strip()}")
                    
                    # Store in queue for display
                    if self.process_queues.get(process_name):
                        try:
                            self.process_queues[process_name].put_nowait(line.strip())
                        except queue.Full:
                            # Remove oldest item and add new one
                            try:
                                self.process_queues[process_name].get_nowait()
                                self.process_queues[process_name].put_nowait(line.strip())
                            except queue.Empty:
                                pass
                                
        except Exception as e:
            self.logger.error(f"Error monitoring {process_name} output: {e}")
    
    def run_data_export(self):
        """Run the data export process with enhanced logging."""
        try:
            self.logger.info("📊 Starting scheduled data export...")
            
            # Run the export
            result = self.data_manager.run_scheduled_job()
            
            self.last_export_time = datetime.now()
            self.logger.info("✅ Data export completed successfully")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Data export failed: {e}")
            return None
    
    def run_ml_retraining(self):
        """Run ML model retraining with enhanced logging."""
        try:
            self.logger.info("🤖 Starting ML model retraining...")
            
            # Get active symbols
            symbols = self.data_manager.get_active_symbols()
            if symbols:
                result = self.data_manager.retrain_ml_models(symbols)
                self.last_ml_retrain_time = datetime.now()
                
                if result.get('success'):
                    self.logger.info(f"✅ ML retraining completed: {result['retrained_models']} models")
                else:
                    self.logger.error(f"❌ ML retraining failed: {result.get('error')}")
                
                return result
            else:
                self.logger.warning("⚠️ No symbols found for ML retraining")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ ML retraining failed: {e}")
            return None
    
    def display_enhanced_status(self):
        """Display comprehensive status with enhanced visuals."""
        # Clear screen for better visual effect
        os.system('clear' if os.name == 'posix' else 'cls')
        
        print("=" * 90)
        print("🐳 CRYPTO TRADING SYSTEM - DOCKER CONTAINER")
        print("=" * 90)
        print()
        
        # Current time and uptime
        current_time = datetime.now()
        total_uptime = current_time - self.start_time
        uptime_str = str(total_uptime).split('.')[0]  # Remove microseconds
        
        print(f"🕐 Current Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"⏱️  System Uptime: {uptime_str}")
        print()
        
        # Trading Engine Status
        print("🚀 TRADING ENGINE STATUS")
        print("-" * 50)
        trading_uptime = self.get_process_uptime('trading')
        trading_uptime_str = str(trading_uptime).split('.')[0]
        
        if 'trading' in self.processes and self.processes['trading'] != 'demo':
            process = self.processes['trading']
            if process.poll() is None:
                print(f"   Status: ✅ RUNNING (PID: {process.pid})")
            else:
                print(f"   Status: ❌ STOPPED (Exit Code: {process.returncode})")
        else:
            print(f"   Status: 🎭 ENHANCED DEMO MODE")
        
        print(f"   Uptime: {trading_uptime_str}")
        
        # Next trading cycle (every 5 minutes)
        next_cycle_seconds = 300 - (int(trading_uptime.total_seconds()) % 300)
        print(f"   Next Cycle: {self.format_countdown(next_cycle_seconds)}")
        
        # Show recent trading activity
        self.display_recent_activity('trading')
        print()
        
        # Data Export Status
        print("📊 DATA EXPORT STATUS")
        print("-" * 50)
        next_export_time = self.get_next_export_time()
        time_until_export = max(0, (next_export_time - current_time).total_seconds())
        
        print(f"   Status: ⏰ SCHEDULED")
        if self.last_export_time:
            print(f"   Last Export: {self.last_export_time.strftime('%H:%M:%S')}")
        else:
            print(f"   Last Export: Never (scheduled soon)")
        
        print(f"   Next Export: {self.format_countdown(int(time_until_export))}")
        print(f"   Scheduled At: {next_export_time.strftime('%H:%M:%S')}")
        print()
        
        # ML Training Status
        print("🤖 ML TRAINING STATUS")
        print("-" * 50)
        next_ml_time = self.get_next_ml_retrain_time()
        time_until_ml = max(0, (next_ml_time - current_time).total_seconds())
        
        print(f"   Status: 🧠 SCHEDULED")
        if self.last_ml_retrain_time:
            print(f"   Last Training: {self.last_ml_retrain_time.strftime('%H:%M:%S')}")
        else:
            print(f"   Last Training: Never (scheduled soon)")
        
        print(f"   Next Training: {self.format_countdown(int(time_until_ml))}")
        print(f"   Scheduled At: {next_ml_time.strftime('%H:%M:%S')}")
        print()
        
        # System Statistics
        try:
            status = self.data_manager.get_status_report()
            current_data = status.get('current_data', {})
            
            print("💾 DATA STATISTICS")
            print("-" * 50)
            print(f"   Total Files: {current_data.get('total_files', 0)}")
            print(f"   Symbols: {current_data.get('symbols_with_data', 0)}")
            print(f"   Data Size: {current_data.get('total_size_mb', 0):.2f} MB")
            print(f"   Container Storage: /app/historical_exports")
            print()
        except Exception:
            print("💾 DATA STATISTICS")
            print("-" * 50)
            print("   Status: Loading...")
            print()
        
        # Enhanced Progress Bars
        self.display_progress_bars(next_cycle_seconds, time_until_export, time_until_ml)
        
        # Container Information
        print("🐳 CONTAINER INFO")
        print("-" * 50)
        print(f"   Image: crypto-trading:latest")
        print(f"   Mode: Enhanced Docker Deployment")
        print(f"   Processes: {len([p for p in self.processes.values() if p != 'demo'])} active")
        print()
        
        print("=" * 90)
        print("💡 All processes running with enhanced output | Press Ctrl+C to stop")
        print("=" * 90)
    
    def display_recent_activity(self, process_name: str):
        """Display recent activity from process queue."""
        if process_name in self.process_queues:
            recent_lines = []
            try:
                while not self.process_queues[process_name].empty():
                    recent_lines.append(self.process_queues[process_name].get_nowait())
            except queue.Empty:
                pass
            
            if recent_lines:
                print(f"   Recent Activity:")
                for line in recent_lines[-3:]:  # Show last 3 lines
                    print(f"     • {line[:60]}{'...' if len(line) > 60 else ''}")
    
    def display_progress_bars(self, trading_seconds: int, export_seconds: int, ml_seconds: int):
        """Display visual progress bars for all processes."""
        print("📊 PROCESS CYCLES")
        print("-" * 50)
        
        # Trading progress (5-minute cycle)
        trading_progress = max(0, (300 - trading_seconds) / 300 * 40)
        print(f"   Trading:  [{'█' * int(trading_progress)}{'░' * (40 - int(trading_progress))}] "
              f"{(300 - trading_seconds)/300*100:.1f}%")
        
        # Export progress (2-hour cycle or until next export)
        if export_seconds > 0:
            export_total = min(7200, export_seconds + 300)  # Max 2 hours display
            export_progress = max(0, (export_total - export_seconds) / export_total * 40)
            print(f"   Export:   [{'█' * int(export_progress)}{'░' * (40 - int(export_progress))}] "
                  f"{(export_total - export_seconds)/export_total*100:.1f}%")
        else:
            print(f"   Export:   [{'█' * 40}] 100.0% (Ready)")
        
        # ML training progress (6-hour cycle)
        if ml_seconds > 0:
            ml_total = min(21600, ml_seconds + 300)  # Max 6 hours display
            ml_progress = max(0, (ml_total - ml_seconds) / ml_total * 40)
            print(f"   ML Train: [{'█' * int(ml_progress)}{'░' * (40 - int(ml_progress))}] "
                  f"{(ml_total - ml_seconds)/ml_total*100:.1f}%")
        else:
            print(f"   ML Train: [{'█' * 40}] 100.0% (Ready)")
        
        print()
    
    def monitor_processes(self):
        """Monitor and restart processes if needed."""
        while self.running:
            try:
                current_time = datetime.now()
                
                # Check if trading engine needs restart
                if 'trading' in self.processes and self.processes['trading'] != 'demo':
                    process = self.processes['trading']
                    if process.poll() is not None:  # Process has terminated
                        self.logger.warning(f"⚠️ Trading engine stopped (exit code: {process.returncode})")
                        self.logger.info("🔄 Restarting trading engine...")
                        time.sleep(5)  # Wait before restart
                        self.start_trading_engine()
                
                # Check if it's time for data export
                next_export_time = self.get_next_export_time()
                if current_time >= next_export_time:
                    threading.Thread(target=self.run_data_export, daemon=True).start()
                
                # Check if it's time for ML retraining
                next_ml_time = self.get_next_ml_retrain_time()
                if current_time >= next_ml_time:
                    threading.Thread(target=self.run_ml_retraining, daemon=True).start()
                
                time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"❌ Error in process monitoring: {e}")
                time.sleep(5)
    
    def start(self):
        """Start the Docker process manager."""
        self.logger.info("🐳 Starting Docker Process Manager")
        self.running = True
        
        # Create PID file
        pid_file = self.output_dir / 'docker_process_manager.pid'
        with open(pid_file, 'w') as f:
            f.write(str(os.getpid()))
        
        # Print startup banner
        print("=" * 90)
        print("🐳 CRYPTO TRADING SYSTEM - DOCKER CONTAINER STARTING")
        print("=" * 90)
        print("🚀 Initializing all components...")
        print("📊 Data Export Service: Scheduled")
        print("🤖 ML Training Service: Scheduled") 
        print("💹 Trading Engine: Starting...")
        print("=" * 90)
        
        # Start trading engine
        self.start_trading_engine()
        
        # Start process monitoring thread
        monitor_thread = threading.Thread(target=self.monitor_processes, daemon=True)
        monitor_thread.start()
        
        self.logger.info("✅ Docker Process Manager started successfully")
        
        # Main display loop
        try:
            while self.running:
                self.display_enhanced_status()
                time.sleep(2)  # Update every 2 seconds for better responsiveness
        except KeyboardInterrupt:
            self.logger.info("🛑 Received keyboard interrupt")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the process manager and all child processes."""
        if not self.running:
            return
        
        self.logger.info("🛑 Stopping Docker Process Manager")
        self.running = False
        
        # Stop all child processes
        for name, process in self.processes.items():
            if process == 'demo':
                continue
                
            try:
                if process.poll() is None:  # Process is still running
                    self.logger.info(f"🛑 Stopping {name} process (PID: {process.pid})")
                    process.terminate()
                    
                    # Wait for graceful shutdown
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        self.logger.warning(f"⚡ Force killing {name} process")
                        process.kill()
                        
            except Exception as e:
                self.logger.error(f"❌ Error stopping {name} process: {e}")
        
        # Remove PID file
        try:
            pid_file = self.output_dir / 'docker_process_manager.pid'
            if pid_file.exists():
                pid_file.unlink()
        except Exception as e:
            self.logger.error(f"❌ Error removing PID file: {e}")
        
        self.logger.info("✅ Docker Process Manager stopped")


def main():
    """Main function for Docker deployment."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Docker Process Manager with Enhanced Output")
    parser.add_argument('--output-dir', default='historical_exports', help='Output directory')
    
    args = parser.parse_args()
    
    # Initialize and start Docker process manager
    manager = DockerProcessManager(output_dir=args.output_dir)
    
    try:
        manager.start()
    except Exception as e:
        print(f"❌ Docker Process Manager error: {e}")
        manager.stop()


if __name__ == "__main__":
    main()
