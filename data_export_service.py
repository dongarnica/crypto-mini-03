#!/usr/bin/env python3
"""
Data Export Scheduler Service
=============================

Systemd-compatible service script for running scheduled historical data exports.
This script is designed to run as a background service and manage the periodic
export and cleanup of cryptocurrency historical data.

Features:
- Runs every 2 hours
- Automatic cleanup of old files
- Health monitoring and logging
- Graceful shutdown handling
- Status reporting

Author: Crypto Trading System
Date: June 2025
"""

import os
import sys
import signal
import time
import logging
from datetime import datetime
from pathlib import Path

# Add project paths
sys.path.append(os.path.dirname(__file__))

from scheduled_data_manager import ScheduledDataManager


class DataExportService:
    """Service wrapper for scheduled data exports."""
    
    def __init__(self, output_dir: str = "historical_exports"):
        """Initialize the service."""
        self.output_dir = Path(output_dir)
        self.running = False
        self.manager = None
        
        # Setup service logging
        self.setup_service_logging()
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self.signal_handler)
        signal.signal(signal.SIGINT, self.signal_handler)
        
        self.logger.info("DataExportService initialized")
    
    def setup_service_logging(self):
        """Setup logging for the service."""
        log_dir = self.output_dir / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / 'data_export_service.log'
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # File handler with rotation
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            log_file, maxBytes=50*1024*1024, backupCount=10
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        
        # Setup logger
        self.logger = logging.getLogger('DataExportService')
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        self.logger.propagate = False
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.stop()
    
    def start(self):
        """Start the data export service."""
        self.logger.info("Starting Data Export Service")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"Process PID: {os.getpid()}")
        
        try:
            # Initialize the data manager
            self.manager = ScheduledDataManager(output_dir=str(self.output_dir))
            
            # Mark service as running
            self.running = True
            
            # Create PID file
            pid_file = self.output_dir / 'service.pid'
            with open(pid_file, 'w') as f:
                f.write(str(os.getpid()))
            
            self.logger.info("Service started successfully")
            
            # Start the scheduler
            self.manager.start_scheduler()
            
        except Exception as e:
            self.logger.error(f"Failed to start service: {e}")
            self.stop()
            raise
    
    def stop(self):
        """Stop the data export service."""
        if not self.running:
            return
        
        self.logger.info("Stopping Data Export Service")
        self.running = False
        
        # Remove PID file
        try:
            pid_file = self.output_dir / 'service.pid'
            if pid_file.exists():
                pid_file.unlink()
        except Exception as e:
            self.logger.error(f"Error removing PID file: {e}")
        
        self.logger.info("Data Export Service stopped")
    
    def is_running(self) -> bool:
        """Check if service is running."""
        return self.running
    
    def get_status(self):
        """Get service status."""
        if self.manager:
            return self.manager.get_status()
        return {'service_status': 'stopped'}


def create_systemd_service_file():
    """Create a systemd service file for the data export service."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    python_path = sys.executable
    
    service_content = f"""[Unit]
Description=Crypto Historical Data Export Service
After=network.target
Wants=network.target

[Service]
Type=simple
User=root
WorkingDirectory={current_dir}
Environment=PATH={os.environ.get('PATH', '')}
Environment=PYTHONPATH={current_dir}
ExecStart={python_path} {current_dir}/data_export_service.py --daemon
Restart=always
RestartSec=30
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
"""
    
    service_file = Path('/etc/systemd/system/crypto-data-export.service')
    
    try:
        with open(service_file, 'w') as f:
            f.write(service_content)
        
        print(f"✅ Systemd service file created: {service_file}")
        print("\nTo enable and start the service:")
        print("sudo systemctl daemon-reload")
        print("sudo systemctl enable crypto-data-export.service")
        print("sudo systemctl start crypto-data-export.service")
        print("\nTo check service status:")
        print("sudo systemctl status crypto-data-export.service")
        print("sudo journalctl -u crypto-data-export.service -f")
        
        return str(service_file)
        
    except PermissionError:
        print("❌ Permission denied. Run with sudo to create systemd service file.")
        return None
    except Exception as e:
        print(f"❌ Error creating systemd service file: {e}")
        return None


def create_cron_job():
    """Create a cron job for the data export service."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    python_path = sys.executable
    
    # Create a wrapper script for cron
    cron_script = Path(current_dir) / 'run_data_export.sh'
    
    script_content = f"""#!/bin/bash
# Crypto Data Export Cron Job
# Runs every 2 hours

cd {current_dir}
export PATH={os.environ.get('PATH', '')}
export PYTHONPATH={current_dir}

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '#' | xargs)
fi

# Run the data export
{python_path} scheduled_data_manager.py --manual >> {current_dir}/historical_exports/logs/cron.log 2>&1
"""
    
    try:
        with open(cron_script, 'w') as f:
            f.write(script_content)
        
        # Make executable
        os.chmod(cron_script, 0o755)
        
        # Create cron entry
        cron_entry = f"0 */2 * * * {cron_script}"
        
        print(f"✅ Cron script created: {cron_script}")
        print(f"\nTo add to crontab (runs every 2 hours):")
        print(f"crontab -e")
        print(f"Then add this line:")
        print(f"{cron_entry}")
        print(f"\nOr run this command to add automatically:")
        print(f'(crontab -l 2>/dev/null; echo "{cron_entry}") | crontab -')
        
        return str(cron_script)
        
    except Exception as e:
        print(f"❌ Error creating cron job: {e}")
        return None


def main():
    """Main function for the service."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Data Export Service")
    parser.add_argument('--daemon', action='store_true', help='Run as daemon service')
    parser.add_argument('--output-dir', default='historical_exports', help='Output directory')
    parser.add_argument('--create-systemd', action='store_true', help='Create systemd service file')
    parser.add_argument('--create-cron', action='store_true', help='Create cron job')
    parser.add_argument('--status', action='store_true', help='Check service status')
    
    args = parser.parse_args()
    
    if args.create_systemd:
        create_systemd_service_file()
        return
    
    if args.create_cron:
        create_cron_job()
        return
    
    if args.status:
        # Check if service is running
        output_dir = Path(args.output_dir)
        pid_file = output_dir / 'service.pid'
        
        if pid_file.exists():
            try:
                with open(pid_file, 'r') as f:
                    pid = int(f.read().strip())
                
                # Check if process is still running
                try:
                    os.kill(pid, 0)  # Doesn't actually kill, just checks if process exists
                    print(f"✅ Service is running (PID: {pid})")
                    
                    # Try to get status from manager
                    try:
                        manager = ScheduledDataManager(output_dir=args.output_dir)
                        status = manager.get_status()
                        print("\n📊 Service Status:")
                        for key, value in status.items():
                            print(f"  {key:20}: {value}")
                    except Exception:
                        pass
                        
                except OSError:
                    print(f"❌ Service not running (PID file exists but process {pid} not found)")
                    pid_file.unlink()  # Remove stale PID file
                    
            except (ValueError, FileNotFoundError):
                print("❌ Invalid or missing PID file")
        else:
            print("❌ Service not running (no PID file found)")
        
        return
    
    # Start the service
    service = DataExportService(output_dir=args.output_dir)
    
    try:
        service.start()
    except KeyboardInterrupt:
        print("\nService interrupted by user")
    except Exception as e:
        print(f"Service error: {e}")
    finally:
        service.stop()


if __name__ == "__main__":
    main()
