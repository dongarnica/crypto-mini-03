#!/usr/bin/env python3
"""Health check script for the trading application"""
import os
import sys
from datetime import datetime, timedelta

def check_log_activity():
    """Check if the application is actively logging"""
    try:
        log_dir = "/app/trading/logs"
        if not os.path.exists(log_dir):
            return False
            
        # Find the most recent log file
        log_files = [f for f in os.listdir(log_dir) if f.startswith("trading_") and f.endswith(".log")]
        if not log_files:
            return False
            
        latest_log = sorted(log_files)[-1]
        log_path = os.path.join(log_dir, latest_log)
        
        # Check if the log was modified in the last 10 minutes
        mod_time = datetime.fromtimestamp(os.path.getmtime(log_path))
        if datetime.now() - mod_time < timedelta(minutes=10):
            return True
            
    except Exception as e:
        print(f"Health check error: {e}")
        
    return False

def main():
    """Main health check function"""
    if check_log_activity():
        print("✅ Application is healthy")
        sys.exit(0)
    else:
        print("❌ Application is unhealthy")
        sys.exit(1)

if __name__ == "__main__":
    main()
