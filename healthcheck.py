#!/usr/bin/env python3
"""
Docker Health Check Script
==========================

Comprehensive health check for the crypto trading application
that validates all critical components including ML models.

Author: Crypto Trading Strategy Engine
Date: June 4, 2025
"""

import os
import sys
import glob
import json
import time
from datetime import datetime


def check_environment():
    """Check if running in proper environment."""
    print("🔍 Checking environment...")
    
    required_dirs = [
        '/app/trading',
        '/app/ml',
        '/app/ml_results',
        '/app/historical_exports'
    ]
    
    for directory in required_dirs:
        if not os.path.exists(directory):
            print(f"❌ Missing required directory: {directory}")
            return False
    
    print("✅ Environment directories OK")
    return True


def check_process_manager():
    """Check if process manager is running."""
    print("🔍 Checking process manager...")
    
    pid_file = '/app/historical_exports/process_manager.pid'
    
    if not os.path.exists(pid_file):
        print(f"❌ Process manager PID file not found: {pid_file}")
        return False
    
    try:
        with open(pid_file, 'r') as f:
            pid = int(f.read().strip())
        
        # Check if process is still running
        try:
            os.kill(pid, 0)  # Doesn't actually kill, just checks if process exists
            print(f"✅ Process manager running (PID: {pid})")
            return True
        except OSError:
            print(f"❌ Process manager not running (PID {pid} not found)")
            return False
            
    except (ValueError, FileNotFoundError) as e:
        print(f"❌ Error reading PID file: {e}")
        return False


def check_ml_models():
    """Check if ML models are available."""
    print("🔍 Checking ML models...")
    
    model_dir = '/app/ml_results'
    
    # Look for 3-class enhanced models
    model_files = glob.glob(f"{model_dir}/*_3class_enhanced.h5")
    scaler_files = glob.glob(f"{model_dir}/*_scaler.pkl")
    
    if not model_files:
        print("❌ No 3-class enhanced models found")
        return False
    
    if not scaler_files:
        print("❌ No scaler files found")
        return False
    
    print(f"✅ Found {len(model_files)} ML models and {len(scaler_files)} scalers")
    
    # List available models
    for model_file in model_files[:5]:  # Show first 5
        symbol = os.path.basename(model_file).split('_')[0]
        print(f"   📊 {symbol}")
    
    return True


def check_historical_data():
    """Check if historical data is available."""
    print("🔍 Checking historical data...")
    
    data_dir = '/app/historical_exports'
    csv_files = glob.glob(f"{data_dir}/*.csv")
    
    if not csv_files:
        print("❌ No historical data files found")
        return False
    
    print(f"✅ Found {len(csv_files)} historical data files")
    return True


def check_configuration():
    """Check if configuration files are available."""
    print("🔍 Checking configuration...")
    
    required_files = [
        '/app/.env',
        '/app/trading/models.py',
        '/app/config/symbol_manager.py'
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"❌ Missing required file: {file_path}")
            return False
    
    print("✅ Configuration files OK")
    return True


def test_ml_import():
    """Test if ML components can be imported."""
    print("🔍 Testing ML imports...")
    
    try:
        # Test basic ML imports
        sys.path.append('/app')
        from ml.ml_pipeline_improved_components import ImprovedCryptoLSTMPipeline
        from trading.ml_engine import MLEngine
        print("✅ ML imports successful")
        return True
    except Exception as e:
        print(f"❌ ML import failed: {e}")
        return False


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
    """Main health check routine."""
    print("🏥 Docker Health Check Starting...")
    print("=" * 50)
    
    checks = [
        ("Environment", check_environment),
        ("Process Manager", check_process_manager),
        ("ML Models", check_ml_models),
        ("Historical Data", check_historical_data),
        ("Configuration", check_configuration),
        ("ML Imports", test_ml_import)
    ]
    
    passed = 0
    total = len(checks)
    
    for name, check_func in checks:
        print(f"\n📋 {name} Check:")
        try:
            if check_func():
                passed += 1
            else:
                print(f"❌ {name} check failed")
        except Exception as e:
            print(f"❌ {name} check error: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Health Check Results: {passed}/{total} checks passed")
    
    if passed == total:
        print("✅ All health checks passed - Container is ready!")
        sys.exit(0)
    else:
        print("❌ Some health checks failed - Container needs attention")
        sys.exit(1)

if __name__ == "__main__":
    main()
