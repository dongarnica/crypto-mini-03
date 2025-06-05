#!/usr/bin/env python3
"""
Production Health Check Endpoint
===============================

Flask-based health check endpoint for Docker containers
and production monitoring systems.

Author: Crypto Trading Strategy Engine
Date: June 5, 2025
"""

import os
import sys
import json
import time
import traceback
from datetime import datetime
from flask import Flask, jsonify, Response

app = Flask(__name__)

def check_environment():
    """Check if running in proper environment."""
    required_dirs = [
        '/app/trading',
        '/app/ml',
        '/app/ml_results',
        '/app/historical_exports'
    ]
    
    missing_dirs = []
    for directory in required_dirs:
        if not os.path.exists(directory):
            missing_dirs.append(directory)
    
    return {
        'status': 'healthy' if not missing_dirs else 'unhealthy',
        'missing_directories': missing_dirs,
        'checked_directories': required_dirs
    }

def check_files():
    """Check critical files exist."""
    critical_files = [
        '/app/.env',
        '/app/requirements.txt',
        '/app/start.sh'
    ]
    
    missing_files = []
    for file_path in critical_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    # Count data files
    csv_count = len([f for f in os.listdir('/app/historical_exports') if f.endswith('.csv')]) if os.path.exists('/app/historical_exports') else 0
    
    # Count ML models
    ml_count = 0
    if os.path.exists('/app/ml_results'):
        ml_count = len([f for f in os.listdir('/app/ml_results') if f.endswith('.h5')])
    
    return {
        'status': 'healthy' if not missing_files else 'unhealthy',
        'missing_files': missing_files,
        'data_files_count': csv_count,
        'ml_models_count': ml_count
    }

def check_system_resources():
    """Check system resources."""
    try:
        import psutil
        
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/app')
        
        return {
            'status': 'healthy',
            'memory_percent': memory.percent,
            'memory_available_gb': round(memory.available / (1024**3), 2),
            'disk_percent': round((disk.used / disk.total) * 100, 2),
            'disk_free_gb': round(disk.free / (1024**3), 2)
        }
    except ImportError:
        return {
            'status': 'warning',
            'message': 'psutil not available for system monitoring'
        }
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e)
        }

def check_trading_components():
    """Check trading-specific components."""
    try:
        # Check if we can import key modules
        sys.path.append('/app')
        
        status = {'status': 'healthy', 'components': {}}
        
        # Test imports
        try:
            from trading import trading_logic
            status['components']['trading_logic'] = 'available'
        except Exception as e:
            status['components']['trading_logic'] = f'error: {str(e)}'
            status['status'] = 'degraded'
        
        try:
            from ml import ml_strategy
            status['components']['ml_strategy'] = 'available'
        except Exception as e:
            status['components']['ml_strategy'] = f'error: {str(e)}'
            status['status'] = 'degraded'
        
        try:
            from binance import binance_client
            status['components']['binance_client'] = 'available'
        except Exception as e:
            status['components']['binance_client'] = f'error: {str(e)}'
            status['status'] = 'degraded'
        
        return status
        
    except Exception as e:
        return {
            'status': 'unhealthy',
            'error': str(e),
            'traceback': traceback.format_exc()
        }

@app.route('/health')
def health_check():
    """Main health check endpoint."""
    start_time = time.time()
    
    health_data = {
        'timestamp': datetime.utcnow().isoformat(),
        'status': 'healthy',
        'version': '1.0.0',
        'uptime_check': True,
        'checks': {}
    }
    
    # Run all health checks
    checks = {
        'environment': check_environment,
        'files': check_files,
        'system': check_system_resources,
        'trading': check_trading_components
    }
    
    overall_status = 'healthy'
    
    for check_name, check_func in checks.items():
        try:
            result = check_func()
            health_data['checks'][check_name] = result
            
            if result.get('status') in ['unhealthy', 'error']:
                overall_status = 'unhealthy'
            elif result.get('status') in ['degraded', 'warning'] and overall_status == 'healthy':
                overall_status = 'degraded'
                
        except Exception as e:
            health_data['checks'][check_name] = {
                'status': 'error',
                'error': str(e)
            }
            overall_status = 'unhealthy'
    
    health_data['status'] = overall_status
    health_data['response_time_ms'] = round((time.time() - start_time) * 1000, 2)
    
    # Return appropriate HTTP status
    status_code = 200
    if overall_status == 'unhealthy':
        status_code = 503
    elif overall_status == 'degraded':
        status_code = 200  # Still operational
    
    return jsonify(health_data), status_code

@app.route('/health/simple')
def simple_health():
    """Simple health check for basic monitoring."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'service': 'crypto-trading-engine'
    })

@app.route('/health/ready')
def readiness_check():
    """Kubernetes-style readiness probe."""
    # Check if application is ready to receive traffic
    env_check = check_environment()
    file_check = check_files()
    
    if env_check['status'] == 'healthy' and file_check['status'] == 'healthy':
        return jsonify({
            'status': 'ready',
            'timestamp': datetime.utcnow().isoformat()
        })
    else:
        return jsonify({
            'status': 'not_ready',
            'timestamp': datetime.utcnow().isoformat(),
            'issues': {
                'environment': env_check,
                'files': file_check
            }
        }), 503

@app.route('/health/live')
def liveness_check():
    """Kubernetes-style liveness probe."""
    # Basic liveness check
    return jsonify({
        'status': 'alive',
        'timestamp': datetime.utcnow().isoformat(),
        'pid': os.getpid()
    })

if __name__ == '__main__':
    # For standalone health check (Docker HEALTHCHECK)
    if len(sys.argv) > 1 and sys.argv[1] == 'check':
        try:
            import requests
            response = requests.get('http://localhost:8080/health', timeout=5)
            if response.status_code == 200:
                print("Health check passed")
                sys.exit(0)
            else:
                print(f"Health check failed with status {response.status_code}")
                sys.exit(1)
        except Exception as e:
            print(f"Health check error: {e}")
            sys.exit(1)
    else:
        # Run as Flask app for development/testing
        app.run(host='0.0.0.0', port=5000, debug=False)
