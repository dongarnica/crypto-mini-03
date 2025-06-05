#!/usr/bin/env python3
"""
Scheduled Historical Data Manager
=================================

Manages scheduled retrieval and maintenance of historical crypto data exports.
- Runs every 2 hours to export latest data
- Automatically cleans up outdated files (keeps only latest per symbol)
- Logs all operations for monitoring
- Provides health checks and status reports

Author: Crypto Trading System
Date: June 2025
"""

import os
import sys
import glob
import json
import logging
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from pathlib import Path
import schedule
import shutil

# Add project paths
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), 'binance'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'trading'))

try:
    from export_historical_data import HistoricalDataExporter
except ImportError:
    # Fallback for Docker environment
    import importlib.util
    spec = importlib.util.spec_from_file_location("export_historical_data", 
                                                  os.path.join(os.path.dirname(__file__), "export_historical_data.py"))
    export_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(export_module)
    HistoricalDataExporter = export_module.HistoricalDataExporter

from config.symbols_config import CryptoSymbolsConfig

try:
    from trading.ml_engine import MLEngine
except ImportError:
    # Create a dummy ML engine for Docker environment
    class MLEngine:
        def __init__(self, config):
            self.ml_pipelines = {}
            
        def initialize_pipelines(self, symbols):
            for symbol in symbols:
                self.ml_pipelines[symbol] = type('Pipeline', (), {'model': None})()
            return len(symbols)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()


class ScheduledDataManager:
    """Manages scheduled historical data exports and cleanup."""
    
    def __init__(self, output_dir: str = "historical_exports"):
        """
        Initialize the scheduled data manager.
        
        Args:
            output_dir: Directory for historical data exports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Initialize components
        self.exporter = HistoricalDataExporter(output_dir=str(self.output_dir))
        self.symbols_config = CryptoSymbolsConfig()
        
        # Initialize ML Engine with basic config
        self.ml_engine = None
        self._initialize_ml_engine()
        
        # Configuration
        self.export_period_hours = 2  # Export every 2 hours
        self.keep_latest_only = True  # Keep only latest file per symbol
        self.max_files_per_symbol = 1  # Maximum files to keep per symbol
        self.data_retention_days = 7  # Keep data for 7 days max
        self.enable_ml_retraining = True  # Enable automatic ML retraining
        
        # Status tracking
        self.last_export_time = None
        self.last_cleanup_time = None
        self.last_retrain_time = None
        self.export_count = 0
        self.cleanup_count = 0
        self.retrain_count = 0
        
        self.logger.info("ScheduledDataManager initialized")
        self.logger.info(f"Export directory: {self.output_dir}")
        self.logger.info(f"Export interval: {self.export_period_hours} hours")
        self.logger.info(f"Data retention: {self.data_retention_days} days")
        self.logger.info(f"ML Retraining enabled: {self.enable_ml_retraining}")
    
    def setup_logging(self):
        """Setup comprehensive logging for scheduled operations."""
        log_dir = self.output_dir / 'logs'
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / 'scheduled_data_manager.log'
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # File handler with rotation
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            log_file, maxBytes=10*1024*1024, backupCount=5
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        
        # Setup logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        self.logger.propagate = False
    
    def get_active_symbols(self) -> List[str]:
        """Get list of active symbols for export."""
        try:
            # Combine all symbol categories
            all_symbols = []
            
            # Primary symbols
            primary_symbols = [
                self.symbols_config.primary_symbol,
                self.symbols_config.secondary_symbol,
                self.symbols_config.tertiary_symbol
            ]
            all_symbols.extend([s for s in primary_symbols if s])
            
            # Additional symbol categories
            all_symbols.extend(self.symbols_config.crypto_symbols)
            all_symbols.extend(self.symbols_config.defi_symbols)
            all_symbols.extend(self.symbols_config.altcoin_symbols)
            
            # Remove duplicates and empty strings
            unique_symbols = list(set([s.upper() for s in all_symbols if s and s.strip()]))
            
            self.logger.info(f"Active symbols for export: {len(unique_symbols)} symbols")
            self.logger.debug(f"Symbols: {', '.join(unique_symbols)}")
            
            return unique_symbols
            
        except Exception as e:
            self.logger.error(f"Error getting active symbols: {e}")
            # Fallback to default symbols
            return ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT', 'LINKUSDT']
    
    def find_existing_files(self, symbol: str) -> List[Path]:
        """
        Find existing export files for a symbol.
        
        Args:
            symbol: Trading symbol to search for
            
        Returns:
            List of file paths sorted by modification time (newest first)
        """
        pattern = f"{symbol}_1year_hourly_*.csv"
        matching_files = list(self.output_dir.glob(pattern))
        
        # Sort by modification time (newest first)
        matching_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        return matching_files
    
    def cleanup_old_files(self) -> Dict[str, int]:
        """
        Clean up outdated export files.
        
        Returns:
            Dictionary with cleanup statistics
        """
        self.logger.info("Starting cleanup of old export files")
        cleanup_stats = {
            'symbols_processed': 0,
            'files_deleted': 0,
            'space_freed_mb': 0,
            'errors': 0
        }
        
        try:
            # Get all symbols from existing files
            all_files = list(self.output_dir.glob("*_1year_hourly_*.csv"))
            symbols_found = set()
            
            for file_path in all_files:
                # Extract symbol from filename
                filename = file_path.name
                symbol = filename.split('_1year_hourly_')[0]
                symbols_found.add(symbol)
            
            self.logger.info(f"Found files for {len(symbols_found)} symbols")
            
            # Process each symbol
            for symbol in symbols_found:
                try:
                    cleanup_stats['symbols_processed'] += 1
                    existing_files = self.find_existing_files(symbol)
                    
                    self.logger.debug(f"Processing {symbol}: {len(existing_files)} files found")
                    
                    if len(existing_files) <= self.max_files_per_symbol:
                        continue  # No cleanup needed
                    
                    # Keep only the newest files
                    files_to_keep = existing_files[:self.max_files_per_symbol]
                    files_to_delete = existing_files[self.max_files_per_symbol:]
                    
                    self.logger.info(f"{symbol}: Keeping {len(files_to_keep)} files, deleting {len(files_to_delete)} files")
                    
                    # Delete old files
                    for file_path in files_to_delete:
                        try:
                            file_size = file_path.stat().st_size
                            file_path.unlink()
                            
                            cleanup_stats['files_deleted'] += 1
                            cleanup_stats['space_freed_mb'] += file_size / (1024 * 1024)
                            
                            self.logger.info(f"Deleted: {file_path.name} ({file_size / (1024 * 1024):.2f} MB)")
                            
                        except Exception as e:
                            self.logger.error(f"Error deleting {file_path}: {e}")
                            cleanup_stats['errors'] += 1
                
                except Exception as e:
                    self.logger.error(f"Error processing cleanup for {symbol}: {e}")
                    cleanup_stats['errors'] += 1
            
            # Also cleanup old files based on age
            cutoff_date = datetime.now() - timedelta(days=self.data_retention_days)
            cutoff_timestamp = cutoff_date.timestamp()
            
            for file_path in all_files:
                try:
                    if file_path.stat().st_mtime < cutoff_timestamp:
                        file_size = file_path.stat().st_size
                        file_path.unlink()
                        
                        cleanup_stats['files_deleted'] += 1
                        cleanup_stats['space_freed_mb'] += file_size / (1024 * 1024)
                        
                        self.logger.info(f"Deleted old file: {file_path.name} (older than {self.data_retention_days} days)")
                
                except Exception as e:
                    self.logger.error(f"Error deleting old file {file_path}: {e}")
                    cleanup_stats['errors'] += 1
            
            self.cleanup_count += 1
            self.last_cleanup_time = datetime.now()
            
            self.logger.info(f"Cleanup completed: {cleanup_stats}")
            return cleanup_stats
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
            cleanup_stats['errors'] += 1
            return cleanup_stats
    
    def export_historical_data(self) -> Dict[str, any]:
        """
        Export historical data for all active symbols.
        
        Returns:
            Dictionary with export results
        """
        self.logger.info("Starting scheduled historical data export")
        
        try:
            # Get active symbols
            symbols = self.get_active_symbols()
            
            if not symbols:
                self.logger.warning("No symbols found for export")
                return {'success': False, 'error': 'No symbols configured'}
            
            # Set date range (rolling 12 months of hourly data for ML training)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)  # 12 months of hourly data
            
            self.logger.info(f"Exporting data for {len(symbols)} symbols")
            self.logger.info(f"Date range (12 months): {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            
            # Export data
            results = self.exporter.export_multiple_symbols(
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                format='csv'
            )
            
            self.export_count += 1
            self.last_export_time = datetime.now()
            
            # Log results
            summary = results.get('summary', {})
            self.logger.info(f"Export completed - Success: {summary.get('successful_exports', 0)}, "
                           f"Failed: {summary.get('failed_exports', 0)}")
            
            return {'success': True, 'results': results}
            
        except Exception as e:
            self.logger.error(f"Error during export: {e}")
            return {'success': False, 'error': str(e)}
    
    def run_scheduled_job(self):
        """Run the complete scheduled job (export + cleanup + ML retraining)."""
        self.logger.info("=" * 80)
        self.logger.info("SCHEDULED DATA MANAGEMENT JOB STARTED")
        self.logger.info("=" * 80)
        
        job_start_time = datetime.now()
        
        try:
            # Step 1: Export latest data
            self.logger.info("Step 1: Exporting historical data")
            export_result = self.export_historical_data()
            
            if export_result['success']:
                self.logger.info("✅ Historical data export completed successfully")
            else:
                self.logger.error(f"❌ Historical data export failed: {export_result.get('error')}")
            
            # Step 2: Cleanup old files
            self.logger.info("Step 2: Cleaning up old files")
            cleanup_result = self.cleanup_old_files()
            
            if cleanup_result['files_deleted'] > 0:
                self.logger.info(f"✅ Cleanup completed: {cleanup_result['files_deleted']} files deleted, "
                               f"{cleanup_result['space_freed_mb']:.2f} MB freed")
            else:
                self.logger.info("✅ Cleanup completed: No files needed cleanup")
            
            # Step 3: ML Model Retraining (if enabled)
            if self.enable_ml_retraining:
                self.logger.info("Step 3: Starting ML model retraining")
                
                # Get active symbols for retraining
                active_symbols = self.get_active_symbols()
                
                if active_symbols:
                    self.logger.info(f"🤖 Starting ML retraining for {len(active_symbols)} symbols")
                    retrain_result = self.retrain_ml_models(active_symbols)
                    
                    if retrain_result['success']:
                        self.logger.info(f"✅ ML retraining completed: {retrain_result['retrained_models']} models updated")
                    else:
                        self.logger.error(f"❌ ML retraining failed: {retrain_result.get('error', 'Unknown error')}")
                else:
                    self.logger.info("✅ No symbols found for ML retraining")
            else:
                self.logger.info("Step 3: ML retraining disabled")
            
            # Step 4: Generate status report
            self.generate_status_report()
            
        except Exception as e:
            self.logger.error(f"❌ Scheduled job failed: {e}")
        
        finally:
            job_duration = datetime.now() - job_start_time
            self.logger.info(f"Scheduled job completed in {job_duration}")
            self.logger.info("=" * 80)
    
    def generate_status_report(self):
        """Generate and save a status report."""
        try:
            # Count current files
            all_files = list(self.output_dir.glob("*_1year_hourly_*.csv"))
            symbols_with_data = set()
            total_size_mb = 0
            
            for file_path in all_files:
                filename = file_path.name
                symbol = filename.split('_1year_hourly_')[0]
                symbols_with_data.add(symbol)
                total_size_mb += file_path.stat().st_size / (1024 * 1024)
            
            # Create status report
            status_report = {
                'timestamp': datetime.now().isoformat(),
                'manager_info': {
                    'export_interval_hours': self.export_period_hours,
                    'data_retention_days': self.data_retention_days,
                    'max_files_per_symbol': self.max_files_per_symbol
                },
                'statistics': {
                    'total_exports_run': self.export_count,
                    'total_cleanups_run': self.cleanup_count,
                    'total_retrains_run': self.retrain_count,
                    'last_export_time': self.last_export_time.isoformat() if self.last_export_time else None,
                    'last_cleanup_time': self.last_cleanup_time.isoformat() if self.last_cleanup_time else None,
                    'last_retrain_time': self.last_retrain_time.isoformat() if self.last_retrain_time else None
                },
                'current_data': {
                    'total_files': len(all_files),
                    'symbols_with_data': len(symbols_with_data),
                    'total_size_mb': round(total_size_mb, 2),
                    'symbols_list': sorted(list(symbols_with_data))
                }
            }
            
            # Save status report
            status_file = self.output_dir / 'status_report.json'
            with open(status_file, 'w') as f:
                json.dump(status_report, f, indent=2, default=str)
            
            self.logger.info(f"Status report saved: {status_file}")
            self.logger.info(f"Current status: {len(all_files)} files, {len(symbols_with_data)} symbols, "
                           f"{total_size_mb:.2f} MB total")
            
        except Exception as e:
            self.logger.error(f"Error generating status report: {e}")
    
    def start_scheduler(self):
        """Start the scheduled data management service."""
        self.logger.info("Starting Scheduled Data Management Service")
        self.logger.info(f"Schedule: Every {self.export_period_hours} hours")
        
        # Schedule the job
        schedule.every(self.export_period_hours).hours.do(self.run_scheduled_job)
        
        # Run initial job
        self.logger.info("Running initial data management job")
        self.run_scheduled_job()
        
        # Start the scheduler loop
        self.logger.info("Scheduler started - waiting for next scheduled run")
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
                
        except KeyboardInterrupt:
            self.logger.info("Scheduler stopped by user")
        except Exception as e:
            self.logger.error(f"Scheduler error: {e}")
    
    def run_manual_job(self):
        """Run a manual job (for testing or one-time use)."""
        self.logger.info("Running manual data management job")
        self.run_scheduled_job()
    
    def get_status(self) -> Dict[str, any]:
        """Get current status of the data manager."""
        all_files = list(self.output_dir.glob("*_1year_hourly_*.csv"))
        symbols_with_data = set()
        
        for file_path in all_files:
            filename = file_path.name
            symbol = filename.split('_1year_hourly_')[0]
            symbols_with_data.add(symbol)
        
        return {
            'service_status': 'running',
            'export_count': self.export_count,
            'cleanup_count': self.cleanup_count,
            'retrain_count': self.retrain_count,
            'last_export': self.last_export_time.isoformat() if self.last_export_time else None,
            'last_cleanup': self.last_cleanup_time.isoformat() if self.last_cleanup_time else None,
            'last_retrain': self.last_retrain_time.isoformat() if self.last_retrain_time else None,
            'ml_retraining_enabled': self.enable_ml_retraining,
            'current_files': len(all_files),
            'symbols_with_data': len(symbols_with_data),
            'output_directory': str(self.output_dir)
        }
    
    def get_status_report(self) -> Dict[str, any]:
        """Get comprehensive status report with detailed information."""
        try:
            # Get basic status
            basic_status = self.get_status()
            
            # Get file statistics
            all_files = list(self.output_dir.glob("*_1year_hourly_*.csv"))
            files_by_symbol = {}
            total_size_mb = 0
            oldest_file = None
            newest_file = None
            
            for file_path in all_files:
                filename = file_path.name
                symbol = filename.split('_1year_hourly_')[0]
                file_size = file_path.stat().st_size
                file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                
                if symbol not in files_by_symbol:
                    files_by_symbol[symbol] = []
                
                files_by_symbol[symbol].append({
                    'filename': filename,
                    'size_mb': file_size / (1024 * 1024),
                    'modified_time': file_mtime.isoformat()
                })
                
                total_size_mb += file_size / (1024 * 1024)
                
                if oldest_file is None or file_mtime < datetime.fromtimestamp(oldest_file.stat().st_mtime):
                    oldest_file = file_path
                if newest_file is None or file_mtime > datetime.fromtimestamp(newest_file.stat().st_mtime):
                    newest_file = file_path
            
            # Sort files by modification time for each symbol
            for symbol in files_by_symbol:
                files_by_symbol[symbol].sort(key=lambda x: x['modified_time'], reverse=True)
            
            # Create comprehensive report
            status_report = {
                'timestamp': datetime.now().isoformat(),
                'system_status': basic_status,
                'configuration': {
                    'export_interval_hours': self.export_period_hours,
                    'data_retention_days': self.data_retention_days,
                    'max_files_per_symbol': self.max_files_per_symbol,
                    'keep_latest_only': self.keep_latest_only,
                    'ml_retraining_enabled': self.enable_ml_retraining,
                    'output_directory': str(self.output_dir)
                },
                'file_statistics': {
                    'total_files': len(all_files),
                    'symbols_count': len(files_by_symbol),
                    'total_size_mb': round(total_size_mb, 2),
                    'total_size_gb': round(total_size_mb / 1024, 2),
                    'oldest_file': oldest_file.name if oldest_file else None,
                    'newest_file': newest_file.name if newest_file else None,
                    'oldest_file_date': datetime.fromtimestamp(oldest_file.stat().st_mtime).isoformat() if oldest_file else None,
                    'newest_file_date': datetime.fromtimestamp(newest_file.stat().st_mtime).isoformat() if newest_file else None
                },
                'symbols_detail': files_by_symbol,
                'health_check': {
                    'logs_directory_exists': (self.output_dir / 'logs').exists(),
                    'exporter_initialized': self.exporter is not None,
                    'symbols_config_loaded': self.symbols_config is not None,
                    'ml_engine_initialized': self.ml_engine is not None,
                    'recent_activity': self.last_export_time is not None or self.last_cleanup_time is not None
                }
            }
            
            return status_report
            
        except Exception as e:
            self.logger.error(f"Error generating status report: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'error': f"Failed to generate status report: {str(e)}",
                'basic_status': self.get_status()
            }
    
    def _initialize_ml_engine(self):
        """Initialize ML engine for model retraining."""
        try:
            # Create a basic config object for ML engine
            class BasicMLConfig:
                def __init__(self):
                    self.lookback_period = 60
                    self.prediction_horizon = 24
                    self.min_confidence = 0.6
                    self.log_level = 'INFO'
                    self.save_trades = True
            
            ml_config = BasicMLConfig()
            self.ml_engine = MLEngine(ml_config)
            self.logger.info("✅ ML Engine initialized for retraining")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize ML Engine: {e}")
            self.ml_engine = None
            self.enable_ml_retraining = False
    
    def detect_new_data_files(self, symbols: List[str]) -> Dict[str, str]:
        """
        Detect symbols that have new historical data files since last export.
        
        Args:
            symbols: List of symbols to check
            
        Returns:
            Dictionary mapping symbols to their newest data file paths
        """
        new_data_files = {}
        
        try:
            # Check if we have a previous export time to compare against
            if not self.last_export_time:
                # First run - consider all existing files as "new"
                self.logger.info("First run detected - will retrain all available models")
                cutoff_time = datetime.now() - timedelta(hours=self.export_period_hours * 2)
            else:
                # Use last export time as cutoff
                cutoff_time = self.last_export_time
            
            for symbol in symbols:
                try:
                    # Find the newest file for this symbol
                    existing_files = self.find_existing_files(symbol)
                    
                    if existing_files:
                        newest_file = existing_files[0]  # Already sorted newest first
                        file_time = datetime.fromtimestamp(newest_file.stat().st_mtime)
                        
                        # Check if file is newer than cutoff
                        if file_time > cutoff_time:
                            new_data_files[symbol] = str(newest_file)
                            self.logger.debug(f"New data detected for {symbol}: {newest_file.name}")
                        else:
                            self.logger.debug(f"No new data for {symbol} (file: {file_time}, cutoff: {cutoff_time})")
                    else:
                        self.logger.debug(f"No data files found for {symbol}")
                        
                except Exception as e:
                    self.logger.error(f"Error checking new data for {symbol}: {e}")
            
            self.logger.info(f"Detected new data for {len(new_data_files)} symbols: {list(new_data_files.keys())}")
            return new_data_files
            
        except Exception as e:
            self.logger.error(f"Error detecting new data files: {e}")
            return {}
    
    def retrain_ml_models(self, symbols_to_retrain: List[str]) -> Dict[str, any]:
        """
        Retrain ML models for symbols using symbol-based data loading.
        
        Args:
            symbols_to_retrain: List of symbols to retrain
            
        Returns:
            Dictionary with retraining results
        """
        if not self.enable_ml_retraining or not self.ml_engine:
            self.logger.warning("ML retraining is disabled or ML engine not available")
            return {'success': False, 'error': 'ML retraining disabled'}
        
        if not symbols_to_retrain:
            self.logger.info("No symbols provided for retraining - skipping ML retraining")
            return {'success': True, 'retrained_models': 0, 'message': 'No retraining needed'}
        
        self.logger.info("🤖 Starting ML model retraining process")
        self.logger.info(f"Symbols to retrain: {symbols_to_retrain}")
        
        retrain_results = {
            'success': True,
            'retrained_models': 0,
            'failed_models': 0,
            'results_by_symbol': {},
            'errors': []
        }
        
        try:
            # Initialize pipelines for symbols - this will use symbol-based data loading
            # which automatically fetches comprehensive historical data from Binance
            self.logger.info(f"Initializing ML pipelines for {len(symbols_to_retrain)} symbols...")
            self.logger.info("Using symbol-based data loading for comprehensive historical data...")
            
            self.ml_engine.initialize_pipelines(symbols_to_retrain)
            
            # Check how many pipelines were successfully initialized
            if hasattr(self.ml_engine, 'ml_pipelines'):
                initialized_count = len(self.ml_engine.ml_pipelines)
                self.logger.info(f"Successfully initialized {initialized_count}/{len(symbols_to_retrain)} ML pipelines")
                
                # The initialize_pipelines method already handles training, but we can 
                # also explicitly retrain models that had issues
                for symbol in symbols_to_retrain:
                    try:
                        start_time = datetime.now()
                        
                        if symbol in self.ml_engine.ml_pipelines:
                            # Pipeline exists, check if model is properly trained
                            pipeline = self.ml_engine.ml_pipelines[symbol]
                            
                            if hasattr(pipeline, 'model') and pipeline.model is not None:
                                # Model is trained successfully
                                retrain_results['retrained_models'] += 1
                                duration = datetime.now() - start_time
                                retrain_results['results_by_symbol'][symbol] = {
                                    'success': True,
                                    'duration_seconds': duration.total_seconds(),
                                    'data_source': 'symbol_based_loading'
                                }
                                self.logger.info(f"✅ Successfully trained/loaded model for {symbol}")
                            else:
                                # Model training failed
                                retrain_results['failed_models'] += 1
                                duration = datetime.now() - start_time
                                retrain_results['results_by_symbol'][symbol] = {
                                    'success': False,
                                    'error': 'Model training failed during initialization',
                                    'duration_seconds': duration.total_seconds(),
                                    'data_source': 'symbol_based_loading'
                                }
                                self.logger.error(f"❌ Failed to train model for {symbol}")
                        else:
                            # Pipeline initialization failed
                            retrain_results['failed_models'] += 1
                            duration = datetime.now() - start_time
                            retrain_results['results_by_symbol'][symbol] = {
                                'success': False,
                                'error': 'Pipeline initialization failed',
                                'duration_seconds': duration.total_seconds(),
                                'data_source': 'symbol_based_loading'
                            }
                            self.logger.error(f"❌ Failed to initialize pipeline for {symbol}")
                            
                    except Exception as e:
                        error_msg = f"Error processing {symbol}: {str(e)}"
                        self.logger.error(error_msg)
                        retrain_results['failed_models'] += 1
                        retrain_results['errors'].append(error_msg)
                        retrain_results['results_by_symbol'][symbol] = {
                            'success': False,
                            'error': str(e),
                            'data_source': 'symbol_based_loading'
                        }
            else:
                self.logger.error("ML engine pipelines not accessible")
                retrain_results['success'] = False
                retrain_results['errors'].append("ML engine pipelines not accessible")
            
            # Update tracking
            self.retrain_count += 1
            self.last_retrain_time = datetime.now()
            
            # Log summary
            total_attempts = retrain_results['retrained_models'] + retrain_results['failed_models']
            success_rate = (retrain_results['retrained_models'] / total_attempts * 100) if total_attempts > 0 else 0
            
            self.logger.info(f"🎯 ML Retraining Summary:")
            self.logger.info(f"   Total models processed: {total_attempts}")
            self.logger.info(f"   Successfully retrained: {retrain_results['retrained_models']}")
            self.logger.info(f"   Failed retraining: {retrain_results['failed_models']}")
            self.logger.info(f"   Success rate: {success_rate:.1f}%")
            
            if retrain_results['failed_models'] > 0:
                retrain_results['success'] = retrain_results['retrained_models'] > 0  # Partial success if at least one succeeded
            
            return retrain_results
            
        except Exception as e:
            error_msg = f"Critical error in ML retraining process: {e}"
            self.logger.error(error_msg)
            retrain_results['success'] = False
            retrain_results['errors'].append(error_msg)
            return retrain_results
    
    def export_ml_training_data(self, symbols: List[str] = None) -> Dict[str, any]:
        """
        Export longer-term historical data specifically for ML training.
        
        Args:
            symbols: List of symbols to export (if None, uses active symbols)
            
        Returns:
            Dictionary with export results
        """
        self.logger.info("Starting ML training data export")
        
        try:
            if symbols is None:
                symbols = self.get_active_symbols()
            
            if not symbols:
                self.logger.warning("No symbols found for ML training data export")
                return {'success': False, 'error': 'No symbols configured'}
            
            # For ML training, we need 12 months of hourly data for robust training
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)  # 12 months of hourly data
            
            self.logger.info(f"Exporting ML training data for {len(symbols)} symbols")
            self.logger.info(f"Date range (12 months): {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            
            # Export data with larger timeframe
            results = self.exporter.export_multiple_symbols(
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                format='csv'
            )
            
            # Log results
            summary = results.get('summary', {})
            self.logger.info(f"ML training data export completed - Success: {summary.get('successful_exports', 0)}, "
                           f"Failed: {summary.get('failed_exports', 0)}")
            
            return {'success': True, 'results': results}
            
        except Exception as e:
            self.logger.error(f"Error during ML training data export: {e}")
            return {'success': False, 'error': str(e)}

def main():
    """Main function for running the scheduled data manager."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Scheduled Historical Data Manager")
    parser.add_argument('--manual', action='store_true', help='Run manual job instead of starting scheduler')
    parser.add_argument('--cleanup-only', action='store_true', help='Run cleanup only')
    parser.add_argument('--export-only', action='store_true', help='Run export only')
    parser.add_argument('--export-ml-data', action='store_true', help='Export 12-month ML training data')
    parser.add_argument('--retrain-only', action='store_true', help='Run ML retraining only')
    parser.add_argument('--no-ml-retrain', action='store_true', help='Disable ML retraining')
    parser.add_argument('--status', action='store_true', help='Show current status')
    parser.add_argument('--output-dir', default='historical_exports', help='Output directory')
    
    args = parser.parse_args()
    
    # Initialize manager
    manager = ScheduledDataManager(output_dir=args.output_dir)
    
    # Disable ML retraining if requested
    if args.no_ml_retrain:
        manager.enable_ml_retraining = False
        print("🚫 ML retraining disabled")
    
    if args.status:
        # Show status
        status = manager.get_status()
        print("\n📊 Scheduled Data Manager Status")
        print("=" * 50)
        for key, value in status.items():
            print(f"{key:20}: {value}")
        
    elif args.cleanup_only:
        # Run cleanup only
        print("🧹 Running cleanup only...")
        result = manager.cleanup_old_files()
        print(f"Cleanup completed: {result}")
        
    elif args.export_only:
        # Run export only
        print("📥 Running export only...")
        result = manager.export_historical_data()
        print(f"Export completed: {result.get('success', False)}")
    
    elif args.export_ml_data:
        # Export 12-month ML training data
        print("📊 Exporting 12-month ML training data...")
        result = manager.export_ml_training_data()
        print(f"ML training data export completed: {result.get('success', False)}")
    
    elif args.retrain_only:
        # Run ML retraining only
        print("🤖 Running ML retraining only...")
        if not manager.enable_ml_retraining:
            print("❌ ML retraining is disabled")
        else:
            active_symbols = manager.get_active_symbols()
            
            if active_symbols:
                result = manager.retrain_ml_models(active_symbols)
                print(f"ML retraining completed: {result}")
            else:
                print("No active symbols found for retraining")
        
    elif args.manual:
        # Run manual job
        manager.run_manual_job()
        
    else:
        # Start scheduler service
        manager.start_scheduler()


if __name__ == "__main__":
    main()
