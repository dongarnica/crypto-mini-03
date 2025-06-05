#!/usr/bin/env python3
"""
Demonstration of Scheduled Data Management System
================================================

This script demonstrates all the capabilities of the scheduled data management system.
"""

import os
import sys
from datetime import datetime
from pathlib import Path

def main():
    """Demonstrate the scheduled data management system."""
    print("🚀 SCHEDULED DATA MANAGEMENT SYSTEM DEMONSTRATION")
    print("=" * 70)
    
    # Import the components
    from scheduled_data_manager import ScheduledDataManager
    from file_manager import HistoricalDataFileManager
    from data_export_service import DataExportService
    
    # Initialize the manager
    print("\n1️⃣  INITIALIZING SYSTEM")
    print("-" * 30)
    manager = ScheduledDataManager()
    file_manager = HistoricalDataFileManager('historical_exports')
    print("✅ All components initialized successfully")
    
    # Get comprehensive status
    print("\n2️⃣  SYSTEM STATUS REPORT")
    print("-" * 30)
    status = manager.get_status_report()
    
    print(f"📊 System Statistics:")
    print(f"  • Total Files: {status['file_statistics']['total_files']}")
    print(f"  • Symbols: {status['file_statistics']['symbols_count']}")
    print(f"  • Total Size: {status['file_statistics']['total_size_mb']:.2f} MB")
    print(f"  • Export Interval: {status['configuration']['export_interval_hours']} hours")
    print(f"  • Data Retention: {status['configuration']['data_retention_days']} days")
    
    if status['file_statistics']['newest_file']:
        print(f"  • Newest File: {status['file_statistics']['newest_file']}")
        print(f"  • Oldest File: {status['file_statistics']['oldest_file']}")
    
    # Show symbols with data
    print(f"\n📈 Symbols with Historical Data:")
    symbols_detail = status.get('symbols_detail', {})
    for i, (symbol, files) in enumerate(list(symbols_detail.items())[:10], 1):
        latest_file = files[0] if files else {}
        size_mb = latest_file.get('size_mb', 0)
        print(f"  {i:2d}. {symbol:<12} {len(files)} file(s) ({size_mb:.2f} MB)")
    
    if len(symbols_detail) > 10:
        print(f"  ... and {len(symbols_detail) - 10} more symbols")
    
    # Demonstrate cleanup functionality
    print("\n3️⃣  FILE CLEANUP DEMONSTRATION")
    print("-" * 30)
    
    # Show files before cleanup
    files_by_symbol = file_manager.get_files_by_symbol()
    print(f"📂 Files by symbol before cleanup:")
    symbols_with_multiple = 0
    for symbol, files in files_by_symbol.items():
        if len(files) > 1:
            symbols_with_multiple += 1
            print(f"  • {symbol}: {len(files)} files")
    
    if symbols_with_multiple == 0:
        print("  ✅ All symbols already have optimal file count")
    else:
        print(f"  ⚠️  {symbols_with_multiple} symbols have multiple files")
        
        # Run cleanup
        print("\n🧹 Running cleanup...")
        cleanup_result = manager.cleanup_old_files()
        print(f"✅ Cleanup completed:")
        print(f"  • Symbols processed: {cleanup_result['symbols_processed']}")
        print(f"  • Files deleted: {cleanup_result['files_deleted']}")
        print(f"  • Space freed: {cleanup_result['space_freed_mb']:.2f} MB")
        print(f"  • Errors: {cleanup_result['errors']}")
    
    # Demonstrate active symbols detection
    print("\n4️⃣  ACTIVE SYMBOLS CONFIGURATION")
    print("-" * 30)
    active_symbols = manager.get_active_symbols()
    print(f"🎯 Active symbols for export ({len(active_symbols)} total):")
    for i, symbol in enumerate(active_symbols[:10], 1):
        print(f"  {i:2d}. {symbol}")
    
    if len(active_symbols) > 10:
        print(f"  ... and {len(active_symbols) - 10} more symbols")
    
    # Show health check
    print("\n5️⃣  SYSTEM HEALTH CHECK")
    print("-" * 30)
    health = status.get('health_check', {})
    health_items = [
        ('Logs Directory', health.get('logs_directory_exists', False)),
        ('Exporter Initialized', health.get('exporter_initialized', False)),
        ('Symbols Config Loaded', health.get('symbols_config_loaded', False)),
        ('Recent Activity', health.get('recent_activity', False)),
    ]
    
    for item_name, item_status in health_items:
        status_icon = "✅" if item_status else "❌"
        print(f"  {status_icon} {item_name}")
    
    # Show file management capabilities
    print("\n6️⃣  FILE MANAGEMENT CAPABILITIES")
    print("-" * 30)
    
    # Test file validation
    try:
        all_files = file_manager.get_all_files()
        print(f"📁 File Management Features:")
        print(f"  • Total files discovered: {len(all_files)}")
        
        # Test duplicate detection
        duplicates = file_manager.find_duplicate_files()
        if duplicates:
            print(f"  • Duplicate groups found: {len(duplicates)}")
            total_duplicate_size = sum(dup['total_size_mb'] for dup in duplicates)
            print(f"  • Space used by duplicates: {total_duplicate_size:.2f} MB")
        else:
            print(f"  • No duplicate files detected ✅")
        
        # Test file validation
        validation_results = file_manager.validate_files()
        valid_files = validation_results.get('valid_files', 0)
        invalid_files = validation_results.get('invalid_files', 0)
        print(f"  • Valid files: {valid_files}")
        print(f"  • Invalid files: {invalid_files}")
        
    except Exception as e:
        print(f"  ⚠️  File management test error: {e}")
    
    # Show service wrapper capabilities
    print("\n7️⃣  SERVICE DEPLOYMENT OPTIONS")
    print("-" * 30)
    service = DataExportService()
    print("🔧 Available deployment methods:")
    print("  • Background daemon service")
    print("  • Systemd service integration")  
    print("  • Cron job scheduling")
    print("  • Manual execution")
    print("  • Docker container deployment")
    
    # Show usage commands
    print("\n8️⃣  USAGE COMMANDS")
    print("-" * 30)
    print("📋 How to use the scheduled data manager:")
    print()
    print("# Show current status:")
    print("python3 scheduled_data_manager.py --status")
    print()
    print("# Run manual job (export + cleanup):")
    print("python3 scheduled_data_manager.py --manual")
    print()
    print("# Run cleanup only:")
    print("python3 scheduled_data_manager.py --cleanup-only")
    print()
    print("# Run export only:")
    print("python3 scheduled_data_manager.py --export-only")
    print()
    print("# Start the scheduler service (runs every 2 hours):")
    print("python3 scheduled_data_manager.py")
    print()
    print("# Start as background service:")
    print("python3 data_export_service.py start")
    print()
    print("# Stop background service:")
    print("python3 data_export_service.py stop")
    print()
    print("# Check service status:")
    print("python3 data_export_service.py status")
    
    # Configuration summary
    print("\n9️⃣  SYSTEM CONFIGURATION")
    print("-" * 30)
    config = status.get('configuration', {})
    print("⚙️  Current configuration:")
    print(f"  • Export Interval: Every {config.get('export_interval_hours', 2)} hours")
    print(f"  • Data Retention: {config.get('data_retention_days', 7)} days")
    print(f"  • Max Files per Symbol: {config.get('max_files_per_symbol', 1)}")
    print(f"  • Keep Latest Only: {config.get('keep_latest_only', True)}")
    print(f"  • Output Directory: {config.get('output_directory', 'historical_exports')}")
    
    print("\n✨ SUMMARY")
    print("-" * 30)
    print("🎯 The Scheduled Data Management System provides:")
    print("   ✅ Automated data exports every 2 hours")
    print("   ✅ Automatic cleanup of outdated files")
    print("   ✅ Comprehensive logging and monitoring")
    print("   ✅ Health checks and status reporting")
    print("   ✅ Service deployment options")
    print("   ✅ File management and validation")
    print("   ✅ Duplicate detection and removal")
    print("   ✅ Configurable retention policies")
    
    print(f"\n🏁 Demonstration completed successfully!")
    print(f"💾 Check logs in: {manager.output_dir}/logs/")
    print(f"📊 Status reports saved to: {manager.output_dir}/status_report.json")

if __name__ == "__main__":
    main()
