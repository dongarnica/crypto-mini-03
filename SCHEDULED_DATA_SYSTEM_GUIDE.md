# Scheduled Data Management System
# Complete Setup and Usage Guide

## 🎯 Overview

The Scheduled Data Management System provides automated retrieval and maintenance of historical cryptocurrency data exports with the following features:

- **Automated Exports**: Runs every 2 hours to export latest data
- **Smart Cleanup**: Automatically removes outdated files (keeps only latest per symbol)
- **Comprehensive Logging**: All operations logged with rotation
- **Health Monitoring**: Status reports and health checks
- **Service Integration**: Multiple deployment options

## 📁 System Components

### Core Files
- `scheduled_data_manager.py` - Main scheduler with export and cleanup logic
- `data_export_service.py` - Service wrapper for daemon operation
- `file_manager.py` - File management and cleanup utilities
- `export_historical_data.py` - Historical data exporter (existing)
- `config/symbols_config.py` - Symbol configuration (existing)

### Supporting Files
- `demo_scheduled_system.py` - Demonstration script
- `test_scheduled_system.py` - Testing utilities

## 🚀 Quick Start

### 1. Basic Usage Commands

```bash
# Show current status
python3 scheduled_data_manager.py --status

# Run manual job (export + cleanup)
python3 scheduled_data_manager.py --manual

# Run cleanup only
python3 scheduled_data_manager.py --cleanup-only

# Run export only
python3 scheduled_data_manager.py --export-only

# Start the scheduler service (runs every 2 hours)
python3 scheduled_data_manager.py
```

### 2. Service Management

```bash
# Start as background service
python3 data_export_service.py start

# Stop background service
python3 data_export_service.py stop

# Check service status
python3 data_export_service.py status

# Restart service
python3 data_export_service.py restart
```

### 3. System Demonstration

```bash
# Run comprehensive system demonstration
python3 demo_scheduled_system.py
```

## ⚙️ Configuration

### Default Settings
- **Export Interval**: Every 2 hours
- **Data Retention**: 7 days maximum
- **Files per Symbol**: 1 (keeps only latest)
- **Export Format**: CSV
- **Data Range**: Last 2 days on each export
- **Output Directory**: `historical_exports/`

### Customization
You can modify these settings in the `ScheduledDataManager.__init__()` method:

```python
self.export_period_hours = 2  # Export every 2 hours
self.max_files_per_symbol = 1  # Maximum files to keep per symbol
self.data_retention_days = 7  # Keep data for 7 days max
```

## 📊 Monitoring and Logs

### Log Files
- **Main Log**: `historical_exports/logs/scheduled_data_manager.log`
- **Service Log**: `data_export_service.log`
- **Export Logs**: `historical_exports/export_log.log`

### Status Reports
- **JSON Report**: `historical_exports/status_report.json`
- **Console Status**: Use `--status` flag

### Log Rotation
- Automatic log rotation (10MB max, 5 backup files)
- Comprehensive logging with timestamps and function names

## 🔄 Automated Scheduling

### Schedule Library
The system uses the `schedule` library for timing:
```python
schedule.every(2).hours.do(self.run_scheduled_job)
```

### What Happens Every 2 Hours
1. **Export Latest Data**: Downloads last 2 days of data for all configured symbols
2. **Cleanup Old Files**: Removes outdated files per retention policy
3. **Generate Status Report**: Creates comprehensive status report
4. **Log Operations**: Records all activities with detailed logging

## 🗂️ File Management Features

### Automatic Cleanup
- Keeps only the latest file per symbol
- Removes files older than retention period
- Frees disk space automatically
- Logs all cleanup operations

### File Validation
- Validates CSV file integrity
- Checks for missing or corrupted data
- Reports validation results

### Duplicate Detection
- Identifies duplicate files by content hash
- Reports potential space savings
- Prevents data redundancy

## 🏗️ Deployment Options

### 1. Manual Execution
```bash
# Run once manually
python3 scheduled_data_manager.py --manual
```

### 2. Background Service
```bash
# Start persistent background service
python3 data_export_service.py start
```

### 3. Systemd Service (Linux)
```bash
# Create systemd service
python3 data_export_service.py create-systemd-service

# Enable and start
sudo systemctl enable crypto-data-export
sudo systemctl start crypto-data-export
```

### 4. Cron Job
```bash
# Add to crontab for every 2 hours
0 */2 * * * cd /workspaces/crypto-mini-03 && python3 scheduled_data_manager.py --manual
```

### 5. Docker Container
```bash
# Run in Docker (using existing Dockerfile)
docker run -d --name crypto-scheduler \
  -v $(pwd)/historical_exports:/app/historical_exports \
  -v $(pwd)/.env:/app/.env \
  crypto-trading-system python3 scheduled_data_manager.py
```

## 📈 Symbol Configuration

The system automatically loads symbols from your environment configuration:
- **Primary Symbols**: BTCUSDT, ETHUSDT, SOLUSDT
- **DeFi Symbols**: UNIUSDT, LINKUSDT, DOTUSDT, AVAXUSDT, AAVEUSDT, etc.
- **Altcoins**: DOGEUSDT, SHIBUSDT, XRPUSDT
- **Additional**: From CRYPTO_SYMBOLS environment variable

## 🔍 Health Monitoring

### Health Checks
- Logs directory exists
- Exporter properly initialized
- Symbols configuration loaded
- Recent activity detected

### Status Information
- Total files and symbols
- Storage usage (MB/GB)
- Last export/cleanup times
- File age and validity
- System configuration

## 🛠️ Troubleshooting

### Common Issues

1. **No symbols configured**
   - Check `.env` file for CRYPTO_SYMBOLS
   - Verify symbols_config.py loads correctly

2. **API rate limits**
   - System includes rate limiting delays
   - Adjust request_delay in exporter if needed

3. **Disk space issues**
   - Cleanup runs automatically
   - Adjust data_retention_days if needed

4. **Permission errors**
   - Ensure write permissions on historical_exports/
   - Check log directory permissions

### Debug Mode
Add debug logging:
```python
self.logger.setLevel(logging.DEBUG)
```

## 📝 Status Report Example

```json
{
  "timestamp": "2025-06-05T01:19:00",
  "system_status": {
    "service_status": "running",
    "export_count": 5,
    "cleanup_count": 3,
    "current_files": 12,
    "symbols_with_data": 12
  },
  "configuration": {
    "export_interval_hours": 2,
    "data_retention_days": 7,
    "max_files_per_symbol": 1
  },
  "file_statistics": {
    "total_files": 12,
    "symbols_count": 12,
    "total_size_mb": 145.67,
    "total_size_gb": 0.14
  }
}
```

## ✅ Verification

To verify the system is working correctly:

1. **Check Status**:
   ```bash
   python3 scheduled_data_manager.py --status
   ```

2. **Run Manual Test**:
   ```bash
   python3 scheduled_data_manager.py --manual
   ```

3. **Check Logs**:
   ```bash
   tail -f historical_exports/logs/scheduled_data_manager.log
   ```

4. **Verify Files**:
   ```bash
   ls -la historical_exports/*.csv
   ```

## 🎉 Success Indicators

- ✅ Status shows `service_status: running`
- ✅ Files present in `historical_exports/`
- ✅ Logs show successful exports and cleanups
- ✅ Status report generated in JSON format
- ✅ No errors in log files

---

**The system is now ready for production use with automated 2-hour scheduling!**
