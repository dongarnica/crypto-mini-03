#!/usr/bin/env python3
"""
Historical Data File Manager
============================

Utility for managing historical data export files.
Provides cleanup, organization, and maintenance functions.

Features:
- Clean up old/duplicate files
- Organize files by symbol and date
- Generate file inventory reports
- Validate file integrity
- Archive old data

Author: Crypto Trading System
Date: June 2025
"""

import os
import json
import glob
import shutil
import hashlib
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import pandas as pd


class HistoricalDataFileManager:
    """Manages historical data export files."""
    
    def __init__(self, data_dir: str = "historical_exports"):
        """
        Initialize the file manager.
        
        Args:
            data_dir: Directory containing historical data files
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # File patterns
        self.file_pattern = "*_1year_hourly_*.csv"
        self.timestamp_format = "%Y%m%d_%H%M%S"
        
        self.logger.info(f"HistoricalDataFileManager initialized for {self.data_dir}")
    
    def setup_logging(self):
        """Setup logging for file operations."""
        log_dir = self.data_dir / 'logs'
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / 'file_manager.log'
        
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        self.logger.propagate = False
    
    def get_all_files(self) -> List[Path]:
        """Get all historical data files."""
        files = list(self.data_dir.glob(self.file_pattern))
        self.logger.debug(f"Found {len(files)} historical data files")
        return files
    
    def parse_filename(self, filepath: Path) -> Dict[str, str]:
        """
        Parse information from filename.
        
        Args:
            filepath: Path to the file
            
        Returns:
            Dictionary with parsed information
        """
        try:
            filename = filepath.stem  # Remove .csv extension
            
            # Expected format: SYMBOL_1year_hourly_YYYYMMDD_HHMMSS
            parts = filename.split('_')
            
            if len(parts) >= 5:
                symbol = parts[0]
                timestamp_str = '_'.join(parts[-2:])  # Last two parts are date and time
                
                # Parse timestamp
                try:
                    timestamp = datetime.strptime(timestamp_str, self.timestamp_format)
                except ValueError:
                    timestamp = None
                
                return {
                    'symbol': symbol,
                    'timestamp_str': timestamp_str,
                    'timestamp': timestamp,
                    'file_path': str(filepath),
                    'file_size': filepath.stat().st_size,
                    'file_size_mb': filepath.stat().st_size / (1024 * 1024),
                    'modification_time': datetime.fromtimestamp(filepath.stat().st_mtime)
                }
            else:
                self.logger.warning(f"Could not parse filename: {filename}")
                return {
                    'symbol': 'UNKNOWN',
                    'timestamp_str': '',
                    'timestamp': None,
                    'file_path': str(filepath),
                    'file_size': filepath.stat().st_size,
                    'file_size_mb': filepath.stat().st_size / (1024 * 1024),
                    'modification_time': datetime.fromtimestamp(filepath.stat().st_mtime)
                }
                
        except Exception as e:
            self.logger.error(f"Error parsing filename {filepath}: {e}")
            return {
                'symbol': 'ERROR',
                'timestamp_str': '',
                'timestamp': None,
                'file_path': str(filepath),
                'file_size': 0,
                'file_size_mb': 0,
                'modification_time': None
            }
    
    def get_files_by_symbol(self) -> Dict[str, List[Dict]]:
        """
        Group files by symbol.
        
        Returns:
            Dictionary with symbol as key and list of file info as value
        """
        files_by_symbol = {}
        
        for filepath in self.get_all_files():
            file_info = self.parse_filename(filepath)
            symbol = file_info['symbol']
            
            if symbol not in files_by_symbol:
                files_by_symbol[symbol] = []
            
            files_by_symbol[symbol].append(file_info)
        
        # Sort files by timestamp for each symbol (newest first)
        for symbol in files_by_symbol:
            files_by_symbol[symbol].sort(
                key=lambda x: x['timestamp'] or datetime.min, 
                reverse=True
            )
        
        return files_by_symbol
    
    def find_duplicate_files(self) -> List[Dict]:
        """
        Find potential duplicate files.
        
        Returns:
            List of dictionaries describing duplicate file groups
        """
        self.logger.info("Searching for duplicate files...")
        
        files_by_symbol = self.get_files_by_symbol()
        duplicates = []
        
        for symbol, files in files_by_symbol.items():
            if len(files) > 1:
                # Group by file size first (quick check)
                size_groups = {}
                for file_info in files:
                    size = file_info['file_size']
                    if size not in size_groups:
                        size_groups[size] = []
                    size_groups[size].append(file_info)
                
                # Check for duplicates within same size groups
                for size, size_files in size_groups.items():
                    if len(size_files) > 1:
                        # Check if files are actually duplicates by comparing content hash
                        hash_groups = {}
                        for file_info in size_files:
                            try:
                                file_hash = self.calculate_file_hash(Path(file_info['file_path']))
                                if file_hash not in hash_groups:
                                    hash_groups[file_hash] = []
                                hash_groups[file_hash].append(file_info)
                            except Exception as e:
                                self.logger.error(f"Error calculating hash for {file_info['file_path']}: {e}")
                        
                        # Report duplicate groups
                        for file_hash, hash_files in hash_groups.items():
                            if len(hash_files) > 1:
                                duplicates.append({
                                    'symbol': symbol,
                                    'file_hash': file_hash,
                                    'file_count': len(hash_files),
                                    'files': hash_files,
                                    'total_size_mb': sum(f['file_size_mb'] for f in hash_files)
                                })
        
        self.logger.info(f"Found {len(duplicates)} duplicate file groups")
        return duplicates
    
    def calculate_file_hash(self, filepath: Path) -> str:
        """Calculate MD5 hash of file content."""
        hash_md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            # Read in chunks to handle large files
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def clean_old_files(self, keep_latest: int = 1, max_age_days: int = 7) -> Dict[str, int]:
        """
        Clean up old files, keeping only the latest N files per symbol.
        
        Args:
            keep_latest: Number of latest files to keep per symbol
            max_age_days: Maximum age of files to keep (in days)
            
        Returns:
            Dictionary with cleanup statistics
        """
        self.logger.info(f"Starting cleanup: keep_latest={keep_latest}, max_age_days={max_age_days}")
        
        stats = {
            'symbols_processed': 0,
            'files_deleted': 0,
            'space_freed_mb': 0,
            'files_kept': 0
        }
        
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        files_by_symbol = self.get_files_by_symbol()
        
        for symbol, files in files_by_symbol.items():
            stats['symbols_processed'] += 1
            self.logger.info(f"Processing {symbol}: {len(files)} files found")
            
            # Separate files to keep and delete
            files_to_keep = []
            files_to_delete = []
            
            # Sort by timestamp (newest first)
            sorted_files = sorted(
                files, 
                key=lambda x: x['timestamp'] or datetime.min, 
                reverse=True
            )
            
            for i, file_info in enumerate(sorted_files):
                # Keep if within the latest N files
                if i < keep_latest:
                    files_to_keep.append(file_info)
                    continue
                
                # Also keep if file is recent (within max_age_days)
                file_timestamp = file_info['timestamp']
                if file_timestamp and file_timestamp > cutoff_date:
                    files_to_keep.append(file_info)
                    continue
                
                # Mark for deletion
                files_to_delete.append(file_info)
            
            # Delete old files
            for file_info in files_to_delete:
                try:
                    filepath = Path(file_info['file_path'])
                    if filepath.exists():
                        filepath.unlink()
                        stats['files_deleted'] += 1
                        stats['space_freed_mb'] += file_info['file_size_mb']
                        self.logger.info(f"Deleted: {filepath.name} ({file_info['file_size_mb']:.2f} MB)")
                except Exception as e:
                    self.logger.error(f"Error deleting {file_info['file_path']}: {e}")
            
            stats['files_kept'] += len(files_to_keep)
            self.logger.info(f"{symbol}: Kept {len(files_to_keep)} files, deleted {len(files_to_delete)} files")
        
        self.logger.info(f"Cleanup completed: {stats}")
        return stats
    
    def remove_duplicates(self, dry_run: bool = True) -> Dict[str, int]:
        """
        Remove duplicate files, keeping the newest version.
        
        Args:
            dry_run: If True, only report what would be deleted
            
        Returns:
            Dictionary with removal statistics
        """
        self.logger.info(f"Removing duplicates (dry_run={dry_run})")
        
        stats = {
            'duplicate_groups': 0,
            'files_removed': 0,
            'space_freed_mb': 0
        }
        
        duplicates = self.find_duplicate_files()
        
        for duplicate_group in duplicates:
            stats['duplicate_groups'] += 1
            files = duplicate_group['files']
            
            # Keep the newest file (first in the sorted list)
            files_sorted = sorted(
                files, 
                key=lambda x: x['timestamp'] or datetime.min, 
                reverse=True
            )
            
            keep_file = files_sorted[0]
            delete_files = files_sorted[1:]
            
            self.logger.info(f"Duplicate group for {duplicate_group['symbol']}:")
            self.logger.info(f"  Keeping: {Path(keep_file['file_path']).name}")
            
            for file_info in delete_files:
                filepath = Path(file_info['file_path'])
                self.logger.info(f"  {'Would delete' if dry_run else 'Deleting'}: {filepath.name}")
                
                if not dry_run:
                    try:
                        if filepath.exists():
                            filepath.unlink()
                            stats['files_removed'] += 1
                            stats['space_freed_mb'] += file_info['file_size_mb']
                    except Exception as e:
                        self.logger.error(f"Error deleting {filepath}: {e}")
                else:
                    stats['files_removed'] += 1
                    stats['space_freed_mb'] += file_info['file_size_mb']
        
        self.logger.info(f"Duplicate removal completed: {stats}")
        return stats
    
    def validate_files(self) -> Dict[str, List]:
        """
        Validate file integrity and content.
        
        Returns:
            Dictionary with validation results
        """
        self.logger.info("Validating files...")
        
        results = {
            'valid_files': [],
            'invalid_files': [],
            'empty_files': [],
            'corrupted_files': []
        }
        
        for filepath in self.get_all_files():
            try:
                file_info = self.parse_filename(filepath)
                
                # Check if file is empty
                if file_info['file_size'] == 0:
                    results['empty_files'].append(file_info)
                    continue
                
                # Try to read as CSV
                try:
                    df = pd.read_csv(filepath, nrows=5)  # Read first 5 rows to test
                    
                    # Check if required columns exist
                    required_columns = ['open_time', 'open', 'high', 'low', 'close', 'volume']
                    missing_columns = [col for col in required_columns if col not in df.columns]
                    
                    if missing_columns:
                        file_info['error'] = f"Missing columns: {missing_columns}"
                        results['invalid_files'].append(file_info)
                    else:
                        results['valid_files'].append(file_info)
                        
                except Exception as e:
                    file_info['error'] = f"CSV read error: {str(e)}"
                    results['corrupted_files'].append(file_info)
                
            except Exception as e:
                file_info = {'file_path': str(filepath), 'error': f"General error: {str(e)}"}
                results['corrupted_files'].append(file_info)
        
        # Log summary
        for category, files in results.items():
            self.logger.info(f"{category}: {len(files)} files")
        
        return results
    
    def generate_inventory_report(self) -> Dict:
        """
        Generate a comprehensive inventory report.
        
        Returns:
            Dictionary with inventory information
        """
        self.logger.info("Generating inventory report...")
        
        files_by_symbol = self.get_files_by_symbol()
        
        # Calculate statistics
        total_files = sum(len(files) for files in files_by_symbol.values())
        total_size_mb = sum(
            sum(f['file_size_mb'] for f in files) 
            for files in files_by_symbol.values()
        )
        
        # Symbol statistics
        symbol_stats = {}
        for symbol, files in files_by_symbol.items():
            if files:
                symbol_stats[symbol] = {
                    'file_count': len(files),
                    'total_size_mb': sum(f['file_size_mb'] for f in files),
                    'latest_file': files[0]['file_path'] if files else None,
                    'latest_timestamp': files[0]['timestamp'].isoformat() if files and files[0]['timestamp'] else None,
                    'oldest_file': files[-1]['file_path'] if files else None,
                    'oldest_timestamp': files[-1]['timestamp'].isoformat() if files and files[-1]['timestamp'] else None
                }
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'data_directory': str(self.data_dir),
            'summary': {
                'total_files': total_files,
                'total_symbols': len(files_by_symbol),
                'total_size_mb': round(total_size_mb, 2),
                'total_size_gb': round(total_size_mb / 1024, 2)
            },
            'symbols': symbol_stats
        }
        
        # Save report
        report_file = self.data_dir / f'inventory_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Inventory report saved: {report_file}")
        return report
    
    def archive_old_data(self, archive_dir: str = None, max_age_days: int = 30) -> Dict[str, int]:
        """
        Archive old data files to a separate directory.
        
        Args:
            archive_dir: Directory to archive old files (defaults to data_dir/archive)
            max_age_days: Maximum age of files to keep in main directory
            
        Returns:
            Dictionary with archive statistics
        """
        if archive_dir is None:
            archive_dir = self.data_dir / 'archive'
        else:
            archive_dir = Path(archive_dir)
        
        archive_dir.mkdir(exist_ok=True)
        
        self.logger.info(f"Archiving files older than {max_age_days} days to {archive_dir}")
        
        stats = {
            'files_archived': 0,
            'space_archived_mb': 0
        }
        
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        
        for filepath in self.get_all_files():
            file_info = self.parse_filename(filepath)
            
            # Check if file should be archived
            should_archive = False
            
            if file_info['timestamp'] and file_info['timestamp'] < cutoff_date:
                should_archive = True
            elif file_info['modification_time'] and file_info['modification_time'] < cutoff_date:
                should_archive = True
            
            if should_archive:
                try:
                    # Create archive subdirectory for symbol
                    symbol_archive_dir = archive_dir / file_info['symbol']
                    symbol_archive_dir.mkdir(exist_ok=True)
                    
                    # Move file to archive
                    archive_path = symbol_archive_dir / filepath.name
                    shutil.move(str(filepath), str(archive_path))
                    
                    stats['files_archived'] += 1
                    stats['space_archived_mb'] += file_info['file_size_mb']
                    
                    self.logger.info(f"Archived: {filepath.name} -> {archive_path}")
                    
                except Exception as e:
                    self.logger.error(f"Error archiving {filepath}: {e}")
        
        self.logger.info(f"Archive completed: {stats}")
        return stats


def main():
    """Main function for the file manager."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Historical Data File Manager")
    parser.add_argument('--data-dir', default='historical_exports', help='Data directory')
    parser.add_argument('--clean', action='store_true', help='Clean old files')
    parser.add_argument('--keep-latest', type=int, default=1, help='Number of latest files to keep per symbol')
    parser.add_argument('--max-age-days', type=int, default=7, help='Maximum age of files to keep')
    parser.add_argument('--remove-duplicates', action='store_true', help='Remove duplicate files')
    parser.add_argument('--dry-run', action='store_true', help='Dry run mode (report only)')
    parser.add_argument('--validate', action='store_true', help='Validate file integrity')
    parser.add_argument('--inventory', action='store_true', help='Generate inventory report')
    parser.add_argument('--archive', action='store_true', help='Archive old files')
    parser.add_argument('--archive-days', type=int, default=30, help='Archive files older than N days')
    
    args = parser.parse_args()
    
    # Initialize file manager
    manager = HistoricalDataFileManager(data_dir=args.data_dir)
    
    if args.clean:
        print(f"🧹 Cleaning old files (keep latest {args.keep_latest}, max age {args.max_age_days} days)")
        result = manager.clean_old_files(keep_latest=args.keep_latest, max_age_days=args.max_age_days)
        print(f"Cleanup result: {result}")
    
    if args.remove_duplicates:
        print(f"🔄 Removing duplicates (dry_run={args.dry_run})")
        result = manager.remove_duplicates(dry_run=args.dry_run)
        print(f"Duplicate removal result: {result}")
    
    if args.validate:
        print("✅ Validating files")
        result = manager.validate_files()
        print(f"Validation result:")
        for category, files in result.items():
            print(f"  {category}: {len(files)} files")
    
    if args.inventory:
        print("📊 Generating inventory report")
        report = manager.generate_inventory_report()
        print(f"Inventory summary: {report['summary']}")
    
    if args.archive:
        print(f"📦 Archiving files older than {args.archive_days} days")
        result = manager.archive_old_data(max_age_days=args.archive_days)
        print(f"Archive result: {result}")
    
    if not any([args.clean, args.remove_duplicates, args.validate, args.inventory, args.archive]):
        print("📋 Current status:")
        files_by_symbol = manager.get_files_by_symbol()
        total_files = sum(len(files) for files in files_by_symbol.values())
        total_size = sum(sum(f['file_size_mb'] for f in files) for files in files_by_symbol.values())
        
        print(f"  Total files: {total_files}")
        print(f"  Total symbols: {len(files_by_symbol)}")
        print(f"  Total size: {total_size:.2f} MB")
        print(f"  Data directory: {manager.data_dir}")


if __name__ == "__main__":
    main()
