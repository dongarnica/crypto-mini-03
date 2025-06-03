#!/usr/bin/env python3
"""
Cleanup Binary Models Script
============================

Removes old binary models to prevent confusion and force 3-class model usage.
This ensures the trading engine only uses profitable 3-class models.

Author: Crypto Trading Strategy Engine  
Date: June 2, 2025
"""

import os
import glob

def cleanup_binary_models():
    """Remove binary model files to force 3-class usage."""
    
    # Directories to check
    directories = [
        '/workspaces/crypto-mini-03/ml_results',
        '/workspaces/crypto-mini-03/ml_results/models'
    ]
    
    removed_count = 0
    
    for directory in directories:
        if not os.path.exists(directory):
            continue
            
        # Find binary model files
        binary_patterns = [
            f"{directory}/*_binary_*.h5",
            f"{directory}/*_binary_*.keras",
            f"{directory}/*_improved_model.keras"
        ]
        
        for pattern in binary_patterns:
            files = glob.glob(pattern)
            for file_path in files:
                try:
                    print(f"🗑️  Removing binary model: {os.path.basename(file_path)}")
                    os.remove(file_path)
                    removed_count += 1
                except Exception as e:
                    print(f"❌ Failed to remove {file_path}: {e}")
    
    print(f"\n✅ Cleanup complete: Removed {removed_count} binary model files")
    print("🎯 Only 3-class enhanced models will be used going forward")
    
    # List remaining 3-class models
    remaining_3class = []
    for directory in directories:
        if os.path.exists(directory):
            files = glob.glob(f"{directory}/*_3class_enhanced.h5")
            remaining_3class.extend(files)
    
    if remaining_3class:
        print(f"\n📊 Found {len(remaining_3class)} 3-class models:")
        for model_path in remaining_3class:
            print(f"   ✅ {os.path.basename(model_path)}")
    else:
        print("\n⚠️  No 3-class models found - they will be trained on first run")

if __name__ == "__main__":
    print("🧹 Cleaning up binary models to enforce 3-class model usage...")
    print("=" * 60)
    cleanup_binary_models()
