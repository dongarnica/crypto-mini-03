#!/usr/bin/env python3
"""
Simple test script to run the ML pipeline
"""

import os
import sys
sys.path.append('/workspaces/crypto-mini-03')

from ml.ml_pipeline import run_complete_analysis

if __name__ == "__main__":
    print("Starting ML Pipeline Test...")
    
    # Use the BTCUSDT data
    csv_path = "/workspaces/crypto-mini-03/historical_exports/BTCUSDT_1year_hourly_20250601_025148.csv"
    
    # Run the analysis
    pipeline, history, results = run_complete_analysis(
        csv_path=csv_path,
        symbol='BTCUSDT',
        save_results=True
    )
    
    if pipeline is not None:
        print("✅ Pipeline completed successfully!")
    else:
        print("❌ Pipeline failed!")
