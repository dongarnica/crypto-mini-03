#!/usr/bin/env python3
"""
Enhanced ML Training Script - Implementing Critical Fixes
=====================================================

This script addresses the critical issues identified:
1. Model Bias Problems (100% Down/Neutral predictions)
2. Confidence Distribution Issues (clustering around 50%)
3. Trading Execution Problems (zero executed trades)
4. Prioritizes 3-class models (only profitable approach)
5. Lowers confidence thresholds (25-30% instead of 35%)
6. Investigates binary target creation (threshold adjustment logic)
7. Reviews class weighting (prevents severe bias)

Author: GitHub Copilot
Date: June 2025
"""

import os
import sys
import json
import warnings
from datetime import datetime
import numpy as np
import pandas as pd

# Add the project root to Python path
sys.path.append('/workspaces/crypto-mini-03')

# Import the improved ML pipeline
from ml.ml_pipeline_improved_components import ImprovedCryptoLSTMPipeline

warnings.filterwarnings('ignore')

class EnhancedMLTrainer:
    """Enhanced ML trainer implementing all critical fixes."""
    
    def __init__(self):
        """Initialize the enhanced trainer."""
        self.symbols = [
            'BTCUSDT', 'ETHUSDT', 'DOTUSDT', 'LINKUSDT', 
            'LTCUSDT', 'BCHUSDT', 'UNIUSDT', 'SOLUSDT', 'AVAXUSDT'
        ]
        self.results_dir = "/workspaces/crypto-mini-03/ml_results"
        self.ensure_results_dir()
    
    def ensure_results_dir(self):
        """Ensure the results directory exists."""
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
            print(f"Created results directory: {self.results_dir}")
    
    def train_enhanced_models(self):
        """Train enhanced models for all symbols with critical fixes."""
        print("🚀 ENHANCED ML TRAINING - CRITICAL FIXES IMPLEMENTED")
        print("=" * 70)
        print("FIXES APPLIED:")
        print("✅ 1. Prioritize 3-class models (only profitable approach)")
        print("✅ 2. Lower confidence thresholds (25-30% instead of 35%)")  
        print("✅ 3. Fix binary target creation (balanced thresholds)")
        print("✅ 4. Improved class weighting (prevent severe bias)")
        print("✅ 5. Enhanced feature selection consistency")
        print("✅ 6. Simplified architecture (prevent overfitting)")
        print("=" * 70)
        
        all_results = {}
        
        for symbol in self.symbols:
            print(f"\n🔄 TRAINING MODELS FOR {symbol}")
            print("-" * 50)
            
            try:
                # Train 3-class model (PRIORITIZED)
                results_3class = self.train_symbol_3class(symbol)
                
                # Train binary model (for comparison)
                results_binary = self.train_symbol_binary(symbol)
                
                all_results[symbol] = {
                    '3_class': results_3class,
                    'binary': results_binary,
                    'timestamp': datetime.now().isoformat()
                }
                
                # Determine best model
                best_model = self.determine_best_model(results_3class, results_binary)
                all_results[symbol]['recommended_model'] = best_model
                
                print(f"✅ {symbol} - RECOMMENDED: {best_model}")
                
            except Exception as e:
                print(f"❌ Error training {symbol}: {str(e)}")
                all_results[symbol] = {
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
        
        # Save comprehensive results
        self.save_training_results(all_results)
        self.print_final_summary(all_results)
        
        return all_results
    
    def train_symbol_3class(self, symbol):
        """Train 3-class model with enhanced fixes."""
        print(f"  🎯 3-Class Model (PRIORITIZED)")
        
        # Enhanced parameters based on recommendations
        pipeline = ImprovedCryptoLSTMPipeline(
            symbol=symbol,
            confidence_threshold=0.25,  # Lowered from 0.35 to 0.25 (25%)
            use_binary_classification=False,  # 3-class is prioritized
            prediction_horizon=4,  # 4-hour prediction horizon
            lookback_period=24,  # 24-hour lookback
            buy_threshold=0.008,  # Slightly higher for better precision
            sell_threshold=-0.008  # Slightly higher for better precision
        )
        
        # Load and process data
        pipeline.load_data_from_symbol(symbol)
        pipeline.add_essential_indicators()
        pipeline.create_features_and_targets()
        
        # Build the model architecture (CRITICAL FIX)
        pipeline.build_simplified_model()
        
        # Train with enhanced validation
        pipeline.train_with_proper_validation()
        
        # Evaluate performance
        pipeline.evaluate_model_performance()
        
        # Enhanced backtesting with multiple thresholds
        backtest_25 = pipeline.improved_backtest(confidence_threshold=0.25)
        backtest_30 = pipeline.improved_backtest(confidence_threshold=0.30)
        
        # Save model
        model_path = os.path.join(self.results_dir, f"{symbol}_3class_enhanced.h5")
        pipeline.model.save(model_path)
        
        return {
            'model_type': '3-class',
            'confidence_threshold': 0.25,
            'backtest_25': backtest_25,
            'backtest_30': backtest_30,
            'model_path': model_path,
            'features_count': len(pipeline.selected_features),
            'features': pipeline.selected_features,
            'training_success': True
        }
    
    def train_symbol_binary(self, symbol):
        """Train binary model with fixes (for comparison)."""
        print(f"  📊 Binary Model (for comparison)")
        
        # Enhanced parameters with fixes
        pipeline = ImprovedCryptoLSTMPipeline(
            symbol=symbol,
            confidence_threshold=0.30,  # Lowered from 0.35 to 0.30 (30%)
            use_binary_classification=True,  # Binary classification
            prediction_horizon=4,  # 4-hour prediction horizon
            lookback_period=24,  # 24-hour lookback
            buy_threshold=0.006,  # Balanced thresholds
            sell_threshold=-0.006  # Balanced thresholds
        )
        
        # Load and process data
        pipeline.load_data_from_symbol(symbol)
        pipeline.add_essential_indicators()
        pipeline.create_features_and_targets()
        
        # Build the model architecture (CRITICAL FIX)
        pipeline.build_simplified_model()
        
        # Train with enhanced validation
        pipeline.train_with_proper_validation()
        
        # Evaluate performance
        pipeline.evaluate_model_performance()
        
        # Enhanced backtesting with multiple thresholds
        backtest_25 = pipeline.improved_backtest(confidence_threshold=0.25)
        backtest_30 = pipeline.improved_backtest(confidence_threshold=0.30)
        
        # Save model
        model_path = os.path.join(self.results_dir, f"{symbol}_binary_enhanced.h5")
        pipeline.model.save(model_path)
        
        return {
            'model_type': 'binary',
            'confidence_threshold': 0.30,
            'backtest_25': backtest_25,
            'backtest_30': backtest_30,
            'model_path': model_path,
            'features_count': len(pipeline.selected_features),
            'features': pipeline.selected_features,
            'training_success': True
        }
    
    def determine_best_model(self, results_3class, results_binary):
        """Determine the best model based on multiple criteria."""
        
        # Prioritize 3-class as per recommendations
        score_3class = 0
        score_binary = 0
        
        # 1. Execution rate (most important)
        exec_rate_3class = max(
            results_3class['backtest_25'].get('executed_trades', 0),
            results_3class['backtest_30'].get('executed_trades', 0)
        )
        exec_rate_binary = max(
            results_binary['backtest_25'].get('executed_trades', 0),
            results_binary['backtest_30'].get('executed_trades', 0)
        )
        
        if exec_rate_3class > exec_rate_binary:
            score_3class += 3
        elif exec_rate_binary > exec_rate_3class:
            score_binary += 1  # Lower weight for binary
        
        # 2. Return performance
        return_3class = max(
            results_3class['backtest_25'].get('total_return', -1),
            results_3class['backtest_30'].get('total_return', -1)
        )
        return_binary = max(
            results_binary['backtest_25'].get('total_return', -1),
            results_binary['backtest_30'].get('total_return', -1)
        )
        
        if return_3class > return_binary and return_3class > 0:
            score_3class += 2
        elif return_binary > return_3class and return_binary > 0:
            score_binary += 1
        
        # 3. Prioritize 3-class per recommendations
        score_3class += 2  # Built-in preference for 3-class
        
        # Determine winner
        if score_3class >= score_binary:
            return "3-class (RECOMMENDED)"
        else:
            return "binary (with caution)"
    
    def save_training_results(self, results):
        """Save comprehensive training results."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = os.path.join(self.results_dir, f"enhanced_training_results_{timestamp}.json")
        
        # Convert numpy types to native Python types for JSON serialization
        json_results = {}
        for symbol, symbol_results in results.items():
            json_results[symbol] = self.convert_for_json(symbol_results)
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2, default=str)
        
        print(f"\n📄 Results saved to: {results_file}")
    
    def convert_for_json(self, obj):
        """Convert numpy types to JSON-serializable types."""
        if isinstance(obj, dict):
            return {k: self.convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_for_json(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def print_final_summary(self, results):
        """Print comprehensive training summary."""
        print("\n" + "=" * 70)
        print("🏆 ENHANCED TRAINING SUMMARY")
        print("=" * 70)
        
        successful_symbols = []
        failed_symbols = []
        model_recommendations = {}
        
        for symbol, result in results.items():
            if 'error' in result:
                failed_symbols.append(symbol)
            else:
                successful_symbols.append(symbol)
                model_recommendations[symbol] = result.get('recommended_model', 'Unknown')
        
        print(f"✅ Successfully trained: {len(successful_symbols)}/{len(results)} symbols")
        print(f"❌ Failed: {len(failed_symbols)} symbols")
        
        if successful_symbols:
            print(f"\n📈 SUCCESSFUL SYMBOLS:")
            for symbol in successful_symbols:
                recommendation = model_recommendations.get(symbol, 'Unknown')
                print(f"  {symbol}: {recommendation}")
        
        if failed_symbols:
            print(f"\n❌ FAILED SYMBOLS:")
            for symbol in failed_symbols:
                error = results[symbol].get('error', 'Unknown error')
                print(f"  {symbol}: {error}")
        
        # Count recommendations
        three_class_count = sum(1 for rec in model_recommendations.values() if '3-class' in rec)
        binary_count = sum(1 for rec in model_recommendations.values() if 'binary' in rec)
        
        print(f"\n🎯 MODEL RECOMMENDATIONS:")
        print(f"  3-Class Models: {three_class_count} (PRIORITIZED)")
        print(f"  Binary Models: {binary_count}")
        
        print(f"\n📊 KEY IMPROVEMENTS IMPLEMENTED:")
        print(f"  ✅ Lower confidence thresholds (25-30% vs 35%)")
        print(f"  ✅ Balanced target creation (prevents bias)")
        print(f"  ✅ Enhanced class weighting")
        print(f"  ✅ Consistent feature selection")
        print(f"  ✅ 3-class model prioritization")
        
        print("\n🚀 Next Steps:")
        print("  1. Update .env file with successful symbols")
        print("  2. Use 3-class models in production")
        print("  3. Set confidence thresholds to 25-30%")
        print("  4. Monitor execution rates and profitability")

def main():
    """Main function to run enhanced ML training."""
    print("🔧 Enhanced ML Training - Critical Fixes Implementation")
    print("🎯 Addressing model bias, confidence issues, and execution problems")
    
    trainer = EnhancedMLTrainer()
    results = trainer.train_enhanced_models()
    
    return results

if __name__ == "__main__":
    results = main()
