#!/usr/bin/env python3
"""
ML Prediction Engine
====================

Handles machine learning pipeline management, predictions, 
and model training for trading signals.

Author: Crypto Trading Strategy Engine
Date: June 2, 2025
"""

import logging
import os
import sys
import traceback
from datetime import datetime
from typing import Dict, List, Optional

from ml.ml_pipeline_improved_components import ImprovedCryptoLSTMPipeline


class MLEngine:
    """Manages ML pipelines and predictions for trading."""
    
    def __init__(self, config):
        """
        Initialize ML engine.
        
        Args:
            config: Trading configuration
        """
        self.config = config
        self.ml_pipelines: Dict[str, ImprovedCryptoLSTMPipeline] = {}
        self.data_cache: Dict[str, any] = {}
        self.logger = logging.getLogger('MLEngine')
        
        # Configure logger level to match trading engine
        log_level = getattr(logging, self.config.log_level.upper())
        self.logger.setLevel(log_level)
        
        # Add file handler if save_trades is enabled
        if self.config.save_trades and not self.logger.handlers:
            # Determine log directory based on environment
            if os.path.exists('/app'):
                # Running in Docker container
                log_file = f'/app/trading/logs/trading_{datetime.now().strftime("%Y%m%d")}.log'
            else:
                # Running in development environment
                log_file = f'/workspaces/crypto-mini-03/trading/logs/trading_{datetime.now().strftime("%Y%m%d")}.log'
            
            file_handler = logging.FileHandler(log_file)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            file_handler.setLevel(log_level)
            self.logger.addHandler(file_handler)
    
    def initialize_pipelines(self, symbols: List[str]) -> None:
        """
        Initialize ML pipelines for given symbols.
        
        Args:
            symbols: List of crypto symbols to initialize
        """
        self.logger.info("🤖 Initializing ML pipelines...")
        
        for symbol in symbols:
            try:
                # Create improved ML pipeline
                pipeline = ImprovedCryptoLSTMPipeline(
                    symbol=symbol,
                    lookback_period=self.config.lookback_period,
                    prediction_horizon=self.config.prediction_horizon,
                    confidence_threshold=self.config.min_confidence,
                    use_binary_classification=self.config.use_binary_classification
                )
                
                # Check for existing trained model
                model_path = self._get_model_path(symbol)
                
                if os.path.exists(model_path):
                    self.logger.info(f"Found existing model for {symbol}")
                    try:
                        # First load the data (required for predictions)
                        pipeline.load_data_from_symbol(symbol)
                        pipeline.add_essential_indicators()
                        
                        # Then load the trained model
                        pipeline.load_trained_model(model_path)
                        self.logger.info(f"✅ Loaded trained model for {symbol}")
                    except Exception as load_error:
                        self.logger.warning(f"Failed to load model for {symbol}: {load_error}")
                        self.logger.info(f"Will train new model for {symbol}")
                        self._train_pipeline_model(pipeline, symbol)
                else:
                    self.logger.info(f"No existing model for {symbol}, training new model...")
                    # Load data before training
                    pipeline.load_data_from_symbol(symbol)
                    pipeline.add_essential_indicators()
                    self._train_pipeline_model(pipeline, symbol)
                
                self.ml_pipelines[symbol] = pipeline
                self.logger.info(f"✅ ML pipeline ready for {symbol}")
                
            except Exception as e:
                self.logger.error(f"Failed to initialize ML pipeline for {symbol}: {e}")
                # Continue with other symbols
        
        self.logger.info(f"Initialized {len(self.ml_pipelines)} ML pipelines")
    
    def get_prediction(self, symbol: str) -> Optional[Dict]:
        """
        Get ML prediction for a symbol.
        
        Args:
            symbol: Crypto symbol
            
        Returns:
            Dict with prediction information or None
        """
        if symbol not in self.ml_pipelines:
            self.logger.warning(f"No ML pipeline for {symbol}")
            return None
        
        try:
            pipeline = self.ml_pipelines[symbol]
            
            # Check if model is trained
            if pipeline.model is None:
                self.logger.warning(f"Model not trained for {symbol}")
                return None
            
            # Get prediction using the improved method
            prediction = pipeline.predict_with_lower_threshold()
            
            if prediction and not prediction.get('error'):
                # Enhanced prediction processing
                signal = prediction.get('signal', 'HOLD')
                confidence = prediction.get('confidence', 0.0)
                current_price = prediction.get('current_price', 0.0)
                recommendation = prediction.get('recommendation', '')
                probabilities = prediction.get('probabilities', {})
                tradeable = prediction.get('tradeable', False)
                high_confidence = prediction.get('high_confidence', False)
                
                # Log prediction details
                self.logger.debug(f"🔮 ML Prediction for {symbol}:")
                self.logger.debug(f"   Signal: {signal} (confidence: {confidence:.1%})")
                self.logger.debug(f"   Price: ${current_price:.2f}")
                self.logger.debug(f"   Tradeable: {tradeable}")
                self.logger.debug(f"   Recommendation: {recommendation}")
                
                # Enhanced prediction with additional metadata
                enhanced_prediction = {
                    'symbol': symbol,
                    'signal': signal,
                    'confidence': confidence,
                    'current_price': current_price,
                    'recommendation': recommendation,
                    'probabilities': probabilities,
                    'tradeable': tradeable,
                    'high_confidence': high_confidence,
                    'prediction_timestamp': datetime.now().isoformat(),
                    'classification_type': prediction.get('classification_type', 'unknown'),
                    'features_used': prediction.get('features_used', 0),
                    'features_missing': prediction.get('features_missing', 0)
                }
                
                return enhanced_prediction
            
            else:
                error = prediction.get('error', 'Unknown prediction error') if prediction else 'No prediction returned'
                self.logger.error(f"Prediction failed for {symbol}: {error}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting ML prediction for {symbol}: {e}")
            return None
    
    def refresh_model_data(self, symbol: str) -> bool:
        """
        Refresh model data for a specific symbol by retraining.
        
        Args:
            symbol: Symbol to refresh
            
        Returns:
            True if successful
        """
        if symbol not in self.ml_pipelines:
            self.logger.error(f"No pipeline for {symbol}")
            return False
        
        try:
            self.logger.info(f"🔄 Refreshing model data for {symbol}...")
            pipeline = self.ml_pipelines[symbol]
            
            # Retrain with latest data
            self._train_pipeline_model(pipeline, symbol)
            
            self.logger.info(f"✅ Model data refreshed for {symbol}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error refreshing model for {symbol}: {e}")
            return False
    
    def refresh_data_pipeline(self, symbol: str, force_retrain: bool = False) -> bool:
        """
        Refresh data pipeline for a symbol with latest historical data.
        
        Args:
            symbol: Symbol to refresh
            force_retrain: Force model retraining
            
        Returns:
            True if successful
        """
        try:
            if symbol not in self.ml_pipelines:
                self.logger.error(f"No pipeline for {symbol}")
                return False
            
            pipeline = self.ml_pipelines[symbol]
            
            self.logger.info(f"🔄 Refreshing data pipeline for {symbol}...")
            
            # Find latest historical data file
            historical_dir = '/workspaces/crypto-mini-03/historical_exports'
            if os.path.exists(historical_dir):
                files = [f for f in os.listdir(historical_dir) 
                        if f.startswith(symbol) and f.endswith('.csv')]
                
                if files:
                    # Get the most recent file
                    latest_file = max(files, key=lambda x: os.path.getctime(
                        os.path.join(historical_dir, x)
                    ))
                    latest_path = os.path.join(historical_dir, latest_file)
                    
                    self.logger.info(f"Loading latest data from: {latest_file}")
                    
                    # Load new data
                    pipeline.load_data_from_csv(latest_path)
                    pipeline.add_essential_indicators()
            
            # Validate data
            if self._validate_pipeline_data(pipeline, symbol):
                # Refresh features if needed
                if force_retrain or not hasattr(pipeline, 'model') or pipeline.model is None:
                    self.logger.info(f"Retraining model for {symbol}...")
                    self._train_pipeline_model(pipeline, symbol)
                
                self.logger.info(f"✅ Data pipeline refreshed for {symbol}")
                return True
            else:
                self.logger.error(f"Data validation failed for {symbol}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error refreshing data pipeline for {symbol}: {e}")
            return False
    
    def _train_pipeline_model(self, pipeline: ImprovedCryptoLSTMPipeline, symbol: str) -> None:
        """Train ML model for a specific symbol."""
        try:
            self.logger.info(f"🎓 Training model for {symbol}...")
            
            # Check if data is already loaded, if not load it
            if not hasattr(pipeline, 'processed_data') or pipeline.processed_data is None:
                self.logger.info(f"Loading data for training {symbol}...")
                pipeline.load_data_from_symbol(symbol)
                pipeline.add_essential_indicators()
            
            # Run complete analysis which includes training
            result = pipeline.run_complete_analysis(symbol=symbol)
            
            if result and result.get('success'):
                self.logger.info(f"✅ Model training completed for {symbol}")
                
                # Log performance metrics
                performance = result.get('performance', {})
                accuracy = performance.get('classification_report', {}).get('accuracy', 0)
                self.logger.info(f"   Validation accuracy: {accuracy:.1%}")
                
                # Log backtest results
                backtest = result.get('backtest_result', {})
                strategy_return = backtest.get('total_return', 0)
                executed_trades = backtest.get('executed_trades', 0)
                self.logger.info(f"   Strategy return: {strategy_return:.1%}")
                self.logger.info(f"   Executed trades: {executed_trades}")
                
                # Save model
                model_path = self._get_model_path(symbol)
                try:
                    if hasattr(pipeline, 'save_model_components'):
                        pipeline.save_model_components(model_path)
                        self.logger.info(f"💾 Model and components saved to {model_path}")
                    else:
                        pipeline.model.save(model_path)
                        self.logger.info(f"💾 Model saved to {model_path}")
                except Exception as save_error:
                    self.logger.warning(f"Failed to save model: {save_error}")
                
            else:
                error = result.get('error', 'Unknown error') if result else 'No result returned'
                self.logger.error(f"❌ Model training failed for {symbol}: {error}")
                
        except Exception as e:
            self.logger.error(f"Error training model for {symbol}: {e}")
            self.logger.error(traceback.format_exc())
    
    def _validate_pipeline_data(self, pipeline, symbol: str) -> bool:
        """Validate pipeline data quality."""
        try:
            if not hasattr(pipeline, 'processed_data') or pipeline.processed_data is None:
                self.logger.error(f"No processed data available for {symbol}")
                return False
            
            data = pipeline.processed_data
            
            if len(data) < 100:  # Minimum data points
                self.logger.error(f"Insufficient data for {symbol}: {len(data)} rows")
                return False
            
            # Check for required columns
            required_cols = ['close', 'open', 'high', 'low', 'volume']
            missing_cols = [col for col in required_cols if col not in data.columns]
            
            if missing_cols:
                self.logger.error(f"Missing required columns for {symbol}: {missing_cols}")
                return False
            
            # Check for null values
            null_counts = data[required_cols].isnull().sum()
            total_nulls = null_counts.sum()
            null_percentage = total_nulls / (len(data) * len(required_cols))
            
            if null_percentage > 0.05:  # More than 5% nulls
                self.logger.warning(f"High null percentage for {symbol}: {null_percentage:.1%}")
                return False
            
            # Check for reasonable price values
            price_cols = ['close', 'open', 'high', 'low']
            for col in price_cols:
                if (data[col] <= 0).any():
                    self.logger.error(f"Non-positive prices found in {col} for {symbol}")
                    return False
            
            self.logger.debug(f"Data validation passed for {symbol}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating data for {symbol}: {e}")
            return False
    
    def _get_model_path(self, symbol: str) -> str:
        """Get model file path for a symbol."""
        # Determine model directory based on environment
        if os.path.exists('/app'):
            # Running in Docker container
            model_dir = '/app/ml_results/models'
        else:
            # Running in development environment
            model_dir = '/workspaces/crypto-mini-03/ml_results/models'
        
        os.makedirs(model_dir, exist_ok=True)
        return f'{model_dir}/{symbol}_improved_model.keras'
    
    def get_pipeline_count(self) -> int:
        """Get number of initialized pipelines."""
        return len(self.ml_pipelines)
    
    def get_pipeline_symbols(self) -> List[str]:
        """Get list of symbols with initialized pipelines."""
        return list(self.ml_pipelines.keys())
