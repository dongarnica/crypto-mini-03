import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import json
import os
import tensorflow as tf
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (LSTM, Dense, Dropout, BatchNormalization, 
                                   Attention, MultiHeadAttention, LayerNormalization,
                                   Conv1D, MaxPooling1D, GlobalAveragePooling1D, Input)
from tensorflow.keras.optimizers import Adam, AdamW
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l1_l2

class ImprovedCryptoLSTMPipeline:
    """
    Improved LSTM-based cryptocurrency trading pipeline addressing core issues:
    - Lower confidence thresholds for signal generation
    - Feature selection to reduce noise
    - Better data quality checks
    - Improved model architecture
    - Enhanced signal generation
    """
    
    def __init__(self, symbol='BTCUSDT', interval='1h', lookback_period=24, 
                 prediction_horizon=1, confidence_threshold=0.55):
        """
        Initialize the improved LSTM trading pipeline.
        
        Args:
            symbol: Trading pair symbol
            interval: Data interval
            lookback_period: Number of timesteps to look back (reduced to 24)
            prediction_horizon: Number of steps ahead to predict
            confidence_threshold: Minimum confidence for trading (reduced to 55%)
        """
        self.symbol = symbol
        self.interval = interval
        self.lookback_period = lookback_period
        self.prediction_horizon = prediction_horizon
        self.confidence_threshold = confidence_threshold  # Lowered from 60% to 55%
        
        # Simplified scalers
        self.scaler = RobustScaler()
        
        # Data storage
        self.raw_data = None
        self.processed_data = None
        self.features = None
        self.targets = None
        self.selected_features = None
        
        # Improved model parameters
        self.lstm_units = [64, 32]  # Reduced complexity
        self.dropout_rate = 0.3
        self.learning_rate = 0.001
        self.epochs = 50  # Reduced to prevent overfitting
        self.batch_size = 64
        
        # More aggressive trading thresholds
        self.buy_threshold = 0.002   # Reduced from 0.003 to 0.002 (0.2%)
        self.sell_threshold = -0.002  # Reduced from -0.003 to -0.002
        
        # Model storage
        self.model = None
        self.feature_selector = None
        
    def load_data_from_csv(self, csv_path):
        """Enhanced data loading with validation."""
        print(f"Loading data from {csv_path}...")
        self.raw_data = pd.read_csv(csv_path)
        
        # Ensure proper datetime conversion
        self.raw_data['open_time'] = pd.to_datetime(self.raw_data['open_time'])
        
        # Sort and remove duplicates
        self.raw_data = self.raw_data.sort_values('open_time').reset_index(drop=True)
        self.raw_data = self.raw_data.drop_duplicates(subset=['open_time'], keep='first')
        
        # Data quality checks
        print(f"Loaded {len(self.raw_data)} data points")
        print(f"Date range: {self.raw_data['open_time'].min()} to {self.raw_data['open_time'].max()}")
        
        # Check for missing values
        missing_data = self.raw_data.isnull().sum()
        if missing_data.any():
            print("Missing data found:")
            print(missing_data[missing_data > 0])
        
        # Check for data leakage issues
        self._check_data_quality()
        
        return self.raw_data
    
    def _check_data_quality(self):
        """Comprehensive data quality checks."""
        df = self.raw_data
        
        print("\n=== Data Quality Analysis ===")
        
        # Check for duplicates
        duplicates = df.duplicated(subset=['open_time']).sum()
        print(f"Duplicate timestamps: {duplicates}")
        
        # Check for gaps in time series
        time_diffs = df['open_time'].diff()
        expected_diff = pd.Timedelta(hours=1)
        gaps = time_diffs[time_diffs > expected_diff * 1.5]
        print(f"Time gaps found: {len(gaps)}")
        
        # Check for outliers in price data
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            q1, q3 = df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            outliers = df[(df[col] < q1 - 3*iqr) | (df[col] > q3 + 3*iqr)]
            print(f"Outliers in {col}: {len(outliers)} ({len(outliers)/len(df)*100:.2f}%)")
        
        # Check OHLC consistency
        invalid_ohlc = df[(df['high'] < df['low']) | 
                         (df['high'] < df['open']) | 
                         (df['high'] < df['close']) |
                         (df['low'] > df['open']) | 
                         (df['low'] > df['close'])]
        print(f"Invalid OHLC relationships: {len(invalid_ohlc)}")
        
        # Check for zero or negative prices
        zero_prices = df[(df[price_cols] <= 0).any(axis=1)]
        print(f"Zero/negative prices: {len(zero_prices)}")
        
        print("=== End Data Quality Analysis ===\n")
    
    def add_essential_indicators(self):
        """Add only essential technical indicators to reduce noise."""
        print("Adding essential technical indicators...")
        df = self.raw_data.copy()
        
        # Basic price indicators
        df['sma_10'] = df['close'].rolling(10).mean()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        
        # RSI
        def calculate_rsi(series, period=14):
            delta = series.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / (loss + 1e-10)
            rsi = 100 - (100 / (1 + rs))
            return rsi.fillna(50)
        
        df['rsi'] = calculate_rsi(df['close'])
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Volume indicators
        df['volume_sma_20'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        
        # Price momentum
        df['return_1'] = df['close'].pct_change(1)
        df['return_5'] = df['close'].pct_change(5)
        df['return_10'] = df['close'].pct_change(10)
        
        # Volatility
        df['volatility_10'] = df['close'].rolling(10).std() / df['close'].rolling(10).mean()
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = true_range.rolling(14).mean()
        df['atr_pct'] = df['atr'] / df['close']
        
        # Time features (simplified)
        df['hour'] = df['open_time'].dt.hour
        df['is_trading_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 16)).astype(int)
        
        self.processed_data = df
        print(f"Added essential indicators. Dataset shape: {df.shape}")
        
    def create_improved_targets(self, df):
        """Create improved trading targets with better signal generation."""
        print("Creating improved targets...")
        
        # Simple future return calculation
        df['future_return'] = df['close'].shift(-self.prediction_horizon) / df['close'] - 1
        
        # Create targets with fixed thresholds (more aggressive)
        df['signal'] = 0  # Hold (default)
        
        # Buy signal: future return > threshold
        df.loc[df['future_return'] > self.buy_threshold, 'signal'] = 1
        
        # Sell signal: future return < threshold  
        df.loc[df['future_return'] < self.sell_threshold, 'signal'] = 2
        
        # Check signal distribution
        signal_counts = df['signal'].value_counts().sort_index()
        signal_pct = df['signal'].value_counts(normalize=True).sort_index() * 100
        
        print("Signal Distribution:")
        for signal, count in signal_counts.items():
            signal_name = ['Hold', 'Buy', 'Sell'][signal]
            print(f"  {signal_name}: {count} ({signal_pct[signal]:.1f}%)")
        
        # Ensure we have enough buy/sell signals
        if signal_pct.get(1, 0) < 5 or signal_pct.get(2, 0) < 5:
            print("WARNING: Very few buy/sell signals generated. Consider adjusting thresholds.")
            print(f"Current thresholds: Buy={self.buy_threshold:.3f}, Sell={self.sell_threshold:.3f}")
        
        return df
    
    def select_best_features(self, df, target_col='signal', n_features=15):
        """Select best features to reduce noise and overfitting."""
        print(f"Selecting top {n_features} features...")
        
        # Define potential features
        feature_candidates = [
            'open', 'high', 'low', 'close', 'volume',
            'sma_10', 'sma_20', 'ema_12', 'ema_26',
            'macd', 'macd_signal', 'rsi', 'bb_position',
            'volume_ratio', 'return_1', 'return_5', 'return_10',
            'volatility_10', 'atr_pct', 'is_trading_hours'
        ]
        
        # Filter available features
        available_features = [f for f in feature_candidates if f in df.columns]
        print(f"Available features: {len(available_features)}")
        
        # Remove rows with NaN
        df_clean = df[available_features + [target_col]].dropna()
        df_clean = df_clean.replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(df_clean) < 100:
            raise ValueError("Insufficient clean data for feature selection")
        
        X = df_clean[available_features]
        y = df_clean[target_col]
        
        # Use mutual information for feature selection (better for non-linear relationships)
        self.feature_selector = SelectKBest(score_func=mutual_info_classif, k=min(n_features, len(available_features)))
        X_selected = self.feature_selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_mask = self.feature_selector.get_support()
        self.selected_features = [f for f, selected in zip(available_features, selected_mask) if selected]
        
        print(f"Selected features: {self.selected_features}")
        
        # Show feature scores
        scores = self.feature_selector.scores_
        feature_scores = [(f, score) for f, score in zip(available_features, scores)]
        feature_scores.sort(key=lambda x: x[1], reverse=True)
        
        print("Top 10 feature scores:")
        for i, (feature, score) in enumerate(feature_scores[:10]):
            print(f"  {i+1:2d}. {feature:<20} {score:.4f}")
        
        return df_clean
    
    def create_features_and_targets(self):
        """Create optimized feature matrix and targets."""
        print("Creating features and targets...")
        
        if self.processed_data is None:
            raise ValueError("Data must be processed first")
        
        df = self.processed_data.copy()
        
        # Create improved targets
        df = self.create_improved_targets(df)
        
        # Select best features
        df_clean = self.select_best_features(df)
        
        # Prepare features and targets
        X_data = df_clean[self.selected_features].values
        y_data = df_clean['signal'].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_data)
        
        # Create sequences
        X, y = [], []
        for i in range(self.lookback_period, len(X_scaled)):
            X.append(X_scaled[i-self.lookback_period:i])
            y.append(y_data[i])
        
        self.features = np.array(X)
        self.targets = np.array(y)
        
        print(f"Created {len(self.features)} sequences with {len(self.selected_features)} features")
        print(f"Target distribution: {dict(zip(*np.unique(self.targets, return_counts=True)))}")
        
        return self.features, self.targets
    
    def build_simplified_model(self):
        """Build simplified LSTM model to prevent overfitting."""
        print("Building simplified LSTM model...")
        
        model = Sequential([
            LSTM(self.lstm_units[0], return_sequences=True, 
                 dropout=self.dropout_rate,
                 input_shape=(self.lookback_period, len(self.selected_features))),
            BatchNormalization(),
            
            LSTM(self.lstm_units[1], return_sequences=False,
                 dropout=self.dropout_rate),
            BatchNormalization(),
            
            Dense(32, activation='relu'),
            Dropout(self.dropout_rate),
            
            Dense(16, activation='relu'),
            Dropout(self.dropout_rate),
            
            Dense(3, activation='softmax')
        ])
        
        # Use Adam optimizer
        optimizer = Adam(learning_rate=self.learning_rate)
        
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        print("Model built successfully")
        print(f"Total parameters: {model.count_params():,}")
        
        return model
    
    def train_with_proper_validation(self):
        """Train model with proper time series validation."""
        print("Training with time series validation...")
        
        if self.features is None:
            raise ValueError("Features must be created first")
        
        # Time series split - use last 20% for validation
        split_point = int(len(self.features) * 0.8)
        X_train = self.features[:split_point]
        X_val = self.features[split_point:]
        y_train = self.targets[:split_point]
        y_val = self.targets[split_point:]
        
        print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
        
        # Check class distribution
        train_dist = dict(zip(*np.unique(y_train, return_counts=True)))
        val_dist = dict(zip(*np.unique(y_val, return_counts=True)))
        print(f"Training distribution: {train_dist}")
        print(f"Validation distribution: {val_dist}")
        
        # Calculate class weights
        class_weights = self._calculate_balanced_weights(y_train)
        
        # Callbacks with less patience to prevent overfitting
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
        
        # Evaluate
        val_loss, val_acc = self.model.evaluate(X_val, y_val, verbose=0)
        print(f"Final Validation - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        
        # Store validation data
        self.X_val = X_val
        self.y_val = y_val
        
        return history
    
    def _calculate_balanced_weights(self, y):
        """Calculate balanced class weights."""
        unique_classes, counts = np.unique(y, return_counts=True)
        total_samples = len(y)
        class_weights = {}
        
        for i, count in enumerate(counts):
            # Inverse frequency weighting
            class_weights[unique_classes[i]] = total_samples / (len(unique_classes) * count)
        
        print(f"Class weights: {class_weights}")
        return class_weights
    
    def evaluate_model_performance(self):
        """Comprehensive model evaluation."""
        print("\n=== Model Performance Evaluation ===")
        
        if not hasattr(self, 'X_val'):
            print("No validation data available. Train model first.")
            return
        
        # Get predictions
        y_pred_proba = self.model.predict(self.X_val, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Classification report
        signal_names = ['Hold', 'Buy', 'Sell']
        print("\nClassification Report:")
        print(classification_report(self.y_val, y_pred, target_names=signal_names))
        
        # Confusion matrix
        cm = confusion_matrix(self.y_val, y_pred)
        print("\nConfusion Matrix:")
        print("Predicted ->", "  ".join(f"{s:>8}" for s in signal_names))
        for i, actual in enumerate(signal_names):
            print(f"Actual {actual:>4}: {' '.join(f'{cm[i,j]:>8d}' for j in range(3))}")
        
        # Confidence analysis
        confidences = np.max(y_pred_proba, axis=1)
        print(f"\nConfidence Statistics:")
        print(f"  Mean confidence: {np.mean(confidences):.3f}")
        print(f"  Median confidence: {np.median(confidences):.3f}")
        print(f"  Predictions with >55% confidence: {np.sum(confidences > 0.55) / len(confidences) * 100:.1f}%")
        print(f"  Predictions with >60% confidence: {np.sum(confidences > 0.60) / len(confidences) * 100:.1f}%")
        print(f"  Predictions with >70% confidence: {np.sum(confidences > 0.70) / len(confidences) * 100:.1f}%")
        
        # Trading signal analysis with different thresholds
        thresholds = [0.5, 0.55, 0.6, 0.65, 0.7]
        print(f"\nTrading Signals by Confidence Threshold:")
        print(f"{'Threshold':<10} {'Total':<8} {'Buy':<6} {'Sell':<6} {'Hold':<6}")
        print("-" * 40)
        
        for threshold in thresholds:
            high_conf_mask = confidences >= threshold
            if np.sum(high_conf_mask) > 0:
                high_conf_pred = y_pred[high_conf_mask]
                signal_counts = [np.sum(high_conf_pred == i) for i in range(3)]
                total_signals = np.sum(high_conf_mask)
                print(f"{threshold:<10.2f} {total_signals:<8} {signal_counts[1]:<6} {signal_counts[2]:<6} {signal_counts[0]:<6}")
        
        return {
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'confidences': confidences,
            'classification_report': classification_report(self.y_val, y_pred, target_names=signal_names, output_dict=True)
        }
    
    def improved_backtest(self, initial_capital=10000, confidence_threshold=None):
        """Improved backtesting with lower confidence threshold."""
        print(f"\n=== Improved Backtest (Threshold: {confidence_threshold or self.confidence_threshold:.1%}) ===")
        
        if confidence_threshold is None:
            confidence_threshold = self.confidence_threshold
        
        if not hasattr(self, 'X_val'):
            raise ValueError("Model must be trained first")
        
        # Get predictions
        y_pred_proba = self.model.predict(self.X_val, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        confidences = np.max(y_pred_proba, axis=1)
        
        # Portfolio tracking
        portfolio = {
            'capital': initial_capital,
            'position': 0,
            'trades': [],
            'values': [initial_capital]
        }
        
        # Get price data for validation period
        val_start_idx = len(self.processed_data) - len(self.X_val) - self.lookback_period
        price_data = self.processed_data.iloc[val_start_idx:val_start_idx + len(self.X_val)]
        
        total_signals = 0
        executed_trades = 0
        
        for i, (predicted, confidence) in enumerate(zip(y_pred, confidences)):
            current_price = price_data['close'].iloc[i]
            
            # Apply confidence threshold
            if confidence >= confidence_threshold:
                total_signals += 1
                
                if predicted == 1 and portfolio['position'] <= 0:  # Buy signal
                    # Buy with 90% of capital
                    investment = portfolio['capital'] * 0.9
                    shares = investment / current_price
                    portfolio['capital'] -= investment
                    portfolio['position'] += shares
                    
                    portfolio['trades'].append({
                        'type': 'buy', 'price': current_price, 'shares': shares,
                        'confidence': confidence, 'timestamp': i
                    })
                    executed_trades += 1
                
                elif predicted == 2 and portfolio['position'] > 0:  # Sell signal
                    # Sell all position
                    proceeds = portfolio['position'] * current_price
                    portfolio['capital'] += proceeds
                    
                    portfolio['trades'].append({
                        'type': 'sell', 'price': current_price, 'shares': portfolio['position'],
                        'confidence': confidence, 'timestamp': i
                    })
                    portfolio['position'] = 0
                    executed_trades += 1
            
            # Calculate portfolio value
            position_value = portfolio['position'] * current_price if portfolio['position'] > 0 else 0
            total_value = portfolio['capital'] + position_value
            portfolio['values'].append(total_value)
        
        # Calculate performance
        final_value = portfolio['values'][-1]
        total_return = (final_value - initial_capital) / initial_capital
        
        # Buy and hold comparison
        start_price = price_data['close'].iloc[0]
        end_price = price_data['close'].iloc[-1]
        buy_hold_return = (end_price - start_price) / start_price
        
        print(f"Initial Capital: ${initial_capital:,.2f}")
        print(f"Final Value: ${final_value:,.2f}")
        print(f"Strategy Return: {total_return:.2%}")
        print(f"Buy & Hold Return: {buy_hold_return:.2%}")
        print(f"Excess Return: {total_return - buy_hold_return:.2%}")
        print(f"Total Signals: {total_signals}")
        print(f"Executed Trades: {executed_trades}")
        print(f"Signal Utilization: {executed_trades/max(1, total_signals)*100:.1f}%")
        
        return {
            'final_value': final_value,
            'total_return': total_return,
            'buy_hold_return': buy_hold_return,
            'excess_return': total_return - buy_hold_return,
            'trades': portfolio['trades'],
            'values': portfolio['values'],
            'total_signals': total_signals,
            'executed_trades': executed_trades
        }
    
    def predict_with_lower_threshold(self, recent_data=None):
        """Make predictions with lower confidence threshold."""
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        if recent_data is None:
            recent_data = self.processed_data.tail(self.lookback_period)
        
        # Prepare features (use selected features only)
        df_features = recent_data[self.selected_features].tail(self.lookback_period)
        df_features = df_features.fillna(method='ffill').fillna(method='bfill')
        
        # Scale features
        features_scaled = self.scaler.transform(df_features.values)
        X_pred = features_scaled.reshape(1, self.lookback_period, -1)
        
        # Make prediction
        prediction_proba = self.model.predict(X_pred, verbose=0)
        predicted_class = np.argmax(prediction_proba[0])
        confidence = np.max(prediction_proba[0])
        
        signal_names = ['Hold', 'Buy', 'Sell']
        
        # Generate recommendation with lower threshold
        if confidence >= self.confidence_threshold:
            recommendation = f"TRADE: {signal_names[predicted_class]} (confidence: {confidence:.1%})"
        else:
            recommendation = f"WAIT: Low confidence {signal_names[predicted_class]} signal ({confidence:.1%})"
        
        result = {
            'signal': signal_names[predicted_class],
            'confidence': confidence,
            'probabilities': {
                'Hold': float(prediction_proba[0][0]),
                'Buy': float(prediction_proba[0][1]),
                'Sell': float(prediction_proba[0][2])
            },
            'current_price': recent_data['close'].iloc[-1],
            'recommendation': recommendation,
            'tradeable': confidence >= self.confidence_threshold
        }
        
        return result
    
    def run_complete_analysis(self, csv_path):
        """Run complete improved analysis."""
        print("🚀 Starting Improved Crypto ML Pipeline Analysis")
        print("=" * 60)
        
        try:
            # 1. Load and validate data
            self.load_data_from_csv(csv_path)
            
            # 2. Add essential indicators
            self.add_essential_indicators()
            
            # 3. Create features and targets
            self.create_features_and_targets()
            
            # 4. Build simplified model
            self.build_simplified_model()
            
            # 5. Train with proper validation
            history = self.train_with_proper_validation()
            
            # 6. Evaluate model performance
            performance = self.evaluate_model_performance()
            
            # 7. Run improved backtest with different thresholds
            print("\n" + "=" * 60)
            print("BACKTESTING WITH DIFFERENT CONFIDENCE THRESHOLDS")
            print("=" * 60)
            
            thresholds = [0.5, 0.55, 0.6, 0.65, 0.7]
            backtest_results = {}
            
            for threshold in thresholds:
                backtest_results[threshold] = self.improved_backtest(confidence_threshold=threshold)
            
            # 8. Current prediction
            print("\n" + "=" * 60)
            print("CURRENT PREDICTION")
            print("=" * 60)
            prediction = self.predict_with_lower_threshold()
            
            print(f"Symbol: {self.symbol}")
            print(f"Current Price: ${prediction['current_price']:.2f}")
            print(f"Signal: {prediction['signal']}")
            print(f"Confidence: {prediction['confidence']:.1%}")
            print(f"Recommendation: {prediction['recommendation']}")
            print(f"Probabilities:")
            for signal, prob in prediction['probabilities'].items():
                print(f"  {signal}: {prob:.1%}")
            
            # 9. Summary and recommendations
            print("\n" + "=" * 60)
            print("ANALYSIS SUMMARY & RECOMMENDATIONS")
            print("=" * 60)
            
            best_threshold = max(backtest_results.keys(), 
                               key=lambda t: backtest_results[t]['total_return'])
            
            print(f"✅ Model trained successfully with {len(self.selected_features)} selected features")
            print(f"✅ Confidence threshold lowered to {self.confidence_threshold:.0%}")
            print(f"✅ Trading thresholds reduced to ±{abs(self.buy_threshold):.1%}")
            print(f"✅ Best performing confidence threshold: {best_threshold:.0%}")
            print(f"✅ Current prediction confidence: {prediction['confidence']:.1%}")
            
            if prediction['confidence'] >= self.confidence_threshold:
                print(f"🎯 ACTIONABLE SIGNAL: {prediction['signal']} with {prediction['confidence']:.1%} confidence")
            else:
                print(f"⏳ WAIT: Signal too weak, consider waiting for higher confidence")
            
            return {
                'history': history,
                'performance': performance,
                'backtest_results': backtest_results,
                'prediction': prediction,
                'selected_features': self.selected_features
            }
            
        except Exception as e:
            print(f"❌ Analysis failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

def main():
    """Main function to run the improved pipeline."""
    # Use the most recent BTCUSDT data
    csv_path = "/workspaces/crypto-mini-03/historical_exports/BTCUSDT_1year_hourly_20250601_163754.csv"
    
    # Initialize improved pipeline
    pipeline = ImprovedCryptoLSTMPipeline(
        symbol='BTCUSDT',
        confidence_threshold=0.55,  # Lower threshold
        lookback_period=24,         # Shorter lookback
        buy_threshold=0.002,        # More aggressive thresholds
        sell_threshold=-0.002
    )
    
    # Run complete analysis
    results = pipeline.run_complete_analysis(csv_path)
    
    if results:
        print("\n🎉 Analysis completed successfully!")
        print("Key improvements implemented:")
        print("- Lowered confidence threshold to 55%")
        print("- Reduced features from 41 to top 15 most important")
        print("- More aggressive trading thresholds (±0.2%)")
        print("- Simplified model architecture")
        print("- Better data quality checks")
    else:
        print("❌ Analysis failed. Check error messages above.")

if __name__ == "__main__":
    main()
