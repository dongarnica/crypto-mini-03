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
warnings.filterwarnings('ignore')

# Machine Learning imports
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import TimeSeriesSplit
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (LSTM, Dense, Dropout, BatchNormalization, 
                                   Attention, MultiHeadAttention, LayerNormalization,
                                   Conv1D, MaxPooling1D, GlobalAveragePooling1D, Input)
from tensorflow.keras.optimizers import Adam, AdamW
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l1_l2
import joblib

class OptimizedCryptoLSTMPipeline:
    """
    Optimized LSTM-based cryptocurrency trading pipeline with enhanced features:
    - Advanced feature engineering with market microstructure indicators
    - Attention mechanisms and hybrid architectures
    - Multi-timeframe analysis
    - Advanced backtesting with risk metrics
    - Robust validation with walk-forward analysis
    """
    
    def __init__(self, symbol='BTCUSDT', interval='1h', lookback_period=48, 
                 prediction_horizon=1, use_attention=True):
        """
        Initialize the optimized LSTM trading pipeline.
        
        Args:
            symbol: Trading pair symbol
            interval: Data interval
            lookback_period: Number of timesteps to look back (reduced from 60 to 48)
            prediction_horizon: Number of steps ahead to predict
            use_attention: Whether to use attention mechanism
        """
        self.symbol = symbol
        self.interval = interval
        self.lookback_period = lookback_period
        self.prediction_horizon = prediction_horizon
        self.use_attention = use_attention
        
        # Enhanced scalers
        self.price_scaler = RobustScaler()  # More robust to outliers
        self.volume_scaler = StandardScaler()
        self.feature_scaler = RobustScaler()
        
        # Data storage
        self.raw_data = None
        self.processed_data = None
        self.features = None
        self.targets = None
        
        # Optimized model parameters
        self.lstm_units = [128, 64, 32]  # Increased capacity
        self.dropout_rate = 0.2  # Reduced for better learning
        self.recurrent_dropout = 0.1
        self.learning_rate = 0.0005  # Lower learning rate
        self.epochs = 100
        self.batch_size = 32
        self.l1_reg = 1e-5
        self.l2_reg = 1e-4
        
        # Trading thresholds (dynamic)
        self.buy_threshold = 0.003  # 0.3%
        self.sell_threshold = -0.003
        
        # Model storage
        self.model = None
        self.best_model_path = 'best_model.keras'
        
    def load_data_from_csv(self, csv_path):
        """Enhanced data loading with validation."""
        print(f"Loading data from {csv_path}...")
        self.raw_data = pd.read_csv(csv_path)
        
        # Ensure proper datetime conversion
        self.raw_data['open_time'] = pd.to_datetime(self.raw_data['open_time'])
        if 'close_time' in self.raw_data.columns:
            self.raw_data['close_time'] = pd.to_datetime(self.raw_data['close_time'])
        
        # Sort and remove duplicates
        self.raw_data = self.raw_data.sort_values('open_time').reset_index(drop=True)
        self.raw_data = self.raw_data.drop_duplicates(subset=['open_time'], keep='first')
        
        # Data validation
        print(f"Loaded {len(self.raw_data)} data points")
        print(f"Date range: {self.raw_data['open_time'].min()} to {self.raw_data['open_time'].max()}")
        
        # Check for missing values
        missing_data = self.raw_data.isnull().sum()
        if missing_data.any():
            print("Missing data found:")
            print(missing_data[missing_data > 0])
        
        return self.raw_data
    
    def _calculate_advanced_indicators(self, df):
        """Enhanced technical indicators including market microstructure."""
        
        # Price-based indicators (existing)
        df['sma_10'] = df['close'].rolling(10).mean()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        df['ema_9'] = df['close'].ewm(span=9).mean()
        
        # MACD with signal line
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        df['macd_normalized'] = df['macd'] / df['close']
        
        # RSI with multiple periods
        def calculate_rsi(series, period):
            delta = series.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            # Handle division by zero
            rs = gain / (loss + 1e-10)  # Add small epsilon to avoid division by zero
            rsi = 100 - (100 / (1 + rs))
            return rsi.fillna(50)  # Fill NaN with neutral RSI value
        
        df['rsi_14'] = calculate_rsi(df['close'], 14)
        df['rsi_7'] = calculate_rsi(df['close'], 7)  # Faster RSI
        
        # Bollinger Bands with additional metrics
        df['bb_middle'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        df['bb_squeeze'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # Market microstructure indicators (NEW)
        df['spread'] = df['high'] - df['low']
        df['spread_pct'] = df['spread'] / df['close']
        df['body_size'] = abs(df['close'] - df['open'])
        df['body_pct'] = df['body_size'] / df['close']
        df['upper_shadow'] = df['high'] - np.maximum(df['open'], df['close'])
        df['lower_shadow'] = np.minimum(df['open'], df['close']) - df['low']
        df['shadow_ratio'] = (df['upper_shadow'] + df['lower_shadow']) / df['body_size']
        
        # Volume analysis (ENHANCED)
        df['volume_sma_10'] = df['volume'].rolling(10).mean()
        df['volume_sma_20'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        df['volume_trend'] = df['volume_sma_10'] / df['volume_sma_20']
        
        # Order flow indicators (if available)
        if 'taker_buy_base_asset_volume' in df.columns:
            df['buy_sell_ratio'] = df['taker_buy_base_asset_volume'] / df['volume']
            df['buy_pressure'] = df['buy_sell_ratio'] * df['volume_ratio']
        
        # Price momentum and volatility
        for period in [3, 7, 14]:
            df[f'return_{period}'] = df['close'].pct_change(period)
            df[f'volatility_{period}'] = df['close'].rolling(period).std() / df['close'].rolling(period).mean()
        
        # Support and resistance levels
        df['resistance'] = df['high'].rolling(20).max()
        df['support'] = df['low'].rolling(20).min()
        df['resistance_distance'] = (df['resistance'] - df['close']) / df['close']
        df['support_distance'] = (df['close'] - df['support']) / df['close']
        
        # Trend indicators
        df['price_position'] = (df['close'] - df['low'].rolling(50).min()) / (df['high'].rolling(50).max() - df['low'].rolling(50).min())
        
        # Williams %R
        highest_high = df['high'].rolling(14).max()
        lowest_low = df['low'].rolling(14).min()
        df['williams_r'] = -100 * ((highest_high - df['close']) / (highest_high - lowest_low))
        
        # ATR (Average True Range)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = true_range.rolling(14).mean()
        df['atr_pct'] = df['atr'] / df['close']
        
        return df
    
    def add_technical_indicators(self):
        """Add comprehensive technical indicators."""
        print("Adding advanced technical indicators...")
        df = self.raw_data.copy()
        
        # Apply advanced indicators
        df = self._calculate_advanced_indicators(df)
        
        # Time-based features
        df['hour'] = df['open_time'].dt.hour
        df['day_of_week'] = df['open_time'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Cyclical encoding for time features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        self.processed_data = df
        print(f"Added technical indicators. Dataset shape: {df.shape}")
        
        # Show feature statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        print(f"Total numeric features: {len(numeric_cols)}")
        
    def create_dynamic_targets(self, df):
        """Create dynamic trading targets based on volatility."""
        # Calculate dynamic thresholds based on rolling volatility
        volatility = df['close'].rolling(20).std() / df['close'].rolling(20).mean()
        median_vol = volatility.median()
        
        # Adjust thresholds based on market conditions
        dynamic_buy_threshold = np.where(volatility > median_vol, 
                                       self.buy_threshold * 1.5, 
                                       self.buy_threshold * 0.8)
        dynamic_sell_threshold = np.where(volatility > median_vol, 
                                        self.sell_threshold * 1.5, 
                                        self.sell_threshold * 0.8)
        
        # Create future returns
        df['future_return'] = df['close'].shift(-self.prediction_horizon).pct_change()
        
        # Create targets with dynamic thresholds
        df['signal'] = 0  # Hold
        for i in range(len(df)):
            if pd.notna(df['future_return'].iloc[i]):
                if df['future_return'].iloc[i] > dynamic_buy_threshold[i]:
                    df.loc[df.index[i], 'signal'] = 1  # Buy
                elif df['future_return'].iloc[i] < dynamic_sell_threshold[i]:
                    df.loc[df.index[i], 'signal'] = 2  # Sell
        
        return df
    
    def create_features_and_targets(self):
        """Create optimized feature matrix and targets."""
        print("Creating enhanced features and targets...")
        
        if self.processed_data is None:
            raise ValueError("Data must be processed first")
        
        df = self.processed_data.copy()
        
        # Create dynamic targets
        df = self.create_dynamic_targets(df)
        
        # Select features (optimized selection)
        price_features = ['open', 'high', 'low', 'close']
        volume_features = ['volume', 'volume_ratio', 'volume_trend']
        if 'buy_sell_ratio' in df.columns:
            volume_features.extend(['buy_sell_ratio', 'buy_pressure'])
        
        technical_features = [
            'sma_10', 'sma_20', 'sma_50', 'ema_12', 'ema_26',
            'macd', 'macd_signal', 'macd_hist', 'macd_normalized',
            'rsi_14', 'rsi_7', 'bb_position', 'bb_squeeze',
            'williams_r', 'atr_pct'
        ]
        
        microstructure_features = [
            'spread_pct', 'body_pct', 'shadow_ratio',
            'resistance_distance', 'support_distance', 'price_position'
        ]
        
        momentum_features = [
            'return_3', 'return_7', 'return_14',
            'volatility_3', 'volatility_7', 'volatility_14'
        ]
        
        time_features = ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'is_weekend']
        
        all_features = (price_features + volume_features + technical_features + 
                       microstructure_features + momentum_features + time_features)
        
        # Filter features that exist in the dataframe
        available_features = [f for f in all_features if f in df.columns]
        print(f"Using {len(available_features)} features")
        
        # Store the used features for consistency in prediction
        self.used_features = available_features.copy()
        
        # Remove rows with NaN values
        df_clean = df[available_features + ['signal']].dropna()
        
        # Replace infinity values with NaN and then drop them
        df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
        df_clean = df_clean.dropna()
        
        if len(df_clean) < self.lookback_period + 10:
            raise ValueError(f"Insufficient data after cleaning: {len(df_clean)} rows")
        
        # Separate scaling for different feature types
        price_data = df_clean[price_features].values
        volume_data = df_clean[[f for f in volume_features if f in available_features]].values
        other_data = df_clean[[f for f in available_features if f not in price_features + volume_features]].values
        
        # Scale features
        price_scaled = self.price_scaler.fit_transform(price_data)
        volume_scaled = self.volume_scaler.fit_transform(volume_data) if volume_data.size > 0 else np.array([]).reshape(len(df_clean), 0)
        other_scaled = self.feature_scaler.fit_transform(other_data) if other_data.size > 0 else np.array([]).reshape(len(df_clean), 0)
        
        # Combine scaled features
        features_scaled = np.hstack([price_scaled, volume_scaled, other_scaled])
        targets = df_clean['signal'].values
        
        # Create sequences
        X, y = [], []
        for i in range(self.lookback_period, len(features_scaled)):
            X.append(features_scaled[i-self.lookback_period:i])
            y.append(targets[i])
        
        self.features = np.array(X)
        self.targets = np.array(y)
        
        print(f"Created {len(self.features)} sequences with {self.features.shape[2]} features")
        print(f"Target distribution: {dict(zip(*np.unique(self.targets, return_counts=True)))}")
        
        return self.features, self.targets
    
    def build_hybrid_model(self):
        """Build hybrid CNN-LSTM model with attention."""
        print("Building hybrid CNN-LSTM model with attention...")
        
        input_shape = (self.lookback_period, self.features.shape[2])
        inputs = Input(shape=input_shape)
        
        # CNN layers for feature extraction
        x = Conv1D(filters=64, kernel_size=3, activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = Conv1D(filters=32, kernel_size=3, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(self.dropout_rate)(x)
        
        # LSTM layers
        x = LSTM(self.lstm_units[0], return_sequences=True, 
                dropout=self.dropout_rate, recurrent_dropout=self.recurrent_dropout,
                kernel_regularizer=l1_l2(self.l1_reg, self.l2_reg))(x)
        x = BatchNormalization()(x)
        
        x = LSTM(self.lstm_units[1], return_sequences=True,
                dropout=self.dropout_rate, recurrent_dropout=self.recurrent_dropout,
                kernel_regularizer=l1_l2(self.l1_reg, self.l2_reg))(x)
        x = BatchNormalization()(x)
        
        # Attention mechanism
        if self.use_attention:
            x = MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
            x = LayerNormalization()(x)
        
        x = LSTM(self.lstm_units[2], return_sequences=False,
                dropout=self.dropout_rate, recurrent_dropout=self.recurrent_dropout,
                kernel_regularizer=l1_l2(self.l1_reg, self.l2_reg))(x)
        x = BatchNormalization()(x)
        
        # Dense layers
        x = Dense(64, activation='relu', kernel_regularizer=l1_l2(self.l1_reg, self.l2_reg))(x)
        x = Dropout(self.dropout_rate)(x)
        x = Dense(32, activation='relu', kernel_regularizer=l1_l2(self.l1_reg, self.l2_reg))(x)
        x = Dropout(self.dropout_rate)(x)
        
        # Output layer
        outputs = Dense(3, activation='softmax', name='predictions')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        # Use AdamW optimizer with weight decay
        optimizer = AdamW(learning_rate=self.learning_rate, weight_decay=0.01)
        
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        print("Hybrid model built successfully")
        print(f"Total parameters: {model.count_params():,}")
        
        return model
    
    def train_with_validation(self, validation_method='time_series_split'):
        """Train model with proper time series validation."""
        print(f"Training with {validation_method} validation...")
        
        if self.features is None:
            raise ValueError("Features must be created first")
        
        # Prepare validation strategy
        if validation_method == 'time_series_split':
            # Use TimeSeriesSplit for proper temporal validation
            tscv = TimeSeriesSplit(n_splits=5)
            splits = list(tscv.split(self.features))
            train_idx, val_idx = splits[-1]  # Use last split for final training
        else:
            # Simple temporal split
            split_point = int(len(self.features) * 0.8)
            train_idx = np.arange(split_point)
            val_idx = np.arange(split_point, len(self.features))
        
        X_train, X_val = self.features[train_idx], self.features[val_idx]
        y_train, y_val = self.targets[train_idx], self.targets[val_idx]
        
        print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
        
        # Calculate class weights
        class_weights = self._calculate_class_weights(y_train)
        
        # Enhanced callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                self.best_model_path,
                monitor='val_loss',
                save_best_only=True,
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
        
        # Load best model
        import tensorflow as tf
        self.model = tf.keras.models.load_model(self.best_model_path)
        
        # Evaluate
        val_loss, val_acc = self.model.evaluate(X_val, y_val, verbose=0)
        print(f"Final Validation - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        
        # Store validation data
        self.X_val = X_val
        self.y_val = y_val
        
        return history
    
    def _calculate_class_weights(self, y):
        """Calculate balanced class weights."""
        unique_classes, counts = np.unique(y, return_counts=True)
        total_samples = len(y)
        class_weights = {}
        
        for i, count in enumerate(counts):
            class_weights[unique_classes[i]] = total_samples / (len(unique_classes) * count)
        
        return class_weights
    
    def advanced_backtest(self, initial_capital=10000, transaction_cost=0.001, 
                         slippage=0.0005, max_position_size=0.95):
        """Enhanced backtesting with risk management."""
        print("Running advanced backtest...")
        
        if not hasattr(self, 'X_val') or not hasattr(self, 'y_val'):
            raise ValueError("Model must be trained first")
        
        # Get predictions
        y_pred_proba = self.model.predict(self.X_val, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Initialize portfolio tracking
        portfolio = {
            'capital': initial_capital,
            'position': 0,
            'position_size': 0,
            'entry_price': 0,
            'trades': [],
            'portfolio_values': [initial_capital],
            'returns': [],
            'drawdowns': []
        }
        
        # Get price data for validation period
        val_start_idx = len(self.processed_data) - len(self.X_val) - self.lookback_period
        price_data = self.processed_data.iloc[val_start_idx:val_start_idx + len(self.X_val)]
        
        for i, (actual, predicted, proba) in enumerate(zip(self.y_val, y_pred, y_pred_proba)):
            current_price = price_data['close'].iloc[i]
            confidence = np.max(proba)
            
            # Only trade with high confidence
            if confidence < 0.6:
                predicted = 0  # Force hold
            
            # Trading logic with risk management
            if predicted == 1 and portfolio['position'] <= 0:  # Buy signal
                # Calculate position size based on confidence and available capital
                position_value = portfolio['capital'] * max_position_size * confidence
                shares = position_value / current_price
                cost = shares * current_price * (1 + transaction_cost + slippage)
                
                if cost <= portfolio['capital']:
                    portfolio['capital'] -= cost
                    portfolio['position'] += shares
                    portfolio['position_size'] += position_value
                    portfolio['entry_price'] = current_price
                    
                    portfolio['trades'].append({
                        'type': 'buy',
                        'price': current_price,
                        'shares': shares,
                        'confidence': confidence,
                        'timestamp': price_data['open_time'].iloc[i]
                    })
            
            elif predicted == 2 and portfolio['position'] > 0:  # Sell signal
                # Sell all position
                proceeds = portfolio['position'] * current_price * (1 - transaction_cost - slippage)
                portfolio['capital'] += proceeds
                
                portfolio['trades'].append({
                    'type': 'sell',
                    'price': current_price,
                    'shares': portfolio['position'],
                    'confidence': confidence,
                    'timestamp': price_data['open_time'].iloc[i]
                })
                
                portfolio['position'] = 0
                portfolio['position_size'] = 0
                portfolio['entry_price'] = 0
            
            # Calculate portfolio value
            if portfolio['position'] > 0:
                position_value = portfolio['position'] * current_price
                total_value = portfolio['capital'] + position_value
            else:
                total_value = portfolio['capital']
            
            portfolio['portfolio_values'].append(total_value)
            
            # Calculate returns and drawdown
            if len(portfolio['portfolio_values']) > 1:
                daily_return = (total_value - portfolio['portfolio_values'][-2]) / portfolio['portfolio_values'][-2]
                portfolio['returns'].append(daily_return)
                
                # Calculate drawdown
                peak = max(portfolio['portfolio_values'])
                drawdown = (peak - total_value) / peak
                portfolio['drawdowns'].append(drawdown)
        
        # Calculate performance metrics
        final_value = portfolio['portfolio_values'][-1]
        total_return = (final_value - initial_capital) / initial_capital
        
        # Buy and hold comparison
        start_price = price_data['close'].iloc[0]
        end_price = price_data['close'].iloc[-1]
        buy_hold_return = (end_price - start_price) / start_price
        
        # Risk metrics
        returns_array = np.array(portfolio['returns'])
        if len(returns_array) > 0:
            sharpe_ratio = np.mean(returns_array) / np.std(returns_array) * np.sqrt(252) if np.std(returns_array) > 0 else 0
            max_drawdown = max(portfolio['drawdowns']) if portfolio['drawdowns'] else 0
            win_rate = len([t for t in portfolio['trades'] if t['type'] == 'sell' and 
                          any(t2['type'] == 'buy' and t2['timestamp'] < t['timestamp'] 
                              for t2 in portfolio['trades'])]) / max(1, len([t for t in portfolio['trades'] if t['type'] == 'sell']))
        else:
            sharpe_ratio = 0
            max_drawdown = 0
            win_rate = 0
        
        results = {
            'initial_capital': initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'buy_hold_return': buy_hold_return,
            'excess_return': total_return - buy_hold_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'num_trades': len(portfolio['trades']),
            'trades': portfolio['trades'],
            'portfolio_values': portfolio['portfolio_values'],
            'returns': portfolio['returns']
        }
        
        # Print results
        print(f"\n=== Advanced Backtest Results ===")
        print(f"Initial Capital: ${initial_capital:,.2f}")
        print(f"Final Value: ${final_value:,.2f}")
        print(f"Total Return: {total_return:.2%}")
        print(f"Buy & Hold Return: {buy_hold_return:.2%}")
        print(f"Excess Return: {results['excess_return']:.2%}")
        print(f"Sharpe Ratio: {sharpe_ratio:.3f}")
        print(f"Max Drawdown: {max_drawdown:.2%}")
        print(f"Win Rate: {win_rate:.2%}")
        print(f"Number of Trades: {len(portfolio['trades'])}")
        
        return results
    
    def predict_with_confidence(self, recent_data=None):
        """Enhanced prediction with confidence intervals."""
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        if recent_data is None:
            recent_data = self.processed_data.tail(self.lookback_period)
        
        # Use the stored feature list if available, otherwise compute it
        if hasattr(self, 'used_features') and self.used_features is not None:
            available_features = [f for f in self.used_features if f in recent_data.columns]
            print(f"Using stored feature list: {len(available_features)} features")
        else:
            # Use the same feature engineering as training
            price_features = ['open', 'high', 'low', 'close']
            volume_features = ['volume', 'volume_ratio', 'volume_trend']
            if 'buy_sell_ratio' in recent_data.columns:
                volume_features.extend(['buy_sell_ratio', 'buy_pressure'])
            
            technical_features = [
                'sma_10', 'sma_20', 'sma_50', 'ema_12', 'ema_26',
                'macd', 'macd_signal', 'macd_hist', 'macd_normalized',
                'rsi_14', 'rsi_7', 'bb_position', 'bb_squeeze',
                'williams_r', 'atr_pct'
            ]
            
            microstructure_features = [
                'spread_pct', 'body_pct', 'shadow_ratio',
                'resistance_distance', 'support_distance', 'price_position'
            ]
            
            momentum_features = [
                'return_3', 'return_7', 'return_14',
                'volatility_3', 'volatility_7', 'volatility_14'
            ]
            
            time_features = ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'is_weekend']
            
            all_features = (price_features + volume_features + technical_features + 
                           microstructure_features + momentum_features + time_features)
            
            # Filter features that exist in the dataframe (same as training)
            available_features = [f for f in all_features if f in recent_data.columns]
            print(f"Computed feature list: {len(available_features)} features")
        
        if len(available_features) < 4:
            raise ValueError("Insufficient features available for prediction")
        
        # Prepare data in the same way as training
        df_clean = recent_data[available_features].copy()
        df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
        df_clean = df_clean.fillna(method='ffill').fillna(method='bfill')
        
        # Separate scaling for different feature types (same as training)
        price_features = ['open', 'high', 'low', 'close']
        volume_features = ['volume', 'volume_ratio', 'volume_trend']
        if 'buy_sell_ratio' in df_clean.columns:
            volume_features.extend(['buy_sell_ratio', 'buy_pressure'])
        
        price_data = df_clean[[f for f in price_features if f in available_features]].values
        volume_data = df_clean[[f for f in volume_features if f in available_features]].values
        other_data = df_clean[[f for f in available_features if f not in price_features + volume_features]].values
        
        # Scale features using the same approach as training
        price_scaled = self.price_scaler.transform(price_data)
        volume_scaled = self.volume_scaler.transform(volume_data) if volume_data.size > 0 else np.array([]).reshape(len(df_clean), 0)
        other_scaled = self.feature_scaler.transform(other_data) if other_data.size > 0 else np.array([]).reshape(len(df_clean), 0)
        
        # Combine scaled features
        features_scaled = np.hstack([price_scaled, volume_scaled, other_scaled])
        
        # Ensure we have enough data for lookback period
        if len(features_scaled) < self.lookback_period:
            raise ValueError(f"Need at least {self.lookback_period} data points for prediction, got {len(features_scaled)}")
        
        # Use the last lookback_period rows for prediction
        prediction_features = features_scaled[-self.lookback_period:]
        
        # Reshape for prediction
        X_pred = prediction_features.reshape(1, self.lookback_period, -1)
        
        # Make prediction
        prediction_proba = self.model.predict(X_pred, verbose=0)
        predicted_class = np.argmax(prediction_proba[0])
        confidence = np.max(prediction_proba[0])
        
        signal_names = ['Hold', 'Buy', 'Sell']
        
        result = {
            'signal': signal_names[predicted_class],
            'confidence': confidence,
            'probabilities': {
                'Hold': float(prediction_proba[0][0]),
                'Buy': float(prediction_proba[0][1]),
                'Sell': float(prediction_proba[0][2])
            },
            'current_price': recent_data['close'].iloc[-1],
            'timestamp': recent_data['open_time'].iloc[-1] if 'open_time' in recent_data.columns else None,
            'recommendation': self._generate_recommendation(predicted_class, confidence)
        }
        
        return result
    
    def _generate_recommendation(self, predicted_class, confidence):
        """Generate trading recommendation based on prediction and confidence."""
        signal_names = ['Hold', 'Buy', 'Sell']
        signal = signal_names[predicted_class]
        
        if confidence < 0.6:
            return f"Weak {signal} signal (confidence: {confidence:.1%}). Consider waiting for stronger confirmation."
        elif confidence < 0.8:
            return f"Moderate {signal} signal (confidence: {confidence:.1%}). Proceed with caution."
        else:
            return f"Strong {signal} signal (confidence: {confidence:.1%}). High confidence trade."
    
    def plot_enhanced_results(self, history, backtest_results):
        """Create comprehensive visualization of results."""
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Training History
        ax1 = plt.subplot(3, 3, 1)
        plt.plot(history.history['loss'], label='Training Loss', alpha=0.7)
        plt.plot(history.history['val_loss'], label='Validation Loss', alpha=0.7)
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        ax2 = plt.subplot(3, 3, 2)
        plt.plot(history.history['accuracy'], label='Training Accuracy', alpha=0.7)
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy', alpha=0.7)
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        # 2. Portfolio Performance
        ax3 = plt.subplot(3, 3, 3)
        portfolio_values = backtest_results['portfolio_values']
        plt.plot(portfolio_values, label='Strategy Portfolio', linewidth=2)
        plt.axhline(y=backtest_results['initial_capital'], color='red', 
                   linestyle='--', label='Initial Capital', alpha=0.7)
        
        # Add buy/sell markers
        trades = backtest_results['trades']
        buy_points = [(i, portfolio_values[i]) for i, trade in enumerate(trades) if trade['type'] == 'buy']
        sell_points = [(i, portfolio_values[i]) for i, trade in enumerate(trades) if trade['type'] == 'sell']
        
        if buy_points:
            buy_x, buy_y = zip(*buy_points)
            plt.scatter(buy_x, buy_y, color='green', marker='^', s=100, label='Buy', alpha=0.8)
        if sell_points:
            sell_x, sell_y = zip(*sell_points)
            plt.scatter(sell_x, sell_y, color='red', marker='v', s=100, label='Sell', alpha=0.8)
        
        plt.title('Portfolio Performance')
        plt.xlabel('Time Steps')
        plt.ylabel('Portfolio Value ($)')
        plt.legend()
        plt.grid(True)
        
        # 3. Returns Distribution
        ax4 = plt.subplot(3, 3, 4)
        if backtest_results['returns']:
            plt.hist(backtest_results['returns'], bins=50, alpha=0.7, edgecolor='black')
            plt.axvline(x=np.mean(backtest_results['returns']), color='red', 
                       linestyle='--', label=f'Mean: {np.mean(backtest_results["returns"]):.4f}')
            plt.title('Returns Distribution')
            plt.xlabel('Daily Returns')
            plt.ylabel('Frequency')
            plt.legend()
            plt.grid(True)
        
        # 4. Drawdown
        ax5 = plt.subplot(3, 3, 5)
        if 'drawdowns' in backtest_results and backtest_results['drawdowns']:
            drawdowns = [-d * 100 for d in backtest_results['drawdowns']]  # Convert to percentage
            plt.plot(drawdowns, color='red', alpha=0.7)
            plt.fill_between(range(len(drawdowns)), drawdowns, 0, alpha=0.3, color='red')
            plt.title('Drawdown Over Time')
            plt.xlabel('Time Steps')
            plt.ylabel('Drawdown (%)')
            plt.grid(True)
        
        # 5. Confusion Matrix
        ax6 = plt.subplot(3, 3, 6)
        if hasattr(self, 'y_val'):
            y_pred = np.argmax(self.model.predict(self.X_val, verbose=0), axis=1)
            cm = confusion_matrix(self.y_val, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Hold', 'Buy', 'Sell'],
                       yticklabels=['Hold', 'Buy', 'Sell'])
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
        
        # 6. Feature Importance (approximation using model weights)
        ax7 = plt.subplot(3, 3, 7)
        try:
            # Get first layer weights as proxy for feature importance
            first_layer_weights = self.model.layers[0].get_weights()[0]
            feature_importance = np.mean(np.abs(first_layer_weights), axis=1)
            
            # Use top 10 features
            top_indices = np.argsort(feature_importance)[-10:]
            top_importance = feature_importance[top_indices]
            
            plt.barh(range(len(top_importance)), top_importance)
            plt.title('Feature Importance (Approximation)')
            plt.xlabel('Importance Score')
            plt.ylabel('Feature Index')
            plt.grid(True)
        except:
            plt.text(0.5, 0.5, 'Feature importance\nnot available', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Feature Importance')
        
        # 7. Performance Metrics Summary
        ax8 = plt.subplot(3, 3, 8)
        metrics = {
            'Total Return': f"{backtest_results['total_return']:.2%}",
            'Buy & Hold': f"{backtest_results['buy_hold_return']:.2%}",
            'Excess Return': f"{backtest_results['excess_return']:.2%}",
            'Sharpe Ratio': f"{backtest_results['sharpe_ratio']:.3f}",
            'Max Drawdown': f"{backtest_results['max_drawdown']:.2%}",
            'Win Rate': f"{backtest_results['win_rate']:.2%}",
            'Num Trades': f"{backtest_results['num_trades']}"
        }
        
        y_pos = np.arange(len(metrics))
        plt.barh(y_pos, [float(v.rstrip('%')) if '%' in v else float(v) for v in metrics.values()])
        plt.yticks(y_pos, list(metrics.keys()))
        plt.title('Performance Metrics')
        plt.grid(True, axis='x')
        
        # Add text annotations
        for i, (key, value) in enumerate(metrics.items()):
            plt.text(0.1, i, value, va='center', fontweight='bold')
        
        # 8. Price vs Signals
        ax9 = plt.subplot(3, 3, 9)
        if hasattr(self, 'processed_data') and len(self.processed_data) > 0:
            # Plot recent price data
            recent_data = self.processed_data.tail(100)
            plt.plot(recent_data['close'], label='Close Price', alpha=0.7)
            
            # Add moving averages if available
            if 'sma_20' in recent_data.columns:
                plt.plot(recent_data['sma_20'], label='SMA 20', alpha=0.5, linestyle='--')
            if 'sma_50' in recent_data.columns:
                plt.plot(recent_data['sma_50'], label='SMA 50', alpha=0.5, linestyle='--')
            
            plt.title('Recent Price Action')
            plt.xlabel('Time')
            plt.ylabel('Price')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def save_model_and_scalers(self, model_path='optimized_crypto_lstm.keras', 
                              scalers_path='optimized_scalers.joblib'):
        """Save the trained model and all scalers."""
        if self.model is None:
            raise ValueError("No model to save")
        
        # Save model
        self.model.save(model_path)
        
        # Save all scalers and feature information
        scalers = {
            'feature_scaler': self.feature_scaler,
            'price_scaler': self.price_scaler,
            'volume_scaler': self.volume_scaler,
            'feature_list': getattr(self, 'used_features', None)  # Save the feature list used
        }
        joblib.dump(scalers, scalers_path)
        
        # Save configuration
        config = {
            'symbol': self.symbol,
            'interval': self.interval,
            'lookback_period': self.lookback_period,
            'prediction_horizon': self.prediction_horizon,
            'use_attention': self.use_attention,
            'model_params': {
                'lstm_units': self.lstm_units,
                'dropout_rate': self.dropout_rate,
                'learning_rate': self.learning_rate
            }
        }
        joblib.dump(config, 'model_config.joblib')
        
        print(f"Model saved to {model_path}")
        print(f"Scalers saved to {scalers_path}")
        print("Configuration saved to model_config.joblib")
    
    def load_model_and_scalers(self, model_path='optimized_crypto_lstm.keras', 
                              scalers_path='optimized_scalers.joblib'):
        """Load a pre-trained model and scalers."""
        # Load model
        self.model = tf.keras.models.load_model(model_path)
        
        # Load scalers and feature information
        scalers = joblib.load(scalers_path)
        self.feature_scaler = scalers['feature_scaler']
        self.price_scaler = scalers['price_scaler']
        self.volume_scaler = scalers['volume_scaler']
        
        # Load the feature list if available
        if 'feature_list' in scalers and scalers['feature_list'] is not None:
            self.used_features = scalers['feature_list']
            print(f"Loaded feature list with {len(self.used_features)} features")
        
        # Load configuration if available
        try:
            config = joblib.load('model_config.joblib')
            self.symbol = config['symbol']
            self.interval = config['interval']
            self.lookback_period = config['lookback_period']
            self.prediction_horizon = config['prediction_horizon']
            self.use_attention = config['use_attention']
            print("Configuration loaded successfully")
        except:
            print("Configuration file not found, using current settings")
        
        print(f"Model loaded from {model_path}")
        print(f"Scalers loaded from {scalers_path}")


def run_optimized_pipeline(csv_path):
    """
    Complete optimized pipeline execution.
    
    Args:
        csv_path: Path to the CSV file with OHLCV data
    """
    print("=== Optimized Crypto LSTM Trading Pipeline ===")
    
    # Initialize pipeline with optimized parameters
    pipeline = OptimizedCryptoLSTMPipeline(
        symbol='BTCUSDT',
        interval='1h',
        lookback_period=48,  # Reduced for better performance
        prediction_horizon=1,
        use_attention=True
    )
    
    try:
        # Load and validate data
        pipeline.load_data_from_csv(csv_path)
        
        # Check data quality
        if len(pipeline.raw_data) < 500:
            print("Warning: Small dataset detected. Consider gathering more data for better results.")
        
        # Process data with advanced indicators
        pipeline.add_technical_indicators()
        
        # Create features and targets
        pipeline.create_features_and_targets()
        
        # Build hybrid model
        pipeline.build_hybrid_model()
        
        # Train with proper validation
        print("\nStarting model training...")
        history = pipeline.train_with_validation(validation_method='time_series_split')
        
        # Make predictions
        prediction = pipeline.predict_with_confidence()
        print(f"\n=== Latest Trading Signal ===")
        print(f"Signal: {prediction['signal']}")
        print(f"Confidence: {prediction['confidence']:.1%}")
        print(f"Current Price: ${prediction['current_price']:,.4f}")
        print(f"Recommendation: {prediction['recommendation']}")
        print("Probabilities:")
        for signal, prob in prediction['probabilities'].items():
            print(f"  {signal}: {prob:.1%}")
        
        # Run advanced backtest
        print(f"\n=== Running Advanced Backtest ===")
        backtest_results = pipeline.advanced_backtest(
            initial_capital=10000,
            transaction_cost=0.001,
            slippage=0.0005
        )
        
        # Create comprehensive visualizations
        pipeline.plot_enhanced_results(history, backtest_results)
        
        # Save model
        pipeline.save_model_and_scalers()
        
        # Performance summary
        print(f"\n=== Performance Summary ===")
        print(f"Strategy vs Buy & Hold:")
        print(f"  Strategy Return: {backtest_results['total_return']:.2%}")
        print(f"  Buy & Hold Return: {backtest_results['buy_hold_return']:.2%}")
        print(f"  Excess Return: {backtest_results['excess_return']:.2%}")
        print(f"\nRisk Metrics:")
        print(f"  Sharpe Ratio: {backtest_results['sharpe_ratio']:.3f}")
        print(f"  Maximum Drawdown: {backtest_results['max_drawdown']:.2%}")
        print(f"  Win Rate: {backtest_results['win_rate']:.1%}")
        
        return pipeline, history, backtest_results
        
    except Exception as e:
        print(f"Error in pipeline execution: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None


# Utility functions for data preprocessing
def preprocess_binance_data(df):
    """
    Preprocess raw Binance data to ensure compatibility.
    
    Args:
        df: DataFrame with Binance OHLCV data
    """
    # Ensure proper column names
    required_columns = ['open_time', 'open', 'high', 'low', 'close', 'volume']
    
    # Check if all required columns exist
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Convert data types
    numeric_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Handle datetime
    df['open_time'] = pd.to_datetime(df['open_time'])
    
    # Remove any rows with NaN values in critical columns
    df = df.dropna(subset=required_columns)
    
    # Sort by time
    df = df.sort_values('open_time').reset_index(drop=True)
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['open_time'], keep='first')
    
    print(f"Preprocessed data: {len(df)} rows, {df.shape[1]} columns")
    print(f"Date range: {df['open_time'].min()} to {df['open_time'].max()}")
    
    return df

def validate_data_quality(df):
    """
    Validate data quality and provide recommendations.
    
    Args:
        df: DataFrame with OHLCV data
    """
    print("\n=== Data Quality Report ===")
    
    # Basic statistics
    print(f"Total data points: {len(df)}")
    print(f"Date range: {(df['open_time'].max() - df['open_time'].min()).days} days")
    
    # Check for gaps in time series
    time_diff = df['open_time'].diff()
    expected_interval = time_diff.mode()[0] if len(time_diff.mode()) > 0 else pd.Timedelta(hours=1)
    gaps = time_diff[time_diff > expected_interval * 1.5]
    
    if len(gaps) > 0:
        print(f"Warning: Found {len(gaps)} time gaps in data")
        print(f"Largest gap: {gaps.max()}")
    
    # Check for price anomalies
    price_changes = df['close'].pct_change()
    extreme_changes = price_changes[abs(price_changes) > 0.5]  # >50% changes
    
    if len(extreme_changes) > 0:
        print(f"Warning: Found {len(extreme_changes)} extreme price changes (>50%)")
    
    # Volume analysis
    zero_volume = (df['volume'] == 0).sum()
    if zero_volume > 0:
        print(f"Warning: {zero_volume} periods with zero volume")
    
    # Data completeness
    completeness = (1 - df.isnull().sum() / len(df)) * 100
    print("\nData completeness:")
    for col, pct in completeness.items():
        if pct < 100:
            print(f"  {col}: {pct:.1f}%")
    
    # Recommendations
    print("\n=== Recommendations ===")
    if len(df) < 1000:
        print("• Dataset is small (<1000 points). Consider gathering more data for better model performance.")
    elif len(df) < 2000:
        print("• Dataset is moderate (1000-2000 points). Results may be acceptable but more data would help.")
    else:
        print("• Dataset size is good (>2000 points). Should provide reliable results.")
    
    if len(gaps) > len(df) * 0.01:  # More than 1% gaps
        print("• High number of time gaps detected. Consider filling gaps or using gap-aware preprocessing.")
    
    if zero_volume > len(df) * 0.05:  # More than 5% zero volume
        print("• High number of zero-volume periods. Consider filtering or data source quality.")

def create_sample_data():
    """Create sample data for testing if no CSV is available."""
    print("Creating sample data for testing...")
    
    # Generate synthetic crypto-like data
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=2000, freq='1H')
    
    # Generate price data with trend and volatility
    base_price = 0.45
    trend = np.linspace(0, 0.1, len(dates))
    volatility = np.random.normal(0, 0.02, len(dates))
    
    prices = []
    current_price = base_price
    
    for i in range(len(dates)):
        # Add trend and random walk
        price_change = trend[i] / len(dates) + volatility[i]
        current_price = max(0.01, current_price * (1 + price_change))
        prices.append(current_price)
    
    # Create OHLCV data
    df = pd.DataFrame()
    df['open_time'] = dates
    df['close'] = prices
    
    # Generate OHLC from close prices
    for i in range(len(df)):
        volatility = abs(np.random.normal(0, 0.01))
        df.loc[i, 'open'] = df.loc[i-1, 'close'] if i > 0 else df.loc[i, 'close']
        df.loc[i, 'high'] = df.loc[i, 'close'] * (1 + volatility)
        df.loc[i, 'low'] = df.loc[i, 'close'] * (1 - volatility)
    
    # Generate volume data
    df['volume'] = np.random.lognormal(8, 1, len(df))
    
    # Add optional columns for more realistic data
    df['close_time'] = df['open_time'] + pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=1)
    df['quote_asset_volume'] = df['volume'] * df['close']
    df['number_of_trades'] = np.random.poisson(15, len(df))
    df['taker_buy_base_asset_volume'] = df['volume'] * np.random.uniform(0.3, 0.7, len(df))
    df['taker_buy_quote_asset_volume'] = df['taker_buy_base_asset_volume'] * df['close']
    
    print(f"Generated {len(df)} synthetic data points")
    return df

def run_complete_analysis(csv_path=None, symbol='CRYPTO', save_results=True):
    """
    Run complete analysis with comprehensive reporting.
    
    Args:
        csv_path: Path to CSV file, if None creates sample data
        symbol: Symbol name for reporting
        save_results: Whether to save model and results
    """
    print("=" * 60)
    print("🚀 OPTIMIZED CRYPTO LSTM TRADING PIPELINE")
    print("=" * 60)
    
    try:
        # Load or create data
        if csv_path and os.path.exists(csv_path):
            print(f"📊 Loading data from: {csv_path}")
            df = pd.read_csv(csv_path)
            df = preprocess_binance_data(df)
        else:
            print("📊 No CSV provided or file not found. Creating sample data...")
            df = create_sample_data()
            csv_path = "sample_crypto_data.csv"
            df.to_csv(csv_path, index=False)
            print(f"💾 Sample data saved to: {csv_path}")
        
        # Validate data quality
        validate_data_quality(df)
        
        # Initialize optimized pipeline
        pipeline = OptimizedCryptoLSTMPipeline(
            symbol=symbol,
            interval='1h',
            lookback_period=48,
            prediction_horizon=1,
            use_attention=True
        )
        
        # Load data into pipeline
        pipeline.raw_data = df
        
        # Process data
        print("\n🔧 Processing data and adding technical indicators...")
        pipeline.add_technical_indicators()
        
        # Create features
        print("🏗️ Creating features and targets...")
        pipeline.create_features_and_targets()
        
        # Build model
        print("🧠 Building hybrid CNN-LSTM model...")
        pipeline.build_hybrid_model()
        
        # Train model
        print("🎯 Training model with time series validation...")
        history = pipeline.train_with_validation()
        
        # Generate prediction
        print("🔮 Generating latest prediction...")
        prediction = pipeline.predict_with_confidence()
        
        # Run backtest
        print("📈 Running advanced backtest...")
        results = pipeline.advanced_backtest()
        
        # Create visualizations
        print("📊 Creating comprehensive visualizations...")
        pipeline.plot_enhanced_results(history, results)
        
        # Save results if requested
        if save_results:
            print("💾 Saving model and results...")
            pipeline.save_model_and_scalers()
            
            # Save results summary
            summary = {
                'symbol': symbol,
                'data_points': len(df),
                'features_created': pipeline.features.shape[2],
                'model_params': pipeline.model.count_params(),
                'backtest_results': results,
                'latest_prediction': prediction,
                'timestamp': datetime.now().isoformat()
            }
            
            with open('analysis_summary.json', 'w') as f:
                import json
                # Convert numpy types to Python types for JSON serialization
                def convert_numpy(obj):
                    if isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, pd.Timestamp):
                        return obj.isoformat()
                    return obj
                
                json_summary = json.loads(json.dumps(summary, default=convert_numpy))
                json.dump(json_summary, f, indent=2)
            
            print("📁 Results saved to analysis_summary.json")
        
        # Print final summary
        print("\n" + "=" * 60)
        print("🎉 ANALYSIS COMPLETE - SUMMARY REPORT")
        print("=" * 60)
        print(f"📊 Symbol: {symbol}")
        print(f"📈 Data Points: {len(df):,}")
        print(f"🧠 Model Parameters: {pipeline.model.count_params():,}")
        print(f"🎯 Features Used: {pipeline.features.shape[2]}")
        print()
        print("🔮 LATEST PREDICTION:")
        print(f"   Signal: {prediction['signal']}")
        print(f"   Confidence: {prediction['confidence']:.1%}")
        print(f"   Recommendation: {prediction['recommendation']}")
        print()
        print("📈 BACKTEST PERFORMANCE:")
        print(f"   Total Return: {results['total_return']:.2%}")
        print(f"   Buy & Hold: {results['buy_hold_return']:.2%}")
        print(f"   Excess Return: {results['excess_return']:.2%}")
        print(f"   Sharpe Ratio: {results['sharpe_ratio']:.3f}")
        print(f"   Max Drawdown: {results['max_drawdown']:.2%}")
        print(f"   Win Rate: {results['win_rate']:.1%}")
        print(f"   Trades: {results['num_trades']}")
        print("=" * 60)
        
        return pipeline, history, results
        
    except Exception as e:
        print(f"\n❌ Error in analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None

# Example usage for your specific dataset
if __name__ == "__main__":
    # Option 1: Use your CSV file
    csv_file_path = "historical_exports/BTCUSDT_1year_hourly_20250601_025148.csv"  # Replace with your actual file path
    
    # Option 2: Use sample data for testing
    # csv_file_path = None
    
    # Run complete analysis
    pipeline, history, results = run_complete_analysis(
        csv_path=csv_file_path,
        symbol='BTCUSDT',  # Or your actual symbol
        save_results=True
    )
    
    if pipeline is not None:
        print("\n🎊 Pipeline completed successfully!")
        
        # Example of making additional predictions
        print("\n🔄 Making additional prediction...")
        latest_prediction = pipeline.predict_with_confidence()
        print(f"Signal: {latest_prediction['signal']} "
              f"(Confidence: {latest_prediction['confidence']:.1%})")
        
        # Example of loading saved model later
        print("\n💾 Example: Loading saved model...")
        new_pipeline = OptimizedCryptoLSTMPipeline()
        try:
            new_pipeline.load_model_and_scalers()
            print("✅ Model loaded successfully!")
        except:
            print("ℹ️ No saved model found (this is normal on first run)")
    else:
        print("\n❌ Pipeline failed. Please check your data and try again.")
        print("\nTroubleshooting tips:")
        print("1. Ensure your CSV has columns: open_time,open,high,low,close,volume")
        print("2. Check that dates are in proper format")
        print("3. Verify numeric columns contain valid numbers")
        print("4. Make sure you have at least 500+ data points") 