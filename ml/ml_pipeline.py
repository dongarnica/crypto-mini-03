import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import joblib

# Technical indicators
import talib

# Import our Binance client
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from binance.binance_client import BinanceUSClient

class CryptoLSTMPipeline:
    """
    LSTM-based cryptocurrency trading pipeline with buy/sell/hold recommendations.
    
    This class handles:
    - Data collection from Binance
    - Feature engineering with technical indicators
    - LSTM model training for price prediction
    - Trading signal generation (Buy/Sell/Hold)
    - Backtesting and performance evaluation
    """
    
    def __init__(self, symbol='BTCUSDT', interval='1h', lookback_period=60):
        """
        Initialize the LSTM trading pipeline.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            interval: Data interval ('1h', '4h', '1d', etc.)
            lookback_period: Number of timesteps to look back for LSTM
        """
        self.symbol = symbol
        self.interval = interval
        self.lookback_period = lookback_period
        
        # Initialize Binance client
        self.client = BinanceUSClient()
        
        # Model and scalers
        self.model = None
        self.price_scaler = MinMaxScaler()
        self.feature_scaler = MinMaxScaler()
        
        # Data storage
        self.raw_data = None
        self.processed_data = None
        self.features = None
        self.targets = None
        
        # Model parameters
        self.lstm_units = [100, 50, 25]
        self.dropout_rate = 0.2
        self.learning_rate = 0.001
        self.epochs = 100
        self.batch_size = 32
        
    def fetch_data(self, days_back=365):
        """
        Fetch historical data from Binance.
        
        Args:
            days_back: Number of days of historical data to fetch
        """
        print(f"Fetching {days_back} days of {self.symbol} data at {self.interval} interval...")
        
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days_back)
        
        # Fetch data in chunks to avoid API limits
        all_data = []
        current_start = start_time
        
        while current_start < end_time:
            current_end = min(current_start + timedelta(days=30), end_time)
            
            chunk_data = self.client.get_candlestick_data(
                symbol=self.symbol,
                interval=self.interval,
                start_time=current_start,
                end_time=current_end,
                limit=1000
            )
            
            if not chunk_data.empty:
                all_data.append(chunk_data)
            
            current_start = current_end
        
        if all_data:
            self.raw_data = pd.concat(all_data, ignore_index=True)
            self.raw_data = self.raw_data.drop_duplicates(subset=['open_time'])
            self.raw_data = self.raw_data.sort_values('open_time').reset_index(drop=True)
            print(f"Fetched {len(self.raw_data)} data points")
        else:
            raise ValueError("No data fetched from Binance")
    
    def load_data_from_csv(self, csv_path):
        """
        Load data from existing CSV file.
        
        Args:
            csv_path: Path to the CSV file
        """
        print(f"Loading data from {csv_path}...")
        self.raw_data = pd.read_csv(csv_path)
        self.raw_data['open_time'] = pd.to_datetime(self.raw_data['open_time'])
        self.raw_data = self.raw_data.sort_values('open_time').reset_index(drop=True)
        print(f"Loaded {len(self.raw_data)} data points")
    
    def add_technical_indicators(self):
        """
        Add technical indicators to the dataset.
        """
        print("Adding technical indicators...")
        df = self.raw_data.copy()
        
        # Price-based indicators
        df['sma_20'] = talib.SMA(df['close'], timeperiod=20)
        df['sma_50'] = talib.SMA(df['close'], timeperiod=50)
        df['ema_12'] = talib.EMA(df['close'], timeperiod=12)
        df['ema_26'] = talib.EMA(df['close'], timeperiod=26)
        
        # MACD
        df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(df['close'])
        
        # RSI
        df['rsi'] = talib.RSI(df['close'], timeperiod=14)
        
        # Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(df['close'])
        
        # Stochastic
        df['stoch_k'], df['stoch_d'] = talib.STOCH(df['high'], df['low'], df['close'])
        
        # Williams %R
        df['williams_r'] = talib.WILLR(df['high'], df['low'], df['close'])
        
        # Average True Range
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'])
        
        # Volume indicators
        df['volume_sma'] = talib.SMA(df['volume'], timeperiod=20)
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Price change indicators
        df['price_change'] = df['close'].pct_change()
        df['high_low_ratio'] = (df['high'] - df['low']) / df['close']
        df['open_close_ratio'] = (df['close'] - df['open']) / df['open']
        
        # Volatility (rolling standard deviation)
        df['volatility'] = df['close'].rolling(window=20).std()
        
        self.processed_data = df
        print("Technical indicators added successfully")
    
    def create_features_and_targets(self):
        """
        Create feature matrix and target variables for LSTM training.
        """
        print("Creating features and targets...")
        
        if self.processed_data is None:
            raise ValueError("Data must be processed first. Call add_technical_indicators()")
        
        df = self.processed_data.copy()
        
        # Select features for the model
        feature_columns = [
            'open', 'high', 'low', 'close', 'volume',
            'sma_20', 'sma_50', 'ema_12', 'ema_26',
            'macd', 'macd_signal', 'macd_hist',
            'rsi', 'bb_upper', 'bb_middle', 'bb_lower',
            'stoch_k', 'stoch_d', 'williams_r', 'atr',
            'volume_ratio', 'price_change', 'high_low_ratio',
            'open_close_ratio', 'volatility'
        ]
        
        # Create target variable (future price movement)
        # 0: Hold, 1: Buy, 2: Sell
        df['future_price'] = df['close'].shift(-1)
        df['price_change_future'] = (df['future_price'] - df['close']) / df['close']
        
        # Define thresholds for buy/sell signals
        buy_threshold = 0.02  # 2% increase
        sell_threshold = -0.02  # 2% decrease
        
        df['signal'] = 0  # Hold
        df.loc[df['price_change_future'] > buy_threshold, 'signal'] = 1  # Buy
        df.loc[df['price_change_future'] < sell_threshold, 'signal'] = 2  # Sell
        
        # Remove rows with NaN values
        df = df.dropna()
        
        # Prepare features
        features = df[feature_columns].values
        targets = df['signal'].values
        
        # Scale features
        features_scaled = self.feature_scaler.fit_transform(features)
        
        # Create sequences for LSTM
        X, y = [], []
        for i in range(self.lookback_period, len(features_scaled)):
            X.append(features_scaled[i-self.lookback_period:i])
            y.append(targets[i])
        
        self.features = np.array(X)
        self.targets = np.array(y)
        
        print(f"Created {len(self.features)} sequences with {self.features.shape[2]} features")
        print(f"Target distribution: {np.bincount(self.targets)}")
    
    def build_model(self):
        """
        Build the LSTM model architecture.
        """
        print("Building LSTM model...")
        
        model = Sequential()
        
        # First LSTM layer
        model.add(LSTM(
            units=self.lstm_units[0],
            return_sequences=True,
            input_shape=(self.lookback_period, self.features.shape[2])
        ))
        model.add(Dropout(self.dropout_rate))
        model.add(BatchNormalization())
        
        # Second LSTM layer
        model.add(LSTM(
            units=self.lstm_units[1],
            return_sequences=True
        ))
        model.add(Dropout(self.dropout_rate))
        model.add(BatchNormalization())
        
        # Third LSTM layer
        model.add(LSTM(
            units=self.lstm_units[2],
            return_sequences=False
        ))
        model.add(Dropout(self.dropout_rate))
        model.add(BatchNormalization())
        
        # Dense layers for classification
        model.add(Dense(50, activation='relu'))
        model.add(Dropout(self.dropout_rate))
        
        model.add(Dense(25, activation='relu'))
        model.add(Dropout(self.dropout_rate))
        
        # Output layer (3 classes: Hold, Buy, Sell)
        model.add(Dense(3, activation='softmax'))
        
        # Compile model
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        print("Model built successfully")
        print(model.summary())
    
    def train_model(self, test_size=0.2, validation_size=0.2):
        """
        Train the LSTM model.
        
        Args:
            test_size: Proportion of data for testing
            validation_size: Proportion of training data for validation
        """
        print("Training LSTM model...")
        
        if self.features is None or self.targets is None:
            raise ValueError("Features and targets must be created first")
        
        # Split data chronologically
        total_samples = len(self.features)
        train_size = int(total_samples * (1 - test_size))
        
        X_train = self.features[:train_size]
        X_test = self.features[train_size:]
        y_train = self.targets[:train_size]
        y_test = self.targets[train_size:]
        
        # Calculate class weights to handle imbalance
        class_weights = {}
        unique_classes, counts = np.unique(y_train, return_counts=True)
        total_samples = len(y_train)
        
        for i, count in enumerate(counts):
            class_weights[unique_classes[i]] = total_samples / (len(unique_classes) * count)
        
        print(f"Class weights: {class_weights}")
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        )
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=validation_size,
            callbacks=[early_stopping, reduce_lr],
            class_weight=class_weights,
            verbose=1
        )
        
        # Evaluate on test set
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"Test Accuracy: {test_accuracy:.4f}")
        
        # Predictions on test set
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_classes, 
                                  target_names=['Hold', 'Buy', 'Sell']))
        
        # Store test data for further analysis
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = y_pred_classes
        
        return history
    
    def predict_signal(self, recent_data=None):
        """
        Predict trading signal for the most recent data.
        
        Args:
            recent_data: Optional DataFrame with recent data. If None, uses latest data.
            
        Returns:
            Dictionary with prediction details
        """
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        if recent_data is None:
            # Use the most recent data from our dataset
            if self.processed_data is None:
                raise ValueError("No processed data available")
            recent_data = self.processed_data.tail(self.lookback_period)
        
        # Prepare features
        feature_columns = [
            'open', 'high', 'low', 'close', 'volume',
            'sma_20', 'sma_50', 'ema_12', 'ema_26',
            'macd', 'macd_signal', 'macd_hist',
            'rsi', 'bb_upper', 'bb_middle', 'bb_lower',
            'stoch_k', 'stoch_d', 'williams_r', 'atr',
            'volume_ratio', 'price_change', 'high_low_ratio',
            'open_close_ratio', 'volatility'
        ]
        
        features = recent_data[feature_columns].values
        features_scaled = self.feature_scaler.transform(features)
        
        # Reshape for prediction
        X_pred = features_scaled.reshape(1, self.lookback_period, -1)
        
        # Make prediction
        prediction = self.model.predict(X_pred, verbose=0)
        predicted_class = np.argmax(prediction[0])
        confidence = np.max(prediction[0])
        
        signal_names = ['Hold', 'Buy', 'Sell']
        
        result = {
            'signal': signal_names[predicted_class],
            'confidence': confidence,
            'probabilities': {
                'Hold': prediction[0][0],
                'Buy': prediction[0][1],
                'Sell': prediction[0][2]
            },
            'current_price': recent_data['close'].iloc[-1],
            'timestamp': recent_data['open_time'].iloc[-1]
        }
        
        return result
    
    def backtest_strategy(self, initial_capital=10000, transaction_cost=0.001):
        """
        Backtest the trading strategy.
        
        Args:
            initial_capital: Starting capital
            transaction_cost: Transaction cost as a fraction (0.001 = 0.1%)
            
        Returns:
            Dictionary with backtest results
        """
        print("Running backtest...")
        
        if self.X_test is None or self.y_test is None:
            raise ValueError("Model must be trained and tested first")
        
        # Get test data dates (approximate)
        total_samples = len(self.processed_data)
        test_start_idx = int(total_samples * 0.8)
        test_data = self.processed_data.iloc[test_start_idx:test_start_idx + len(self.y_test)]
        
        capital = initial_capital
        position = 0  # 0: no position, 1: long position
        trades = []
        portfolio_values = [initial_capital]
        
        for i, (actual, predicted) in enumerate(zip(self.y_test, self.y_pred)):
            current_price = test_data['close'].iloc[i]
            
            # Trading logic
            if predicted == 1 and position == 0:  # Buy signal and no position
                shares = capital / current_price
                position = 1
                capital = 0
                trade_cost = shares * current_price * transaction_cost
                capital -= trade_cost
                
                trades.append({
                    'type': 'buy',
                    'price': current_price,
                    'shares': shares,
                    'timestamp': test_data['open_time'].iloc[i]
                })
                
            elif predicted == 2 and position == 1:  # Sell signal and have position
                capital = shares * current_price
                trade_cost = capital * transaction_cost
                capital -= trade_cost
                position = 0
                
                trades.append({
                    'type': 'sell',
                    'price': current_price,
                    'shares': shares,
                    'timestamp': test_data['open_time'].iloc[i]
                })
            
            # Calculate portfolio value
            if position == 1:
                portfolio_value = shares * current_price
            else:
                portfolio_value = capital
            
            portfolio_values.append(portfolio_value)
        
        # Final portfolio value
        final_value = portfolio_values[-1]
        total_return = (final_value - initial_capital) / initial_capital
        
        # Calculate buy and hold return
        buy_hold_return = (test_data['close'].iloc[-1] - test_data['close'].iloc[0]) / test_data['close'].iloc[0]
        
        results = {
            'initial_capital': initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'buy_hold_return': buy_hold_return,
            'excess_return': total_return - buy_hold_return,
            'num_trades': len(trades),
            'trades': trades,
            'portfolio_values': portfolio_values
        }
        
        print(f"Backtest Results:")
        print(f"Initial Capital: ${initial_capital:,.2f}")
        print(f"Final Value: ${final_value:,.2f}")
        print(f"Total Return: {total_return:.2%}")
        print(f"Buy & Hold Return: {buy_hold_return:.2%}")
        print(f"Excess Return: {results['excess_return']:.2%}")
        print(f"Number of Trades: {len(trades)}")
        
        return results
    
    def save_model(self, model_path='crypto_lstm_model.h5', scaler_path='scalers.joblib'):
        """
        Save the trained model and scalers.
        
        Args:
            model_path: Path to save the Keras model
            scaler_path: Path to save the scalers
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        # Save model
        self.model.save(model_path)
        
        # Save scalers
        scalers = {
            'feature_scaler': self.feature_scaler,
            'price_scaler': self.price_scaler
        }
        joblib.dump(scalers, scaler_path)
        
        print(f"Model saved to {model_path}")
        print(f"Scalers saved to {scaler_path}")
    
    def load_model(self, model_path='crypto_lstm_model.h5', scaler_path='scalers.joblib'):
        """
        Load a pre-trained model and scalers.
        
        Args:
            model_path: Path to the saved Keras model
            scaler_path: Path to the saved scalers
        """
        # Load model
        self.model = tf.keras.models.load_model(model_path)
        
        # Load scalers
        scalers = joblib.load(scaler_path)
        self.feature_scaler = scalers['feature_scaler']
        self.price_scaler = scalers['price_scaler']
        
        print(f"Model loaded from {model_path}")
        print(f"Scalers loaded from {scaler_path}")
    
    def plot_training_history(self, history):
        """
        Plot training history.
        
        Args:
            history: Training history from model.fit()
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def plot_backtest_results(self, backtest_results):
        """
        Plot backtest results.
        
        Args:
            backtest_results: Results from backtest_strategy()
        """
        portfolio_values = backtest_results['portfolio_values']
        
        plt.figure(figsize=(12, 6))
        plt.plot(portfolio_values, label='Strategy Portfolio Value')
        plt.axhline(y=backtest_results['initial_capital'], color='red', 
                   linestyle='--', label='Initial Capital')
        plt.title('Portfolio Value Over Time')
        plt.xlabel('Time Steps')
        plt.ylabel('Portfolio Value ($)')
        plt.legend()
        plt.grid(True)
        plt.show()


def run_example_pipeline():
    """
    Example usage of the LSTM trading pipeline.
    """
    print("=== Crypto LSTM Trading Pipeline Example ===")
    
    # Initialize pipeline
    pipeline = CryptoLSTMPipeline(symbol='BTCUSDT', interval='1h', lookback_period=60)
    
    # Load data from existing CSV (or fetch new data)
    csv_path = '/workspaces/crypto-mini-03/binance_exports/BTCUSDT_1h_data.csv'
    
    try:
        pipeline.load_data_from_csv(csv_path)
    except Exception as e:
        print(f"Could not load from CSV: {e}")
        print("Fetching new data from Binance...")
        pipeline.fetch_data(days_back=90)
    
    # Process data and add technical indicators
    pipeline.add_technical_indicators()
    
    # Create features and targets
    pipeline.create_features_and_targets()
    
    # Build and train model
    pipeline.build_model()
    history = pipeline.train_model()
    
    # Make a prediction on recent data
    prediction = pipeline.predict_signal()
    print("\n=== Latest Trading Signal ===")
    print(f"Signal: {prediction['signal']}")
    print(f"Confidence: {prediction['confidence']:.2%}")
    print(f"Current Price: ${prediction['current_price']:,.2f}")
    print(f"Probabilities: {prediction['probabilities']}")
    
    # Run backtest
    backtest_results = pipeline.backtest_strategy()
    
    # Save model
    pipeline.save_model('btc_lstm_model.h5', 'btc_scalers.joblib')
    
    return pipeline, history, backtest_results


if __name__ == "__main__":
    # Run the example pipeline
    pipeline, history, results = run_example_pipeline()