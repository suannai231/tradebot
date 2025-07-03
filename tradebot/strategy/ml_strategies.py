import numpy as np
import pandas as pd
import logging
from collections import deque, defaultdict
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import asyncio
import aiohttp
import json

# Machine Learning imports
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import VotingClassifier

# Deep Learning imports
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU, Attention
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

# Reinforcement Learning imports
try:
    import gymnasium as gym
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False

# Sentiment Analysis imports
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    from textblob import TextBlob
    SENTIMENT_AVAILABLE = True
except ImportError:
    SENTIMENT_AVAILABLE = False

# Technical Analysis
try:
    import ta
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False

from tradebot.common.models import PriceTick, Signal, Side

logger = logging.getLogger("ml_strategies")

@dataclass
class MLStrategyConfig:
    """Configuration for ML strategies"""
    # Data parameters (reduced for testing)
    lookback_period: int = 20  # Reduced from 50
    prediction_horizon: int = 3  # Reduced from 5
    min_data_points: int = 30  # Reduced from 100
    
    # Model parameters (more aggressive)
    retrain_frequency: int = 500  # Reduced from 1000
    confidence_threshold: float = 0.3  # Reduced from 0.6
    
    # Feature engineering
    use_technical_indicators: bool = True
    use_sentiment: bool = True
    use_market_data: bool = True
    
    # Risk management (more aggressive)
    stop_loss: float = 0.05  # Increased from 0.02
    take_profit: float = 0.10  # Increased from 0.05
    max_position_size: float = 0.2  # Increased from 0.1


class FeatureEngineer:
    """Advanced feature engineering for ML strategies"""
    
    def __init__(self, config: MLStrategyConfig):
        self.config = config
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators"""
        if not TA_AVAILABLE:
            return df
            
        # Price-based indicators
        df['sma_5'] = ta.trend.sma_indicator(df['close'], window=5)
        df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
        df['ema_12'] = ta.trend.ema_indicator(df['close'], window=12)
        df['ema_26'] = ta.trend.ema_indicator(df['close'], window=26)
        
        # Momentum indicators
        df['rsi'] = ta.momentum.rsi(df['close'], window=14)
        df['stoch'] = ta.momentum.stoch(df['high'], df['low'], df['close'])
        df['williams_r'] = ta.momentum.williams_r(df['high'], df['low'], df['close'])
        df['macd'] = ta.trend.macd_diff(df['close'])
        
        # Volatility indicators
        df['bb_upper'] = ta.volatility.bollinger_hband(df['close'])
        df['bb_lower'] = ta.volatility.bollinger_lband(df['close'])
        df['bb_width'] = ta.volatility.bollinger_wband(df['close'])
        df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
        
        # Volume indicators
        if 'volume' in df.columns:
            df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
            df['vwap'] = ta.volume.volume_weighted_average_price(df['high'], df['low'], df['close'], df['volume'])
            df['volume_sma'] = ta.volume.volume_sma(df['close'], df['volume'])
        
        # Price action features
        df['price_change'] = df['close'].pct_change()
        df['price_change_5'] = df['close'].pct_change(5)
        df['price_change_20'] = df['close'].pct_change(20)
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']
        
        # Lagged features
        for lag in [1, 2, 3, 5]:
            df[f'close_lag_{lag}'] = df['close'].shift(lag)
            df[f'volume_lag_{lag}'] = df['volume'].shift(lag) if 'volume' in df.columns else 0
        
        return df
    
    def create_features(self, prices: List[float], volumes: List[float] = None, 
                       opens: List[float] = None, highs: List[float] = None, 
                       lows: List[float] = None) -> np.ndarray:
        """Create feature matrix from price data"""
        if len(prices) < self.config.lookback_period:
            return np.array([])
        
        # Create DataFrame
        data = {'close': prices}
        if volumes:
            data['volume'] = volumes
        if opens:
            data['open'] = opens
        if highs:
            data['high'] = highs
        if lows:
            data['low'] = lows
        
        df = pd.DataFrame(data)
        
        # Calculate technical indicators
        if self.config.use_technical_indicators:
            df = self.calculate_technical_indicators(df)
        
        # Remove NaN values
        df = df.dropna()
        
        if len(df) == 0:
            return np.array([])
        
        # Select features (exclude target variables)
        feature_cols = [col for col in df.columns if col not in ['target', 'signal']]
        features = df[feature_cols].values
        
        # Store feature names
        self.feature_names = feature_cols
        
        return features


class EnsembleMLStrategy:
    """Ensemble learning strategy combining multiple ML models"""
    
    def __init__(self, config: MLStrategyConfig):
        self.config = config
        self.feature_engineer = FeatureEngineer(config)
        
        # Initialize models
        self.rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.xgb_classifier = xgb.XGBClassifier(n_estimators=100, random_state=42)
        self.lgb_classifier = lgb.LGBMClassifier(n_estimators=100, random_state=42)
        
        # Ensemble model
        self.ensemble = VotingClassifier(
            estimators=[
                ('rf', self.rf_classifier),
                ('xgb', self.xgb_classifier),
                ('lgb', self.lgb_classifier)
            ],
            voting='soft'
        )
        
        # Data storage
        self.prices: Dict[str, deque] = defaultdict(lambda: deque(maxlen=200))
        self.volumes: Dict[str, deque] = defaultdict(lambda: deque(maxlen=200))
        self.opens: Dict[str, deque] = defaultdict(lambda: deque(maxlen=200))
        self.highs: Dict[str, deque] = defaultdict(lambda: deque(maxlen=200))
        self.lows: Dict[str, deque] = defaultdict(lambda: deque(maxlen=200))
        
        self.last_side: Dict[str, Side] = defaultdict(lambda: None)
        self.last_signals: Dict[str, datetime] = defaultdict(lambda: datetime.min.replace(tzinfo=timezone.utc))
        self.tick_count: Dict[str, int] = defaultdict(int)
        
        # Model state
        self.models_trained = False
        self.scaler = StandardScaler()
        
    def update_data(self, tick: PriceTick):
        """Update internal data structures"""
        symbol = tick.symbol
        self.prices[symbol].append(tick.price)
        self.volumes[symbol].append(tick.volume if hasattr(tick, 'volume') else 1000)
        self.opens[symbol].append(tick.open if hasattr(tick, 'open') else tick.price)
        self.highs[symbol].append(tick.high if hasattr(tick, 'high') else tick.price)
        self.lows[symbol].append(tick.low if hasattr(tick, 'low') else tick.price)
        
    def create_labels(self, prices: List[float], horizon: int = 5) -> np.ndarray:
        """Create binary labels for classification (1 for price increase, 0 for decrease)"""
        if len(prices) < horizon + 1:
            return np.array([])
        
        labels = []
        for i in range(len(prices) - horizon):
            future_price = prices[i + horizon]
            current_price = prices[i]
            label = 1 if future_price > current_price else 0
            labels.append(label)
        
        return np.array(labels)
    
    def train_models(self, symbol: str):
        """Train ensemble models"""
        if len(self.prices[symbol]) < self.config.min_data_points:
            return
        
        # Create features
        features = self.feature_engineer.create_features(
            list(self.prices[symbol]),
            list(self.volumes[symbol]),
            list(self.opens[symbol]),
            list(self.highs[symbol]),
            list(self.lows[symbol])
        )
        
        if len(features) == 0:
            return
        
        # Create labels
        labels = self.create_labels(list(self.prices[symbol]), self.config.prediction_horizon)
        
        if len(labels) == 0 or len(features) != len(labels):
            return
        
        # Align features and labels
        min_len = min(len(features), len(labels))
        features = features[:min_len]
        labels = labels[:min_len]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42, shuffle=False
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train ensemble model
        try:
            self.ensemble.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = self.ensemble.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            
            logger.info(f"Ensemble model trained for {symbol}. Accuracy: {accuracy:.3f}, Precision: {precision:.3f}")
            self.models_trained = True
            
        except Exception as e:
            logger.error(f"Error training ensemble model: {e}")
    
    def predict(self, symbol: str) -> Tuple[float, float]:
        """Make prediction using ensemble model"""
        if not self.models_trained:
            return 0.5, 0.0
        
        # Create features for current state
        features = self.feature_engineer.create_features(
            list(self.prices[symbol]),
            list(self.volumes[symbol]),
            list(self.opens[symbol]),
            list(self.highs[symbol]),
            list(self.lows[symbol])
        )
        
        if len(features) == 0:
            return 0.5, 0.0
        
        # Use most recent feature vector
        current_features = features[-1:].reshape(1, -1)
        current_features_scaled = self.scaler.transform(current_features)
        
        # Get prediction probabilities
        try:
            probabilities = self.ensemble.predict_proba(current_features_scaled)[0]
            buy_probability = probabilities[1] if len(probabilities) > 1 else 0.5
            confidence = max(probabilities)
            return buy_probability, confidence
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return 0.5, 0.0
    
    async def _log_signal(self, signal: Signal, strategy_name: str):
        """Log signal to performance tracker"""
        try:
            # Import here to avoid circular imports
            from tradebot.strategy.ml_service import get_performance_tracker
            
            tracker = await get_performance_tracker()
            if tracker:
                await tracker.log_signal(strategy_name, signal.symbol, signal)
        except Exception as e:
            logger.error(f"Error logging signal to performance tracker: {e}")
    
    def on_tick(self, tick: PriceTick) -> Optional[Signal]:
        """Process tick and generate signals"""
        symbol = tick.symbol
        self.update_data(tick)
        
        # Check if we have enough data
        if len(self.prices[symbol]) < self.config.min_data_points:
            return None
        
        # Check if we should generate a signal (reduced frequency for more signals)
        time_since_last = datetime.now(timezone.utc) - self.last_signals[symbol]
        if time_since_last.total_seconds() < 60:  # Reduced from 300 to 60 seconds
            return None
        
        # Try to get prediction
        buy_probability, confidence = self.predict(symbol)
        
        # If models aren't trained yet, use simple technical analysis as fallback
        if confidence == 0.0:
            # Simple fallback: RSI-based signals
            prices = list(self.prices[symbol])
            if len(prices) >= 14:
                # Calculate simple RSI
                deltas = np.diff(prices[-15:])
                gains = np.where(deltas > 0, deltas, 0)
                losses = np.where(deltas < 0, -deltas, 0)
                avg_gain = np.mean(gains)
                avg_loss = np.mean(losses)
                
                if avg_loss > 0:
                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))
                    
                    if rsi < 30:  # Oversold
                        buy_probability = 0.7
                        confidence = 0.5
                    elif rsi > 70:  # Overbought
                        buy_probability = 0.3
                        confidence = 0.5
        
        # Generate signal with lower threshold
        if buy_probability > 0.6 and confidence > 0.2:  # Lowered from 0.6 confidence
            if self.last_side[symbol] != Side.buy:
                self.last_side[symbol] = Side.buy
                self.last_signals[symbol] = tick.timestamp
                logger.info(f"Ensemble ML buy signal for {symbol}: prob={buy_probability:.3f}, conf={confidence:.3f}")
                signal = Signal(
                    symbol=symbol, 
                    side=Side.buy, 
                    price=tick.price, 
                    timestamp=tick.timestamp, 
                    confidence=confidence
                )
                # Log signal asynchronously
                asyncio.create_task(self._log_signal(signal, "ensemble_ml"))
                return signal
        
        elif buy_probability < 0.4 and confidence > 0.2:  # Lowered from 0.6 confidence
            if self.last_side[symbol] == Side.buy:
                self.last_side[symbol] = Side.sell
                self.last_signals[symbol] = tick.timestamp
                logger.info(f"Ensemble ML sell signal for {symbol}: prob={buy_probability:.3f}, conf={confidence:.3f}")
                signal = Signal(
                    symbol=symbol, 
                    side=Side.sell, 
                    price=tick.price, 
                    timestamp=tick.timestamp, 
                    confidence=confidence
                )
                # Log signal asynchronously
                asyncio.create_task(self._log_signal(signal, "ensemble_ml"))
                return signal
        
        return None


class LSTMStrategy:
    """LSTM-based time series forecasting strategy"""
    
    def __init__(self, config: MLStrategyConfig):
        self.config = config
        self.feature_engineer = FeatureEngineer(config)
        
        # Data storage
        self.prices: Dict[str, deque] = defaultdict(lambda: deque(maxlen=200))
        self.volumes: Dict[str, deque] = defaultdict(lambda: deque(maxlen=200))
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, MinMaxScaler] = {}
        
        self.last_side: Dict[str, Side] = defaultdict(lambda: None)
        self.last_signals: Dict[str, datetime] = defaultdict(lambda: datetime.min.replace(tzinfo=timezone.utc))
        self.tick_count: Dict[str, int] = defaultdict(int)
        
        if not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow not available. LSTM strategy will not work.")
    
    def update_data(self, tick: PriceTick):
        """Update internal data structures"""
        symbol = tick.symbol
        self.prices[symbol].append(tick.price)
        self.volumes[symbol].append(tick.volume if hasattr(tick, 'volume') else 1000)
    
    def create_sequences(self, data: np.ndarray, lookback: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training"""
        X, y = [], []
        for i in range(lookback, len(data)):
            X.append(data[i-lookback:i])
            y.append(data[i])
        return np.array(X), np.array(y)
    
    def build_lstm_model(self, input_shape: Tuple[int, int]):
        """Build LSTM model architecture"""
        if not TENSORFLOW_AVAILABLE:
            return None
            
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model
    
    def train_model(self, symbol: str):
        """Train LSTM model for a symbol"""
        if not TENSORFLOW_AVAILABLE or len(self.prices[symbol]) < self.config.min_data_points:
            return
        
        # Create features
        features = self.feature_engineer.create_features(
            list(self.prices[symbol]),
            list(self.volumes[symbol])
        )
        
        if len(features) == 0:
            return
        
        # Scale features
        scaler = MinMaxScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Create sequences
        X, y = self.create_sequences(features_scaled, self.config.lookback_period)
        
        if len(X) < 10:  # Need minimum sequences
            return
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Build and train model
        try:
            model = self.build_lstm_model((X_train.shape[1], X_train.shape[2]))
            model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=0)
            
            # Store model and scaler
            self.models[symbol] = model
            self.scalers[symbol] = scaler
            
            logger.info(f"LSTM model trained for {symbol}")
            
        except Exception as e:
            logger.error(f"Error training LSTM model: {e}")
    
    def predict(self, symbol: str) -> Tuple[float, float]:
        """Make prediction using LSTM model"""
        if symbol not in self.models or not TENSORFLOW_AVAILABLE:
            return 0.5, 0.0
        
        # Create features
        features = self.feature_engineer.create_features(
            list(self.prices[symbol]),
            list(self.volumes[symbol])
        )
        
        if len(features) < self.config.lookback_period:
            return 0.5, 0.0
        
        # Scale features
        features_scaled = self.scalers[symbol].transform(features)
        
        # Create sequence for prediction
        sequence = features_scaled[-self.config.lookback_period:].reshape(1, -1, features_scaled.shape[1])
        
        try:
            # Make prediction
            prediction = self.models[symbol].predict(sequence, verbose=0)[0][0]
            
            # Convert prediction to probability (simplified)
            current_price = self.prices[symbol][-1]
            predicted_price = prediction * current_price  # Simplified conversion
            
            # Calculate probability based on predicted price movement
            price_change = (predicted_price - current_price) / current_price
            probability = 0.5 + np.tanh(price_change * 10) * 0.5  # Convert to 0-1 range
            
            confidence = min(abs(price_change) * 10, 1.0)  # Higher change = higher confidence
            
            return probability, confidence
            
        except Exception as e:
            logger.error(f"Error making LSTM prediction: {e}")
            return 0.5, 0.0
    
    async def _log_signal(self, signal: Signal, strategy_name: str):
        """Log signal to performance tracker"""
        try:
            # Import here to avoid circular imports
            from tradebot.strategy.ml_service import get_performance_tracker
            
            tracker = await get_performance_tracker()
            if tracker:
                await tracker.log_signal(strategy_name, signal.symbol, signal)
        except Exception as e:
            logger.error(f"Error logging signal to performance tracker: {e}")
    
    def on_tick(self, tick: PriceTick) -> Optional[Signal]:
        """Process tick and generate signals"""
        symbol = tick.symbol
        self.update_data(tick)
        self.tick_count[symbol] += 1
        
        # Check if we should retrain
        if self.tick_count[symbol] % self.config.retrain_frequency == 0:
            self.train_model(symbol)
        
        # Check if we have enough data
        if len(self.prices[symbol]) < self.config.lookback_period:
            return None
        
        # Check if we should generate a signal
        time_since_last = datetime.now(timezone.utc) - self.last_signals[symbol]
        if time_since_last.total_seconds() < 300:  # 5 minutes minimum
            return None
        
        # Make prediction
        buy_probability, confidence = self.predict(symbol)
        
        # Generate signal based on prediction
        if buy_probability > 0.6 and confidence > self.config.confidence_threshold:
            if self.last_side[symbol] != Side.buy:
                self.last_side[symbol] = Side.buy
                self.last_signals[symbol] = tick.timestamp
                logger.info(f"LSTM buy signal for {symbol}: prob={buy_probability:.3f}, conf={confidence:.3f}")
                signal = Signal(
                    symbol=symbol, 
                    side=Side.buy, 
                    price=tick.price, 
                    timestamp=tick.timestamp, 
                    confidence=confidence
                )
                # Log signal asynchronously
                asyncio.create_task(self._log_signal(signal, "lstm_ml"))
                return signal
        
        elif buy_probability < 0.4 and confidence > self.config.confidence_threshold:
            if self.last_side[symbol] == Side.buy:
                self.last_side[symbol] = Side.sell
                self.last_signals[symbol] = tick.timestamp
                logger.info(f"LSTM sell signal for {symbol}: prob={buy_probability:.3f}, conf={confidence:.3f}")
                signal = Signal(
                    symbol=symbol, 
                    side=Side.sell, 
                    price=tick.price, 
                    timestamp=tick.timestamp, 
                    confidence=confidence
                )
                # Log signal asynchronously
                asyncio.create_task(self._log_signal(signal, "lstm_ml"))
                return signal
        
        return None


class SentimentStrategy:
    """Sentiment analysis-based trading strategy"""
    
    def __init__(self, config: MLStrategyConfig):
        self.config = config
        self.feature_engineer = FeatureEngineer(config)
        
        # Simplified sentiment analysis (avoid problematic transformers)
        self.sentiment_analyzer = None  # Disable transformers to avoid segfaults
        print("Sentiment strategy initialized with simplified analysis (no transformers)")
        
        # Data storage
        self.prices: Dict[str, deque] = defaultdict(lambda: deque(maxlen=200))
        self.sentiment_scores: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))
        self.news_cache: Dict[str, List[Dict]] = defaultdict(list)
        
        self.last_side: Dict[str, Side] = defaultdict(lambda: None)
        self.last_signals: Dict[str, datetime] = defaultdict(lambda: datetime.min.replace(tzinfo=timezone.utc))
        self.last_sentiment_update: Dict[str, datetime] = defaultdict(lambda: datetime.min.replace(tzinfo=timezone.utc))
        
    def update_data(self, tick: PriceTick):
        """Update internal data structures"""
        symbol = tick.symbol
        self.prices[symbol].append(tick.price)
    
    def get_simple_sentiment(self, symbol: str) -> float:
        """Get simplified sentiment score based on price movement"""
        try:
            # Simple sentiment based on recent price movement
            if len(self.prices[symbol]) < 2:
                return 0.0
            
            # Calculate recent price changes with safety checks
            prices = list(self.prices[symbol])
            
            # Safety check for extremely small prices (penny stocks)
            if any(p <= 0 or p < 1e-10 for p in prices):
                logger.warning(f"Extremely small prices detected for {symbol}, using neutral sentiment")
                return 0.0
            
            # Calculate price changes with safety checks
            if len(prices) >= 5:
                # Check for division by zero
                if prices[-3] == 0 or prices[-5] == 0:
                    return 0.0
                short_change = (prices[-1] - prices[-3]) / prices[-3]  # 3-period change
                medium_change = (prices[-1] - prices[-5]) / prices[-5]  # 5-period change
            else:
                # Check for division by zero
                if prices[-2] == 0:
                    return 0.0
                short_change = (prices[-1] - prices[-2]) / prices[-2]
                medium_change = short_change
            
            # Safety check for extreme values
            if abs(short_change) > 100 or abs(medium_change) > 100:
                logger.warning(f"Extreme price changes detected for {symbol}, capping values")
                short_change = max(-1, min(1, short_change))
                medium_change = max(-1, min(1, medium_change))
            
            # Combine changes for sentiment (positive change = positive sentiment)
            sentiment = (short_change * 0.6 + medium_change * 0.4) * 5  # Scale and weight
            sentiment = np.tanh(sentiment)  # Bound to [-1, 1]
            
            # Final safety check
            if not np.isfinite(sentiment):
                logger.warning(f"Non-finite sentiment calculated for {symbol}, using neutral")
                sentiment = 0.0
            
            # Store sentiment score
            self.sentiment_scores[symbol].append(sentiment)
            self.last_sentiment_update[symbol] = datetime.now(timezone.utc)
            
            return sentiment
            
        except Exception as e:
            logger.error(f"Error calculating sentiment for {symbol}: {e}")
            return 0.0
    
    async def fetch_news_sentiment(self, symbol: str) -> float:
        """Fetch and analyze news sentiment for a symbol"""
        if not SENTIMENT_AVAILABLE:
            return 0.0
        
        # Check if we should update sentiment (every 30 minutes)
        time_since_update = datetime.now(timezone.utc) - self.last_sentiment_update[symbol]
        if time_since_update.total_seconds() < 1800:  # 30 minutes
            return np.mean(list(self.sentiment_scores[symbol])) if self.sentiment_scores[symbol] else 0.0
        
        try:
            # Simulate news sentiment (in real implementation, you'd fetch from news APIs)
            # For now, we'll use a simple random sentiment based on price movement
            if len(self.prices[symbol]) < 2:
                sentiment = 0.0
            else:
                price_change = (self.prices[symbol][-1] - self.prices[symbol][-2]) / self.prices[symbol][-2]
                # Positive price change = positive sentiment
                sentiment = np.tanh(price_change * 5)  # Scale and bound to [-1, 1]
            
            self.sentiment_scores[symbol].append(sentiment)
            self.last_sentiment_update[symbol] = datetime.now(timezone.utc)
            
            return sentiment
            
        except Exception as e:
            logger.error(f"Error fetching sentiment for {symbol}: {e}")
            return 0.0
    
    def analyze_text_sentiment(self, text: str) -> float:
        """Analyze sentiment of text using simplified approach"""
        # Simplified sentiment analysis to avoid segfaults
        try:
            # Simple keyword-based sentiment
            positive_words = ['good', 'great', 'excellent', 'positive', 'up', 'rise', 'gain', 'profit', 'bull']
            negative_words = ['bad', 'terrible', 'negative', 'down', 'fall', 'loss', 'bear', 'decline']
            
            text_lower = text.lower()
            positive_count = sum(1 for word in positive_words if word in text_lower)
            negative_count = sum(1 for word in negative_words if word in text_lower)
            
            if positive_count > negative_count:
                return 0.6 + (positive_count - negative_count) * 0.1
            elif negative_count > positive_count:
                return 0.4 - (negative_count - positive_count) * 0.1
            else:
                return 0.5
                
        except Exception as e:
            logger.error(f"Error analyzing text sentiment: {e}")
            return 0.5
    
    async def _log_signal(self, signal: Signal, strategy_name: str):
        """Log signal to performance tracker"""
        try:
            # Import here to avoid circular imports
            from tradebot.strategy.ml_service import get_performance_tracker
            
            tracker = await get_performance_tracker()
            if tracker:
                await tracker.log_signal(strategy_name, signal.symbol, signal)
        except Exception as e:
            logger.error(f"Error logging signal to performance tracker: {e}")
    
    def on_tick(self, tick: PriceTick) -> Optional[Signal]:
        """Process tick and generate signals - SIMPLIFIED STABLE VERSION"""
        symbol = tick.symbol
        
        try:
            # Simple data storage without complex operations
            self.prices[symbol].append(tick.price)
            
            # Basic requirements check
            if len(self.prices[symbol]) < 10:
                return None
            
            # Skip if price is too small (prevents numerical issues)
            if tick.price < 0.01:
                return None
            
            # Simple time throttling
            time_since_last = datetime.now(timezone.utc) - self.last_signals[symbol]
            if time_since_last.total_seconds() < 300:  # 5 minutes
                return None
            
            # Ultra-simple sentiment based on price momentum only
            prices = list(self.prices[symbol])
            if len(prices) < 5:
                return None
            
            # Simple price-based sentiment (no complex calculations)
            recent_prices = prices[-5:]
            price_trend = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
            
            # Simple buy/sell logic based on trend
            if price_trend > 0.02:  # 2% upward trend = positive sentiment
                if self.last_side[symbol] != Side.buy:
                    self.last_side[symbol] = Side.buy
                    self.last_signals[symbol] = tick.timestamp
                    
                    logger.info(f"Simple sentiment buy signal for {symbol}: trend={price_trend:.3f}")
                    return Signal(
                        symbol=symbol,
                        side=Side.buy,
                        price=tick.price,
                        timestamp=tick.timestamp,
                        confidence=min(abs(price_trend) * 10, 1.0)
                    )
            
            elif price_trend < -0.02:  # 2% downward trend = negative sentiment
                if self.last_side[symbol] == Side.buy:
                    self.last_side[symbol] = Side.sell
                    self.last_signals[symbol] = tick.timestamp
                    
                    logger.info(f"Simple sentiment sell signal for {symbol}: trend={price_trend:.3f}")
                    return Signal(
                        symbol=symbol,
                        side=Side.sell,
                        price=tick.price,
                        timestamp=tick.timestamp,
                        confidence=min(abs(price_trend) * 10, 1.0)
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Error in simplified sentiment strategy for {symbol}: {e}")
            return None
    
    def calculate_rsi(self, prices: List[float], period: int = 14) -> Optional[float]:
        """Calculate RSI with safety checks"""
        try:
            if len(prices) < period + 1:
                return None
            
            # Safety check for extremely small or invalid prices
            if any(p <= 0 or not np.isfinite(p) for p in prices):
                return None
            
            deltas = np.diff(prices)
            
            # Safety check for deltas
            if not np.all(np.isfinite(deltas)):
                return None
                
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            avg_gain = np.mean(gains[-period:])
            avg_loss = np.mean(losses[-period:])
            
            # Safety checks for averages
            if not np.isfinite(avg_gain) or not np.isfinite(avg_loss):
                return None
            
            if avg_loss == 0:
                return 100
            
            rs = avg_gain / avg_loss
            
            # Safety check for RS
            if not np.isfinite(rs):
                return None
                
            rsi = 100 - (100 / (1 + rs))
            
            # Final safety check
            if not np.isfinite(rsi):
                return None
                
            return rsi
            
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return None


# Factory function to create ML strategies
def create_ml_strategy(strategy_type: str, config: MLStrategyConfig = None) -> Any:
    """Factory function to create ML strategy instances"""
    if config is None:
        config = MLStrategyConfig()
    
    if strategy_type == "ensemble":
        return EnsembleMLStrategy(config)
    elif strategy_type == "lstm":
        return LSTMStrategy(config)
    elif strategy_type == "sentiment":
        return SentimentStrategy(config)
    else:
        raise ValueError(f"Unknown ML strategy type: {strategy_type}")


# Example usage and testing
if __name__ == "__main__":
    # Test the strategies
    config = MLStrategyConfig()
    
    # Test ensemble strategy
    ensemble_strategy = create_ml_strategy("ensemble", config)
    
    # Test LSTM strategy
    lstm_strategy = create_ml_strategy("lstm", config)
    
    # Test sentiment strategy
    sentiment_strategy = create_ml_strategy("sentiment", config)
    
    print("ML strategies created successfully!") 