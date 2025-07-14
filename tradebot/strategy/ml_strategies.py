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
import os
import pickle
from pathlib import Path

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
    use_technical_indicators: bool = False  # Temporarily disabled for debugging
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
            
        # Ensure required columns exist
        required_columns = ['close']
        for col in required_columns:
            if col not in df.columns:
                return df
                
        # Add missing OHLC columns if they don't exist
        if 'open' not in df.columns:
            df['open'] = df['close']
        if 'high' not in df.columns:
            df['high'] = df['close']
        if 'low' not in df.columns:
            df['low'] = df['close']
            
        try:
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
                
        except Exception as e:
            logger.warning(f"Error calculating technical indicators: {e}")
            return df
            df[f'volume_lag_{lag}'] = df['volume'].shift(lag) if 'volume' in df.columns else 0
        
        return df
    
    def create_features(self, prices: List[float], volumes: List[float] = None, 
                       opens: List[float] = None, highs: List[float] = None, 
                       lows: List[float] = None) -> np.ndarray:
        """Create feature matrix from price data with robust error handling"""
        try:
            if len(prices) < 2:
                return np.array([0] * 5)  # Return default features
            
            # Clean and validate input data
            clean_prices = [float(p) for p in prices if p is not None and p > 0]
            if len(clean_prices) < 2:
                return np.array([0] * 5)
            
            # Calculate simple features without technical indicators
            features = []
            
            # Price returns
            returns = [(clean_prices[i] - clean_prices[i-1]) / clean_prices[i-1] 
                      for i in range(1, len(clean_prices))]
            
            # Basic price features
            features.extend([
                returns[-1] if returns else 0,  # Latest return
                np.mean(returns[-5:]) if len(returns) >= 5 else 0,  # Short term mean
                np.std(returns[-5:]) if len(returns) >= 5 else 0,   # Short term volatility
            ])
            
            # Simple moving averages
            if len(clean_prices) >= 5:
                sma_5 = np.mean(clean_prices[-5:])
                features.append((clean_prices[-1] - sma_5) / sma_5 if sma_5 > 0 else 0)
            else:
                features.append(0)
            
            # Volume feature (if available)
            if volumes and len(volumes) > 0:
                clean_volumes = [float(v) for v in volumes if v is not None and v > 0]
                if len(clean_volumes) >= 10:
                    vol_mean = np.mean(clean_volumes[-10:])
                    features.append(clean_volumes[-1] / vol_mean if vol_mean > 0 else 1)
                else:
                    features.append(1)
            else:
                features.append(1)
            
            # Ensure we have exactly 5 features
            while len(features) < 5:
                features.append(0)
            
            result = np.array(features[:5])
            
            # Replace any inf or nan values
            result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Store simple feature names
            self.feature_names = ['latest_return', 'mean_return', 'volatility', 'sma_ratio', 'volume_ratio']
            
            return result
            
        except Exception as e:
            logger.error(f"Feature engineering failed: {e}")
            return np.array([0] * 5)  # Return default features


class EnsembleMLStrategy:
    """Ensemble learning strategy combining multiple ML models"""
    
    def __init__(self, config: MLStrategyConfig):
        self.config = config
        self.feature_engineer = FeatureEngineer(config)
        
        # GPU status detection
        self.gpu_enabled = torch.cuda.is_available()
        self.mps_enabled = torch.backends.mps.is_available()
        
        # Initialize models with GPU support where available
        self.rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # XGBoost with GPU support (if available)
        try:
            # Try GPU first (Apple Silicon doesn't support XGBoost GPU, but CUDA does)
            self.xgb_classifier = xgb.XGBClassifier(
                n_estimators=100, 
                random_state=42,
                tree_method='gpu_hist' if self.gpu_enabled else 'hist',
                gpu_id=0 if self.gpu_enabled else None
            )
            if self.gpu_enabled:
                print("✅ XGBoost GPU (CUDA) support enabled")
            else:
                print("⚠️  XGBoost using CPU (CUDA GPU not available)")
        except:
            # Fallback to CPU
            self.xgb_classifier = xgb.XGBClassifier(n_estimators=100, random_state=42)
            print("⚠️  XGBoost fallback to CPU")
        
        # LightGBM with GPU support
        try:
            self.lgb_classifier = lgb.LGBMClassifier(
                n_estimators=100, 
                random_state=42,
                device='gpu' if self.gpu_enabled else 'cpu'
            )
            if self.gpu_enabled:
                print("✅ LightGBM GPU (CUDA) support enabled")
            else:
                print("⚠️  LightGBM using CPU (CUDA GPU not available)")
        except:
            # Fallback to CPU
            self.lgb_classifier = lgb.LGBMClassifier(n_estimators=100, random_state=42)
            print("⚠️  LightGBM fallback to CPU")
        
        # Random Forest status (always CPU)
        print("⚠️  Random Forest using CPU (scikit-learn limitation)")
        
        # Overall GPU status summary
        if self.gpu_enabled:
            print("🚀 Ensemble ML Strategy: GPU-accelerated (CUDA)")
        elif self.mps_enabled:
            print("🚀 Ensemble ML Strategy: MPS-accelerated (Apple Silicon)")
        else:
            print("💻 Ensemble ML Strategy: CPU-only")
        
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
        
        # Model persistence
        self.models_dir = Path("models") / "ensemble"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Auto-load pre-trained models for common symbols
        self._auto_load_models()
    
    def _auto_load_models(self):
        """Auto-load pre-trained models for common symbols"""
        common_symbols = ['AAPL', 'GOOGL', 'MSFT', 'TNXP']
        for symbol in common_symbols:
            try:
                self.load_model(symbol)
                logger.info(f"Auto-loaded ensemble model for {symbol}")
            except Exception as e:
                logger.debug(f"No pre-trained ensemble model found for {symbol}: {e}")
        
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
        
        # Create features for multiple time points (sliding window)
        prices_list = list(self.prices[symbol])
        volumes_list = list(self.volumes[symbol])
        opens_list = list(self.opens[symbol])
        highs_list = list(self.highs[symbol])
        lows_list = list(self.lows[symbol])
        
        # Create features for each time window
        features_list = []
        for i in range(10, len(prices_list)):  # Start from index 10 to have enough history
            window_prices = prices_list[i-10:i+1]
            window_volumes = volumes_list[i-10:i+1] if volumes_list else None
            window_opens = opens_list[i-10:i+1] if opens_list else None
            window_highs = highs_list[i-10:i+1] if highs_list else None
            window_lows = lows_list[i-10:i+1] if lows_list else None
            
            features = self.feature_engineer.create_features(
                window_prices, window_volumes, window_opens, window_highs, window_lows
            )
            features_list.append(features)
        
        if len(features_list) == 0:
            return
        
        # Convert to numpy array
        features = np.array(features_list)
        
        # Create labels
        labels = self.create_labels(list(self.prices[symbol]), self.config.prediction_horizon)
        
        # Align features and labels by trimming labels to match features
        if len(labels) > len(features):
            labels = labels[len(labels) - len(features):]
        elif len(features) > len(labels):
            features = features[:len(labels)]
        
        if len(features) == 0 or len(labels) == 0:
            return
        
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
        """Process signal through position manager"""
        try:
            # Import here to avoid circular imports
            from tradebot.strategy.position_manager import process_ml_signal
            
            action = await process_ml_signal(strategy_name, signal.symbol, signal)
            logger.debug(f"Position manager action for {signal.symbol}: {action}")
        except Exception as e:
            logger.error(f"Error processing signal through position manager: {e}")
    
    def on_tick(self, tick: PriceTick) -> Optional[Signal]:
        """Process tick and generate signals"""
        symbol = tick.symbol
        self.update_data(tick)
        self.tick_count[symbol] += 1
        
        # Check if we should train models (every retrain_frequency ticks)
        if self.tick_count[symbol] % self.config.retrain_frequency == 0:
            if len(self.prices[symbol]) >= self.config.min_data_points:
                logger.info(f"Training ensemble models for {symbol} (tick #{self.tick_count[symbol]})")
                self.train_models(symbol)
        
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
                # Log signal asynchronously (if event loop is running)
                try:
                    asyncio.create_task(self._log_signal(signal, "ensemble_ml"))
                except RuntimeError:
                    # No event loop running (e.g., in tests)
                    pass
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
                # Log signal asynchronously (if event loop is running)
                try:
                    asyncio.create_task(self._log_signal(signal, "ensemble_ml"))
                except RuntimeError:
                    # No event loop running (e.g., in tests)
                    pass
                return signal
        
        return None

    def save_model(self, symbol: str = "global") -> Optional[str]:
        """Save the ensemble model to disk"""
        try:
            if not self.models_trained:
                logger.warning("No trained model to save")
                return None
            
            # Create symbol-specific directory
            symbol_dir = self.models_dir / symbol
            symbol_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = symbol_dir / f"model_{timestamp}.pkl"
            
            # Save model data
            model_data = {
                'strategy_type': 'ensemble',
                'symbol': symbol,
                'timestamp': timestamp,
                'model_state': {
                    'ensemble': self.ensemble,
                    'scaler': self.scaler,
                    'models_trained': self.models_trained
                },
                'config': self.config
            }
            
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"Ensemble model saved: {model_path}")
            return str(model_path)
            
        except Exception as e:
            logger.error(f"Failed to save ensemble model: {e}")
            return None

    def load_model(self, symbol: str = "global", model_path: str = None) -> bool:
        """Load the ensemble model from disk"""
        try:
            if model_path is None:
                # Find the latest model file
                symbol_dir = self.models_dir / symbol
                if not symbol_dir.exists():
                    logger.warning(f"No model directory found for symbol: {symbol}")
                    return False
                
                model_files = list(symbol_dir.glob("model_*.pkl"))
                if not model_files:
                    logger.warning(f"No model files found for symbol: {symbol}")
                    return False
                
                # Get the latest model file
                model_path = max(model_files, key=lambda p: p.stat().st_mtime)
            
            # Load model data
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Restore model state
            self.ensemble = model_data['model_state']['ensemble']
            self.scaler = model_data['model_state']['scaler']
            self.models_trained = model_data['model_state']['models_trained']
            
            logger.info(f"Ensemble model loaded: {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load ensemble model: {e}")
            return False


class LSTMStrategy:
    """LSTM-based time series forecasting strategy"""
    
    def __init__(self, config: MLStrategyConfig):
        self.config = config
        self.feature_engineer = FeatureEngineer(config)
        
        # Data storage
        self.prices: Dict[str, deque] = defaultdict(lambda: deque(maxlen=200))
        self.volumes: Dict[str, deque] = defaultdict(lambda: deque(maxlen=200))
        self.opens: Dict[str, deque] = defaultdict(lambda: deque(maxlen=200))
        self.highs: Dict[str, deque] = defaultdict(lambda: deque(maxlen=200))
        self.lows: Dict[str, deque] = defaultdict(lambda: deque(maxlen=200))
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, MinMaxScaler] = {}
        
        self.last_side: Dict[str, Side] = defaultdict(lambda: None)
        self.last_signals: Dict[str, datetime] = defaultdict(lambda: datetime.min.replace(tzinfo=timezone.utc))
        self.tick_count: Dict[str, int] = defaultdict(int)
        
        # Setup device for PyTorch LSTM (fallback if TensorFlow not available)
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"LSTM Strategy using device: {self.device}")
        
        if not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow not available. Using PyTorch LSTM with GPU support.")
        
        # Model persistence
        self.models_dir = Path("models") / "lstm"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # PyTorch models for persistence
        self.pytorch_models: Dict[str, Any] = {}
        self.pytorch_optimizers: Dict[str, Any] = {}
        
        # Track training status per symbol
        self.models_trained: Dict[str, bool] = defaultdict(bool)
        
        # Auto-load pre-trained models for common symbols
        self._auto_load_models()
    
    def _auto_load_models(self):
        """Auto-load pre-trained models for common symbols"""
        common_symbols = ['AAPL', 'GOOGL', 'MSFT', 'TNXP']
        for symbol in common_symbols:
            try:
                self.load_model(symbol)
                logger.info(f"Auto-loaded LSTM model for {symbol}")
            except Exception as e:
                logger.debug(f"No pre-trained LSTM model found for {symbol}: {e}")
    
    def update_data(self, tick: PriceTick):
        """Update internal data structures"""
        symbol = tick.symbol
        self.prices[symbol].append(tick.price)
        self.volumes[symbol].append(tick.volume if hasattr(tick, 'volume') else 1000)
        self.opens[symbol].append(getattr(tick, 'open_price', tick.price))
        self.highs[symbol].append(getattr(tick, 'high_price', tick.price))
        self.lows[symbol].append(getattr(tick, 'low_price', tick.price))
    
    def create_sequences(self, data: np.ndarray, lookback: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training"""
        X, y = [], []
        for i in range(lookback, len(data)):
            X.append(data[i-lookback:i])
            y.append(data[i])
        return np.array(X), np.array(y)
    
    def build_lstm_model(self, input_shape: Tuple[int, int]):
        """Build LSTM model architecture - PyTorch implementation with GPU support"""
        if not TENSORFLOW_AVAILABLE:
            # Use PyTorch LSTM with GPU support
            return self.build_pytorch_lstm(input_shape)
            
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
    
    def build_pytorch_lstm(self, input_shape: Tuple[int, int]):
        """Build PyTorch LSTM model with GPU support"""
        import torch
        import torch.nn as nn
        
        class LSTMModel(nn.Module):
            def __init__(self, input_size, hidden_size=50, num_layers=2, dropout=0.2):
                super(LSTMModel, self).__init__()
                self.hidden_size = hidden_size
                self.num_layers = num_layers
                
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                                   batch_first=True, dropout=dropout)
                self.dropout = nn.Dropout(dropout)
                self.fc1 = nn.Linear(hidden_size, 25)
                self.fc2 = nn.Linear(25, 1)
                self.relu = nn.ReLU()
                
            def forward(self, x):
                # Initialize hidden state
                h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
                c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
                
                # LSTM forward pass
                out, _ = self.lstm(x, (h0, c0))
                out = self.dropout(out[:, -1, :])  # Take last output
                out = self.relu(self.fc1(out))
                out = self.fc2(out)
                return out
        
        # Create model and move to device
        model = LSTMModel(input_shape[1]).to(self.device)
        print(f"PyTorch LSTM model created on device: {self.device}")
        return model
    
    def train_model(self, symbol: str):
        """Train LSTM model for a symbol"""
        if not TENSORFLOW_AVAILABLE or len(self.prices[symbol]) < self.config.min_data_points:
            return
        
        # Create time series data from price history
        prices_list = list(self.prices[symbol])
        
        if len(prices_list) < self.config.lookback_period + 10:
            return
        
        # Create sequences directly from price data for LSTM
        scaler = MinMaxScaler()
        
        # Reshape prices for scaling
        prices_array = np.array(prices_list).reshape(-1, 1)
        prices_scaled = scaler.fit_transform(prices_array).flatten()
        
        # Create sequences for LSTM training
        X, y = self.create_sequences(prices_scaled, self.config.lookback_period)
        
        if len(X) < 10:  # Need minimum sequences
            return
        
        # Reshape X for LSTM (samples, timesteps, features)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Build and train model
        try:
            model = self.build_lstm_model((X_train.shape[1], X_train.shape[2]))
            
            # Train with reduced epochs for testing
            model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_test, y_test), verbose=0)
            
            # Store model and scaler
            self.models[symbol] = model
            self.scalers[symbol] = scaler
            self.models_trained[symbol] = True
            
            logger.info(f"LSTM model trained for {symbol} with {len(X)} sequences")
            
        except Exception as e:
            logger.error(f"Error training LSTM model for {symbol}: {e}")
            import traceback
            logger.error(f"LSTM training traceback: {traceback.format_exc()}")
    
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
        """Process signal through position manager"""
        try:
            # Import here to avoid circular imports
            from tradebot.strategy.position_manager import process_ml_signal
            
            action = await process_ml_signal(strategy_name, signal.symbol, signal)
            logger.debug(f"Position manager action for {signal.symbol}: {action}")
        except Exception as e:
            logger.error(f"Error processing signal through position manager: {e}")
    
    def on_tick(self, tick: PriceTick) -> Optional[Signal]:
        """Process tick and generate signals - WITH FALLBACK FOR NO TENSORFLOW"""
        symbol = tick.symbol
        self.update_data(tick)
        self.tick_count[symbol] += 1
        
        # Check if we should train model (every retrain_frequency ticks)
        if self.tick_count[symbol] % 1000 == 0:  # Train every 1000 ticks
            if len(self.prices[symbol]) >= self.config.min_data_points:
                logger.info(f"Training LSTM model for {symbol} (tick #{self.tick_count[symbol]})")
                self.train_model(symbol)
        
        # Check if we have enough data
        if len(self.prices[symbol]) < self.config.lookback_period:
            return None
        
        # Check if we should generate a signal
        time_since_last = datetime.now(timezone.utc) - self.last_signals[symbol]
        if time_since_last.total_seconds() < 180:  # 3 minutes minimum
            return None
        
        # Make prediction (with fallback if TensorFlow not available)
        buy_probability, confidence = self.predict(symbol)
        
        # If TensorFlow not available, use LSTM-inspired time series analysis
        if not TENSORFLOW_AVAILABLE:
            buy_probability, confidence = self.lstm_fallback_prediction(symbol)
        
        # Generate signal based on prediction with ADAPTIVE thresholds based on stock quality
        
        # Check if this is a penny stock (use cached calculation from prediction)
        prices = list(self.prices[symbol])
        if len(prices) >= 20:
            max_price = max(prices)
            min_price = min(prices)
            price_range = (max_price - min_price) / min_price if min_price > 0 else 0
            long_term_trend = (prices[-1] - prices[0]) / prices[0] if prices[0] > 0 else 0
            daily_changes = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(-10, 0) if prices[i-1] > 0]
            avg_volatility = np.std(daily_changes) if daily_changes else 0
            current_price = prices[-1]
            
            penny_stock_risk = 0
            if price_range > 5.0: penny_stock_risk += 0.3
            if long_term_trend < -0.5: penny_stock_risk += 0.4
            if avg_volatility > 0.2: penny_stock_risk += 0.3
            if current_price < 5.0: penny_stock_risk += 0.2
            
            # ADAPTIVE THRESHOLDS based on stock quality
            if penny_stock_risk > 0.6:
                # High-risk penny stocks: No trading
                return None
            elif penny_stock_risk > 0.3:
                # Medium-risk stocks: Very conservative
                buy_threshold, sell_threshold, conf_threshold = 0.72, 0.28, 0.65
            else:
                # Quality stocks: More reasonable thresholds
                buy_threshold, sell_threshold, conf_threshold = 0.65, 0.35, 0.45
        else:
            # Default for insufficient data
            buy_threshold, sell_threshold, conf_threshold = 0.65, 0.35, 0.45
        
        if buy_probability > buy_threshold and confidence > conf_threshold:
            if self.last_side[symbol] != Side.buy:
                self.last_side[symbol] = Side.buy
                self.last_signals[symbol] = tick.timestamp
                logger.info(f"Adaptive LSTM buy signal for {symbol}: prob={buy_probability:.3f}, conf={confidence:.3f}, thresholds=({buy_threshold:.2f}, {conf_threshold:.2f})")
                signal = Signal(
                    symbol=symbol, 
                    side=Side.buy, 
                    price=tick.price, 
                    timestamp=tick.timestamp, 
                    confidence=confidence
                )
                # Log signal asynchronously (if event loop is running)
                try:
                    asyncio.create_task(self._log_signal(signal, "lstm_ml"))
                except RuntimeError:
                    # No event loop running (e.g., in tests)
                    pass
                return signal
        
        elif buy_probability < sell_threshold and confidence > conf_threshold:
            if self.last_side[symbol] == Side.buy:
                self.last_side[symbol] = Side.sell
                self.last_signals[symbol] = tick.timestamp
                logger.info(f"Adaptive LSTM sell signal for {symbol}: prob={buy_probability:.3f}, conf={confidence:.3f}, thresholds=({sell_threshold:.2f}, {conf_threshold:.2f})")
                signal = Signal(
                    symbol=symbol, 
                    side=Side.sell, 
                    price=tick.price, 
                    timestamp=tick.timestamp, 
                    confidence=confidence
                )
                # Log signal asynchronously (if event loop is running)
                try:
                    asyncio.create_task(self._log_signal(signal, "lstm_ml"))
                except RuntimeError:
                    # No event loop running (e.g., in tests)
                    pass
                return signal
        
        return None
    
    def lstm_fallback_prediction(self, symbol: str) -> Tuple[float, float]:
        """IMPROVED LSTM-inspired prediction with PENNY STOCK PROTECTION"""
        try:
            prices = list(self.prices[symbol])
            volumes = list(self.volumes[symbol])
            
            if len(prices) < 20:
                return 0.5, 0.0
            
            current_price = prices[-1]
            
            # === PENNY STOCK DETECTION & PROTECTION ===
            # Detect if this is a problematic penny stock
            max_price = max(prices)
            min_price = min(prices)
            price_range = (max_price - min_price) / min_price if min_price > 0 else 0
            
            # Calculate overall trend (is stock in long-term decline?)
            long_term_trend = (prices[-1] - prices[0]) / prices[0] if prices[0] > 0 else 0
            
            # Check for extreme volatility patterns
            daily_changes = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(-10, 0) if prices[i-1] > 0]
            avg_volatility = np.std(daily_changes) if daily_changes else 0
            
            # PENNY STOCK WARNING SIGNS:
            penny_stock_risk = 0
            
            # 1. Extreme price range (>500% range suggests pump/dump)
            if price_range > 5.0:
                penny_stock_risk += 0.3
            
            # 2. Strong downward bias (>50% decline suggests fundamental issues)
            if long_term_trend < -0.5:
                penny_stock_risk += 0.4
            
            # 3. Extreme volatility (>20% daily moves suggest manipulation)
            if avg_volatility > 0.2:
                penny_stock_risk += 0.3
            
            # 4. Very low absolute price (classic penny stock)
            if current_price < 5.0:
                penny_stock_risk += 0.2
            
            # If penny stock risk is high, be EXTREMELY conservative
            if penny_stock_risk > 0.6:
                logger.warning(f"High penny stock risk detected for {symbol}: {penny_stock_risk:.2f}")
                # Return neutral signals for dangerous penny stocks
                return 0.5, 0.1  # Very low confidence
            
            # === MARKET REGIME DETECTION ===
            # Determine if market is trending or mean-reverting
            price_changes = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(-10, 0) if prices[i-1] > 0]
            if not price_changes:
                return 0.5, 0.0
                
            volatility = np.std(price_changes)
            trend_strength = abs(np.mean(price_changes))
            
            # Market regime classification
            if trend_strength > volatility * 0.5:
                regime = "trending"
                regime_confidence = min(trend_strength / volatility, 1.0)
            else:
                regime = "mean_reverting"
                regime_confidence = min(volatility / (trend_strength + 0.001), 1.0)
            
            # === STRATEGY SELECTION BASED ON REGIME ===
            
            if regime == "trending":
                # MOMENTUM STRATEGY for trending markets
                
                # 1. Multi-timeframe momentum
                short_ma = np.mean(prices[-5:])
                medium_ma = np.mean(prices[-10:])
                long_ma = np.mean(prices[-20:])
                
                # Strong momentum signal
                if short_ma > medium_ma > long_ma:
                    momentum_score = 0.8  # Strong uptrend
                elif short_ma > medium_ma:
                    momentum_score = 0.65  # Weak uptrend
                elif short_ma < medium_ma < long_ma:
                    momentum_score = 0.2  # Strong downtrend
                elif short_ma < medium_ma:
                    momentum_score = 0.35  # Weak downtrend
                else:
                    momentum_score = 0.5  # Neutral
                
                # 2. Breakout detection
                high_20 = max(prices[-20:])
                low_20 = min(prices[-20:])
                price_position = (current_price - low_20) / (high_20 - low_20) if high_20 > low_20 else 0.5
                
                if price_position > 0.9:  # Near high
                    breakout_score = 0.8  # Potential breakout
                elif price_position < 0.1:  # Near low
                    breakout_score = 0.2  # Potential breakdown
                else:
                    breakout_score = 0.5
                
                # 3. Volume confirmation
                recent_vol = np.mean(volumes[-3:])
                avg_vol = np.mean(volumes[-10:])
                vol_ratio = recent_vol / avg_vol if avg_vol > 0 else 1
                
                if vol_ratio > 1.5:  # High volume
                    volume_score = 0.7
                elif vol_ratio < 0.7:  # Low volume
                    volume_score = 0.4
                else:
                    volume_score = 0.5
                
                # Combine for trending strategy
                combined_score = (momentum_score * 0.5 + breakout_score * 0.3 + volume_score * 0.2)
                confidence = regime_confidence * 0.8
                
            else:  # mean_reverting regime
                # MEAN REVERSION STRATEGY for sideways markets
                
                # 1. RSI-like oscillator
                gains = [max(0, change) for change in price_changes]
                losses = [abs(min(0, change)) for change in price_changes]
                avg_gain = np.mean(gains) if gains else 0
                avg_loss = np.mean(losses) if losses else 0
                
                if avg_loss > 0:
                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))
                else:
                    rsi = 50
                
                # RSI signals
                if rsi < 25:  # Oversold
                    rsi_score = 0.8
                elif rsi > 75:  # Overbought
                    rsi_score = 0.2
                else:
                    rsi_score = 0.5
                
                # 2. Distance from mean
                sma_15 = np.mean(prices[-15:])
                distance = (current_price - sma_15) / sma_15 if sma_15 > 0 else 0
                
                if distance < -0.05:  # 5% below mean
                    mean_score = 0.75  # Buy the dip
                elif distance > 0.05:  # 5% above mean
                    mean_score = 0.25  # Sell the peak
                else:
                    mean_score = 0.5
                
                # 3. Bollinger Band-like analysis
                std_dev = np.std(prices[-15:])
                upper_band = sma_15 + (2 * std_dev)
                lower_band = sma_15 - (2 * std_dev)
                
                if current_price < lower_band:
                    band_score = 0.8  # Oversold
                elif current_price > upper_band:
                    band_score = 0.2  # Overbought
                else:
                    band_score = 0.5
                
                # Combine for mean reversion strategy
                combined_score = (rsi_score * 0.4 + mean_score * 0.35 + band_score * 0.25)
                confidence = regime_confidence * 0.7
            
            # === ENHANCED RISK MANAGEMENT OVERLAY ===
            
            # 1. Volatility adjustment
            if volatility > 0.1:  # Very high volatility (10%+ daily moves)
                confidence *= 0.3  # Be VERY cautious with volatile stocks
                # Pull signals toward neutral in high volatility
                combined_score = combined_score * 0.5 + 0.5 * 0.5
            elif volatility > 0.05:  # High volatility (5%+ daily moves)
                confidence *= 0.6  # Be cautious
                combined_score = combined_score * 0.8 + 0.5 * 0.2
            
            # 2. Recent performance check (enhanced)
            recent_return = (current_price - prices[-5]) / prices[-5] if prices[-5] > 0 else 0
            if abs(recent_return) > 0.3:  # 30% move in 5 periods (extreme)
                confidence *= 0.3  # Very low confidence after extreme moves
                combined_score = 0.5  # Force neutral
            elif abs(recent_return) > 0.2:  # 20% move in 5 periods
                confidence *= 0.5  # Reduce confidence after big moves
            
            # 3. Long-term trend bias protection
            if long_term_trend < -0.3:  # Stock down >30% overall
                # Be very reluctant to buy declining stocks
                if combined_score > 0.5:  # If suggesting buy
                    combined_score = combined_score * 0.6 + 0.5 * 0.4  # Pull toward neutral
                    confidence *= 0.7
            
            # 4. Price momentum divergence check
            short_momentum = (prices[-1] - prices[-3]) / prices[-3] if prices[-3] > 0 else 0
            medium_momentum = (prices[-1] - prices[-7]) / prices[-7] if prices[-7] > 0 else 0
            
            # If short and medium momentum disagree strongly, reduce confidence
            momentum_divergence = abs(short_momentum - medium_momentum)
            if momentum_divergence > 0.1:  # 10% divergence
                confidence *= 0.8
            
            # 5. Penny stock additional protection (only for high-risk stocks)
            if penny_stock_risk > 0.5:  # Only high penny stock risk
                confidence *= (1.0 - penny_stock_risk * 0.5)  # Less aggressive penalty
                # For penny stocks, be more conservative on buy signals
                if combined_score > 0.5:
                    combined_score = combined_score * 0.9 + 0.5 * 0.1  # Less aggressive pull
            
            # 6. Ensure reasonable bounds with stricter limits
            combined_score = max(0.2, min(0.8, combined_score))  # Narrower range
            confidence = max(0.1, min(0.8, confidence))
            
            return combined_score, confidence
            
        except Exception as e:
            logger.error(f"Error in improved LSTM prediction for {symbol}: {e}")
            return 0.5, 0.3

    def save_model(self, symbol: str) -> Optional[str]:
        """Save the LSTM model to disk"""
        try:
            if symbol not in self.models or not self.models_trained[symbol]:
                logger.warning(f"No trained model to save for symbol: {symbol}")
                return None
            
            # Create symbol-specific directory
            symbol_dir = self.models_dir / symbol
            symbol_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = symbol_dir / f"model_{timestamp}.pkl"
            
            # Save model data
            model_data = {
                'strategy_type': 'lstm',
                'symbol': symbol,
                'timestamp': timestamp,
                'model_state': {
                    'model': self.models[symbol],
                    'scaler': self.scalers.get(symbol),
                    'pytorch_model': self.pytorch_models.get(symbol),
                    'pytorch_optimizer': self.pytorch_optimizers.get(symbol)
                },
                'config': self.config
            }
            
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"LSTM model saved for {symbol}: {model_path}")
            return str(model_path)
            
        except Exception as e:
            logger.error(f"Failed to save LSTM model for {symbol}: {e}")
            return None

    def load_model(self, symbol: str, model_path: str = None) -> bool:
        """Load the LSTM model from disk"""
        try:
            if model_path is None:
                # Find the latest model file
                symbol_dir = self.models_dir / symbol
                if not symbol_dir.exists():
                    logger.warning(f"No model directory found for symbol: {symbol}")
                    return False
                
                model_files = list(symbol_dir.glob("model_*.pkl"))
                if not model_files:
                    logger.warning(f"No model files found for symbol: {symbol}")
                    return False
                
                # Get the latest model file
                model_path = max(model_files, key=lambda p: p.stat().st_mtime)
            
            # Load model data
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Restore model state
            self.models[symbol] = model_data['model_state']['model']
            if model_data['model_state']['scaler']:
                self.scalers[symbol] = model_data['model_state']['scaler']
            if model_data['model_state']['pytorch_model']:
                self.pytorch_models[symbol] = model_data['model_state']['pytorch_model']
            if model_data['model_state']['pytorch_optimizer']:
                self.pytorch_optimizers[symbol] = model_data['model_state']['pytorch_optimizer']
            
            logger.info(f"LSTM model loaded for {symbol}: {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load LSTM model for {symbol}: {e}")
            return False


class SentimentStrategy:
    """Sentiment analysis-based trading strategy with GPU support"""
    
    def __init__(self, config: MLStrategyConfig):
        self.config = config
        self.feature_engineer = FeatureEngineer(config)
        
        # Setup GPU device for sentiment analysis
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Sentiment Strategy using device: {self.device}")
        
        # Initialize sentiment analyzer with GPU support
        self.sentiment_analyzer = None  # Will be initialized on first use
        self.gpu_enabled = torch.backends.mps.is_available() or torch.cuda.is_available()
        
        if self.gpu_enabled:
            print("GPU-accelerated sentiment analysis enabled")
        else:
            print("CPU-only sentiment analysis (GPU not available)")
        
        # Data storage
        self.prices: Dict[str, deque] = defaultdict(lambda: deque(maxlen=200))
        self.sentiment_scores: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))
        self.news_cache: Dict[str, List[Dict]] = defaultdict(list)
        
        # Sentiment analysis state
        self.sentiment_threshold = 0.5  # Default threshold
        self.last_sentiment = 0.0
        
        self.last_side: Dict[str, Side] = defaultdict(lambda: None)
        self.last_signals: Dict[str, datetime] = defaultdict(lambda: datetime.min.replace(tzinfo=timezone.utc))
        self.last_sentiment_update: Dict[str, datetime] = defaultdict(lambda: datetime.min.replace(tzinfo=timezone.utc))
        
        # Model persistence
        self.models_dir = Path("models") / "sentiment"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Auto-load pre-trained models for common symbols
        self._auto_load_models()
    
    def _auto_load_models(self):
        """Auto-load pre-trained models for common symbols"""
        common_symbols = ['AAPL', 'GOOGL', 'MSFT', 'TNXP']
        for symbol in common_symbols:
            try:
                self.load_model(symbol)
                logger.info(f"Auto-loaded sentiment model for {symbol}")
            except Exception as e:
                logger.debug(f"No pre-trained sentiment model found for {symbol}: {e}")
        
    def update_data(self, tick: PriceTick):
        """Update internal data structures"""
        symbol = tick.symbol
        self.prices[symbol].append(tick.price)
    
    def train_model(self, symbol: str):
        """Train the sentiment analysis model"""
        try:
            logger.info(f"Training sentiment model for {symbol}")
            
            # For sentiment analysis, we don't need to train traditional ML models
            # Instead, we can calibrate sentiment thresholds based on historical data
            if len(self.prices[symbol]) < self.config.min_data_points:
                logger.warning(f"Insufficient data for sentiment training: {len(self.prices[symbol])}")
                return
            
            # Calculate sentiment scores for historical data
            sentiment_scores = []
            prices = list(self.prices[symbol])
            
            for i in range(1, len(prices)):
                # Calculate price momentum
                price_change = (prices[i] - prices[i-1]) / prices[i-1]
                sentiment_scores.append(price_change)
            
            # Calculate statistics for threshold calibration
            if sentiment_scores:
                mean_sentiment = np.mean(sentiment_scores)
                std_sentiment = np.std(sentiment_scores)
                
                # Store calibrated thresholds
                self.sentiment_thresholds = {
                    'mean': mean_sentiment,
                    'std': std_sentiment,
                    'buy_threshold': mean_sentiment + 0.5 * std_sentiment,
                    'sell_threshold': mean_sentiment - 0.5 * std_sentiment
                }
                
                logger.info(f"Sentiment model trained for {symbol}: mean={mean_sentiment:.4f}, std={std_sentiment:.4f}")
            else:
                logger.warning(f"No sentiment data available for {symbol}")
                
        except Exception as e:
            logger.error(f"Sentiment model training failed for {symbol}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
    
    def get_simple_sentiment(self, symbol: str) -> float:
        """Get sentiment score with GPU acceleration when available"""
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
            
            # GPU-accelerated sentiment calculation when available and beneficial
            # Only use GPU for larger datasets to avoid overhead
            if self.gpu_enabled and len(prices) >= 50:  # Increased threshold
                return self._gpu_sentiment_calculation(prices)
            else:
                return self._cpu_sentiment_calculation(prices)
            
        except Exception as e:
            logger.error(f"Error calculating sentiment for {symbol}: {e}")
            return 0.0
    
    def _gpu_sentiment_calculation(self, prices: List[float]) -> float:
        """GPU-accelerated sentiment calculation using PyTorch"""
        try:
            # Convert to PyTorch tensor on GPU
            price_tensor = torch.tensor(prices, dtype=torch.float32, device=self.device)
            
            # Calculate multiple timeframe changes using GPU
            if len(prices) >= 10:
                # Multi-timeframe momentum analysis
                short_change = (price_tensor[-1] - price_tensor[-3]) / price_tensor[-3]
                medium_change = (price_tensor[-1] - price_tensor[-5]) / price_tensor[-5]
                long_change = (price_tensor[-1] - price_tensor[-10]) / price_tensor[-10]
                
                # Volatility-adjusted sentiment
                recent_prices = price_tensor[-10:]
                volatility = torch.std(recent_prices / recent_prices[0])  # Normalized volatility
                
                # Trend strength calculation
                trend_strength = torch.abs(torch.mean(torch.diff(recent_prices) / recent_prices[:-1]))
                
                # Combine signals with GPU operations
                momentum_signal = (short_change * 0.5 + medium_change * 0.3 + long_change * 0.2)
                volatility_adjusted = momentum_signal / (volatility + 0.01)  # Avoid division by zero
                
                # Apply non-linear transformation
                sentiment = torch.tanh(volatility_adjusted * 3.0)
                
                # Convert back to CPU and return
                return float(sentiment.cpu().item())
            else:
                # Fallback to CPU for small datasets
                return self._cpu_sentiment_calculation(prices)
                
        except Exception as e:
            logger.error(f"Error in GPU sentiment calculation: {e}")
            return self._cpu_sentiment_calculation(prices)
    
    def _cpu_sentiment_calculation(self, prices: List[float]) -> float:
        """CPU-based sentiment calculation (fallback)"""
        try:
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
                logger.warning(f"Extreme price changes detected, capping values")
                short_change = max(-1, min(1, short_change))
                medium_change = max(-1, min(1, medium_change))
            
            # Combine changes for sentiment (positive change = positive sentiment)
            sentiment = (short_change * 0.6 + medium_change * 0.4) * 5  # Scale and weight
            sentiment = np.tanh(sentiment)  # Bound to [-1, 1]
            
            # Final safety check
            if not np.isfinite(sentiment):
                logger.warning(f"Non-finite sentiment calculated, using neutral")
                sentiment = 0.0
            
            return sentiment
            
        except Exception as e:
            logger.error(f"Error in CPU sentiment calculation: {e}")
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
        """Process signal through position manager"""
        try:
            # Import here to avoid circular imports
            from tradebot.strategy.position_manager import process_ml_signal
            
            action = await process_ml_signal(strategy_name, signal.symbol, signal)
            logger.debug(f"Position manager action for {signal.symbol}: {action}")
        except Exception as e:
            logger.error(f"Error processing signal through position manager: {e}")
    
    def on_tick(self, tick: PriceTick) -> Optional[Signal]:
        """Process tick and generate signals with GPU-accelerated sentiment analysis"""
        symbol = tick.symbol
        
        try:
            # Update data storage
            self.prices[symbol].append(tick.price)
            
            # Basic requirements check
            if len(self.prices[symbol]) < 10:
                return None
            
            # Skip if price is too small (prevents numerical issues)
            if tick.price < 0.01:
                return None
            
            # Time throttling - reduced for more frequent signals
            time_since_last = datetime.now(timezone.utc) - self.last_signals[symbol]
            if time_since_last.total_seconds() < 180:  # 3 minutes (reduced from 5)
                return None
            
            # Get GPU-accelerated sentiment score
            sentiment_score = self.get_simple_sentiment(symbol)
            
            # Store sentiment score
            self.sentiment_scores[symbol].append(sentiment_score)
            self.last_sentiment_update[symbol] = datetime.now(timezone.utc)
            
            # Calculate confidence based on sentiment strength and consistency
            recent_sentiments = list(self.sentiment_scores[symbol])[-3:]  # Last 3 sentiment scores
            sentiment_consistency = 1.0 - np.std(recent_sentiments) if len(recent_sentiments) > 1 else 0.5
            sentiment_strength = abs(sentiment_score)
            confidence = min(sentiment_strength * sentiment_consistency * 2.0, 1.0)
            
            # Enhanced buy/sell logic with adaptive thresholds
            buy_threshold = 0.3 if self.gpu_enabled else 0.4  # Lower threshold for GPU (more sensitive)
            sell_threshold = -0.3 if self.gpu_enabled else -0.4
            min_confidence = 0.4 if self.gpu_enabled else 0.5
            
            # Generate buy signal
            if sentiment_score > buy_threshold and confidence > min_confidence:
                if self.last_side[symbol] != Side.buy:
                    self.last_side[symbol] = Side.buy
                    self.last_signals[symbol] = tick.timestamp
                    
                    logger.info(f"GPU sentiment buy signal for {symbol}: score={sentiment_score:.3f}, conf={confidence:.3f}")
                    signal = Signal(
                        symbol=symbol,
                        side=Side.buy,
                        price=tick.price,
                        timestamp=tick.timestamp,
                        confidence=confidence
                    )
                    # Log signal asynchronously (if event loop is running)
                    try:
                        asyncio.create_task(self._log_signal(signal, "sentiment_ml"))
                    except RuntimeError:
                        # No event loop running (e.g., in tests)
                        pass
                    return signal
            
            # Generate sell signal
            elif sentiment_score < sell_threshold and confidence > min_confidence:
                if self.last_side[symbol] == Side.buy:
                    self.last_side[symbol] = Side.sell
                    self.last_signals[symbol] = tick.timestamp
                    
                    logger.info(f"GPU sentiment sell signal for {symbol}: score={sentiment_score:.3f}, conf={confidence:.3f}")
                    signal = Signal(
                        symbol=symbol,
                        side=Side.sell,
                        price=tick.price,
                        timestamp=tick.timestamp,
                        confidence=confidence
                    )
                    # Log signal asynchronously (if event loop is running)
                    try:
                        asyncio.create_task(self._log_signal(signal, "sentiment_ml"))
                    except RuntimeError:
                        # No event loop running (e.g., in tests)
                        pass
                    return signal
            
            return None
            
        except Exception as e:
            logger.error(f"Error in GPU sentiment strategy for {symbol}: {e}")
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
    
    def save_model(self, symbol: str) -> Optional[str]:
        """Save the sentiment model to disk"""
        try:
            # Create symbol-specific directory
            symbol_dir = self.models_dir / symbol
            symbol_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = symbol_dir / f"model_{timestamp}.pkl"
            
            # Save model data
            model_data = {
                'strategy_type': 'sentiment',
                'symbol': symbol,
                'timestamp': timestamp,
                'model_state': {
                    'sentiment_threshold': self.sentiment_threshold,
                    'last_sentiment': self.last_sentiment,
                    'device': self.device
                },
                'config': self.config
            }
            
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"Sentiment model saved for {symbol}: {model_path}")
            return str(model_path)
            
        except Exception as e:
            logger.error(f"Failed to save sentiment model for {symbol}: {e}")
            return None

    def load_model(self, symbol: str, model_path: str = None) -> bool:
        """Load the sentiment model from disk"""
        try:
            if model_path is None:
                # Find the latest model file
                symbol_dir = self.models_dir / symbol
                if not symbol_dir.exists():
                    logger.warning(f"No model directory found for symbol: {symbol}")
                    return False
                
                model_files = list(symbol_dir.glob("model_*.pkl"))
                if not model_files:
                    logger.warning(f"No model files found for symbol: {symbol}")
                    return False
                
                # Get the latest model file
                model_path = max(model_files, key=lambda p: p.stat().st_mtime)
            
            # Load model data
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Restore model state
            self.sentiment_threshold = model_data['model_state']['sentiment_threshold']
            self.last_sentiment = model_data['model_state']['last_sentiment']
            
            logger.info(f"Sentiment model loaded for {symbol}: {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load sentiment model for {symbol}: {e}")
            return False


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