import numpy as np
import pandas as pd
import logging
from collections import deque, defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import pickle
# Try to import RL dependencies
try:
    import gymnasium as gym
    from gymnasium import spaces
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import BaseCallback
    RL_AVAILABLE = True
except ImportError as e:
    RL_AVAILABLE = False
    # Create dummy classes to prevent import errors
    class gym:
        class Env: pass
        class spaces:
            class Box: pass
    spaces = gym.spaces

from tradebot.common.models import PriceTick, Signal, Side

logger = logging.getLogger("rl_strategies")

@dataclass
class RLStrategyConfig:
    """Configuration for RL strategies"""
    # Environment parameters
    initial_balance: float = 10000.0
    max_position_size: float = 0.2
    transaction_cost: float = 0.001  # 0.1% per trade
    
    # Training parameters
    learning_rate: float = 0.0003
    batch_size: int = 64
    n_steps: int = 2048
    n_epochs: int = 10
    gamma: float = 0.99
    
    # Data parameters
    lookback_period: int = 20
    min_data_points: int = 30
    
    # Risk management
    stop_loss: float = 0.05
    take_profit: float = 0.10


class TradingEnvironment(gym.Env):
    """Custom trading environment for reinforcement learning"""
    
    def __init__(self, config: RLStrategyConfig):
        super().__init__()
        self.config = config
        
        # Action space: [buy, sell, hold] with position sizing
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(1,), dtype=np.float32
        )
        
        # Observation space: price data + technical indicators + position info
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32
        )
        
        # Environment state
        self.reset()
    
    def reset(self, seed=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        self.balance = self.config.initial_balance
        self.position = 0.0
        self.position_value = 0.0
        self.current_price = 0.0
        self.entry_price = 0.0
        
        # Data storage
        self.prices = deque(maxlen=100)
        self.volumes = deque(maxlen=100)
        self.returns = deque(maxlen=100)
        
        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.max_drawdown = 0.0
        self.peak_balance = self.balance
        
        return self._get_observation(), {}
    
    def step(self, action):
        """Execute action and return new state, reward, done, info"""
        # Store previous state
        prev_balance = self.balance
        prev_position = self.position
        
        # Execute action
        action_value = action[0]
        
        if action_value > 0.3:  # Buy signal
            if self.position <= 0:  # Not currently long
                # Calculate position size based on action strength
                position_size = min(abs(action_value), self.config.max_position_size)
                shares = (self.balance * position_size) / self.current_price
                
                if shares > 0 and self.balance >= shares * self.current_price:
                    self.position = shares
                    self.entry_price = self.current_price
                    self.position_value = shares * self.current_price
                    
                    # Apply transaction costs
                    cost = self.position_value * self.config.transaction_cost
                    self.balance -= cost
                    self.total_pnl -= cost
                    
                    self.total_trades += 1
        
        elif action_value < -0.3:  # Sell signal
            if self.position > 0:  # Currently long
                # Calculate sell amount based on action strength
                sell_ratio = min(abs(action_value), 1.0)
                shares_to_sell = self.position * sell_ratio
                
                if shares_to_sell > 0:
                    sell_value = shares_to_sell * self.current_price
                    
                    # Calculate PnL
                    pnl = sell_value - (shares_to_sell * self.entry_price)
                    self.total_pnl += pnl
                    
                    if pnl > 0:
                        self.winning_trades += 1
                    
                    # Update position
                    self.position -= shares_to_sell
                    self.position_value = self.position * self.current_price
                    
                    # Apply transaction costs
                    cost = sell_value * self.config.transaction_cost
                    self.balance += sell_value - cost
                    self.total_pnl -= cost
                    
                    self.total_trades += 1
        
        # Update current position value
        if self.position > 0:
            self.position_value = self.position * self.current_price
        
        # Calculate reward
        reward = self._calculate_reward(prev_balance, prev_position)
        
        # Check if done
        done = self._is_done()
        
        # Update peak balance for drawdown calculation
        current_total = self.balance + self.position_value
        if current_total > self.peak_balance:
            self.peak_balance = current_total
        
        # Calculate drawdown
        current_drawdown = (self.peak_balance - current_total) / self.peak_balance
        self.max_drawdown = max(self.max_drawdown, current_drawdown)
        
        info = {
            'balance': self.balance,
            'position': self.position,
            'position_value': self.position_value,
            'total_pnl': self.total_pnl,
            'total_trades': self.total_trades,
            'win_rate': self.winning_trades / max(self.total_trades, 1),
            'max_drawdown': self.max_drawdown
        }
        
        return self._get_observation(), reward, done, False, info
    
    def _get_observation(self):
        """Get current observation vector"""
        obs = []
        
        # Price data (normalized)
        if len(self.prices) >= 10:
            recent_prices = list(self.prices)[-10:]
            price_changes = np.diff(recent_prices) / recent_prices[:-1]
            obs.extend(price_changes)
        else:
            obs.extend([0.0] * 9)
        
        # Volume data (normalized)
        if len(self.volumes) >= 5:
            recent_volumes = list(self.volumes)[-5:]
            volume_ratio = recent_volumes[-1] / np.mean(recent_volumes) if np.mean(recent_volumes) > 0 else 1.0
            obs.append(volume_ratio)
        else:
            obs.append(1.0)
        
        # Position information
        obs.append(self.position / self.config.max_position_size if self.config.max_position_size > 0 else 0.0)
        obs.append(self.position_value / self.config.initial_balance)
        
        # Account information
        total_value = self.balance + self.position_value
        obs.append(self.balance / self.config.initial_balance)
        obs.append(total_value / self.config.initial_balance - 1.0)  # Return
        
        # Risk metrics
        obs.append(self.max_drawdown)
        obs.append(self.total_trades / 100.0)  # Normalized trade count
        
        # Pad to fixed size
        while len(obs) < 20:
            obs.append(0.0)
        
        return np.array(obs[:20], dtype=np.float32)
    
    def _calculate_reward(self, prev_balance: float, prev_position: float) -> float:
        """Calculate reward based on performance"""
        current_total = self.balance + self.position_value
        prev_total = prev_balance + (prev_position * self.current_price if prev_position > 0 else 0)
        
        # Immediate return
        immediate_return = (current_total - prev_total) / prev_total if prev_total > 0 else 0
        
        # Risk-adjusted reward
        risk_penalty = -self.max_drawdown * 0.1  # Penalize drawdown
        
        # Trade efficiency penalty
        trade_penalty = -self.total_trades * 0.001  # Small penalty for overtrading
        
        # Sharpe ratio component (simplified)
        if len(self.returns) >= 10:
            returns_array = np.array(list(self.returns)[-10:])
            sharpe_component = np.mean(returns_array) / (np.std(returns_array) + 1e-8)
        else:
            sharpe_component = 0.0
        
        reward = immediate_return * 100 + risk_penalty + trade_penalty + sharpe_component * 0.1
        
        return reward
    
    def _is_done(self) -> bool:
        """Check if episode should end"""
        # End if balance is too low
        if self.balance < self.config.initial_balance * 0.5:
            return True
        
        # End if drawdown is too high
        if self.max_drawdown > 0.2:  # 20% drawdown
            return True
        
        # End if we have enough data
        if len(self.prices) >= 1000:
            return True
        
        return False
    
    def update_market_data(self, price: float, volume: float = 1000):
        """Update market data"""
        self.current_price = price
        self.prices.append(price)
        self.volumes.append(volume)
        
        # Calculate return
        if len(self.prices) >= 2:
            return_val = (price - self.prices[-2]) / self.prices[-2]
            self.returns.append(return_val)


class RLStrategy:
    """Reinforcement learning trading strategy using PPO"""
    
    def __init__(self, config: RLStrategyConfig):
        self.config = config
        
        if not RL_AVAILABLE:
            print("Warning: RL dependencies not available. Strategy will use fallback logic.")
            self.env = None
            self.vec_env = None
            self.model = None
        else:
            try:
                # Create environment
                self.env = TradingEnvironment(config)
                self.vec_env = DummyVecEnv([lambda: self.env])
                
                # Initialize PPO agent with GPU support
                device = "mps" if torch.backends.mps.is_available() else "cpu"
                print(f"RL Strategy using device: {device}")
                
                self.model = PPO(
                    "MlpPolicy",
                    self.vec_env,
                    learning_rate=config.learning_rate,
                    batch_size=config.batch_size,
                    n_steps=config.n_steps,
                    n_epochs=config.n_epochs,
                    gamma=config.gamma,
                    device=device,
                    verbose=0
                )
            except Exception as e:
                print(f"Error initializing RL components: {e}")
                self.env = None
                self.vec_env = None
                self.model = None
        
        # Data storage
        self.prices: Dict[str, deque] = defaultdict(lambda: deque(maxlen=200))
        self.volumes: Dict[str, deque] = defaultdict(lambda: deque(maxlen=200))
        self.environments: Dict[str, TradingEnvironment] = {}
        
        self.last_side: Dict[str, Side] = defaultdict(lambda: None)
        self.last_signals: Dict[str, datetime] = defaultdict(lambda: datetime.min.replace(tzinfo=timezone.utc))
        self.tick_count: Dict[str, int] = defaultdict(int)
        
        # Training state
        self.models_trained = False
        self.training_episodes = 0
        
        # Model persistence
        self.models_dir = Path("models") / "rl"
        self.models_dir.mkdir(parents=True, exist_ok=True)
    
    def update_data(self, tick: PriceTick):
        """Update internal data structures"""
        symbol = tick.symbol
        self.prices[symbol].append(tick.price)
        self.volumes[symbol].append(tick.volume if hasattr(tick, 'volume') else 1000)
        
        # Update environment if it exists
        if symbol in self.environments:
            self.environments[symbol].update_market_data(tick.price, tick.volume if hasattr(tick, 'volume') else 1000)
    
    def train_model(self, symbol: str):
        """Train RL model for a symbol"""
        if not RL_AVAILABLE or len(self.prices[symbol]) < self.config.min_data_points:
            return
        
        # Create new environment for this symbol
        env = TradingEnvironment(self.config)
        
        # Populate environment with historical data
        for i, price in enumerate(self.prices[symbol]):
            volume = self.volumes[symbol][i] if i < len(self.volumes[symbol]) else 1000
            env.update_market_data(price, volume)
        
        # Train model
        try:
            # Create temporary environment for training with GPU support
            temp_env = DummyVecEnv([lambda: env])
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            
            temp_model = PPO(
                "MlpPolicy",
                temp_env,
                learning_rate=self.config.learning_rate,
                batch_size=self.config.batch_size,
                n_steps=self.config.n_steps,
                n_epochs=self.config.n_epochs,
                gamma=self.config.gamma,
                device=device,
                verbose=0
            )
            
            # Train for a few episodes
            temp_model.learn(total_timesteps=1000)
            
            # Store trained model and environment
            self.environments[symbol] = env
            self.models_trained = True
            self.training_episodes += 1
            
            logger.info(f"RL model trained for {symbol}. Episodes: {self.training_episodes}")
            
        except Exception as e:
            logger.error(f"Error training RL model: {e}")
    
    def predict(self, symbol: str) -> Tuple[float, float]:
        """Make prediction using RL model"""
        if not RL_AVAILABLE or self.model is None or symbol not in self.environments or not self.models_trained:
            return 0.5, 0.0
        
        try:
            # Get current observation
            obs = self.environments[symbol]._get_observation()
            
            # Make prediction using the model
            action, _ = self.model.predict(obs, deterministic=True)
            
            # Convert action to probability
            # Action is in [-1, 1], convert to [0, 1] for buy probability
            buy_probability = (action[0] + 1) / 2
            
            # Confidence based on action strength
            confidence = abs(action[0])
            
            return buy_probability, confidence
            
        except Exception as e:
            logger.error(f"Error making RL prediction: {e}")
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
        """Process tick and generate signals - IMPROVED RL STRATEGY WITH RISK MANAGEMENT"""
        symbol = tick.symbol
        
        try:
            # Simple data storage without complex RL operations
            self.prices[symbol].append(tick.price)
            # Handle volume more carefully - default to 1000 if None or missing
            volume = 1000  # Default volume
            if hasattr(tick, 'volume') and tick.volume is not None:
                volume = tick.volume
            self.volumes[symbol].append(volume)
            
            # Basic requirements check
            if len(self.prices[symbol]) < 20:  # Need more data for better analysis
                return None
            
            # Skip if price is too small (prevents numerical issues)
            if tick.price < 0.01:
                return None
            
            # Simple time throttling
            time_since_last = datetime.now(timezone.utc) - self.last_signals[symbol]
            if time_since_last.total_seconds() < 180:  # 3 minutes between signals
                return None
            
            # Get price and volume data
            prices = list(self.prices[symbol])
            volumes = list(self.volumes[symbol])
            current_price = prices[-1]
            
            if len(prices) < 20:
                return None
            
            # === ADVANCED TECHNICAL ANALYSIS ===
            
            # 1. Trend Analysis (EMA-based)
            def calculate_ema(data, period):
                """Calculate Exponential Moving Average"""
                multiplier = 2 / (period + 1)
                ema = [data[0]]  # Start with first price
                for price in data[1:]:
                    ema.append((price * multiplier) + (ema[-1] * (1 - multiplier)))
                return ema[-1]
            
            try:
                ema_5 = calculate_ema(prices[-10:], 5)
                ema_10 = calculate_ema(prices[-15:], 10)
                ema_20 = calculate_ema(prices[-20:], 20)
                
                # Trend strength (0 = strong down, 1 = strong up)
                trend_score = 0.5  # Default neutral
                if ema_5 > ema_10 > ema_20:
                    trend_score = 0.8  # Strong uptrend
                elif ema_5 > ema_10:
                    trend_score = 0.65  # Weak uptrend
                elif ema_5 < ema_10 < ema_20:
                    trend_score = 0.2  # Strong downtrend
                elif ema_5 < ema_10:
                    trend_score = 0.35  # Weak downtrend
                
            except (IndexError, ZeroDivisionError):
                trend_score = 0.5
            
            # 2. RSI (Relative Strength Index)
            try:
                price_changes = [prices[i] - prices[i-1] for i in range(-14, 0)]
                gains = [max(0, change) for change in price_changes]
                losses = [abs(min(0, change)) for change in price_changes]
                
                avg_gain = sum(gains) / len(gains) if gains else 0
                avg_loss = sum(losses) / len(losses) if losses else 0
                
                if avg_loss == 0:
                    rsi = 100
                else:
                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))
                
                # RSI score (0 = oversold/buy, 1 = overbought/sell)
                if rsi < 30:
                    rsi_score = 0.8  # Oversold - buy signal
                elif rsi > 70:
                    rsi_score = 0.2  # Overbought - sell signal
                else:
                    rsi_score = 0.5  # Neutral
                    
            except (IndexError, ZeroDivisionError):
                rsi_score = 0.5
            
            # 3. Volume Analysis
            try:
                recent_vol = sum(volumes[-3:]) / 3
                avg_vol = sum(volumes[-10:]) / 10
                volume_ratio = recent_vol / avg_vol if avg_vol > 0 else 1
                
                # Volume confirmation score
                if volume_ratio > 1.5:
                    volume_score = 0.7  # High volume confirms move
                elif volume_ratio < 0.5:
                    volume_score = 0.3  # Low volume - weak signal
                else:
                    volume_score = 0.5  # Normal volume
                    
            except (TypeError, ZeroDivisionError):
                volume_score = 0.5
            
            # 4. Volatility Analysis (for risk management)
            try:
                price_changes = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(-10, 0) if prices[i-1] > 0]
                if price_changes:
                    volatility = sum([abs(change) for change in price_changes]) / len(price_changes)
                    
                    # For high volatility stocks like TNXP, be more conservative
                    if volatility > 0.1:  # 10% daily volatility
                        volatility_penalty = 0.3  # Be very conservative
                    elif volatility > 0.05:  # 5% daily volatility
                        volatility_penalty = 0.4  # Be conservative
                    else:
                        volatility_penalty = 0.5  # Normal
                else:
                    volatility_penalty = 0.5
                    
            except (IndexError, ZeroDivisionError):
                volatility_penalty = 0.5
            
            # 5. Price Position Analysis
            try:
                high_20 = max(prices[-20:])
                low_20 = min(prices[-20:])
                price_position = (current_price - low_20) / (high_20 - low_20) if high_20 > low_20 else 0.5
                
                # Avoid buying at peaks, prefer buying at dips
                if price_position > 0.8:
                    position_score = 0.2  # Near high - avoid buying
                elif price_position < 0.3:
                    position_score = 0.8  # Near low - good buying opportunity
                else:
                    position_score = 0.5  # Middle range
                    
            except (IndexError, ZeroDivisionError):
                position_score = 0.5
            
            # === COMBINE ALL FACTORS ===
            # Weighted combination with emphasis on risk management
            combined_score = (
                trend_score * 0.25 +        # Trend direction
                rsi_score * 0.25 +          # Momentum
                volume_score * 0.15 +       # Volume confirmation
                volatility_penalty * 0.15 + # Risk management
                position_score * 0.20       # Entry timing
            )
            
            # Calculate confidence based on factor alignment
            factor_scores = [trend_score, rsi_score, volume_score, volatility_penalty, position_score]
            factor_variance = sum([(score - combined_score) ** 2 for score in factor_scores]) / len(factor_scores)
            confidence = max(0.3, 1.0 - factor_variance * 3)  # Higher agreement = higher confidence
            
            # === SIGNAL GENERATION WITH BALANCED THRESHOLDS ===
            
            # Balanced thresholds - conservative but not too restrictive
            if combined_score > 0.58 and confidence > 0.4:  # Good buy signal
                if self.last_side[symbol] != Side.buy:
                    self.last_side[symbol] = Side.buy
                    self.last_signals[symbol] = tick.timestamp
                    
                    logger.info(f"Improved RL buy signal for {symbol}: score={combined_score:.3f}, conf={confidence:.3f}, trend={trend_score:.3f}, rsi={rsi_score:.3f}, pos={position_score:.3f}")
                    return Signal(
                        symbol=symbol,
                        side=Side.buy,
                        price=tick.price,
                        timestamp=tick.timestamp,
                        confidence=confidence
                    )
            
            elif combined_score < 0.42 and confidence > 0.35:  # Good sell signal
                if self.last_side[symbol] == Side.buy:
                    self.last_side[symbol] = Side.sell
                    self.last_signals[symbol] = tick.timestamp
                    
                    logger.info(f"Improved RL sell signal for {symbol}: score={combined_score:.3f}, conf={confidence:.3f}, trend={trend_score:.3f}, rsi={rsi_score:.3f}, pos={position_score:.3f}")
                    return Signal(
                        symbol=symbol,
                        side=Side.sell,
                        price=tick.price,
                        timestamp=tick.timestamp,
                        confidence=confidence
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Error in improved RL strategy for {symbol}: {e}")
            return None
    
    def save_model(self, symbol: str) -> Optional[str]:
        """Save the RL model to disk"""
        try:
            # Create symbol-specific directory
            symbol_dir = self.models_dir / symbol
            symbol_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = symbol_dir / f"model_{timestamp}.pkl"
            
            # Save model data
            model_data = {
                'strategy_type': 'rl',
                'symbol': symbol,
                'timestamp': timestamp,
                'model_state': {
                    'environments': self.environments.get(symbol),
                    'models_trained': self.models_trained,
                    'training_episodes': self.training_episodes
                },
                'config': self.config
            }
            
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"RL model saved for {symbol}: {model_path}")
            return str(model_path)
            
        except Exception as e:
            logger.error(f"Failed to save RL model for {symbol}: {e}")
            return None

    def load_model(self, symbol: str, model_path: str = None) -> bool:
        """Load the RL model from disk"""
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
            if model_data['model_state']['environments']:
                self.environments[symbol] = model_data['model_state']['environments']
            self.models_trained = model_data['model_state']['models_trained']
            self.training_episodes = model_data['model_state']['training_episodes']
            
            logger.info(f"RL model loaded for {symbol}: {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load RL model for {symbol}: {e}")
            return False


# Factory function to create RL strategies
def create_rl_strategy(config: RLStrategyConfig = None) -> RLStrategy:
    """Factory function to create RL strategy instances"""
    if config is None:
        config = RLStrategyConfig()
    
    return RLStrategy(config)


# Example usage
if __name__ == "__main__":
    # Test the RL strategy
    config = RLStrategyConfig()
    rl_strategy = create_rl_strategy(config)
    
    print("RL strategy created successfully!") 