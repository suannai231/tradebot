import numpy as np
import pandas as pd
import logging
from collections import deque, defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
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
                
                # Initialize PPO agent
                self.model = PPO(
                    "MlpPolicy",
                    self.vec_env,
                    learning_rate=config.learning_rate,
                    batch_size=config.batch_size,
                    n_steps=config.n_steps,
                    n_epochs=config.n_epochs,
                    gamma=config.gamma,
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
            # Create temporary environment for training
            temp_env = DummyVecEnv([lambda: env])
            temp_model = PPO(
                "MlpPolicy",
                temp_env,
                learning_rate=self.config.learning_rate,
                batch_size=self.config.batch_size,
                n_steps=self.config.n_steps,
                n_epochs=self.config.n_epochs,
                gamma=self.config.gamma,
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
            # Simple data storage without complex RL operations
            self.prices[symbol].append(tick.price)
            self.volumes[symbol].append(tick.volume if hasattr(tick, 'volume') else 1000)
            
            # Basic requirements check
            if len(self.prices[symbol]) < 10:
                return None
            
            # Skip if price is too small (prevents numerical issues)
            if tick.price < 0.01:
                return None
            
            # Simple time throttling
            time_since_last = datetime.now(timezone.utc) - self.last_signals[symbol]
            if time_since_last.total_seconds() < 120:  # 2 minutes (more frequent)
                return None
            
            # Simplified "RL-inspired" strategy using multiple factors
            prices = list(self.prices[symbol])
            volumes = list(self.volumes[symbol])
            
            if len(prices) < 10:
                return None
            
            # Factor 1: Price momentum (short vs medium term)
            short_ma = sum(prices[-3:]) / 3
            medium_ma = sum(prices[-7:]) / 7
            momentum_score = 0.7 if short_ma > medium_ma else 0.3
            
            # Factor 2: Volume confirmation
            recent_vol = sum(volumes[-3:]) / 3
            avg_vol = sum(volumes[-10:]) / 10
            volume_score = 0.6 if recent_vol > avg_vol * 1.2 else 0.4
            
            # Factor 3: Price volatility (opportunity factor)
            price_changes = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(-5, 0)]
            volatility = sum([abs(change) for change in price_changes]) / len(price_changes)
            vol_score = 0.6 if volatility > 0.02 else 0.5  # Higher vol = more opportunity
            
            # Factor 4: Mean reversion component
            current_price = prices[-1]
            long_ma = sum(prices[-10:]) / 10
            distance_from_mean = (current_price - long_ma) / long_ma
            reversion_score = 0.7 if distance_from_mean < -0.02 else (0.3 if distance_from_mean > 0.02 else 0.5)
            
            # Combine factors (weighted average)
            combined_score = (momentum_score * 0.3 + volume_score * 0.2 + 
                            vol_score * 0.2 + reversion_score * 0.3)
            
            # Calculate confidence based on factor alignment
            factor_scores = [momentum_score, volume_score, vol_score, reversion_score]
            factor_variance = sum([(score - combined_score) ** 2 for score in factor_scores]) / len(factor_scores)
            confidence = max(0.3, 1.0 - factor_variance * 5)  # Higher agreement = higher confidence
            
            # Generate signals with more aggressive thresholds
            if combined_score > 0.55 and confidence > 0.3:  # Lowered thresholds
                if self.last_side[symbol] != Side.buy:
                    self.last_side[symbol] = Side.buy
                    self.last_signals[symbol] = tick.timestamp
                    
                    logger.info(f"Simplified RL buy signal for {symbol}: score={combined_score:.3f}, conf={confidence:.3f}")
                    return Signal(
                        symbol=symbol,
                        side=Side.buy,
                        price=tick.price,
                        timestamp=tick.timestamp,
                        confidence=confidence
                    )
            
            elif combined_score < 0.45 and confidence > 0.3:  # Lowered thresholds
                if self.last_side[symbol] == Side.buy:
                    self.last_side[symbol] = Side.sell
                    self.last_signals[symbol] = tick.timestamp
                    
                    logger.info(f"Simplified RL sell signal for {symbol}: score={combined_score:.3f}, conf={confidence:.3f}")
                    return Signal(
                        symbol=symbol,
                        side=Side.sell,
                        price=tick.price,
                        timestamp=tick.timestamp,
                        confidence=confidence
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Error in simplified RL strategy for {symbol}: {e}")
            return None


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