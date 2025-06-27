#!/usr/bin/env python3
"""
High Return Strategies

Enhanced strategies designed to achieve >35% returns through:
- More aggressive parameters
- Better risk management
- Optimized entry/exit timing
- Multi-timeframe analysis
"""

import numpy as np
import logging
from collections import deque, defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from tradebot.common.models import PriceTick, Signal, Side

logger = logging.getLogger("high_return_strategies")


@dataclass
class AggressiveConfig:
    """Configuration for aggressive high-return strategies"""
    # More aggressive RSI thresholds
    rsi_oversold: float = 25  # More aggressive than 30
    rsi_overbought: float = 75  # More aggressive than 70
    
    # Tighter stop-losses for better risk management
    stop_loss: float = 0.015  # 1.5% stop-loss
    take_profit: float = 0.05  # 5% take-profit
    
    # More sensitive mean reversion
    z_score_threshold: float = -1.5  # More sensitive than -2.0
    
    # Volume confirmation
    volume_multiplier: float = 2.0  # Higher volume requirement
    
    # Momentum thresholds
    momentum_threshold: float = 0.03  # 3% daily move for momentum


class AggressiveMeanReversionStrategy:
    """Aggressive mean reversion with tight risk management."""
    
    def __init__(self, lookback_period: int = 15, z_score_threshold: float = -1.5, 
                 stop_loss: float = 0.015, take_profit: float = 0.05):
        self.lookback_period = lookback_period
        self.z_score_threshold = z_score_threshold
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        
        self.prices: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.entry_prices: Dict[str, float] = {}
        self.last_side: Dict[str, Side] = defaultdict(lambda: None)
        self.last_signals: Dict[str, datetime] = defaultdict(lambda: datetime.min.replace(tzinfo=timezone.utc))
    
    def update_data(self, tick: PriceTick):
        symbol = tick.symbol
        self.prices[symbol].append(tick.price)
    
    def calculate_z_score(self, symbol: str) -> Optional[float]:
        prices = list(self.prices[symbol])
        if len(prices) < self.lookback_period:
            return None
        mean = np.mean(prices[-self.lookback_period:])
        std = np.std(prices[-self.lookback_period:])
        if std == 0:
            return 0
        return (prices[-1] - mean) / std
    
    def on_tick(self, tick: PriceTick) -> Optional[Signal]:
        symbol = tick.symbol
        self.update_data(tick)
        
        # Check stop-loss/take-profit for existing positions
        if self.last_side[symbol] == Side.buy and symbol in self.entry_prices:
            entry_price = self.entry_prices[symbol]
            current_price = tick.price
            
            # Stop-loss hit
            if current_price <= entry_price * (1 - self.stop_loss):
                self.last_side[symbol] = Side.sell
                logger.info(f"Aggressive stop-loss triggered for {symbol}: entry={entry_price:.2f}, exit={current_price:.2f}")
                return Signal(symbol=symbol, side=Side.sell, price=current_price, timestamp=tick.timestamp, confidence=1.0)
            
            # Take-profit hit
            if current_price >= entry_price * (1 + self.take_profit):
                self.last_side[symbol] = Side.sell
                logger.info(f"Take-profit triggered for {symbol}: entry={entry_price:.2f}, exit={current_price:.2f}")
                return Signal(symbol=symbol, side=Side.sell, price=current_price, timestamp=tick.timestamp, confidence=1.0)
        
        # Generate new signals
        z_score = self.calculate_z_score(symbol)
        if z_score is None:
            return None
        
        # More aggressive entry conditions
        if z_score < self.z_score_threshold and self.last_side[symbol] != Side.buy:
            self.last_side[symbol] = Side.buy
            self.entry_prices[symbol] = tick.price
            logger.info(f"Aggressive mean reversion buy for {symbol}: z_score={z_score:.2f}, price={tick.price:.2f}")
            return Signal(symbol=symbol, side=Side.buy, price=tick.price, timestamp=tick.timestamp, confidence=1.0)
        
        # More aggressive exit conditions
        elif z_score > -0.5 and self.last_side[symbol] == Side.buy:
            self.last_side[symbol] = Side.sell
            logger.info(f"Aggressive mean reversion sell for {symbol}: z_score={z_score:.2f}, price={tick.price:.2f}")
            return Signal(symbol=symbol, side=Side.sell, price=tick.price, timestamp=tick.timestamp, confidence=1.0)
        
        return None


class MomentumBreakoutStrategy:
    """Enhanced momentum breakout with volume confirmation."""
    
    def __init__(self, lookback: int = 15, volume_multiplier: float = 2.0, 
                 momentum_threshold: float = 0.03, stop_loss: float = 0.02):
        self.lookback = lookback
        self.volume_multiplier = volume_multiplier
        self.momentum_threshold = momentum_threshold
        self.stop_loss = stop_loss
        
        self.prices: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.volumes: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.entry_prices: Dict[str, float] = {}
        self.last_side: Dict[str, Side] = defaultdict(lambda: None)
        self.last_signals: Dict[str, datetime] = defaultdict(lambda: datetime.min.replace(tzinfo=timezone.utc))
    
    def update_data(self, tick: PriceTick):
        symbol = tick.symbol
        self.prices[symbol].append(tick.price)
        if tick.volume is not None:
            self.volumes[symbol].append(tick.volume)
    
    def on_tick(self, tick: PriceTick) -> Optional[Signal]:
        symbol = tick.symbol
        self.update_data(tick)
        
        # Check stop-loss for existing positions
        if self.last_side[symbol] == Side.buy and symbol in self.entry_prices:
            entry_price = self.entry_prices[symbol]
            current_price = tick.price
            
            if current_price <= entry_price * (1 - self.stop_loss):
                self.last_side[symbol] = Side.sell
                logger.info(f"Momentum stop-loss triggered for {symbol}: entry={entry_price:.2f}, exit={current_price:.2f}")
                return Signal(symbol=symbol, side=Side.sell, price=current_price, timestamp=tick.timestamp, confidence=1.0)
        
        prices = list(self.prices[symbol])
        volumes = list(self.volumes[symbol])
        
        if len(prices) < self.lookback or len(volumes) < self.lookback:
            return None
        
        current_price = prices[-1]
        current_volume = volumes[-1]
        
        # Calculate momentum indicators
        price_change = (current_price / prices[-2] - 1) if len(prices) > 1 else 0
        rolling_high = max(prices[-self.lookback:-1]) if len(prices) > 1 else current_price
        avg_volume = np.mean(volumes[-self.lookback:])
        
        # Enhanced breakout conditions
        breakout = (current_price > rolling_high * 1.01 and  # 1% above recent high
                   current_volume > avg_volume * self.volume_multiplier and
                   price_change > self.momentum_threshold)
        
        # Momentum fade exit
        momentum_fade = current_price < rolling_high * 0.98  # 2% below recent high
        
        if breakout and self.last_side[symbol] != Side.buy:
            self.last_side[symbol] = Side.buy
            self.entry_prices[symbol] = current_price
            logger.info(f"Enhanced momentum breakout for {symbol}: price={current_price:.2f}, volume={current_volume}")
            return Signal(symbol=symbol, side=Side.buy, price=current_price, timestamp=tick.timestamp, confidence=1.0)
        
        elif momentum_fade and self.last_side[symbol] == Side.buy:
            self.last_side[symbol] = Side.sell
            logger.info(f"Momentum fade exit for {symbol}: price={current_price:.2f}")
            return Signal(symbol=symbol, side=Side.sell, price=current_price, timestamp=tick.timestamp, confidence=1.0)
        
        return None


class MultiTimeframeMomentumStrategy:
    """Multi-timeframe momentum strategy for higher returns."""
    
    def __init__(self, short_lookback: int = 5, medium_lookback: int = 15, 
                 long_lookback: int = 30, stop_loss: float = 0.02):
        self.short_lookback = short_lookback
        self.medium_lookback = medium_lookback
        self.long_lookback = long_lookback
        self.stop_loss = stop_loss
        
        self.prices: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.entry_prices: Dict[str, float] = {}
        self.last_side: Dict[str, Side] = defaultdict(lambda: None)
        self.last_signals: Dict[str, datetime] = defaultdict(lambda: datetime.min.replace(tzinfo=timezone.utc))
    
    def update_data(self, tick: PriceTick):
        symbol = tick.symbol
        self.prices[symbol].append(tick.price)
    
    def on_tick(self, tick: PriceTick) -> Optional[Signal]:
        symbol = tick.symbol
        self.update_data(tick)
        
        # Check stop-loss
        if self.last_side[symbol] == Side.buy and symbol in self.entry_prices:
            entry_price = self.entry_prices[symbol]
            current_price = tick.price
            
            if current_price <= entry_price * (1 - self.stop_loss):
                self.last_side[symbol] = Side.sell
                logger.info(f"Multi-timeframe stop-loss for {symbol}: entry={entry_price:.2f}, exit={current_price:.2f}")
                return Signal(symbol=symbol, side=Side.sell, price=current_price, timestamp=tick.timestamp, confidence=1.0)
        
        prices = list(self.prices[symbol])
        if len(prices) < self.long_lookback:
            return None
        
        current_price = prices[-1]
        
        # Multi-timeframe analysis
        short_ma = np.mean(prices[-self.short_lookback:])
        medium_ma = np.mean(prices[-self.medium_lookback:])
        long_ma = np.mean(prices[-self.long_lookback:])
        
        # All timeframes aligned bullish
        bullish_alignment = (current_price > short_ma > medium_ma > long_ma)
        
        # Short-term momentum
        short_momentum = (current_price / short_ma - 1) > 0.02  # 2% above short MA
        
        # Buy signal: all timeframes bullish + short momentum
        if bullish_alignment and short_momentum and self.last_side[symbol] != Side.buy:
            self.last_side[symbol] = Side.buy
            self.entry_prices[symbol] = current_price
            logger.info(f"Multi-timeframe buy for {symbol}: price={current_price:.2f}")
            return Signal(symbol=symbol, side=Side.buy, price=current_price, timestamp=tick.timestamp, confidence=1.0)
        
        # Sell signal: any timeframe breaks down
        elif (current_price < short_ma or current_price < medium_ma) and self.last_side[symbol] == Side.buy:
            self.last_side[symbol] = Side.sell
            logger.info(f"Multi-timeframe sell for {symbol}: price={current_price:.2f}")
            return Signal(symbol=symbol, side=Side.sell, price=current_price, timestamp=tick.timestamp, confidence=1.0)
        
        return None


def create_high_return_strategy(strategy_type: str, **kwargs):
    """Factory function for high-return strategies."""
    if strategy_type == "aggressive_mean_reversion":
        return AggressiveMeanReversionStrategy(**kwargs)
    elif strategy_type == "enhanced_momentum":
        return MomentumBreakoutStrategy(**kwargs)
    elif strategy_type == "multi_timeframe_momentum":
        return MultiTimeframeMomentumStrategy(**kwargs)
    else:
        raise ValueError(f"Unknown high-return strategy type: {strategy_type}") 