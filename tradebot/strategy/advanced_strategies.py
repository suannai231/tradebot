import numpy as np
import logging
import os
from collections import deque, defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import asyncpg
import aiohttp

from tradebot.common.models import PriceTick, Signal, Side

logger = logging.getLogger("advanced_strategies")

def get_table_name() -> str:
    """Get the price ticks table name based on DATA_SOURCE environment variable."""
    data_source = os.getenv('DATA_SOURCE', 'synthetic')
    return f'price_ticks_{data_source}'

# NOTE: This file contains consolidated strategies from the former high_return_strategies.py
# All high-return strategies have been integrated into this single advanced strategies module
# Available high-return strategies:
# - AggressiveMeanReversionStrategy: Aggressive mean reversion with tight risk management
# - EnhancedMomentumStrategy: Enhanced momentum breakout with volume confirmation  
# - MultiTimeframeMomentumStrategy: Multi-timeframe momentum for higher returns
# Use create_strategy() or create_high_return_strategy() factory functions to create instances

@dataclass
class StrategyConfig:
    """Configuration for advanced strategies"""
    # Moving Average settings
    short_window: int = 5
    long_window: int = 20
    ema_alpha: float = 0.1
    
    # RSI settings
    rsi_period: int = 14
    rsi_oversold: float = 30
    rsi_overbought: float = 70
    
    # Bollinger Bands settings
    bb_period: int = 20
    bb_std_dev: float = 2.0
    
    # Volume settings
    volume_period: int = 20
    volume_threshold: float = 1.5
    
    # Support/Resistance settings
    support_window: int = 20
    support_threshold: float = 0.02
    
    # Composite scoring
    min_composite_score: float = 0.7


class SplitAwareDataProvider:
    """Provides split-aware data for strategy analysis."""
    
    def __init__(self, db_pool: asyncpg.Pool):
        self.db_pool = db_pool
        self._split_cache: Dict[str, List[Dict]] = {}
    
    async def get_splits(self, symbol: str) -> List[Dict]:
        """Get split events for a symbol with caching."""
        if symbol not in self._split_cache:
            async with self.db_pool.acquire() as conn:
                query = """
                    SELECT split_date, split_ratio 
                    FROM stock_splits 
                    WHERE symbol = $1 
                    ORDER BY split_date ASC
                """
                rows = await conn.fetch(query, symbol.upper())
                self._split_cache[symbol] = [
                    {
                        "split_date": row["split_date"],
                        "split_ratio": float(row["split_ratio"])
                    }
                    for row in rows
                ]
        return self._split_cache[symbol]
    
    async def get_raw_volume_data(self, symbol: str, days_back: int = 30) -> List[Dict]:
        """Get raw volume data for accurate volume analysis."""
        async with self.db_pool.acquire() as conn:
            table_name = get_table_name()
            query = f"""
                SELECT timestamp, volume, close_price
                FROM {table_name} 
                WHERE symbol = $1 
                    AND timestamp >= NOW() - INTERVAL '%s days'
                    AND volume IS NOT NULL
                ORDER BY timestamp ASC
            """
            rows = await conn.fetch(query, symbol.upper(), days_back)
            return [
                {
                    "timestamp": row["timestamp"],
                    "volume": int(row["volume"]),
                    "price": float(row["close_price"])
                }
                for row in rows
            ]
    
    async def get_split_adjusted_prices(self, symbol: str, days_back: int = 365) -> List[Dict]:
        """Get split-adjusted prices for trend analysis."""
        # Get raw price data
        async with self.db_pool.acquire() as conn:
            table_name = get_table_name()
            query = f"""
                SELECT timestamp, open_price, high_price, low_price, close_price
                FROM {table_name} 
                WHERE symbol = $1 
                    AND timestamp >= NOW() - INTERVAL '%s days'
                    AND close_price IS NOT NULL
                ORDER BY timestamp ASC
            """
            rows = await conn.fetch(query, symbol.upper(), days_back)
            
            raw_data = [
                {
                    "timestamp": row["timestamp"],
                    "open": float(row["open_price"]),
                    "high": float(row["high_price"]),
                    "low": float(row["low_price"]),
                    "close": float(row["close_price"])
                }
                for row in rows
            ]
        
        # Apply split adjustments
        splits = await self.get_splits(symbol)
        if not splits:
            return raw_data
        
        adjusted_data = []
        for data_point in raw_data:
            adjustment = 1.0
            for split in splits:
                if data_point["timestamp"].date() < split["split_date"]:
                    adjustment *= split["split_ratio"]
            
            adjusted_point = data_point.copy()
            if adjustment != 1.0:
                adjusted_point["open"] /= adjustment
                adjusted_point["high"] /= adjustment
                adjusted_point["low"] /= adjustment
                adjusted_point["close"] /= adjustment
            
            adjusted_data.append(adjusted_point)
        
        return adjusted_data


class SplitAwareStrategy:
    """Base class for strategies that need split-aware data analysis."""
    
    def __init__(self, db_pool: asyncpg.Pool):
        self.data_provider = SplitAwareDataProvider(db_pool)
        self.last_analysis: Dict[str, datetime] = {}
    
    async def analyze_with_split_awareness(self, symbol: str) -> Optional[Dict]:
        """Analyze a symbol using both raw volume and split-adjusted prices."""
        try:
            # Get raw volume data for accurate volume analysis
            volume_data = await self.data_provider.get_raw_volume_data(symbol, days_back=30)
            
            # Get split-adjusted prices for trend analysis
            price_data = await self.data_provider.get_split_adjusted_prices(symbol, days_back=90)
            
            if not volume_data or not price_data:
                return None
            
            # Calculate volume metrics using raw data
            recent_volumes = [d["volume"] for d in volume_data[-20:]]  # Last 20 days
            avg_volume = sum(recent_volumes) / len(recent_volumes)
            current_volume = volume_data[-1]["volume"]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Calculate price metrics using split-adjusted data
            recent_prices = [d["close"] for d in price_data[-20:]]  # Last 20 days
            price_change = (recent_prices[-1] - recent_prices[0]) / recent_prices[0] if recent_prices[0] > 0 else 0
            
            # Combine analysis
            analysis = {
                "symbol": symbol,
                "volume_ratio": volume_ratio,
                "avg_volume": avg_volume,
                "current_volume": current_volume,
                "price_change_20d": price_change,
                "current_price": recent_prices[-1],
                "analysis_timestamp": datetime.now(timezone.utc)
            }
            
            self.last_analysis[symbol] = analysis["analysis_timestamp"]
            return analysis
            
        except Exception as e:
            logger.error(f"Error in split-aware analysis for {symbol}: {e}")
            return None


class AdvancedBuyStrategy:
    """Advanced strategy combining multiple technical indicators for buy signals.
    Accepts any StrategyConfig field as a keyword argument for easy tuning.
    """
    def __init__(self, config: StrategyConfig = None, **kwargs):
        # start with default or provided config
        self.config = config or StrategyConfig()
        # override with supplied keyword args that match StrategyConfig attributes
        for k, v in kwargs.items():
            if hasattr(self.config, k):
                setattr(self.config, k, v)
        
        # Price data storage
        self.prices: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.volumes: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.opens: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.highs: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.lows: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # State tracking
        self.last_side: Dict[str, Side] = defaultdict(lambda: None)
        self.last_signals: Dict[str, datetime] = defaultdict(lambda: datetime.min.replace(tzinfo=timezone.utc))
        
    def update_data(self, tick: PriceTick):
        """Update internal data structures with new tick."""
        symbol = tick.symbol
        
        # Update price data
        self.prices[symbol].append(tick.price)
        
        # Update OHLCV data if available
        if tick.open is not None:
            self.opens[symbol].append(tick.open)
        if tick.high is not None:
            self.highs[symbol].append(tick.high)
        if tick.low is not None:
            self.lows[symbol].append(tick.low)
        if tick.volume is not None:
            self.volumes[symbol].append(tick.volume)
    
    def calculate_sma(self, symbol: str, period: int) -> Optional[float]:
        """Calculate Simple Moving Average."""
        prices = list(self.prices[symbol])
        if len(prices) < period:
            return None
        return np.mean(prices[-period:])
    
    def calculate_ema(self, symbol: str, period: int) -> Optional[float]:
        """Calculate Exponential Moving Average."""
        prices = list(self.prices[symbol])
        if len(prices) < period:
            return None
        
        alpha = 2.0 / (period + 1)
        ema = prices[0]
        for price in prices[1:]:
            ema = alpha * price + (1 - alpha) * ema
        return ema
    
    def calculate_rsi(self, symbol: str) -> Optional[float]:
        """Calculate Relative Strength Index."""
        prices = list(self.prices[symbol])
        if len(prices) < self.config.rsi_period + 1:
            return None
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-self.config.rsi_period:])
        avg_loss = np.mean(losses[-self.config.rsi_period:])
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_bollinger_bands(self, symbol: str) -> Optional[Tuple[float, float, float]]:
        """Calculate Bollinger Bands (upper, middle, lower)."""
        prices = list(self.prices[symbol])
        if len(prices) < self.config.bb_period:
            return None
        
        middle = np.mean(prices[-self.config.bb_period:])
        std = np.std(prices[-self.config.bb_period:])
        upper = middle + (self.config.bb_std_dev * std)
        lower = middle - (self.config.bb_std_dev * std)
        
        return upper, middle, lower
    
    def detect_support_level(self, symbol: str, current_price: float) -> bool:
        """Detect if price is near a support level."""
        prices = list(self.prices[symbol])
        if len(prices) < self.config.support_window:
            return False
        
        # Find local minima in recent prices
        for i in range(self.config.support_window, len(prices) - self.config.support_window):
            if all(prices[i] <= prices[j] for j in range(i-self.config.support_window, i+self.config.support_window+1)):
                support_level = prices[i]
                # Check if current price is near this support level
                if abs(current_price - support_level) / support_level < self.config.support_threshold:
                    return True
        return False
    
    def calculate_volume_score(self, symbol: str) -> float:
        """Calculate volume-based score."""
        volumes = list(self.volumes[symbol])
        if len(volumes) < self.config.volume_period:
            return 0.5  # Neutral score if insufficient data
        
        current_volume = volumes[-1]
        avg_volume = np.mean(volumes[-self.config.volume_period:])
        
        if avg_volume == 0:
            return 0.5
        
        volume_ratio = current_volume / avg_volume
        return min(volume_ratio / self.config.volume_threshold, 1.0)
    
    def detect_bullish_patterns(self, symbol: str) -> float:
        """Detect bullish candlestick patterns."""
        if len(self.opens[symbol]) < 2 or len(self.highs[symbol]) < 2 or len(self.lows[symbol]) < 2:
            return 0.5
        
        opens = list(self.opens[symbol])
        highs = list(self.highs[symbol])
        lows = list(self.lows[symbol])
        prices = list(self.prices[symbol])
        
        score = 0.5  # Base neutral score
        
        # Bullish engulfing pattern
        if len(opens) >= 2:
            prev_open, prev_close = opens[-2], prices[-2]
            curr_open, curr_close = opens[-1], prices[-1]
            
            is_prev_bearish = prev_close < prev_open
            is_curr_bullish = curr_close > curr_open
            is_engulfing = curr_open < prev_close and curr_close > prev_open
            
            if is_prev_bearish and is_curr_bullish and is_engulfing:
                score += 0.3
        
        # Hammer pattern
        if len(highs) >= 1 and len(lows) >= 1:
            high, low, close = highs[-1], lows[-1], prices[-1]
            body_size = abs(close - opens[-1])
            lower_shadow = min(close, opens[-1]) - low
            upper_shadow = high - max(close, opens[-1])
            
            if lower_shadow > 2 * body_size and upper_shadow < body_size:
                score += 0.2
        
        return min(score, 1.0)
    
    def calculate_composite_score(self, symbol: str) -> float:
        """Calculate composite score from all indicators."""
        scores = {}
        
        # Moving Average Crossover Score
        short_ma = self.calculate_sma(symbol, self.config.short_window)
        long_ma = self.calculate_sma(symbol, self.config.long_window)
        if short_ma and long_ma:
            scores['ma_crossover'] = 1.0 if short_ma > long_ma else 0.0
        else:
            scores['ma_crossover'] = 0.5
        
        # RSI Score
        rsi = self.calculate_rsi(symbol)
        if rsi is not None:
            if rsi < self.config.rsi_oversold:
                scores['rsi'] = 1.0  # Oversold - good buy opportunity
            elif rsi < 50:
                scores['rsi'] = 0.7  # Below midpoint - still favorable
            elif rsi < self.config.rsi_overbought:
                scores['rsi'] = 0.3  # Above midpoint - less favorable
            else:
                scores['rsi'] = 0.0  # Overbought - avoid buying
        else:
            scores['rsi'] = 0.5
        
        # Bollinger Bands Score
        bb_result = self.calculate_bollinger_bands(symbol)
        if bb_result:
            upper, middle, lower = bb_result
            current_price = self.prices[symbol][-1]
            if current_price <= lower:
                scores['bollinger'] = 1.0  # At or below lower band
            elif current_price <= middle:
                scores['bollinger'] = 0.7  # Below middle band
            else:
                scores['bollinger'] = 0.3  # Above middle band
        else:
            scores['bollinger'] = 0.5
        
        # Support Level Score
        current_price = self.prices[symbol][-1]
        scores['support'] = 1.0 if self.detect_support_level(symbol, current_price) else 0.0
        
        # Volume Score
        scores['volume'] = self.calculate_volume_score(symbol)
        
        # Pattern Score
        scores['patterns'] = self.detect_bullish_patterns(symbol)
        
        # Calculate weighted composite score
        weights = {
            'ma_crossover': 0.25,
            'rsi': 0.20,
            'bollinger': 0.20,
            'support': 0.15,
            'volume': 0.10,
            'patterns': 0.10
        }
        
        composite_score = sum(scores[factor] * weights[factor] for factor in scores)
        return composite_score
    
    def should_generate_signal(self, symbol: str) -> bool:
        """Check if enough time has passed since last signal."""
        time_since_last = datetime.now(timezone.utc) - self.last_signals[symbol]
        return time_since_last.total_seconds() > 300  # 5 minutes minimum
    
    def on_tick(self, tick: PriceTick) -> Optional[Signal]:
        """Process new tick and generate buy signals."""
        symbol = tick.symbol
        
        # Update internal data
        self.update_data(tick)
        
        # Check if we have enough data
        if len(self.prices[symbol]) < self.config.long_window:
            return None
        
        # Check if we should generate a signal
        if not self.should_generate_signal(symbol):
            return None
        
        # Calculate composite score
        composite_score = self.calculate_composite_score(symbol)
        
        # Generate buy signal if score is high enough and we're not already long
        if (composite_score >= self.config.min_composite_score and 
            self.last_side[symbol] != Side.buy):
            
            self.last_side[symbol] = Side.buy
            self.last_signals[symbol] = datetime.now(timezone.utc)
            
            logger.info(f"Buy signal for {symbol}: score={composite_score:.3f}, price={tick.price}")
            
            return Signal(
                symbol=symbol,
                side=Side.buy,
                price=tick.price,
                confidence=composite_score,
                timestamp=tick.timestamp
            )
        
        # Generate sell signal if score is low and we're long
        elif (composite_score < 0.3 and self.last_side[symbol] == Side.buy):
            self.last_side[symbol] = Side.sell
            self.last_signals[symbol] = datetime.now(timezone.utc)
            
            logger.info(f"Sell signal for {symbol}: score={composite_score:.3f}, price={tick.price}")
            
            return Signal(
                symbol=symbol,
                side=Side.sell,
                price=tick.price,
                confidence=1.0 - composite_score,
                timestamp=tick.timestamp
            )
        
        return None


class MeanReversionStrategy:
    """Mean reversion strategy based on statistical measures."""
    
    def __init__(self, lookback_period: int = 20, z_score_threshold: float = -2.0):
        self.lookback_period = lookback_period
        self.z_score_threshold = z_score_threshold
        self.prices: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
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
        z_score = self.calculate_z_score(symbol)
        if z_score is None:
            return None
        if z_score < self.z_score_threshold and self.last_side[symbol] != Side.buy:
            self.last_side[symbol] = Side.buy
            self.last_signals[symbol] = tick.timestamp
            return Signal(symbol=symbol, side=Side.buy, price=tick.price, timestamp=tick.timestamp, confidence=1.0)
        elif z_score > -self.z_score_threshold and self.last_side[symbol] == Side.buy:
            self.last_side[symbol] = Side.sell
            self.last_signals[symbol] = tick.timestamp
            return Signal(symbol=symbol, side=Side.sell, price=tick.price, timestamp=tick.timestamp, confidence=1.0)
        return None


class LowVolumeStrategy:
    """Strategy optimized for low-volume stocks and limited data scenarios."""
    
    def __init__(self, short_window: int = 3, long_window: int = 8, rsi_period: int = 7):
        self.short_window = short_window
        self.long_window = long_window
        self.rsi_period = rsi_period
        self.prices: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))
        self.last_side: Dict[str, Side] = defaultdict(lambda: None)
        self.last_signals: Dict[str, datetime] = defaultdict(lambda: datetime.min.replace(tzinfo=timezone.utc))
    
    def calculate_simple_rsi(self, symbol: str) -> Optional[float]:
        """Calculate RSI with minimal data requirements."""
        prices = list(self.prices[symbol])
        if len(prices) < self.rsi_period + 1:
            return None
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-self.rsi_period:])
        avg_loss = np.mean(losses[-self.rsi_period:])
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_volatility_score(self, symbol: str) -> float:
        """Calculate volatility-based score for limited data."""
        prices = list(self.prices[symbol])
        if len(prices) < 5:
            return 0.5
        
        # Calculate price momentum
        recent_prices = prices[-5:]
        price_change = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
        
        # Calculate volatility
        returns = np.diff(recent_prices) / recent_prices[:-1]
        volatility = np.std(returns)
        
        # Score based on momentum and volatility
        if price_change > 0.02 and volatility > 0.01:  # 2% gain with some volatility
            return 0.8
        elif price_change > 0.01:  # 1% gain
            return 0.6
        elif price_change < -0.02:  # 2% loss - potential reversal
            return 0.7
        else:
            return 0.3
    
    def should_generate_signal(self, symbol: str) -> bool:
        """Check if enough time has passed since last signal."""
        time_since_last = datetime.now(timezone.utc) - self.last_signals[symbol]
        return time_since_last.total_seconds() > 180  # 3 minutes minimum (reduced from 5)
    
    def on_tick(self, tick: PriceTick) -> Optional[Signal]:
        """Generate signals optimized for low-volume stocks."""
        symbol = tick.symbol
        self.prices[symbol].append(tick.price)
        
        # Check if we have enough data
        if len(self.prices[symbol]) < self.long_window:
            return None
        
        # Check if we should generate a signal
        if not self.should_generate_signal(symbol):
            return None
        
        # Calculate indicators
        short_ma = np.mean(list(self.prices[symbol])[-self.short_window:])
        long_ma = np.mean(list(self.prices[symbol])[-self.long_window:])
        rsi = self.calculate_simple_rsi(symbol)
        volatility_score = self.calculate_volatility_score(symbol)
        
        # Calculate composite score
        ma_score = 1.0 if short_ma > long_ma else 0.0
        rsi_score = 0.5  # Neutral default
        
        if rsi is not None:
            if rsi < 25:  # Very oversold
                rsi_score = 1.0
            elif rsi < 40:  # Oversold
                rsi_score = 0.8
            elif rsi > 75:  # Very overbought
                rsi_score = 0.0
            elif rsi > 60:  # Overbought
                rsi_score = 0.2
        
        # Weighted composite score
        composite_score = (ma_score * 0.4 + rsi_score * 0.3 + volatility_score * 0.3)
        
        # Generate buy signal if score is high enough and we're not already long
        if (composite_score >= 0.6 and self.last_side[symbol] != Side.buy):
            self.last_side[symbol] = Side.buy
            self.last_signals[symbol] = datetime.now(timezone.utc)
            
            logger.info(f"Low-volume buy signal for {symbol}: score={composite_score:.3f}, price={tick.price}")
            
            return Signal(
                symbol=symbol,
                side=Side.buy,
                price=tick.price,
                confidence=composite_score,
                timestamp=tick.timestamp
            )
        
        # Generate sell signal if score is low and we're long
        elif (composite_score < 0.3 and self.last_side[symbol] == Side.buy):
            self.last_side[symbol] = Side.sell
            self.last_signals[symbol] = datetime.now(timezone.utc)
            
            logger.info(f"Low-volume sell signal for {symbol}: score={composite_score:.3f}, price={tick.price}")
            
            return Signal(
                symbol=symbol,
                side=Side.sell,
                price=tick.price,
                confidence=1.0 - composite_score,
                timestamp=tick.timestamp
            )
        
        return None


# ---------------------------------------------------------------------------
# Simple Moving Average crossover strategy for quick back-tests
# ---------------------------------------------------------------------------


class SimpleMovingAverageStrategy:
    """Very simple SMA crossover strategy used by back-tester sample."""

    def __init__(self, short_window: int = 10, long_window: int = 30):
        if short_window >= long_window:
            raise ValueError("short_window must be < long_window")
        self.short_window = short_window
        self.long_window = long_window
        self.prices: Dict[str, deque] = defaultdict(lambda: deque(maxlen=long_window + 5))
        self.last_side: Dict[str, Side] = defaultdict(lambda: None)
        self.last_signals: Dict[str, datetime] = defaultdict(lambda: datetime.min.replace(tzinfo=timezone.utc))

    def on_tick(self, tick: PriceTick) -> Optional[Signal]:
        sym = tick.symbol
        dq = self.prices[sym]
        dq.append(tick.price)
        if len(dq) < self.long_window:
            return None

        # compute SMAs
        prices = list(dq)
        short_ma = sum(prices[-self.short_window:]) / self.short_window
        long_ma = sum(prices[-self.long_window:]) / self.long_window

        # crossover logic
        if short_ma > long_ma and self.last_side[sym] != Side.buy:
            self.last_side[sym] = Side.buy
            self.last_signals[sym] = tick.timestamp
            return Signal(symbol=sym, side=Side.buy, price=tick.price, timestamp=tick.timestamp, confidence=1.0)

        if short_ma < long_ma and self.last_side[sym] == Side.buy:
            self.last_side[sym] = Side.sell
            self.last_signals[sym] = tick.timestamp
            return Signal(symbol=sym, side=Side.sell, price=tick.price, timestamp=tick.timestamp, confidence=1.0)

        return None


# Factory function to create strategies
def create_strategy(strategy_type: str, **kwargs):
    """Factory function to create strategy instances."""
    if strategy_type == "advanced":
        return AdvancedBuyStrategy(**kwargs)
    elif strategy_type == "mean_reversion":
        return MeanReversionStrategy(**kwargs)
    elif strategy_type == "low_volume":
        return LowVolumeStrategy(**kwargs)
    elif strategy_type == "simple_ma":
        return SimpleMovingAverageStrategy(**kwargs)
    elif strategy_type == "momentum_breakout":
        return MomentumBreakoutStrategy(**kwargs)
    elif strategy_type == "volatility_mean_reversion":
        return VolatilityMeanReversionStrategy(**kwargs)
    elif strategy_type == "gap_trading":
        return GapTradingStrategy(**kwargs)
    elif strategy_type == "multi_timeframe":
        return MultiTimeframeStrategy(**kwargs)
    elif strategy_type == "risk_managed":
        return RiskManagedStrategy(**kwargs)
    elif strategy_type == "aggressive_mean_reversion":
        return AggressiveMeanReversionStrategy(**kwargs)
    elif strategy_type == "enhanced_momentum":
        return EnhancedMomentumStrategy(**kwargs)
    elif strategy_type == "high_return_momentum":
        return EnhancedMomentumStrategy(**kwargs)
    elif strategy_type == "multi_timeframe_momentum":
        return MultiTimeframeMomentumStrategy(**kwargs)
    # Machine Learning Strategies
    elif strategy_type == "ensemble_ml":
        from tradebot.strategy.ml_strategies import create_ml_strategy, MLStrategyConfig
        config = MLStrategyConfig(**kwargs)
        return create_ml_strategy("ensemble", config)
    elif strategy_type == "lstm_ml":
        from tradebot.strategy.ml_strategies import create_ml_strategy, MLStrategyConfig
        config = MLStrategyConfig(**kwargs)
        return create_ml_strategy("lstm", config)
    elif strategy_type == "sentiment_ml":
        from tradebot.strategy.ml_strategies import create_ml_strategy, MLStrategyConfig
        config = MLStrategyConfig(**kwargs)
        return create_ml_strategy("sentiment", config)
    elif strategy_type == "rl_ml":
        from tradebot.strategy.rl_strategies import create_rl_strategy, RLStrategyConfig
        config = RLStrategyConfig(**kwargs)
        return create_rl_strategy(config)
    else:
        raise ValueError(f"Unknown strategy type: {strategy_type}")


class MomentumBreakoutStrategy:
    """Momentum breakout strategy for capturing explosive moves."""
    
    def __init__(self, lookback: int = 20, volume_multiplier: float = 1.5, support_window: int = 10):
        self.lookback = lookback
        self.volume_multiplier = volume_multiplier
        self.support_window = support_window
        
        # Data storage
        self.prices: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.volumes: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.last_side: Dict[str, Side] = defaultdict(lambda: None)
        self.last_signals: Dict[str, datetime] = defaultdict(lambda: datetime.min.replace(tzinfo=timezone.utc))
    
    def update_data(self, tick: PriceTick):
        """Update internal data structures."""
        symbol = tick.symbol
        self.prices[symbol].append(tick.price)
        if tick.volume is not None:
            self.volumes[symbol].append(tick.volume)
    
    def on_tick(self, tick: PriceTick) -> Optional[Signal]:
        """Generate buy/sell signals based on momentum breakouts."""
        symbol = tick.symbol
        self.update_data(tick)
        
        prices = list(self.prices[symbol])
        volumes = list(self.volumes[symbol])
        
        if len(prices) < self.lookback or len(volumes) < self.lookback:
            return None
        
        current_price = prices[-1]
        current_volume = volumes[-1]
        
        # Calculate rolling high and average volume
        rolling_high = max(prices[-self.lookback:-1])  # Exclude current price
        avg_volume = np.mean(volumes[-self.lookback:])
        
        # Calculate rolling low for support
        rolling_low = min(prices[-self.support_window:])
        
        # Buy signal: price breaks above recent high with volume confirmation
        breakout = (current_price > rolling_high) and (current_volume > avg_volume * self.volume_multiplier)
        
        # Sell signal: price drops below recent low or momentum fades
        momentum_fade = current_price < rolling_low
        
        # Generate signals
        if breakout and self.last_side[symbol] != Side.buy:
            self.last_side[symbol] = Side.buy
            self.last_signals[symbol] = tick.timestamp
            logger.info(f"Momentum breakout buy signal for {symbol}: price={current_price:.2f}, volume={current_volume}")
            return Signal(symbol=symbol, side=Side.buy, price=current_price, timestamp=tick.timestamp, confidence=1.0)
        
        elif momentum_fade and self.last_side[symbol] == Side.buy:
            self.last_side[symbol] = Side.sell
            self.last_signals[symbol] = tick.timestamp
            logger.info(f"Momentum fade sell signal for {symbol}: price={current_price:.2f}")
            return Signal(symbol=symbol, side=Side.sell, price=current_price, timestamp=tick.timestamp, confidence=1.0)
        
        return None


class VolatilityMeanReversionStrategy:
    """Volatility-based mean reversion strategy with dynamic bands."""
    
    def __init__(self, lookback: int = 20, std_multiplier: float = 2.5, rsi_period: int = 14):
        self.lookback = lookback
        self.std_multiplier = std_multiplier
        self.rsi_period = rsi_period
        
        # Data storage
        self.prices: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.last_side: Dict[str, Side] = defaultdict(lambda: None)
        self.last_signals: Dict[str, datetime] = defaultdict(lambda: datetime.min.replace(tzinfo=timezone.utc))
    
    def update_data(self, tick: PriceTick):
        """Update internal data structures."""
        symbol = tick.symbol
        self.prices[symbol].append(tick.price)
    
    def calculate_rsi(self, symbol: str) -> Optional[float]:
        """Calculate RSI."""
        prices = list(self.prices[symbol])
        if len(prices) < self.rsi_period + 1:
            return None
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-self.rsi_period:])
        avg_loss = np.mean(losses[-self.rsi_period:])
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def on_tick(self, tick: PriceTick) -> Optional[Signal]:
        """Generate buy/sell signals based on volatility-adjusted mean reversion."""
        symbol = tick.symbol
        self.update_data(tick)
        
        prices = list(self.prices[symbol])
        if len(prices) < self.lookback:
            return None
        
        current_price = prices[-1]
        
        # Calculate dynamic Bollinger Bands
        sma = np.mean(prices[-self.lookback:])
        std = np.std(prices[-self.lookback:])
        
        upper_band = sma + (std * self.std_multiplier)
        lower_band = sma - (std * self.std_multiplier)
        
        # Calculate RSI
        rsi = self.calculate_rsi(symbol)
        
        # Buy signal: price touches lower band with RSI oversold
        buy_signal = (current_price <= lower_band) and (rsi is not None and rsi < 30)
        
        # Sell signal: price touches upper band or RSI overbought
        sell_signal = (current_price >= upper_band) or (rsi is not None and rsi > 70)
        
        # Generate signals
        if buy_signal and self.last_side[symbol] != Side.buy:
            self.last_side[symbol] = Side.buy
            self.last_signals[symbol] = tick.timestamp
            logger.info(f"Volatility mean reversion buy signal for {symbol}: price={current_price:.2f}, RSI={rsi:.1f}")
            return Signal(symbol=symbol, side=Side.buy, price=current_price, timestamp=tick.timestamp, confidence=1.0)
        
        elif sell_signal and self.last_side[symbol] == Side.buy:
            self.last_side[symbol] = Side.sell
            self.last_signals[symbol] = tick.timestamp
            logger.info(f"Volatility mean reversion sell signal for {symbol}: price={current_price:.2f}, RSI={rsi:.1f}")
            return Signal(symbol=symbol, side=Side.sell, price=current_price, timestamp=tick.timestamp, confidence=1.0)
        
        return None


class GapTradingStrategy:
    """Gap trading strategy for volatile stocks."""
    
    def __init__(self, gap_threshold: float = 0.05, volume_threshold: float = 1.2, min_price: float = 0.5):
        self.gap_threshold = gap_threshold
        self.volume_threshold = volume_threshold
        self.min_price = min_price
        # Data storage
        self.prices: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
        self.volumes: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
        self.last_side: Dict[str, Side] = defaultdict(lambda: None)
        self.last_signals: Dict[str, datetime] = defaultdict(lambda: datetime.min.replace(tzinfo=timezone.utc))
    
    def update_data(self, tick: PriceTick):
        """Update internal data structures."""
        symbol = tick.symbol
        self.prices[symbol].append(tick.price)
        if tick.volume is not None:
            self.volumes[symbol].append(tick.volume)
    
    def on_tick(self, tick: PriceTick) -> Optional[Signal]:
        """Generate buy/sell signals based on gaps."""
        symbol = tick.symbol
        self.update_data(tick)
        prices = list(self.prices[symbol])
        volumes = list(self.volumes[symbol])
        if len(prices) < 2 or len(volumes) < 2:
            return None
        current_price = prices[-1]
        prev_price = prices[-2]
        current_volume = volumes[-1]
        prev_volume = volumes[-2]
        # Minimum price filter
        if current_price is None or current_price <= 0 or current_price < self.min_price:
            return None
        # Calculate gaps
        gap_up = current_price > prev_price * (1 + self.gap_threshold)
        gap_down = current_price < prev_price * (1 - self.gap_threshold)
        # Volume confirmation
        volume_spike = current_volume > prev_volume * self.volume_threshold
        # Buy signal: gap down with volume confirmation
        buy_signal = gap_down and volume_spike
        # Sell signal: gap up or gap fills
        sell_signal = gap_up or (current_price >= prev_price)
        # Generate signals
        if buy_signal and self.last_side[symbol] != Side.buy:
            self.last_side[symbol] = Side.buy
            self.last_signals[symbol] = tick.timestamp
            logger.info(f"Gap trading buy signal for {symbol}: price={current_price:.2f}, gap={((current_price/prev_price)-1)*100:.1f}%")
            return Signal(symbol=symbol, side=Side.buy, price=current_price, timestamp=tick.timestamp, confidence=1.0)
        elif sell_signal and self.last_side[symbol] == Side.buy:
            self.last_side[symbol] = Side.sell
            self.last_signals[symbol] = tick.timestamp
            logger.info(f"Gap trading sell signal for {symbol}: price={current_price:.2f}")
            return Signal(symbol=symbol, side=Side.sell, price=current_price, timestamp=tick.timestamp, confidence=1.0)
        return None


class MultiTimeframeStrategy:
    """Multi-timeframe strategy combining daily and weekly signals."""
    
    def __init__(self, daily_lookback: int = 20, weekly_lookback: int = 5):
        self.daily_lookback = daily_lookback
        self.weekly_lookback = weekly_lookback
        
        # Data storage
        self.prices: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.last_side: Dict[str, Side] = defaultdict(lambda: None)
        self.last_signals: Dict[str, datetime] = defaultdict(lambda: datetime.min.replace(tzinfo=timezone.utc))
    
    def update_data(self, tick: PriceTick):
        """Update internal data structures."""
        symbol = tick.symbol
        self.prices[symbol].append(tick.price)
    
    def on_tick(self, tick: PriceTick) -> Optional[Signal]:
        """Generate buy/sell signals using multi-timeframe analysis."""
        symbol = tick.symbol
        self.update_data(tick)
        
        prices = list(self.prices[symbol])
        if len(prices) < max(self.daily_lookback, self.weekly_lookback):
            return None
        
        current_price = prices[-1]
        
        # Weekly trend (using 5-day rolling average)
        weekly_sma = np.mean(prices[-self.weekly_lookback:])
        weekly_trend = current_price > weekly_sma
        
        # Daily mean reversion signals
        daily_sma = np.mean(prices[-self.daily_lookback:])
        daily_std = np.std(prices[-self.daily_lookback:])
        
        # Buy when price is below daily mean but weekly trend is up
        buy_signal = (current_price < daily_sma - daily_std) and weekly_trend
        
        # Sell when price is above daily mean or weekly trend turns down
        sell_signal = (current_price > daily_sma + daily_std) or not weekly_trend
        
        # Generate signals
        if buy_signal and self.last_side[symbol] != Side.buy:
            self.last_side[symbol] = Side.buy
            self.last_signals[symbol] = tick.timestamp
            logger.info(f"Multi-timeframe buy signal for {symbol}: price={current_price:.2f}, weekly_trend={weekly_trend}")
            return Signal(symbol=symbol, side=Side.buy, price=current_price, timestamp=tick.timestamp, confidence=1.0)
        
        elif sell_signal and self.last_side[symbol] == Side.buy:
            self.last_side[symbol] = Side.sell
            self.last_signals[symbol] = tick.timestamp
            logger.info(f"Multi-timeframe sell signal for {symbol}: price={current_price:.2f}")
            return Signal(symbol=symbol, side=Side.sell, price=current_price, timestamp=tick.timestamp, confidence=1.0)
        
        return None


class RiskManagedStrategy:
    """Risk-managed strategy with stop-losses and position sizing."""
    
    def __init__(self, base_strategy_type: str = "mean_reversion", stop_loss: float = 0.02, max_drawdown: float = 0.10):
        self.base_strategy_type = base_strategy_type
        self.stop_loss = stop_loss
        self.max_drawdown = max_drawdown
        
        # Create base strategy
        if base_strategy_type == "mean_reversion":
            self.base_strategy = MeanReversionStrategy()
        elif base_strategy_type == "volatility_mean_reversion":
            self.base_strategy = VolatilityMeanReversionStrategy()
        else:
            self.base_strategy = MeanReversionStrategy()  # Default
        
        # Risk management state
        self.entry_prices: Dict[str, float] = {}
        self.portfolio_value: float = 10000  # Starting portfolio value
        self.current_drawdown: float = 0.0
        self.last_side: Dict[str, Side] = defaultdict(lambda: None)
        self.last_signals: Dict[str, datetime] = defaultdict(lambda: datetime.min.replace(tzinfo=timezone.utc))
    
    def update_data(self, tick: PriceTick):
        """Update base strategy data."""
        self.base_strategy.update_data(tick)
    
    def calculate_position_size(self, symbol: str, price: float) -> float:
        """Calculate position size based on volatility and risk."""
        # Simple volatility-based sizing
        prices = list(self.base_strategy.prices[symbol])
        if len(prices) < 20:
            return 0.02  # 2% default
        
        volatility = np.std(prices[-20:]) / np.mean(prices[-20:])
        # Higher volatility = smaller position size
        position_size = max(0.01, min(0.05, 0.02 / volatility))
        return position_size
    
    def on_tick(self, tick: PriceTick) -> Optional[Signal]:
        """Generate signals with risk management."""
        symbol = tick.symbol
        self.update_data(tick)
        
        # Check stop-loss for existing positions
        if self.last_side[symbol] == Side.buy and symbol in self.entry_prices:
            entry_price = self.entry_prices[symbol]
            current_price = tick.price
            
            # Stop-loss hit
            if current_price <= entry_price * (1 - self.stop_loss):
                self.last_side[symbol] = Side.sell
                self.last_signals[symbol] = tick.timestamp
                logger.info(f"Stop-loss triggered for {symbol}: entry={entry_price:.2f}, exit={current_price:.2f}")
                return Signal(symbol=symbol, side=Side.sell, price=current_price, timestamp=tick.timestamp, confidence=1.0)
        
        # Get base strategy signal
        base_signal = self.base_strategy.on_tick(tick)
        
        if base_signal is None:
            return None
        
        # Apply risk management
        if base_signal.side == Side.buy and self.last_side[symbol] != Side.buy:
            # Check drawdown limit
            if self.current_drawdown >= self.max_drawdown:
                logger.info(f"Max drawdown limit reached for {symbol}, skipping buy signal")
                return None
            
            # Calculate position size
            position_size = self.calculate_position_size(symbol, base_signal.price)
            
            # Record entry price for stop-loss
            self.entry_prices[symbol] = base_signal.price
            
            self.last_side[symbol] = Side.buy
            self.last_signals[symbol] = tick.timestamp
            logger.info(f"Risk-managed buy signal for {symbol}: price={base_signal.price:.2f}, position_size={position_size:.1%}")
            return base_signal
        
        elif base_signal.side == Side.sell and self.last_side[symbol] == Side.buy:
            self.last_side[symbol] = Side.sell
            self.last_signals[symbol] = tick.timestamp
            logger.info(f"Risk-managed sell signal for {symbol}: price={base_signal.price:.2f}")
            return base_signal
        
        return None


class AggressiveMeanReversionStrategy:
    """Aggressive mean reversion with tight risk management for high returns."""
    
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


class EnhancedMomentumStrategy:
    """Enhanced momentum breakout with volume confirmation for high returns."""
    
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
                logger.info(f"Enhanced momentum stop-loss for {symbol}: entry={entry_price:.2f}, exit={current_price:.2f}")
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
            logger.info(f"Enhanced momentum fade exit for {symbol}: price={current_price:.2f}")
            return Signal(symbol=symbol, side=Side.sell, price=current_price, timestamp=tick.timestamp, confidence=1.0)
        
        return None


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
    """Factory function to create high-return strategy instances."""
    if strategy_type == "aggressive_mean_reversion":
        return AggressiveMeanReversionStrategy(**kwargs)
    elif strategy_type == "enhanced_momentum":
        return EnhancedMomentumStrategy(**kwargs)
    elif strategy_type == "multi_timeframe_momentum":
        return MultiTimeframeMomentumStrategy(**kwargs)
    elif strategy_type == "high_return_momentum":
        return EnhancedMomentumStrategy(**kwargs)  # Alias for enhanced_momentum
    else:
        # Fall back to regular strategy creation for compatibility
        return create_strategy(strategy_type, **kwargs) 