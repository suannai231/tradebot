import asyncio
import logging
import os
from collections import deque, defaultdict
from statistics import mean
from datetime import datetime, timezone

from tradebot.common.bus import MessageBus
from tradebot.common.models import PriceTick, Signal, Side

# Import strategy factory functions
from tradebot.strategy.advanced_strategies import create_strategy
from tradebot.strategy.ml_strategies import create_ml_strategy, MLStrategyConfig
from tradebot.strategy.rl_strategies import create_rl_strategy, RLStrategyConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("strategy")

SHORT_WINDOW = 5  # in ticks
LONG_WINDOW = 20


class MovingAverageStrategy:
    """Very naive SMA crossover strategy per symbol."""

    def __init__(self):
        self.prices: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=LONG_WINDOW))
        self.last_side: dict[str, Side | None] = defaultdict(lambda: None)

    def on_tick(self, tick: PriceTick) -> Signal | None:
        dq = self.prices[tick.symbol]
        dq.append(tick.price)

        if len(dq) < LONG_WINDOW:
            return None

        short_ma = mean(list(dq)[-SHORT_WINDOW:])
        long_ma = mean(dq)

        if short_ma > long_ma and self.last_side[tick.symbol] != Side.buy:
            self.last_side[tick.symbol] = Side.buy
            return Signal(symbol=tick.symbol, side=Side.buy, price=tick.price, confidence=abs(short_ma - long_ma) / long_ma, timestamp=tick.timestamp)
        elif short_ma < long_ma and self.last_side[tick.symbol] != Side.sell:
            self.last_side[tick.symbol] = Side.sell
            return Signal(symbol=tick.symbol, side=Side.sell, price=tick.price, confidence=abs(short_ma - long_ma) / long_ma, timestamp=tick.timestamp)
        return None


class MultiStrategyService:
    """Service that runs multiple strategies simultaneously"""
    
    def __init__(self):
        self.strategies = {}
        self.strategy_names = {}
        self.setup_strategies()
        
    def setup_strategies(self):
        """Initialize all strategies"""
        logger.info("🚀 Setting up multi-strategy service...")
        
        # Traditional strategies
        self.strategies["simple_ma"] = MovingAverageStrategy()
        self.strategy_names["simple_ma"] = "simple_ma"
        
        try:
            # Advanced strategies - using correct parameters
            self.strategies["advanced"] = create_strategy("advanced", min_composite_score=0.5)
            self.strategy_names["advanced"] = "advanced"
            
            self.strategies["mean_reversion"] = create_strategy("mean_reversion", lookback_period=15, z_score_threshold=-1.5)
            self.strategy_names["mean_reversion"] = "mean_reversion"
            
            self.strategies["low_volume"] = create_strategy("low_volume", short_window=3, long_window=8)
            self.strategy_names["low_volume"] = "low_volume"
            
            # Create momentum_breakout without lookback_period (it doesn't accept this parameter)
            self.strategies["momentum_breakout"] = create_strategy("momentum_breakout")
            self.strategy_names["momentum_breakout"] = "momentum_breakout"
            
            self.strategies["volatility_mean_reversion"] = create_strategy("volatility_mean_reversion")
            self.strategy_names["volatility_mean_reversion"] = "volatility_mean_reversion"
            
            self.strategies["gap_trading"] = create_strategy("gap_trading")
            self.strategy_names["gap_trading"] = "gap_trading"
            
            self.strategies["multi_timeframe"] = create_strategy("multi_timeframe", daily_lookback=20, weekly_lookback=5)
            self.strategy_names["multi_timeframe"] = "multi_timeframe"
            
            self.strategies["risk_managed"] = create_strategy("risk_managed", base_strategy_type="mean_reversion", stop_loss=0.02)
            self.strategy_names["risk_managed"] = "risk_managed"
            
            self.strategies["aggressive_mean_reversion"] = create_strategy("aggressive_mean_reversion", lookback_period=15, z_score_threshold=-1.5)
            self.strategy_names["aggressive_mean_reversion"] = "aggressive_mean_reversion"
            
            self.strategies["enhanced_momentum"] = create_strategy("enhanced_momentum")
            self.strategy_names["enhanced_momentum"] = "enhanced_momentum"
            
            self.strategies["multi_timeframe_momentum"] = create_strategy("multi_timeframe_momentum")
            self.strategy_names["multi_timeframe_momentum"] = "multi_timeframe_momentum"
            
            logger.info("✅ Advanced strategies initialized")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to initialize some advanced strategies: {e}")
            
        try:
            # ML strategies
            ml_config = MLStrategyConfig(
                lookback_period=20,
                confidence_threshold=0.4,
                min_data_points=30
            )
            
            self.strategies["ensemble_ml"] = create_ml_strategy("ensemble", ml_config)
            self.strategy_names["ensemble_ml"] = "ensemble_ml"
            
            self.strategies["lstm_ml"] = create_ml_strategy("lstm", ml_config)
            self.strategy_names["lstm_ml"] = "lstm_ml"
            
            self.strategies["sentiment_ml"] = create_ml_strategy("sentiment", ml_config)
            self.strategy_names["sentiment_ml"] = "sentiment_ml"
            
            logger.info("✅ ML strategies initialized")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to initialize ML strategies: {e}")
            
        try:
            # RL strategies
            rl_config = RLStrategyConfig(
                lookback_period=20,
                min_data_points=30
            )
            
            self.strategies["rl_ml"] = create_rl_strategy(rl_config)
            self.strategy_names["rl_ml"] = "rl_ml"
            
            logger.info("✅ RL strategies initialized")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to initialize RL strategies: {e}")
            
        logger.info(f"🎯 Multi-strategy service ready with {len(self.strategies)} strategies: {list(self.strategies.keys())}")
    
    def process_tick(self, tick: PriceTick) -> list[tuple[str, Signal]]:
        """Process a tick through all strategies and return signals with strategy names"""
        signals = []
        
        for strategy_name, strategy in self.strategies.items():
            try:
                signal = strategy.on_tick(tick)
                if signal:
                    signals.append((strategy_name, signal))
            except Exception as e:
                logger.error(f"Error in strategy {strategy_name}: {e}")
                
        return signals


async def main():
    bus = MessageBus()
    await bus.connect()

    # Use multi-strategy service instead of single strategy
    multi_strategy = MultiStrategyService()

    async for msg in bus.subscribe("price.ticks", last_id="$", block_ms=1000):
        try:
            tick = PriceTick(**msg)
        except Exception as e:
            logger.warning("Malformed tick %s: %s", msg, e)
            continue

        # Process tick through all strategies
        strategy_signals = multi_strategy.process_tick(tick)
        
        for strategy_name, signal in strategy_signals:
            # Create signal data with strategy name
            signal_data = signal.model_dump()
            signal_data["strategy_name"] = strategy_name
            
            await bus.publish("orders.new", signal_data)
            logger.info(f"Generated {strategy_name} signal %s", signal.model_dump_json())


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Multi-strategy service stopped") 