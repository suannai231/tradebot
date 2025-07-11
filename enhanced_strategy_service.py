#!/usr/bin/env python3
"""
Enhanced Strategy Service that includes ML strategies for live signal generation
"""

import asyncio
import logging
import sys
import os
from collections import deque, defaultdict
from statistics import mean
from datetime import datetime, timezone

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from tradebot.common.bus import MessageBus
from tradebot.common.models import PriceTick, Signal, Side
from tradebot.strategy.advanced_strategies import create_strategy

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("enhanced_strategy")

SHORT_WINDOW = 5  # in ticks
LONG_WINDOW = 20


class MovingAverageStrategy:
    """Basic SMA crossover strategy per symbol."""

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


class EnhancedStrategyManager:
    """Manages multiple strategies including ML strategies"""
    
    def __init__(self):
        self.strategies = {}
        self.signal_counts = defaultdict(int)
        
        # Initialize strategies
        self.init_strategies()
        
    def init_strategies(self):
        """Initialize all strategies"""
        logger.info("🚀 Initializing Enhanced Strategy Manager with ML strategies...")
        
        try:
            # Basic moving average strategy
            self.strategies["moving_average"] = MovingAverageStrategy()
            logger.info("✅ Basic Moving Average strategy initialized")
            
            # ML Strategies with conservative settings for live trading
            try:
                # Ensemble ML Strategy
                self.strategies["ensemble_ml"] = create_strategy("ensemble_ml",
                    lookback_period=20,
                    min_data_points=30,
                    confidence_threshold=0.4,  # Slightly higher for live trading
                    retrain_frequency=1000
                )
                logger.info("✅ Ensemble ML strategy initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Ensemble ML: {e}")
            
            try:
                # LSTM ML Strategy  
                self.strategies["lstm_ml"] = create_strategy("lstm_ml",
                    lookback_period=20,
                    min_data_points=30,
                    prediction_horizon=3,
                    confidence_threshold=0.4
                )
                logger.info("✅ LSTM ML strategy initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize LSTM ML: {e}")
            
            try:
                # Sentiment ML Strategy
                self.strategies["sentiment_ml"] = create_strategy("sentiment_ml",
                    lookback_period=20,
                    min_data_points=30,
                    use_sentiment=True,
                    confidence_threshold=0.4
                )
                logger.info("✅ Sentiment ML strategy initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Sentiment ML: {e}")
            
            try:
                # RL ML Strategy - use conservative settings
                self.strategies["rl_ml"] = create_strategy("rl_ml",
                    lookback_period=15,  # Reduced from 20
                    min_data_points=20,  # Reduced from 30
                    initial_balance=10000.0,
                    max_position_size=0.1
                )
                logger.info("✅ RL ML strategy initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize RL ML: {e}")
                logger.warning("⚠️  Skipping RL ML strategy due to initialization error")
                
        except Exception as e:
            logger.error(f"❌ Error initializing strategies: {e}")
        
        logger.info(f"📊 Total strategies initialized: {len(self.strategies)}")
        for name in self.strategies.keys():
            logger.info(f"   • {name}")
    
    async def process_tick(self, tick: PriceTick, bus: MessageBus):
        """Process tick through all strategies"""
        signals_generated = []
        
        for strategy_name, strategy in self.strategies.items():
            try:
                signal = strategy.on_tick(tick)
                if signal:
                    # Publish signal to message bus
                    await bus.publish("orders.new", signal.model_dump())
                    
                    # Log the signal
                    self.signal_counts[strategy_name] += 1
                    logger.info(f"[{strategy_name}] Generated signal #{self.signal_counts[strategy_name]}: {signal.symbol} {signal.side.value} @ {signal.price:.4f} (conf: {signal.confidence:.3f})")
                    
                    signals_generated.append((strategy_name, signal))
                    
            except Exception as e:
                logger.error(f"❌ Error in {strategy_name} strategy: {e}")
        
        return signals_generated
    
    def get_stats(self):
        """Get strategy statistics"""
        return dict(self.signal_counts)


async def main():
    """Main enhanced strategy service"""
    logger.info("🚀 Starting Enhanced Strategy Service with ML strategies...")
    
    try:
        # Initialize message bus
        bus = MessageBus()
        await bus.connect()
        logger.info("✅ Connected to message bus")
        
        # Initialize strategy manager
        strategy_manager = EnhancedStrategyManager()
        
        # Statistics tracking
        tick_count = 0
        last_stats_report = datetime.now(timezone.utc)
        
        logger.info("🎯 Listening for price ticks...")
        
        async for msg in bus.subscribe("price.ticks", last_id="$", block_ms=1000):
            try:
                tick = PriceTick(**msg)
                tick_count += 1
                
                # Process tick through all strategies
                signals = await strategy_manager.process_tick(tick, bus)
                
                # Report statistics every 100 ticks
                if tick_count % 100 == 0:
                    now = datetime.now(timezone.utc)
                    elapsed = (now - last_stats_report).total_seconds()
                    ticks_per_second = 100 / elapsed if elapsed > 0 else 0
                    
                    stats = strategy_manager.get_stats()
                    total_signals = sum(stats.values())
                    
                    logger.info(f"📊 Processed {tick_count} ticks ({ticks_per_second:.1f}/sec), Generated {total_signals} total signals")
                    for strategy_name, count in stats.items():
                        if count > 0:
                            logger.info(f"   • {strategy_name}: {count} signals")
                    
                    # Report position manager statistics
                    try:
                        from tradebot.strategy.position_manager import get_position_manager
                        position_manager = get_position_manager()
                        pos_stats = position_manager.get_stats()
                        logger.info(f"💼 Position Manager: {pos_stats['open_positions']} open, {pos_stats['total_opened']} opened, {pos_stats['total_closed']} closed")
                    except Exception as e:
                        logger.debug(f"Error getting position stats: {e}")
                    
                    last_stats_report = now
                    
            except Exception as e:
                logger.warning(f"⚠️  Malformed tick {msg}: {e}")
                continue
                
    except KeyboardInterrupt:
        logger.info("🛑 Enhanced Strategy Service stopped by user")
    except Exception as e:
        logger.error(f"❌ Enhanced Strategy Service error: {e}")
    finally:
        logger.info("🔌 Disconnecting from message bus...")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Enhanced Strategy Service stopped")
    except Exception as e:
        logger.error(f"Failed to start Enhanced Strategy Service: {e}")
        sys.exit(1) 