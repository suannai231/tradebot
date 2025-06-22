import asyncio
import json
import logging
import os
from datetime import datetime, timezone

import redis.asyncio as redis
from tradebot.common.bus import MessageBus
from tradebot.common.models import Signal

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("execution")

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")


async def execute_order(signal: Signal, bus: MessageBus, redis_client: redis.Redis):
    """Execute order and store signal for dashboard."""
    # TODO: Implement Fidelity brokerage integration here.
    logger.info("EXECUTE %s %s @ %s (confidence=%.2f)" % (
        signal.side.value,
        signal.symbol,
        signal.timestamp.strftime("%H:%M:%S"),
        signal.confidence,
    ))
    
    # Simulate network latency
    await asyncio.sleep(0.1)
    
    # Store signal for dashboard
    await store_signal_for_dashboard(signal, bus, redis_client)


async def store_signal_for_dashboard(signal: Signal, bus: MessageBus, redis_client: redis.Redis):
    """Store trading signal in Redis and publish to dashboard stream."""
    try:
        logger.info(f"Storing signal for dashboard: {signal.symbol} {signal.side.value}")
        
        # Create signal data for dashboard
        signal_data = {
            "symbol": signal.symbol,
            "signal_type": signal.side.value.upper(),
            "price": float(signal.price),
            "timestamp": signal.timestamp.isoformat(),
            "strategy": "moving_average",
            "confidence": float(signal.confidence)
        }
        
        logger.info(f"Signal data created: {signal_data}")
        
        # Store in Redis list for dashboard API
        await redis_client.lpush("trading:signals:recent", json.dumps(signal_data))
        await redis_client.ltrim("trading:signals:recent", 0, 49)  # Keep last 50 signals
        
        # Update signal count
        current_count = await redis_client.get("trading:signals:count") or 0
        await redis_client.set("trading:signals:count", int(current_count) + 1)
        
        # Publish to trading.signals stream for real-time dashboard updates
        await bus.publish("trading.signals", signal_data)
        
        logger.info(f"Successfully stored signal for dashboard: {signal.symbol} {signal.side.value}")
        
    except Exception as e:
        logger.error(f"Failed to store signal for dashboard: {e}", exc_info=True)


async def main():
    bus = MessageBus()
    await bus.connect()
    
    # Connect to Redis for signal storage
    redis_client = redis.from_url(REDIS_URL)
    
    async for msg in bus.subscribe("orders.new", last_id="$", block_ms=1000):
        try:
            signal = Signal(**msg)
        except Exception as e:
            logger.warning("Malformed signal %s: %s", msg, e)
            continue

        await execute_order(signal, bus, redis_client)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Execution service stopped") 