import asyncio
import logging
import os
import random
from datetime import datetime, timezone
from typing import List

from tradebot.common.bus import MessageBus
from tradebot.common.models import PriceTick

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("market_data")

# Symbols to mock – can be overridden with environment variable
SYMBOLS: List[str] = os.getenv("SYMBOLS", "AAPL,MSFT,AMZN,GOOG,TSLA").split(",")

INITIAL_PRICES = {sym: random.uniform(100, 300) for sym in SYMBOLS}


async def generate_ticks(bus: MessageBus, interval: float = 1.0):
    """Generate random walk prices and publish to Redis stream."""
    while True:
        for symbol in SYMBOLS:
            # simple random walk
            price = max(0.01, INITIAL_PRICES[symbol] * (1 + random.uniform(-0.001, 0.001)))
            INITIAL_PRICES[symbol] = price

            tick = PriceTick(symbol=symbol, price=round(price, 2), timestamp=datetime.now(tz=timezone.utc))
            await bus.publish("price.ticks", tick.dict())
            logger.debug("Published tick %s", tick.json())
        await asyncio.sleep(interval)


async def main():
    logger.info("Market data service starting for symbols: %s", SYMBOLS)
    bus = MessageBus()
    await bus.connect()
    await generate_ticks(bus)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Market data service stopped") 