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

# Realistic starting prices based on approximate current market values (as of June 2024)
REALISTIC_PRICES = {
    "AAPL": 220.0,   # Apple around $220
    "MSFT": 430.0,   # Microsoft around $430
    "AMZN": 185.0,   # Amazon around $185
    "GOOG": 175.0,   # Google around $175
    "TSLA": 185.0,   # Tesla around $185
}

INITIAL_PRICES = {sym: REALISTIC_PRICES.get(sym, random.uniform(100, 300)) for sym in SYMBOLS}


async def generate_ticks(bus: MessageBus, interval: float = 1.0):
    """Generate random walk prices with mock OHLCV data and publish to Redis stream."""
    while True:
        for symbol in SYMBOLS:
            # Simple random walk for close price
            close_price = max(0.01, INITIAL_PRICES[symbol] * (1 + random.uniform(-0.001, 0.001)))
            INITIAL_PRICES[symbol] = close_price
            
            # Generate realistic OHLCV data
            # Open is previous close (with small gap)
            open_price = close_price * (1 + random.uniform(-0.0005, 0.0005))
            
            # High and low around the open-close range
            high_price = max(open_price, close_price) * (1 + random.uniform(0, 0.002))
            low_price = min(open_price, close_price) * (1 - random.uniform(0, 0.002))
            
            # Mock volume (realistic range)
            volume = random.randint(1000, 50000)
            trade_count = random.randint(10, 500)
            
            # VWAP around the average of OHLC
            vwap = (open_price + high_price + low_price + close_price) / 4 * (1 + random.uniform(-0.001, 0.001))

            tick = PriceTick(
                symbol=symbol, 
                price=round(close_price, 2),  # Backward compatibility
                timestamp=datetime.now(tz=timezone.utc),
                open=round(open_price, 2),
                high=round(high_price, 2),
                low=round(low_price, 2),
                close=round(close_price, 2),
                volume=volume,
                trade_count=trade_count,
                vwap=round(vwap, 2)
            )
            await bus.publish("price.ticks", tick)
            logger.debug("Published OHLCV tick %s", tick.model_dump_json())
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