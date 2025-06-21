import asyncio
import logging
import os
import random
from datetime import datetime, timezone
from typing import List

from tradebot.common.bus import MessageBus
from tradebot.common.models import PriceTick
from tradebot.common.symbol_manager import SymbolManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("market_data")

# Configuration
SYMBOL_MODE = os.getenv("SYMBOL_MODE", "custom")
LEGACY_SYMBOLS: List[str] = os.getenv("SYMBOLS", "AAPL,MSFT,AMZN,GOOG,TSLA").split(",")
MAX_SYMBOLS = int(os.getenv("MAX_SYMBOLS", "500"))

# Will be populated by symbol manager
SYMBOLS: List[str] = []

# Realistic starting prices based on approximate current market values (as of June 2024)
REALISTIC_PRICES = {
    "AAPL": 220.0,   # Apple around $220
    "MSFT": 430.0,   # Microsoft around $430
    "AMZN": 185.0,   # Amazon around $185
    "GOOG": 175.0,   # Google around $175
    "TSLA": 185.0,   # Tesla around $185
}

# Will be initialized after symbols are loaded
INITIAL_PRICES = {}


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


async def initialize_symbols():
    """Initialize symbols based on configuration."""
    global SYMBOLS, INITIAL_PRICES
    
    if SYMBOL_MODE == "custom":
        SYMBOLS = LEGACY_SYMBOLS
        logger.info("Using custom symbols: %s", SYMBOLS[:10])
    else:
        # Use symbol manager to get symbols
        manager = SymbolManager()
        all_symbols = await manager.initialize(SYMBOL_MODE)
        
        # Limit symbols to prevent overwhelming the system
        SYMBOLS = all_symbols[:MAX_SYMBOLS]
        logger.info("Loaded %d symbols in '%s' mode (limited to %d)", 
                   len(all_symbols), SYMBOL_MODE, len(SYMBOLS))
        logger.info("First 10 symbols: %s", SYMBOLS[:10])
    
    # Initialize prices for all symbols
    INITIAL_PRICES = {sym: REALISTIC_PRICES.get(sym, random.uniform(50, 500)) for sym in SYMBOLS}
    logger.info("Initialized prices for %d symbols", len(INITIAL_PRICES))


async def main():
    logger.info("Market data service starting...")
    
    # Initialize symbols first
    await initialize_symbols()
    
    logger.info("Market data service ready for %d symbols", len(SYMBOLS))
    bus = MessageBus()
    await bus.connect()
    await generate_ticks(bus)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Market data service stopped") 