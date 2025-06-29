import asyncio
import logging
import os
from datetime import datetime, timezone, timedelta
from typing import List

import aiohttp
import asyncpg
from dotenv import load_dotenv

from tradebot.common.models import PriceTick
from tradebot.common.symbol_manager import SymbolManager

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("alpaca_backfill")

ALPACA_KEY = os.getenv("ALPACA_KEY")
ALPACA_SECRET = os.getenv("ALPACA_SECRET")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/tradebot")
# Configuration
SYMBOL_MODE = os.getenv("SYMBOL_MODE", "custom")
LEGACY_SYMBOLS = os.getenv("SYMBOLS", "AAPL,MSFT,AMZN,GOOG,TSLA").split(",")
MAX_BACKFILL_SYMBOLS = int(os.getenv("MAX_SYMBOLS", "500"))

# Will be populated based on mode
SYMBOLS = []

BASE_URL = "https://data.alpaca.markets/v2/stocks"


async def fetch_historical_bars(session: aiohttp.ClientSession, symbol: str, start: str, end: str, timeframe: str = "1Day"):
    """Fetch historical bars from Alpaca REST API."""
    url = f"{BASE_URL}/{symbol}/bars"
    headers = {
        "APCA-API-KEY-ID": ALPACA_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET,
    }
    params = {
        "start": start,
        "end": end,
        "timeframe": timeframe,
        "limit": 10000,
        # Use split-adjusted prices to handle stock splits
        "adjustment": "split",
        "feed": "iex"  # Use IEX feed for free tier
    }
    
    async with session.get(url, headers=headers, params=params) as response:
        # First attempt (possibly with IEX feed)
        if response.status != 200:
            error_text = await response.text()
            logger.error("Failed to fetch %s: %s %s", symbol, response.status, error_text)
            
            # If IEX fails, try without feed parameter (will use SIP/auto)
            if response.status == 403 and "feed" in str(error_text):
                logger.info("Retrying %s without feed parameter", symbol)
                params.pop("feed", None)
                async with session.get(url, headers=headers, params=params) as retry_response:
                    if retry_response.status != 200:
                        logger.error("Retry failed for %s: %s %s", symbol, retry_response.status, await retry_response.text())
                        return []
                    data = await retry_response.json()
                    bars = data.get("bars", [])
                    logger.info("Fetched %d bars for %s (retry)", len(bars), symbol)
                    return bars
            return []
        
        # status 200
        data = await response.json()
        bars = data.get("bars", [])

        # If zero bars returned with IEX feed, try default feed once more
        if not bars and params.get("feed") == "iex":
            logger.info("No bars from IEX for %s, retrying without feed param", symbol)
            params.pop("feed", None)
            async with session.get(url, headers=headers, params=params) as retry_response:
                if retry_response.status != 200:
                    logger.error("Retry failed for %s: %s %s", symbol, retry_response.status, await retry_response.text())
                    return []
                data = await retry_response.json()
                bars = data.get("bars", [])
                logger.info("Fetched %d bars for %s (retry no-feed)", len(bars), symbol)
        else:
            logger.info("Fetched %d bars for %s", len(bars), symbol)
        return bars


async def store_bars(pool: asyncpg.Pool, symbol: str, bars: List[dict]):
    """Store historical bars as price ticks with full OHLCV data."""
    if not bars:
        return
    
    # Convert bars to ticks with full OHLCV data
    ticks = []
    for bar in bars:
        timestamp = datetime.fromisoformat(bar["t"].replace("Z", "+00:00"))
        # Create a tick with full bar data
        tick = PriceTick(
            symbol=symbol,
            price=bar["c"],  # close price (for backward compatibility)
            timestamp=timestamp,
            open=bar.get("o"),
            high=bar.get("h"), 
            low=bar.get("l"),
            close=bar.get("c"),
            volume=bar.get("v"),
            trade_count=bar.get("n"),
            vwap=bar.get("vw")
        )
        ticks.append(tick)
    
    # Batch insert with all OHLCV fields
    async with pool.acquire() as conn:
        await conn.executemany(
            """INSERT INTO price_ticks (symbol, price, timestamp, open_price, high_price, low_price, close_price, volume, trade_count, vwap) 
               VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10) ON CONFLICT DO NOTHING""",
            [(tick.symbol, tick.price, tick.timestamp, tick.open, tick.high, tick.low, tick.close, tick.volume, tick.trade_count, tick.vwap) for tick in ticks]
        )
    
    logger.info("Stored %d OHLCV bars for %s", len(ticks), symbol)


async def backfill_symbol(session: aiohttp.ClientSession, pool: asyncpg.Pool, symbol: str, days_back: int = 365*5):
    """Backfill only the bars that are not yet in price_ticks."""

    async with pool.acquire() as conn:
        last_ts = await conn.fetchval("SELECT max(timestamp) FROM price_ticks WHERE symbol=$1", symbol)

    if last_ts:
        # start the day after the last stored bar
        start_date = last_ts + timedelta(days=1)
        logger.info("Incremental backfill %s starting %s (existing data to %s)", symbol, start_date.date(), last_ts.date())
    else:
        # Full history (fallback)
        start_date = datetime.now(timezone.utc) - timedelta(days=days_back)
        logger.info("Full backfill %s last %d days", symbol, days_back)

    end_date = datetime.now(timezone.utc)

    # If nothing is missing
    if start_date.date() > end_date.date():
        logger.info("%s is already up-to-date – skipping", symbol)
        return

    start_str = start_date.strftime("%Y-%m-%dT%H:%M:%SZ")
    end_str   = end_date.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Fetch daily bars only for the missing range
    bars = await fetch_historical_bars(session, symbol, start_str, end_str, "1Day")
    await store_bars(pool, symbol, bars)


async def init_database(pool: asyncpg.Pool):
    """Ensure database schema exists with full OHLCV support."""
    async with pool.acquire() as conn:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS timescaledb;")
        
        # Check if we need to add new columns to existing table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS price_ticks (
                symbol TEXT NOT NULL,
                price DECIMAL(10,4) NOT NULL,
                timestamp TIMESTAMPTZ NOT NULL,
                volume INTEGER DEFAULT 0,
                UNIQUE(symbol, timestamp)
            );
        """)
        
        # Add new OHLCV columns if they don't exist
        await conn.execute("""
            DO $$ BEGIN
                ALTER TABLE price_ticks ADD COLUMN IF NOT EXISTS open_price DECIMAL(10,4);
                ALTER TABLE price_ticks ADD COLUMN IF NOT EXISTS high_price DECIMAL(10,4);
                ALTER TABLE price_ticks ADD COLUMN IF NOT EXISTS low_price DECIMAL(10,4);
                ALTER TABLE price_ticks ADD COLUMN IF NOT EXISTS close_price DECIMAL(10,4);
                ALTER TABLE price_ticks ADD COLUMN IF NOT EXISTS trade_count INTEGER;
                ALTER TABLE price_ticks ADD COLUMN IF NOT EXISTS vwap DECIMAL(10,4);
            END $$;
        """)
        
        try:
            await conn.execute("""
                SELECT create_hypertable('price_ticks', 'timestamp', if_not_exists => TRUE);
            """)
        except Exception:
            pass  # Already exists
        
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_price_ticks_symbol_time 
            ON price_ticks (symbol, timestamp DESC);
        """)
        
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_price_ticks_symbol_volume 
            ON price_ticks (symbol, volume DESC) WHERE volume IS NOT NULL;
        """)


async def initialize_symbols():
    """Initialize symbols based on configuration."""
    global SYMBOLS
    
    # Use symbol manager for all modes including custom
    manager = SymbolManager()
    all_symbols = await manager.initialize(SYMBOL_MODE)
    
    # Limit symbols for backfill to respect rate limits
    SYMBOLS = all_symbols[:MAX_BACKFILL_SYMBOLS]
    logger.info("Loaded %d symbols in '%s' mode for backfill (limited to %d)", 
               len(all_symbols), SYMBOL_MODE, len(SYMBOLS))


async def main():
    if not (ALPACA_KEY and ALPACA_SECRET):
        logger.error("Set ALPACA_KEY and ALPACA_SECRET environment variables")
        return
    
    # Initialize symbols first
    await initialize_symbols()
    
    logger.info("Starting historical data backfill for %d symbols", len(SYMBOLS))
    
    # Database setup
    pool = await asyncpg.create_pool(DATABASE_URL, min_size=2, max_size=10)
    await init_database(pool)
    
    # HTTP session for API calls
    async with aiohttp.ClientSession() as session:
        # Process symbols in batches to manage rate limits and memory
        batch_size = 10  # Process 10 symbols at a time
        total_symbols = len(SYMBOLS)
        
        for i in range(0, total_symbols, batch_size):
            batch = SYMBOLS[i:i + batch_size]
            logger.info("Processing batch %d/%d: symbols %d-%d", 
                       (i // batch_size) + 1, 
                       (total_symbols + batch_size - 1) // batch_size,
                       i + 1, min(i + batch_size, total_symbols))
            
            # Process batch concurrently with semaphore for rate limiting
            semaphore = asyncio.Semaphore(3)  # Max 3 concurrent requests
            
            async def process_symbol_with_limit(symbol):
                async with semaphore:
                    try:
                        await backfill_symbol(session, pool, symbol.strip(), days_back=365*5)
                        await asyncio.sleep(0.5)  # Rate limiting between requests
                    except Exception as e:
                        logger.error("Failed to backfill %s: %s", symbol, e)
            
            # Process batch concurrently
            tasks = [process_symbol_with_limit(symbol) for symbol in batch]
            await asyncio.gather(*tasks, return_exceptions=True)
            
            # Longer pause between batches
            if i + batch_size < total_symbols:
                logger.info("Batch completed, waiting 5 seconds before next batch...")
                await asyncio.sleep(5)
    
    await pool.close()
    logger.info("Backfill completed")


if __name__ == "__main__":
    asyncio.run(main()) 