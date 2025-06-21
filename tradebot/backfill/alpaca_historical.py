import asyncio
import logging
import os
from datetime import datetime, timezone, timedelta
from typing import List

import aiohttp
import asyncpg
from dotenv import load_dotenv

from tradebot.common.models import PriceTick

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("alpaca_backfill")

ALPACA_KEY = os.getenv("ALPACA_KEY")
ALPACA_SECRET = os.getenv("ALPACA_SECRET")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/tradebot")
SYMBOLS = os.getenv("SYMBOLS", "AAPL,MSFT,AMZN,GOOG,TSLA").split(",")

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
        "adjustment": "raw",
        "feed": "sip"
    }
    
    async with session.get(url, headers=headers, params=params) as response:
        if response.status != 200:
            logger.error("Failed to fetch %s: %s %s", symbol, response.status, await response.text())
            return []
        
        data = await response.json()
        bars = data.get("bars", [])
        logger.info("Fetched %d bars for %s", len(bars), symbol)
        return bars


async def store_bars(pool: asyncpg.Pool, symbol: str, bars: List[dict]):
    """Store historical bars as price ticks."""
    if not bars:
        return
    
    # Convert bars to ticks (using close price)
    ticks = []
    for bar in bars:
        timestamp = datetime.fromisoformat(bar["t"].replace("Z", "+00:00"))
        # Create a tick from the bar's close price
        tick = PriceTick(
            symbol=symbol,
            price=bar["c"],  # close price
            timestamp=timestamp
        )
        ticks.append(tick)
    
    # Batch insert
    async with pool.acquire() as conn:
        await conn.executemany(
            "INSERT INTO price_ticks (symbol, price, timestamp) VALUES ($1, $2, $3) ON CONFLICT DO NOTHING",
            [(tick.symbol, tick.price, tick.timestamp) for tick in ticks]
        )
    
    logger.info("Stored %d ticks for %s", len(ticks), symbol)


async def backfill_symbol(session: aiohttp.ClientSession, pool: asyncpg.Pool, symbol: str, days_back: int = 30):
    """Backfill historical data for one symbol."""
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=days_back)
    
    start_str = start_date.strftime("%Y-%m-%dT%H:%M:%SZ")
    end_str = end_date.strftime("%Y-%m-%dT%H:%M:%SZ")
    
    logger.info("Backfilling %s from %s to %s", symbol, start_str, end_str)
    
    # Fetch daily bars
    bars = await fetch_historical_bars(session, symbol, start_str, end_str, "1Day")
    await store_bars(pool, symbol, bars)
    
    # Fetch hourly bars for recent data
    recent_start = end_date - timedelta(days=7)
    recent_start_str = recent_start.strftime("%Y-%m-%dT%H:%M:%SZ")
    
    logger.info("Fetching hourly data for %s", symbol)
    hourly_bars = await fetch_historical_bars(session, symbol, recent_start_str, end_str, "1Hour")
    await store_bars(pool, symbol, hourly_bars)


async def init_database(pool: asyncpg.Pool):
    """Ensure database schema exists."""
    async with pool.acquire() as conn:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS timescaledb;")
        
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS price_ticks (
                symbol TEXT NOT NULL,
                price DECIMAL(10,4) NOT NULL,
                timestamp TIMESTAMPTZ NOT NULL,
                volume INTEGER DEFAULT 0,
                UNIQUE(symbol, timestamp)
            );
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


async def main():
    if not (ALPACA_KEY and ALPACA_SECRET):
        logger.error("Set ALPACA_KEY and ALPACA_SECRET environment variables")
        return
    
    logger.info("Starting historical data backfill for symbols: %s", SYMBOLS)
    
    # Database setup
    pool = await asyncpg.create_pool(DATABASE_URL, min_size=2, max_size=10)
    await init_database(pool)
    
    # HTTP session for API calls
    async with aiohttp.ClientSession() as session:
        # Backfill each symbol
        for symbol in SYMBOLS:
            try:
                await backfill_symbol(session, pool, symbol.strip(), days_back=90)
                await asyncio.sleep(1)  # Rate limiting
            except Exception as e:
                logger.error("Failed to backfill %s: %s", symbol, e)
    
    await pool.close()
    logger.info("Backfill completed")


if __name__ == "__main__":
    asyncio.run(main()) 