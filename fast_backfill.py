#!/usr/bin/env python3
"""
High-Performance Backfill Script

Optimized version of the backfill service with aggressive performance settings
for faster historical data download.
"""

import asyncio
import logging
import os
import sys
from datetime import datetime, timezone, timedelta
from typing import List

import aiohttp
import asyncpg
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from tradebot.common.models import PriceTick
from tradebot.common.symbol_manager import SymbolManager

load_dotenv()

# Aggressive performance settings
BATCH_SIZE = int(os.getenv("BACKFILL_BATCH_SIZE", "25"))  # 25 symbols per batch
CONCURRENT_REQUESTS = int(os.getenv("BACKFILL_CONCURRENT_REQUESTS", "10"))  # 10 concurrent requests
RATE_LIMIT_DELAY = float(os.getenv("RATE_LIMIT_DELAY", "0.1"))  # 0.1s delay
BATCH_DELAY = 2.0  # Reduced from 5s to 2s between batches

# Database pool settings
DB_POOL_MIN = int(os.getenv("DB_POOL_MIN_SIZE", "5"))
DB_POOL_MAX = int(os.getenv("DB_POOL_MAX_SIZE", "20"))

# Reduce logging for performance
logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("fast_backfill")

ALPACA_KEY = os.getenv("ALPACA_KEY")
ALPACA_SECRET = os.getenv("ALPACA_SECRET")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/tradebot")
SYMBOL_MODE = os.getenv("SYMBOL_MODE", "all")
MAX_SYMBOLS = int(os.getenv("MAX_SYMBOLS", "1000"))

BASE_URL = "https://data.alpaca.markets/v2/stocks"


async def fetch_historical_bars_fast(session: aiohttp.ClientSession, symbol: str, start: str, end: str):
    """Optimized fetch with minimal error handling for speed."""
    url = f"{BASE_URL}/{symbol}/bars"
    headers = {
        "APCA-API-KEY-ID": ALPACA_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET,
    }
    params = {
        "start": start,
        "end": end,
        "timeframe": "1Day",
        "limit": 10000,
        "adjustment": "raw",
        "feed": "iex"
    }
    
    try:
        async with session.get(url, headers=headers, params=params, timeout=aiohttp.ClientTimeout(total=10)) as response:
            if response.status == 200:
                data = await response.json()
                return data.get("bars", [])
            elif response.status == 429:  # Rate limited
                await asyncio.sleep(1)  # Brief pause and retry
                return []
            else:
                return []
    except Exception:
        return []


async def store_bars_batch(pool: asyncpg.Pool, symbol_bars_list: List[tuple]):
    """Optimized batch storage for multiple symbols at once."""
    if not symbol_bars_list:
        return
    
    all_ticks = []
    for symbol, bars in symbol_bars_list:
        for bar in bars:
            timestamp = datetime.fromisoformat(bar["t"].replace("Z", "+00:00"))
            tick_data = (
                symbol, bar["c"], timestamp, bar.get("o"), bar.get("h"), 
                bar.get("l"), bar.get("c"), bar.get("v"), bar.get("n"), bar.get("vw")
            )
            all_ticks.append(tick_data)
    
    if all_ticks:
        async with pool.acquire() as conn:
            await conn.executemany(
                """INSERT INTO price_ticks (symbol, price, timestamp, open_price, high_price, low_price, close_price, volume, trade_count, vwap) 
                   VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10) ON CONFLICT DO NOTHING""",
                all_ticks
            )
        
        logger.warning(f"Stored {len(all_ticks)} bars for {len(symbol_bars_list)} symbols")


async def backfill_batch_fast(session: aiohttp.ClientSession, pool: asyncpg.Pool, symbols: List[str]):
    """Process a batch of symbols with maximum concurrency."""
    end_date = datetime.now(timezone.utc) - timedelta(days=15)
    start_date = end_date - timedelta(days=365)  # Changed from 90 to 365 days
    
    start_str = start_date.strftime("%Y-%m-%dT%H:%M:%SZ")
    end_str = end_date.strftime("%Y-%m-%dT%H:%M:%SZ")
    
    # Create semaphore for rate limiting
    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)
    
    async def fetch_symbol_data(symbol):
        async with semaphore:
            bars = await fetch_historical_bars_fast(session, symbol, start_str, end_str)
            await asyncio.sleep(RATE_LIMIT_DELAY)
            return (symbol, bars)
    
    # Fetch all symbols in batch concurrently
    tasks = [fetch_symbol_data(symbol.strip()) for symbol in symbols]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filter successful results
    successful_results = [r for r in results if isinstance(r, tuple) and r[1]]
    
    # Store all results in one database operation
    if successful_results:
        await store_bars_batch(pool, successful_results)
    
    return len(successful_results)


async def fast_backfill_main():
    """Main high-performance backfill function."""
    if not (ALPACA_KEY and ALPACA_SECRET):
        print("❌ Set ALPACA_KEY and ALPACA_SECRET environment variables")
        return
    
    print("🚀 Starting HIGH-PERFORMANCE backfill...")
    print(f"⚙️  Batch size: {BATCH_SIZE}")
    print(f"⚙️  Concurrent requests: {CONCURRENT_REQUESTS}")
    print(f"⚙️  Rate limit delay: {RATE_LIMIT_DELAY}s")
    print(f"⚙️  Database pool: {DB_POOL_MIN}-{DB_POOL_MAX}")
    
    # Load symbols
    manager = SymbolManager()
    all_symbols = await manager.initialize(SYMBOL_MODE)
    symbols = all_symbols[:MAX_SYMBOLS]
    
    print(f"📊 Processing {len(symbols)} symbols")
    
    # Create optimized database pool
    pool = await asyncpg.create_pool(
        DATABASE_URL, 
        min_size=DB_POOL_MIN, 
        max_size=DB_POOL_MAX,
        command_timeout=60
    )
    
    # Create HTTP session with optimized settings
    connector = aiohttp.TCPConnector(
        limit=100,  # Total connection limit
        limit_per_host=20,  # Per-host connection limit
        ttl_dns_cache=300,  # DNS cache TTL
        use_dns_cache=True,
    )
    
    total_processed = 0
    start_time = datetime.now()
    
    async with aiohttp.ClientSession(connector=connector) as session:
        # Process in large batches
        for i in range(0, len(symbols), BATCH_SIZE):
            batch = symbols[i:i + BATCH_SIZE]
            batch_num = (i // BATCH_SIZE) + 1
            total_batches = (len(symbols) + BATCH_SIZE - 1) // BATCH_SIZE
            
            print(f"⚡ Processing batch {batch_num}/{total_batches} ({len(batch)} symbols)")
            
            # Process batch
            processed = await backfill_batch_fast(session, pool, batch)
            total_processed += processed
            
            # Progress update
            elapsed = (datetime.now() - start_time).total_seconds()
            rate = total_processed / elapsed if elapsed > 0 else 0
            
            print(f"✅ Batch {batch_num} complete: {processed} symbols processed")
            print(f"📈 Overall progress: {total_processed}/{len(symbols)} symbols ({rate:.1f} symbols/sec)")
            
            # Brief pause between batches
            if i + BATCH_SIZE < len(symbols):
                await asyncio.sleep(BATCH_DELAY)
    
    await pool.close()
    
    total_time = (datetime.now() - start_time).total_seconds()
    print(f"🎉 HIGH-PERFORMANCE backfill completed!")
    print(f"📊 Total symbols processed: {total_processed}")
    print(f"⏱️  Total time: {total_time:.1f} seconds")
    print(f"🚀 Average rate: {total_processed/total_time:.1f} symbols/second")


if __name__ == "__main__":
    try:
        asyncio.run(fast_backfill_main())
    except KeyboardInterrupt:
        print("\n⚠️  Fast backfill interrupted by user")
    except Exception as e:
        print(f"❌ Fast backfill failed: {e}") 