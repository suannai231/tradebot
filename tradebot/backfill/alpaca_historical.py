import asyncio
import logging
import os
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Tuple

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
        "adjustment": "raw",  # Changed to raw for accurate volume
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

    # Fetch both raw and split-adjusted bars for split detection
    raw_bars = await fetch_historical_bars(session, symbol, start_str, end_str, "1Day")
    split_bars = await fetch_split_adjusted_bars(session, symbol, start_str, end_str, "1Day")
    
    # Store raw bars (for accurate volume and recent prices)
    await store_bars(pool, symbol, raw_bars)
    
    # Detect and store splits by comparing raw vs split-adjusted data
    if raw_bars and split_bars:
        await detect_and_store_splits(pool, symbol, raw_bars, split_bars)


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
        
        # Create stock splits table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS stock_splits (
                symbol TEXT NOT NULL,
                split_date DATE NOT NULL,
                split_ratio DECIMAL(10,4) NOT NULL,
                raw_price_before DECIMAL(15,4),
                raw_price_after DECIMAL(15,4),
                adjusted_price DECIMAL(15,4),
                created_at TIMESTAMPTZ DEFAULT NOW(),
                PRIMARY KEY (symbol, split_date)
            );
        """)
        
        # Create index for efficient split lookups
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_stock_splits_symbol_date 
            ON stock_splits (symbol, split_date DESC);
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
    """Initialize symbols for backfill - always fetch ALL symbols regardless of SYMBOL_MODE."""
    global SYMBOLS
    
    # Force 'all' mode for backfill to get comprehensive historical data
    backfill_mode = "all"
    logger.info("Backfill mode: ALWAYS fetching ALL symbols (ignoring SYMBOL_MODE=%s)", SYMBOL_MODE)
    
    # Use symbol manager in 'all' mode
    manager = SymbolManager()
    all_symbols = await manager.initialize(backfill_mode)
    
    # Limit symbols for backfill to respect rate limits
    SYMBOLS = all_symbols[:MAX_BACKFILL_SYMBOLS]
    logger.info("Loaded %d symbols for comprehensive backfill (limited to %d)", 
               len(all_symbols), len(SYMBOLS))


async def detect_and_store_splits(pool: asyncpg.Pool, symbol: str, raw_bars: List[dict], split_bars: List[dict]):
    """Detect stock splits by comparing raw and split-adjusted data, then store in database."""
    if len(raw_bars) != len(split_bars) or len(raw_bars) < 2:
        return
    
    splits_detected = []
    
    for i in range(len(raw_bars)):
        raw_bar = raw_bars[i]
        split_bar = split_bars[i]
        
        # Calculate split ratio by comparing raw vs adjusted close prices
        raw_close = raw_bar["c"]
        adj_close = split_bar["c"]
        
        if adj_close > 0:
            split_ratio = raw_close / adj_close
            
            # If ratio is significantly different from 1, it indicates cumulative splits
            if abs(split_ratio - 1.0) > 0.01:  # More than 1% difference
                timestamp = datetime.fromisoformat(raw_bar["t"].replace("Z", "+00:00"))
                
                # Check if this is a new split point (ratio changed from previous bar)
                if i > 0:
                    prev_raw = raw_bars[i-1]["c"]
                    prev_adj = split_bars[i-1]["c"]
                    prev_ratio = prev_raw / prev_adj if prev_adj > 0 else 1.0
                    
                    # Detect actual split event by looking at price jumps
                    raw_price_change = raw_close / prev_raw if prev_raw > 0 else 1.0
                    adj_price_change = adj_close / prev_adj if prev_adj > 0 else 1.0
                    
                    # Split detection: significant difference between raw and adjusted price changes
                    price_change_ratio = raw_price_change / adj_price_change if adj_price_change > 0 else 1.0
                    
                    # Forward split: raw price drops significantly more than adjusted price
                    if price_change_ratio < 0.67 and raw_price_change < 0.67:  # Raw price dropped by >33%
                        split_factor = round(1.0 / raw_price_change)
                        splits_detected.append({
                            'symbol': symbol,
                            'split_date': timestamp.date(),
                            'split_ratio': split_factor,
                            'raw_price_before': prev_raw,
                            'raw_price_after': raw_close,
                            'adj_price': adj_close
                        })
                        logger.info(f"Detected forward split for {symbol} on {timestamp.date()}: {split_factor}:1")
                    
                    # Reverse split: raw price jumps significantly more than adjusted price  
                    elif price_change_ratio > 1.5 and raw_price_change > 1.5:  # Raw price increased by >50%
                        split_factor = round(raw_price_change)  # e.g., 100.0 for 1:100 reverse split (for backward adjustment)
                        splits_detected.append({
                            'symbol': symbol,
                            'split_date': timestamp.date(),
                            'split_ratio': split_factor,
                            'raw_price_before': prev_raw,
                            'raw_price_after': raw_close,
                            'adj_price': adj_close
                        })
                        logger.info(f"Detected reverse split for {symbol} on {timestamp.date()}: 1:{round(raw_price_change)} (stored factor: {split_factor} for backward adjustment)")
    
    # Store detected splits in database
    if splits_detected:
        async with pool.acquire() as conn:
            await conn.executemany(
                """INSERT INTO stock_splits (symbol, split_date, split_ratio, raw_price_before, raw_price_after, adjusted_price) 
                   VALUES ($1, $2, $3, $4, $5, $6) ON CONFLICT (symbol, split_date) DO UPDATE SET
                   split_ratio = EXCLUDED.split_ratio,
                   raw_price_before = EXCLUDED.raw_price_before,
                   raw_price_after = EXCLUDED.raw_price_after,
                   adjusted_price = EXCLUDED.adjusted_price""",
                [(s['symbol'], s['split_date'], s['split_ratio'], s['raw_price_before'], 
                  s['raw_price_after'], s['adj_price']) for s in splits_detected]
            )
        
        logger.info(f"Detected and stored {len(splits_detected)} splits for {symbol}")
        for split in splits_detected:
            logger.info(f"  {split['split_date']}: {split['split_ratio']}:1 split "
                       f"(${split['raw_price_before']:.2f} → ${split['raw_price_after']:.2f})")


async def fetch_split_adjusted_bars(session: aiohttp.ClientSession, symbol: str, start: str, end: str, timeframe: str = "1Day"):
    """Fetch split-adjusted bars for split detection."""
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
        "adjustment": "split",
        "feed": "iex"
    }
    
    try:
        async with session.get(url, headers=headers, params=params) as response:
            if response.status == 200:
                data = await response.json()
                return data.get("bars", [])
            elif response.status == 403:
                # Try without feed parameter
                params.pop("feed", None)
                async with session.get(url, headers=headers, params=params) as retry_response:
                    if retry_response.status == 200:
                        data = await retry_response.json()
                        return data.get("bars", [])
    except Exception as e:
        logger.warning(f"Failed to fetch split-adjusted data for {symbol}: {e}")
    
    return []


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