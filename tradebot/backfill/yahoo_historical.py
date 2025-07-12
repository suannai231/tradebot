"""
Yahoo Finance Historical Data Backfill Service

Fetches historical OHLCV data from Yahoo Finance with comprehensive volume data.
This provides much more accurate volume data compared to Alpaca IEX feed.
Also automatically extracts and stores stock split events for accurate price adjustments.
"""

import asyncio
import logging
import os
import aiohttp
import json
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional
from urllib.parse import quote

import asyncpg
from tradebot.common.models import PriceTick
from tradebot.common.symbol_manager import SymbolManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("yahoo_backfill")

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/tradebot")

# Configuration
SYMBOL_MODE = os.getenv("SYMBOL_MODE", "custom")
LEGACY_SYMBOLS = os.getenv("SYMBOLS", "AAPL,MSFT,AMZN,GOOG,TSLA,TNXP").split(",")
MAX_BACKFILL_SYMBOLS = int(os.getenv("MAX_SYMBOLS", "500"))

# Will be populated based on mode
SYMBOLS = []

# Yahoo Finance API URLs
YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart"


async def store_split_events(pool: asyncpg.Pool, symbol: str, split_events: List[Dict]):
    """Store stock split events in the stock_splits table."""
    if not split_events:
        return
    
    async with pool.acquire() as conn:
        for event in split_events:
            try:
                split_date = datetime.fromtimestamp(event["date"], tz=timezone.utc).date()
                
                # Yahoo Finance provides splits as "numerator:denominator" (e.g., "1:100" for 100:1 reverse split)
                # We need to store the consolidation ratio (100 for 100:1 reverse split)
                numerator = event.get("numerator", 1)
                denominator = event.get("denominator", 1)
                
                # For reverse splits (numerator < denominator), the ratio is denominator/numerator
                # For regular splits (numerator > denominator), the ratio is numerator/denominator  
                if numerator < denominator:
                    # Reverse split: 1:100 -> ratio = 100 (100 old shares become 1 new share)
                    split_ratio = denominator / numerator
                else:
                    # Regular split: 2:1 -> ratio = 2 (1 old share becomes 2 new shares)
                    split_ratio = numerator / denominator
                
                await conn.execute("""
                    INSERT INTO stock_splits (symbol, split_date, split_ratio) 
                    VALUES ($1, $2, $3) 
                    ON CONFLICT (symbol, split_date) DO UPDATE SET split_ratio = $3
                """, symbol, split_date, split_ratio)
                
                logger.info(f"Stored split event for {symbol}: {split_date} ratio {split_ratio:.1f} (Yahoo: {numerator}:{denominator})")
                
            except Exception as e:
                logger.warning(f"Error storing split event for {symbol}: {e}")
                continue


async def fetch_yahoo_historical(session: aiohttp.ClientSession, symbol: str, start_date: datetime, end_date: datetime) -> tuple[List[Dict], List[Dict]]:
    """Fetch historical data and split events from Yahoo Finance."""
    try:
        # Convert dates to Unix timestamps
        start_timestamp = int(start_date.timestamp())
        end_timestamp = int(end_date.timestamp())
        
        # Yahoo Finance chart API
        url = f"{YAHOO_CHART_URL}/{quote(symbol)}"
        params = {
            "period1": start_timestamp,
            "period2": end_timestamp,
            "interval": "1d",
            "includePrePost": "false",
            "events": "div,splits"
        }
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        logger.debug(f"Fetching Yahoo data for {symbol}: {url}")
        
        async with session.get(url, headers=headers, params=params, timeout=30) as response:
            if response.status != 200:
                error_text = await response.text()
                logger.error(f"Yahoo Finance API error {response.status} for {symbol}: {error_text}")
                return [], []
            
            data = await response.json()
            
            # Parse Yahoo Finance response
            chart = data.get("chart", {})
            result = chart.get("result", [])
            
            if not result:
                logger.warning(f"No chart data returned for {symbol}")
                return [], []
            
            chart_data = result[0]
            timestamps = chart_data.get("timestamp", [])
            indicators = chart_data.get("indicators", {})
            
            # Extract split events
            split_events = []
            events = chart_data.get("events", {})
            if events and "splits" in events:
                splits_data = events["splits"]
                for split_timestamp, split_info in splits_data.items():
                    split_events.append({
                        "date": int(split_timestamp),
                        "numerator": split_info.get("numerator", 1),
                        "denominator": split_info.get("denominator", 1),
                        "splitRatio": split_info.get("splitRatio")
                    })
                logger.info(f"Found {len(split_events)} split events for {symbol}")
            
            if not timestamps or not indicators:
                logger.warning(f"No timestamp or indicator data for {symbol}")
                return [], split_events
            
            # Extract OHLCV data
            quotes = indicators.get("quote", [])
            if not quotes:
                logger.warning(f"No quote data in indicators for {symbol}")
                return [], split_events
                
            quote_data = quotes[0]
            opens = quote_data.get("open", [])
            highs = quote_data.get("high", [])
            lows = quote_data.get("low", [])
            closes = quote_data.get("close", [])
            volumes = quote_data.get("volume", [])
            
            if not closes:
                logger.warning(f"No price data found for {symbol}")
                return [], split_events
            
            # Build bars list
            bars = []
            for i, ts in enumerate(timestamps):
                try:
                    # Skip if any required data is missing
                    if i >= len(closes) or closes[i] is None:
                        continue
                    
                    bar = {
                        "t": datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                        "o": opens[i] if i < len(opens) and opens[i] is not None else closes[i],
                        "h": highs[i] if i < len(highs) and highs[i] is not None else closes[i],
                        "l": lows[i] if i < len(lows) and lows[i] is not None else closes[i],
                        "c": closes[i],
                        "v": volumes[i] if i < len(volumes) and volumes[i] is not None else 0
                    }
                    bars.append(bar)
                    
                except (IndexError, TypeError, ValueError) as e:
                    logger.debug(f"Skipping bar {i} for {symbol}: {e}")
                    continue
            
            logger.info(f"Fetched {len(bars)} bars and {len(split_events)} split events for {symbol} from Yahoo Finance")
            return bars, split_events
            
    except Exception as e:
        logger.error(f"Error fetching Yahoo data for {symbol}: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return [], []


async def store_yahoo_bars(pool: asyncpg.Pool, symbol: str, bars: List[Dict]):
    """Store Yahoo Finance bars as price ticks with full OHLCV data."""
    if not bars:
        return
    
    # Convert bars to ticks with full OHLCV data
    ticks = []
    for bar in bars:
        try:
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
                trade_count=None,  # Yahoo doesn't provide trade count
                vwap=None  # Could be calculated if needed
            )
            ticks.append(tick)
        except Exception as e:
            logger.warning(f"Error creating tick for {symbol}: {e}")
            continue
    
    if not ticks:
        logger.warning(f"No valid ticks created for {symbol}")
        return
    
    # Batch insert with all OHLCV fields into Yahoo table
    async with pool.acquire() as conn:
        await conn.executemany(
            """INSERT INTO price_ticks_yahoo (symbol, price, timestamp, open_price, high_price, low_price, close_price, volume, trade_count, vwap) 
               VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10) ON CONFLICT DO NOTHING""",
            [(tick.symbol, tick.price, tick.timestamp, tick.open, tick.high, tick.low, tick.close, tick.volume, tick.trade_count, tick.vwap) for tick in ticks]
        )
    
    logger.info(f"Stored {len(ticks)} OHLCV bars for {symbol} in Yahoo table")


async def backfill_yahoo_symbol(session: aiohttp.ClientSession, pool: asyncpg.Pool, symbol: str, days_back: int = 365*5):
    """Backfill Yahoo Finance data and split events for a single symbol."""
    
    # Check existing data
    async with pool.acquire() as conn:
        last_ts = await conn.fetchval("SELECT max(timestamp) FROM price_ticks_yahoo WHERE symbol=$1", symbol)

    if last_ts:
        # Incremental backfill
        start_date = last_ts + timedelta(days=1)
        logger.info(f"Incremental backfill {symbol} starting {start_date.date()} (existing data to {last_ts.date()})")
    else:
        # Full history backfill
        start_date = datetime.now(timezone.utc) - timedelta(days=days_back)
        logger.info(f"Full backfill {symbol} last {days_back} days")

    end_date = datetime.now(timezone.utc)

    # If nothing is missing
    if start_date.date() > end_date.date():
        logger.info(f"{symbol} is already up-to-date – skipping")
        return

    # Fetch Yahoo Finance data and split events
    bars, split_events = await fetch_yahoo_historical(session, symbol, start_date, end_date)
    
    # Store price data
    await store_yahoo_bars(pool, symbol, bars)
    
    # Store split events
    await store_split_events(pool, symbol, split_events)


async def init_yahoo_database(pool: asyncpg.Pool):
    """Ensure Yahoo Finance table and stock splits table exist."""
    async with pool.acquire() as conn:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS timescaledb;")
        
        # Create Yahoo Finance table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS price_ticks_yahoo (
                symbol TEXT NOT NULL,
                price DECIMAL(15,6) NOT NULL,
                timestamp TIMESTAMPTZ NOT NULL,
                open_price DECIMAL(15,6),
                high_price DECIMAL(15,6),
                low_price DECIMAL(15,6),
                close_price DECIMAL(15,6),
                volume BIGINT DEFAULT 0,
                trade_count INTEGER DEFAULT 0,
                vwap DECIMAL(15,6),
                UNIQUE(symbol, timestamp)
            );
        """)
        
        # Create stock splits table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS stock_splits (
                symbol TEXT NOT NULL,
                split_date DATE NOT NULL,
                split_ratio NUMERIC(10,2) NOT NULL,
                raw_price_before NUMERIC(15,6),
                raw_price_after NUMERIC(15,6),
                adjusted_price NUMERIC(15,6),
                PRIMARY KEY (symbol, split_date)
            );
        """)
        
        # Create hypertable for time-series optimization
        try:
            await conn.execute("""
                SELECT create_hypertable('price_ticks_yahoo', 'timestamp', if_not_exists => TRUE);
            """)
            logger.info("Created Yahoo Finance TimescaleDB hypertable")
        except Exception as e:
            logger.debug(f"Hypertable already exists or creation failed: {e}")
        
        # Create indexes for better query performance
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_price_ticks_yahoo_symbol_time 
            ON price_ticks_yahoo (symbol, timestamp DESC);
        """)
        
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_price_ticks_yahoo_symbol_volume 
            ON price_ticks_yahoo (symbol, volume DESC) WHERE volume IS NOT NULL;
        """)
        
        # Create index for stock splits
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_stock_splits_symbol_date 
            ON stock_splits (symbol, split_date DESC);
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


async def test_yahoo_api():
    """Test Yahoo Finance API with a few symbols."""
    test_symbols = ["TNXP", "AAPL", "MSFT"]
    logger.info("Testing Yahoo Finance API...")
    
    async with aiohttp.ClientSession() as session:
        for symbol in test_symbols:
            start_date = datetime.now(timezone.utc) - timedelta(days=365)  # Check last year for splits
            end_date = datetime.now(timezone.utc)
            
            bars, split_events = await fetch_yahoo_historical(session, symbol, start_date, end_date)
            if bars:
                recent_bar = bars[-1] if bars else {}
                volume = recent_bar.get("v", 0)
                price = recent_bar.get("c", 0)
                logger.info(f"✅ {symbol}: {len(bars)} bars, {len(split_events)} splits, Recent: ${price:.2f}, Volume: {volume:,}")
            else:
                logger.warning(f"❌ {symbol}: No data received")


async def main():
    """Main backfill function."""
    logger.info("🚀 Starting Yahoo Finance historical data backfill...")
    
    # Test API first
    await test_yahoo_api()
    
    # Initialize symbols
    await initialize_symbols()
    
    if not SYMBOLS:
        logger.error("No symbols to backfill")
        return
    
    logger.info(f"Starting Yahoo Finance backfill for {len(SYMBOLS)} symbols")
    
    # Database setup
    pool = await asyncpg.create_pool(DATABASE_URL, min_size=2, max_size=10)
    await init_yahoo_database(pool)
    
    # HTTP session for API calls
    async with aiohttp.ClientSession() as session:
        # Process symbols in batches to manage rate limits
        batch_size = 10  # Yahoo Finance can handle more concurrent requests
        total_symbols = len(SYMBOLS)
        
        for i in range(0, total_symbols, batch_size):
            batch = SYMBOLS[i:i + batch_size]
            logger.info(f"Processing batch {(i // batch_size) + 1}/{(total_symbols + batch_size - 1) // batch_size}: symbols {i + 1}-{min(i + batch_size, total_symbols)}")
            
            # Process batch concurrently with semaphore for rate limiting
            semaphore = asyncio.Semaphore(5)  # Yahoo Finance can handle more requests
            
            async def process_symbol_with_limit(symbol):
                async with semaphore:
                    try:
                        await backfill_yahoo_symbol(session, pool, symbol.strip(), days_back=365*5)
                        await asyncio.sleep(0.2)  # Rate limiting between requests
                    except Exception as e:
                        logger.error(f"Failed to backfill {symbol}: {e}")
            
            # Process batch concurrently
            tasks = [process_symbol_with_limit(symbol) for symbol in batch]
            await asyncio.gather(*tasks, return_exceptions=True)
            
            # Pause between batches
            if i + batch_size < total_symbols:
                logger.info("Batch completed, waiting 2 seconds before next batch...")
                await asyncio.sleep(2)
    
    await pool.close()
    logger.info("✅ Yahoo Finance backfill completed")


if __name__ == "__main__":
    asyncio.run(main()) 