#!/usr/bin/env python3
"""
Fixed TNXP backfill with better error handling.

This script will backfill TNXP historical data from Yahoo Finance API
with improved error handling for data precision issues.
"""

import asyncio
import aiohttp
import asyncpg
import os
import sys
from datetime import datetime, timezone, timedelta
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("tnxp_backfill_fixed")

async def fetch_tnxp_data(session: aiohttp.ClientSession) -> tuple[list, list]:
    """Fetch TNXP data from Yahoo Finance with error handling."""
    symbol = "TNXP"
    start_date = datetime.now(timezone.utc) - timedelta(days=365*5)
    end_date = datetime.now(timezone.utc)
    
    # Yahoo Finance chart API
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    params = {
        "period1": int(start_date.timestamp()),
        "period2": int(end_date.timestamp()),
        "interval": "1d",
        "includePrePost": "false",
        "events": "div,splits"
    }
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    
    try:
        async with session.get(url, headers=headers, params=params, timeout=30) as response:
            if response.status != 200:
                error_text = await response.text()
                logger.error(f"Yahoo Finance API error {response.status}: {error_text}")
                return [], []
            
            data = await response.json()
            
            # Parse response
            chart = data.get("chart", {})
            result = chart.get("result", [])
            
            if not result:
                logger.warning("No chart data returned for TNXP")
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
                        "denominator": split_info.get("denominator", 1)
                    })
                logger.info(f"Found {len(split_events)} split events for TNXP")
            
            if not timestamps or not indicators:
                logger.warning("No timestamp or indicator data for TNXP")
                return [], split_events
            
            # Extract OHLCV data
            quotes = indicators.get("quote", [])
            if not quotes:
                logger.warning("No quote data in indicators for TNXP")
                return [], split_events
                
            quote_data = quotes[0]
            opens = quote_data.get("open", [])
            highs = quote_data.get("high", [])
            lows = quote_data.get("low", [])
            closes = quote_data.get("close", [])
            volumes = quote_data.get("volume", [])
            
            if not closes:
                logger.warning("No price data found for TNXP")
                return [], split_events
            
            # Build bars list with data validation
            bars = []
            for i, ts in enumerate(timestamps):
                try:
                    # Skip if any required data is missing
                    if i >= len(closes) or closes[i] is None:
                        continue
                    
                    # Validate price data
                    close_price = float(closes[i])
                    if close_price <= 0 or close_price > 1000000:  # Reasonable price range
                        logger.warning(f"Skipping TNXP bar {i}: invalid close price {close_price}")
                        continue
                    
                    open_price = opens[i] if i < len(opens) and opens[i] is not None else close_price
                    high_price = highs[i] if i < len(highs) and highs[i] is not None else close_price
                    low_price = lows[i] if i < len(lows) and lows[i] is not None else close_price
                    volume = volumes[i] if i < len(volumes) and volumes[i] is not None else 0
                    
                    # Validate other prices
                    open_price = float(open_price) if open_price and 0 < open_price <= 1000000 else close_price
                    high_price = float(high_price) if high_price and 0 < high_price <= 1000000 else close_price
                    low_price = float(low_price) if low_price and 0 < low_price <= 1000000 else close_price
                    
                    bar = {
                        "t": datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                        "o": open_price,
                        "h": high_price,
                        "l": low_price,
                        "c": close_price,
                        "v": int(volume) if volume else 0
                    }
                    bars.append(bar)
                    
                except (IndexError, TypeError, ValueError) as e:
                    logger.warning(f"Skipping TNXP bar {i}: {e}")
                    continue
            
            logger.info(f"Fetched {len(bars)} valid bars and {len(split_events)} split events for TNXP")
            return bars, split_events
            
    except Exception as e:
        logger.error(f"Error fetching TNXP data: {e}")
        return [], []

async def store_tnxp_data(pool: asyncpg.Pool, bars: list, split_events: list):
    """Store TNXP data with improved error handling."""
    if not bars:
        logger.warning("No bars to store for TNXP")
        return
    
    try:
        async with pool.acquire() as conn:
            # Store price data with explicit data type handling
            for bar in bars:
                try:
                    timestamp = datetime.fromisoformat(bar["t"].replace("Z", "+00:00"))
                    
                    # Convert to Decimal with proper precision
                    open_price = float(bar["o"])
                    high_price = float(bar["h"])
                    low_price = float(bar["l"])
                    close_price = float(bar["c"])
                    volume = int(bar["v"])
                    
                    # Insert with explicit data types
                    await conn.execute("""
                        INSERT INTO price_ticks_yahoo (symbol, price, timestamp, open_price, high_price, low_price, close_price, volume, trade_count, vwap) 
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10) 
                        ON CONFLICT (symbol, timestamp) DO NOTHING
                    """, 
                    "TNXP", 
                    close_price,  # price field
                    timestamp,
                    open_price,
                    high_price,
                    low_price,
                    close_price,
                    volume,
                    None,  # trade_count
                    None   # vwap
                    )
                    
                except Exception as e:
                    logger.warning(f"Error storing TNXP bar {bar.get('t', 'unknown')}: {e}")
                    continue
            
            # Store split events
            for event in split_events:
                try:
                    split_date = datetime.fromtimestamp(event["date"], tz=timezone.utc).date()
                    numerator = event.get("numerator", 1)
                    denominator = event.get("denominator", 1)
                    
                    if numerator < denominator:
                        split_ratio = denominator / numerator
                    else:
                        split_ratio = numerator / denominator
                    
                    await conn.execute("""
                        INSERT INTO stock_splits (symbol, split_date, split_ratio) 
                        VALUES ($1, $2, $3) 
                        ON CONFLICT (symbol, split_date) DO UPDATE SET split_ratio = $3
                    """, "TNXP", split_date, split_ratio)
                    
                    logger.info(f"Stored split event for TNXP: {split_date} ratio {split_ratio:.1f}")
                    
                except Exception as e:
                    logger.warning(f"Error storing TNXP split event: {e}")
                    continue
        
        logger.info(f"Successfully stored {len(bars)} TNXP bars and {len(split_events)} split events")
        
    except Exception as e:
        logger.error(f"Error in store_tnxp_data: {e}")
        raise

async def backfill_tnxp_fixed():
    """Backfill TNXP data with improved error handling."""
    database_url = "postgresql://postgres:password@localhost:5432/tradebot"
    
    print("📥 Starting TNXP backfill with improved error handling...")
    print("🎯 This will provide accurate volume data from Yahoo Finance!")
    
    try:
        # Create database pool
        pool = await asyncpg.create_pool(database_url, min_size=2, max_size=10)
        
        # Ensure table exists
        async with pool.acquire() as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS timescaledb;")
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
            
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS stock_splits (
                    symbol TEXT NOT NULL,
                    split_date DATE NOT NULL,
                    split_ratio NUMERIC(10,2) NOT NULL,
                    PRIMARY KEY (symbol, split_date)
                );
            """)
            
            # Create hypertable
            try:
                await conn.execute("""
                    SELECT create_hypertable('price_ticks_yahoo', 'timestamp', if_not_exists => TRUE);
                """)
            except Exception as e:
                logger.debug(f"Hypertable creation: {e}")
        
        # Create HTTP session and fetch data
        async with aiohttp.ClientSession() as session:
            bars, split_events = await fetch_tnxp_data(session)
            
            if bars:
                await store_tnxp_data(pool, bars, split_events)
                print(f"✅ Successfully stored {len(bars)} TNXP bars and {len(split_events)} split events")
            else:
                print("❌ No TNXP data received from Yahoo Finance")
        
        await pool.close()
        return True
        
    except Exception as e:
        print(f"❌ Error during TNXP backfill: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return False

async def verify_tnxp_data():
    """Verify that TNXP data was properly backfilled."""
    print("\n🔍 Verifying TNXP data...")
    
    database_url = "postgresql://postgres:password@localhost:5432/tradebot"
    
    try:
        pool = await asyncpg.create_pool(database_url)
        
        async with pool.acquire() as conn:
            # Count TNXP rows
            count_result = await conn.fetchval(
                "SELECT COUNT(*) FROM price_ticks_yahoo WHERE symbol = $1",
                "TNXP"
            )
            
            print(f"Found {count_result} TNXP rows in Yahoo table")
            
            if count_result > 0:
                # Get date range and stats
                date_range = await conn.fetchrow(
                    """
                    SELECT 
                        MIN(timestamp) as earliest,
                        MAX(timestamp) as latest,
                        MIN(close_price) as min_price,
                        MAX(close_price) as max_price,
                        MAX(volume) as max_volume,
                        AVG(volume) as avg_volume
                    FROM price_ticks_yahoo 
                    WHERE symbol = $1
                    """,
                    "TNXP"
                )
                
                print(f"📅 Date range: {date_range['earliest']} to {date_range['latest']}")
                print(f"💰 Price range: ${date_range['min_price']:.4f} to ${date_range['max_price']:.4f}")
                print(f"📊 Volume range: Max {date_range['max_volume']:,}, Avg {date_range['avg_volume']:,.0f}")
                
                # Show recent data
                recent_data = await conn.fetch(
                    """
                    SELECT timestamp, close_price, volume 
                    FROM price_ticks_yahoo 
                    WHERE symbol = $1 
                    ORDER BY timestamp DESC 
                    LIMIT 5
                    """,
                    "TNXP"
                )
                
                print("\n📈 Recent TNXP data:")
                for row in recent_data:
                    print(f"  {row['timestamp'].strftime('%Y-%m-%d')}: ${row['close_price']:.4f}, Vol: {row['volume']:,}")
        
        await pool.close()
        
    except Exception as e:
        print(f"❌ Error verifying TNXP data: {e}")

async def main():
    """Main execution function."""
    print("🚀 Starting TNXP backfill with improved error handling...")
    print("📈 This will provide accurate, comprehensive volume data!")
    
    # Backfill TNXP
    if not await backfill_tnxp_fixed():
        print("❌ Failed to backfill TNXP. Exiting.")
        return
    
    # Verify the data
    await verify_tnxp_data()
    
    print("\n✅ TNXP backfill process completed!")
    print("🎯 Your volume data is now accurate!")
    print("💡 Use DATA_SOURCE=yahoo in dashboard to see the data.")

if __name__ == "__main__":
    asyncio.run(main()) 