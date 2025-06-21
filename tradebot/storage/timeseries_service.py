import asyncio
import logging
import os
from datetime import datetime

import asyncpg
from dotenv import load_dotenv

from tradebot.common.bus import MessageBus
from tradebot.common.models import PriceTick

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("timeseries_storage")

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/tradebot")


async def init_database():
    """Create tables and TimescaleDB hypertable with full OHLCV support."""
    conn = await asyncpg.connect(DATABASE_URL)
    try:
        # Create extension
        await conn.execute("CREATE EXTENSION IF NOT EXISTS timescaledb;")
        
        # Create table with basic fields
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS price_ticks (
                symbol TEXT NOT NULL,
                price DECIMAL(10,4) NOT NULL,
                timestamp TIMESTAMPTZ NOT NULL,
                volume INTEGER DEFAULT 0
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
        
        # Convert to hypertable (TimescaleDB)
        try:
            await conn.execute("""
                SELECT create_hypertable('price_ticks', 'timestamp', if_not_exists => TRUE);
            """)
            logger.info("Created TimescaleDB hypertable")
        except Exception as e:
            logger.warning("Hypertable creation failed (may already exist): %s", e)
        
        # Create indexes
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_price_ticks_symbol_time 
            ON price_ticks (symbol, timestamp DESC);
        """)
        
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_price_ticks_symbol_volume 
            ON price_ticks (symbol, volume DESC) WHERE volume IS NOT NULL;
        """)
        
        logger.info("Database initialized successfully with OHLCV support")
    finally:
        await conn.close()


async def store_ticks():
    """Subscribe to price.ticks and store to TimescaleDB with full OHLCV data."""
    bus = MessageBus()
    await bus.connect()
    
    # Database connection pool
    pool = await asyncpg.create_pool(DATABASE_URL, min_size=2, max_size=10)
    
    tick_count = 0
    async for msg in bus.subscribe("price.ticks", last_id="$", block_ms=1000):
        try:
            tick = PriceTick(**msg)
            
            async with pool.acquire() as conn:
                await conn.execute(
                    """INSERT INTO price_ticks (symbol, price, timestamp, open_price, high_price, low_price, close_price, volume, trade_count, vwap) 
                       VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)""",
                    tick.symbol, tick.price, tick.timestamp, tick.open, tick.high, tick.low, tick.close, tick.volume, tick.trade_count, tick.vwap
                )
            
            tick_count += 1
            if tick_count % 100 == 0:
                logger.info("Stored %d ticks", tick_count)
            else:
                if tick.volume:
                    logger.debug("Stored OHLCV: %s @ $%.4f (Vol: %d)", tick.symbol, tick.price, tick.volume)
                else:
                    logger.debug("Stored tick: %s @ $%.4f", tick.symbol, tick.price)
                
        except Exception as e:
            logger.error("Failed to store tick %s: %s", msg, e)


async def main():
    logger.info("Initializing TimescaleDB storage service...")
    await init_database()
    
    logger.info("Starting tick storage...")
    await store_ticks()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("TimescaleDB storage service stopped") 