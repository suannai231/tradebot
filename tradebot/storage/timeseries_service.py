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
    """Create tables and TimescaleDB hypertable."""
    conn = await asyncpg.connect(DATABASE_URL)
    try:
        # Create extension
        await conn.execute("CREATE EXTENSION IF NOT EXISTS timescaledb;")
        
        # Create table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS price_ticks (
                symbol TEXT NOT NULL,
                price DECIMAL(10,4) NOT NULL,
                timestamp TIMESTAMPTZ NOT NULL,
                volume INTEGER DEFAULT 0
            );
        """)
        
        # Convert to hypertable (TimescaleDB)
        try:
            await conn.execute("""
                SELECT create_hypertable('price_ticks', 'timestamp', if_not_exists => TRUE);
            """)
            logger.info("Created TimescaleDB hypertable")
        except Exception as e:
            logger.warning("Hypertable creation failed (may already exist): %s", e)
        
        # Create index on symbol
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_price_ticks_symbol_time 
            ON price_ticks (symbol, timestamp DESC);
        """)
        
        logger.info("Database initialized successfully")
    finally:
        await conn.close()


async def store_ticks():
    """Subscribe to price.ticks and store to TimescaleDB."""
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
                    "INSERT INTO price_ticks (symbol, price, timestamp) VALUES ($1, $2, $3)",
                    tick.symbol, tick.price, tick.timestamp
                )
            
            tick_count += 1
            if tick_count % 100 == 0:
                logger.info("Stored %d ticks", tick_count)
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