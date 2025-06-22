import asyncio
import logging
import os
import asyncpg
from datetime import datetime, timezone
from tradebot.common.bus import MessageBus
from tradebot.common.models import PriceTick

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('timeseries_storage')

# Database configuration
DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://postgres:password@timescaledb:5432/tradebot')

class StorageService:
    def __init__(self):
        self.bus = None
        self.pool = None
        self.tick_count = 0
        self.reconnect_delay = 1
        self.max_reconnect_delay = 30
        
    async def initialize(self):
        """Initialize database connection pool"""
        logger.info("Initializing TimescaleDB storage service...")
        
        # Database connection pool
        self.pool = await asyncpg.create_pool(DATABASE_URL, min_size=2, max_size=10)
        
        # Initialize TimescaleDB hypertable
        async with self.pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS price_ticks (
                    symbol VARCHAR(10) NOT NULL,
                    price DECIMAL(10,4) NOT NULL,
                    timestamp TIMESTAMPTZ NOT NULL,
                    open_price DECIMAL(10,4),
                    high_price DECIMAL(10,4),
                    low_price DECIMAL(10,4),
                    close_price DECIMAL(10,4),
                    volume BIGINT DEFAULT 0,
                    trade_count INTEGER DEFAULT 0,
                    vwap DECIMAL(10,4)
                );
            """)
            
            # Create hypertable if not exists
            try:
                await conn.execute("""
                    SELECT create_hypertable('price_ticks', 'timestamp', if_not_exists => TRUE);
                """)
                logger.info("Created TimescaleDB hypertable")
            except Exception as e:
                logger.info(f"Hypertable already exists or creation failed: {e}")
            
            # Create index for better query performance
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_price_ticks_symbol_time 
                ON price_ticks (symbol, timestamp DESC);
            """)
        
        logger.info("Database initialized successfully with OHLCV support")
    
    async def connect_to_redis(self):
        """Connect to Redis with retry logic"""
        while True:
            try:
                if self.bus:
                    await self.bus.disconnect()
                
                self.bus = MessageBus()
                await self.bus.connect()
                logger.info("Connected to Redis successfully")
                self.reconnect_delay = 1  # Reset delay on successful connection
                return
                
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                logger.info(f"Retrying in {self.reconnect_delay} seconds...")
                await asyncio.sleep(self.reconnect_delay)
                
                # Exponential backoff with max delay
                self.reconnect_delay = min(self.reconnect_delay * 2, self.max_reconnect_delay)
    
    async def store_ticks(self):
        """Subscribe to price.ticks and store to TimescaleDB with automatic reconnection"""
        logger.info("Starting subscription to price.ticks stream...")
        
        while True:
            try:
                await self.connect_to_redis()
                logger.info("Starting tick storage with Redis connection...")
                
                async for msg in self.bus.subscribe("price.ticks", last_id="$", block_ms=5000):
                    try:
                        tick = PriceTick(**msg)
                        
                        async with self.pool.acquire() as conn:
                            await conn.execute(
                                """INSERT INTO price_ticks (symbol, price, timestamp, open_price, high_price, low_price, close_price, volume, trade_count, vwap) 
                                   VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)""",
                                tick.symbol, tick.price, tick.timestamp, tick.open, tick.high, tick.low, tick.close, tick.volume, tick.trade_count, tick.vwap
                            )
                        
                        self.tick_count += 1
                        
                        if self.tick_count % 100 == 0:
                            logger.info(f"Stored {self.tick_count} ticks")
                            
                    except Exception as e:
                        logger.error(f"Error storing tick: {e}")
                        continue
                        
            except Exception as e:
                logger.error(f"Redis connection error: {e}")
                logger.info("Attempting to reconnect to Redis...")
                await asyncio.sleep(2)
                continue

async def main():
    """Main entry point"""
    service = StorageService()
    await service.initialize()
    await service.store_ticks()

if __name__ == "__main__":
    asyncio.run(main()) 