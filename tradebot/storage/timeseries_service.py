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

# Data source configuration - determines which table to use
DATA_SOURCE = os.getenv('DATA_SOURCE', 'synthetic')  # synthetic, alpaca, polygon

class StorageService:
    def __init__(self):
        self.bus = None
        self.pool = None
        self.tick_count = 0
        self.reconnect_delay = 1
        self.max_reconnect_delay = 30
        
        # Determine table name based on data source
        self.table_name = self._get_table_name()
        logger.info(f"Using table: {self.table_name} for data source: {DATA_SOURCE}")
        
    def _get_table_name(self) -> str:
        """Get table name based on data source"""
        valid_sources = ['synthetic', 'alpaca', 'polygon', 'mock', 'test']
        
        if DATA_SOURCE not in valid_sources:
            logger.warning(f"Unknown data source '{DATA_SOURCE}', defaulting to 'synthetic'")
            return 'price_ticks_synthetic'
        
        # Map data sources to table names
        table_mapping = {
            'synthetic': 'price_ticks_synthetic',
            'alpaca': 'price_ticks_alpaca', 
            'polygon': 'price_ticks_polygon',
            'mock': 'price_ticks_mock',
            'test': 'price_ticks_test'
        }
        
        return table_mapping.get(DATA_SOURCE, 'price_ticks_synthetic')
        
    async def initialize(self):
        """Initialize database connection pool and create tables"""
        logger.info("Initializing TimescaleDB storage service...")
        
        # Database connection pool
        self.pool = await asyncpg.create_pool(DATABASE_URL, min_size=2, max_size=10)
        
        # Create all possible tables (idempotent)
        await self._create_all_tables()
        
        logger.info(f"Database initialized successfully with table: {self.table_name}")
    
    async def _create_all_tables(self):
        """Create all data source tables with the same schema"""
        table_names = [
            'price_ticks_synthetic',
            'price_ticks_alpaca', 
            'price_ticks_polygon',
            'price_ticks_mock',
            'price_ticks_test'
        ]
        
        async with self.pool.acquire() as conn:
            for table_name in table_names:
                await self._create_table(conn, table_name)
    
    async def _create_table(self, conn, table_name: str):
        """Create a specific price ticks table with TimescaleDB optimization"""
        try:
            # Create table with OHLCV support
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
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
                await conn.execute(f"""
                    SELECT create_hypertable('{table_name}', 'timestamp', if_not_exists => TRUE);
                """)
                logger.info(f"Created TimescaleDB hypertable: {table_name}")
            except Exception as e:
                logger.debug(f"Hypertable {table_name} already exists or creation failed: {e}")
            
            # Create index for better query performance
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{table_name}_symbol_time 
                ON {table_name} (symbol, timestamp DESC);
            """)
            
        except Exception as e:
            logger.error(f"Error creating table {table_name}: {e}")
    
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
        """Subscribe to price.ticks and store to the appropriate table"""
        logger.info(f"Starting subscription to price.ticks stream for table: {self.table_name}")
        
        while True:
            try:
                await self.connect_to_redis()
                logger.info(f"Starting tick storage with Redis connection to table: {self.table_name}")
                
                async for msg in self.bus.subscribe("price.ticks", last_id="$", block_ms=5000):
                    try:
                        tick = PriceTick(**msg)
                        
                        async with self.pool.acquire() as conn:
                            await conn.execute(
                                f"""INSERT INTO {self.table_name} (symbol, price, timestamp, open_price, high_price, low_price, close_price, volume, trade_count, vwap) 
                                   VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)""",
                                tick.symbol, tick.price, tick.timestamp, tick.open, tick.high, tick.low, tick.close, tick.volume, tick.trade_count, tick.vwap
                            )
                        
                        self.tick_count += 1
                        
                        if self.tick_count % 100 == 0:
                            logger.info(f"Stored {self.tick_count} ticks to {self.table_name}")
                            
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