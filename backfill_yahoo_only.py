#!/usr/bin/env python3
"""
Backfill Yahoo Finance data for testing.

This script will backfill TNXP historical data from Yahoo Finance API.
"""

import asyncio
import aiohttp
import asyncpg
import os
import sys
from datetime import datetime, timezone, timedelta

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from tradebot.backfill.yahoo_historical import backfill_yahoo_symbol, init_yahoo_database

async def backfill_yahoo_test():
    """Backfill Yahoo Finance data for TNXP and a few other symbols for testing."""
    database_url = "postgresql://postgres:password@localhost:5432/tradebot"
    
    print("📥 Starting Yahoo Finance backfill test...")
    
    try:
        # Create database pool
        pool = await asyncpg.create_pool(database_url, min_size=2, max_size=10)
        
        # Initialize database schema
        await init_yahoo_database(pool)
        
        # Test symbols with volume issues
        test_symbols = ["TNXP", "AAPL", "MSFT", "TSLA"]
        
        # Create HTTP session
        async with aiohttp.ClientSession() as session:
            # Backfill each symbol for the last 30 days
            for symbol in test_symbols:
                print(f"\n🔄 Backfilling {symbol}...")
                await backfill_yahoo_symbol(session, pool, symbol, days_back=30)
        
        # Check results
        print("\n📊 Yahoo Finance data verification:")
        async with pool.acquire() as conn:
            for symbol in test_symbols:
                result = await conn.fetchrow("""
                    SELECT COUNT(*) as count, MAX(volume) as max_volume, AVG(volume) as avg_volume
                    FROM price_ticks_yahoo 
                    WHERE symbol = $1 AND timestamp >= NOW() - INTERVAL '30 days'
                """, symbol)
                
                if result['count'] > 0:
                    print(f"✅ {symbol}: {result['count']} records, Max Volume: {result['max_volume']:,}, Avg Volume: {result['avg_volume']:,.0f}")
                else:
                    print(f"❌ {symbol}: No data found")
        
        await pool.close()
        print("\n✅ Yahoo Finance backfill test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error during Yahoo Finance backfill: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(backfill_yahoo_test()) 