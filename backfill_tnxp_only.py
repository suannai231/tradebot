#!/usr/bin/env python3
"""
Backfill TNXP data only.

This script will backfill TNXP historical data from Alpaca API.
"""

import asyncio
import aiohttp
import asyncpg
import os
import sys
from datetime import datetime, timezone, timedelta

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from tradebot.backfill.alpaca_historical import backfill_symbol, init_database

async def backfill_tnxp_only():
    """Backfill TNXP data only."""
    database_url = "postgresql://postgres:password@localhost:5432/tradebot"
    
    print("📥 Starting TNXP backfill...")
    
    try:
        # Create database pool
        pool = await asyncpg.create_pool(database_url, min_size=2, max_size=10)
        
        # Initialize database schema
        await init_database(pool)
        
        # Create HTTP session
        async with aiohttp.ClientSession() as session:
            # Backfill TNXP for the last 5 years
            await backfill_symbol(session, pool, "TNXP", days_back=365*5)
        
        await pool.close()
        print("✅ TNXP backfill completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error during TNXP backfill: {e}")
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
                "SELECT COUNT(*) FROM price_ticks WHERE symbol = $1",
                "TNXP"
            )
            
            print(f"Found {count_result} TNXP rows after backfill")
            
            if count_result > 0:
                # Get date range
                date_range = await conn.fetchrow(
                    """
                    SELECT 
                        MIN(timestamp) as earliest,
                        MAX(timestamp) as latest,
                        MIN(price) as min_price,
                        MAX(price) as max_price
                    FROM price_ticks 
                    WHERE symbol = $1
                    """,
                    "TNXP"
                )
                
                print(f"📅 Date range: {date_range['earliest']} to {date_range['latest']}")
                print(f"💰 Price range: ${date_range['min_price']:.4f} to ${date_range['max_price']:.4f}")
        
        await pool.close()
        
    except Exception as e:
        print(f"❌ Error verifying TNXP data: {e}")

async def main():
    """Main execution function."""
    print("🚀 Starting TNXP backfill process...")
    
    # Backfill TNXP
    if not await backfill_tnxp_only():
        print("❌ Failed to backfill TNXP. Exiting.")
        return
    
    # Verify the data
    await verify_tnxp_data()
    
    print("\n✅ TNXP backfill process completed!")

if __name__ == "__main__":
    asyncio.run(main()) 