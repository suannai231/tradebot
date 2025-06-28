#!/usr/bin/env python3
"""
Delete TNXP data and rerun backfill script.

This script will:
1. Delete all TNXP rows from the price_ticks table
2. Rerun the backfill for TNXP with split-adjusted data
"""

import asyncio
import asyncpg
import aiohttp
import os
import sys
from datetime import datetime, timezone, timedelta

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from tradebot.backfill.alpaca_historical import backfill_symbol

async def delete_tnxp_data():
    """Delete all TNXP data from the database."""
    database_url = "postgresql://postgres:password@localhost:5432/tradebot"
    
    print("🗑️  Deleting all TNXP data from database...")
    
    try:
        pool = await asyncpg.create_pool(database_url)
        
        async with pool.acquire() as conn:
            # Count existing TNXP rows
            count_result = await conn.fetchval(
                "SELECT COUNT(*) FROM price_ticks WHERE symbol = $1",
                "TNXP"
            )
            
            print(f"Found {count_result} TNXP rows in database")
            
            if count_result > 0:
                # Delete all TNXP rows
                deleted_count = await conn.execute(
                    "DELETE FROM price_ticks WHERE symbol = $1",
                    "TNXP"
                )
                
                print(f"✅ Deleted {deleted_count} TNXP rows from database")
            else:
                print("ℹ️  No TNXP data found in database")
        
        await pool.close()
        
    except Exception as e:
        print(f"❌ Error deleting TNXP data: {e}")
        return False
    
    return True

async def rerun_tnxp_backfill():
    """Rerun backfill for TNXP with split-adjusted data."""
    print("\n📥 Rerunning TNXP backfill with split-adjusted data...")
    
    database_url = "postgresql://postgres:password@localhost:5432/tradebot"
    
    try:
        # Create database pool
        pool = await asyncpg.create_pool(database_url, min_size=2, max_size=10)
        
        # Create HTTP session
        async with aiohttp.ClientSession() as session:
            # Use the existing backfill function with correct parameters
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
            # Count new TNXP rows
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
                
                # Check for any suspicious price jumps (indicating split issues)
                price_jumps = await conn.fetch(
                    """
                    SELECT 
                        timestamp,
                        price,
                        LAG(price) OVER (ORDER BY timestamp) as prev_price,
                        (price - LAG(price) OVER (ORDER BY timestamp)) / LAG(price) OVER (ORDER BY timestamp) * 100 as price_change_pct
                    FROM price_ticks 
                    WHERE symbol = $1 
                    ORDER BY timestamp
                    """,
                    "TNXP"
                )
                
                # Find any extreme price changes (>50% in one tick)
                extreme_changes = [row for row in price_jumps if row['price_change_pct'] and abs(row['price_change_pct']) > 50]
                
                if extreme_changes:
                    print(f"⚠️  Found {len(extreme_changes)} extreme price changes (>50%):")
                    for change in extreme_changes[:5]:  # Show first 5
                        print(f"   {change['timestamp']}: ${change['prev_price']:.4f} → ${change['price']:.4f} ({change['price_change_pct']:+.1f}%)")
                else:
                    print("✅ No extreme price changes detected")
        
        await pool.close()
        
    except Exception as e:
        print(f"❌ Error verifying TNXP data: {e}")

async def main():
    """Main execution function."""
    print("🚀 Starting TNXP data cleanup and backfill process...")
    
    # Step 1: Delete existing TNXP data
    if not await delete_tnxp_data():
        print("❌ Failed to delete TNXP data. Exiting.")
        return
    
    # Step 2: Rerun backfill
    if not await rerun_tnxp_backfill():
        print("❌ Failed to rerun TNXP backfill. Exiting.")
        return
    
    # Step 3: Verify the new data
    await verify_tnxp_data()
    
    print("\n✅ TNXP data cleanup and backfill process completed!")

if __name__ == "__main__":
    asyncio.run(main()) 