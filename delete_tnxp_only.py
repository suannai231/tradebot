#!/usr/bin/env python3
"""
Delete TNXP data only - no backfill.

This script will delete all TNXP rows from the price_ticks table
without re-backfilling the data.
"""

import asyncio
import asyncpg
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

async def delete_tnxp_only():
    """Delete all TNXP data from the database without re-backfilling."""
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
                
                # Verify deletion
                remaining_count = await conn.fetchval(
                    "SELECT COUNT(*) FROM price_ticks WHERE symbol = $1",
                    "TNXP"
                )
                
                if remaining_count == 0:
                    print("✅ TNXP data successfully deleted!")
                else:
                    print(f"⚠️  Warning: {remaining_count} TNXP rows still remain")
            else:
                print("ℹ️  No TNXP data found in database")
        
        await pool.close()
        
    except Exception as e:
        print(f"❌ Error deleting TNXP data: {e}")
        return False
    
    return True

async def main():
    """Main execution function."""
    print("🚀 Starting TNXP data deletion...")
    
    # Perform deletion
    if await delete_tnxp_only():
        print("\n✅ TNXP data deletion completed!")
    else:
        print("\n❌ TNXP data deletion failed!")

if __name__ == "__main__":
    asyncio.run(main()) 