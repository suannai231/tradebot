#!/usr/bin/env python3
"""
Database Cleanup Script

This script will delete all price ticks from the database.
Use with caution - this will permanently remove all historical data.
"""

import asyncio
import asyncpg
import os
import sys
from datetime import datetime, timezone

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

async def cleanup_database():
    """Delete all price ticks from the database."""
    database_url = "postgresql://postgres:password@localhost:5432/tradebot"
    
    print("🧹 Starting database cleanup...")
    
    try:
        pool = await asyncpg.create_pool(database_url)
        
        async with pool.acquire() as conn:
            # Count existing rows
            count_result = await conn.fetchval(
                "SELECT COUNT(*) FROM price_ticks"
            )
            
            print(f"Found {count_result} price ticks in database")
            
            if count_result > 0:
                # Delete all rows
                deleted_count = await conn.execute(
                    "DELETE FROM price_ticks"
                )
                
                print(f"✅ Deleted {deleted_count} price ticks from database")
                
                # Verify deletion
                remaining_count = await conn.fetchval(
                    "SELECT COUNT(*) FROM price_ticks"
                )
                
                print(f"Remaining price ticks: {remaining_count}")
                
                if remaining_count == 0:
                    print("✅ Database cleanup completed successfully!")
                else:
                    print(f"⚠️  Warning: {remaining_count} rows still remain")
            else:
                print("ℹ️  No price ticks found in database")
        
        await pool.close()
        
    except Exception as e:
        print(f"❌ Error during database cleanup: {e}")
        return False
    
    return True

async def main():
    """Main execution function."""
    print("🚀 Starting database cleanup process...")
    
    # Confirm with user
    response = input("Are you sure you want to delete ALL price ticks from the database? (yes/no): ")
    
    if response.lower() != 'yes':
        print("❌ Database cleanup cancelled by user")
        return
    
    # Perform cleanup
    if await cleanup_database():
        print("\n✅ Database cleanup process completed!")
    else:
        print("\n❌ Database cleanup failed!")

if __name__ == "__main__":
    asyncio.run(main()) 