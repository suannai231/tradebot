#!/usr/bin/env python3
"""
Data Source Management Utility

Utility for managing separate data source tables in the trading bot system.
Helps with cleaning, migrating, and analyzing data from different sources.
"""

import asyncio
import asyncpg
import argparse
import os
import sys
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/tradebot")

class DataSourceManager:
    """Manages separate data source tables"""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.pool = None
        
        # Available data sources and their tables
        self.data_sources = {
            'synthetic': 'price_ticks_synthetic',
            'alpaca': 'price_ticks_alpaca',
            'polygon': 'price_ticks_polygon',
            'yahoo': 'price_ticks_yahoo'
        }
    
    async def connect(self):
        """Connect to database"""
        self.pool = await asyncpg.create_pool(self.database_url)
        print(f"✅ Connected to database")
    
    async def close(self):
        """Close database connection"""
        if self.pool:
            await self.pool.close()
    
    async def list_tables(self):
        """List all data source tables and their stats"""
        print("\n📊 DATA SOURCE TABLES")
        print("=" * 60)
        
        async with self.pool.acquire() as conn:
            for source, table_name in self.data_sources.items():
                try:
                    # Check if table exists
                    exists = await conn.fetchval(
                        "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = $1)",
                        table_name
                    )
                    
                    if not exists:
                        print(f"❌ {source:12} | {table_name:25} | Not created")
                        continue
                    
                    # Get table stats
                    stats = await conn.fetchrow(f"""
                        SELECT 
                            COUNT(*) as total_rows,
                            COUNT(DISTINCT symbol) as unique_symbols,
                            MIN(timestamp) as earliest_data,
                            MAX(timestamp) as latest_data
                        FROM {table_name}
                    """)
                    
                    if stats['total_rows'] > 0:
                        print(f"✅ {source:12} | {table_name:25} | {stats['total_rows']:,} rows | {stats['unique_symbols']} symbols")
                        print(f"   {'':12} | {'':25} | {stats['earliest_data']} to {stats['latest_data']}")
                    else:
                        print(f"⚪ {source:12} | {table_name:25} | Empty")
                        
                except Exception as e:
                    print(f"❌ {source:12} | {table_name:25} | Error: {e}")
    
    async def clean_table(self, source: str, confirm: bool = False):
        """Clean (empty) a specific data source table"""
        if source not in self.data_sources:
            print(f"❌ Unknown data source: {source}")
            return
        
        table_name = self.data_sources[source]
        
        async with self.pool.acquire() as conn:
            # Check if table exists and has data
            exists = await conn.fetchval(
                "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = $1)",
                table_name
            )
            
            if not exists:
                print(f"❌ Table {table_name} does not exist")
                return
            
            row_count = await conn.fetchval(f"SELECT COUNT(*) FROM {table_name}")
            
            if row_count == 0:
                print(f"⚪ Table {table_name} is already empty")
                return
            
            print(f"⚠️  Table {table_name} contains {row_count:,} rows")
            
            if not confirm:
                response = input(f"Are you sure you want to delete all data from {table_name}? (yes/no): ")
                if response.lower() != 'yes':
                    print("❌ Operation cancelled")
                    return
            
            # Clean the table
            await conn.execute(f"DELETE FROM {table_name}")
            print(f"✅ Cleaned {row_count:,} rows from {table_name}")
    
    async def migrate_data(self, from_source: str, to_source: str, symbol: str = None):
        """Migrate data from one source table to another"""
        if from_source not in self.data_sources or to_source not in self.data_sources:
            print(f"❌ Invalid source. Available: {list(self.data_sources.keys())}")
            return
        
        from_table = self.data_sources[from_source]
        to_table = self.data_sources[to_source]
        
        async with self.pool.acquire() as conn:
            # Build query
            if symbol:
                query = f"""
                    INSERT INTO {to_table} (symbol, price, timestamp, open_price, high_price, low_price, close_price, volume, trade_count, vwap)
                    SELECT symbol, price, timestamp, open_price, high_price, low_price, close_price, volume, trade_count, vwap
                    FROM {from_table}
                    WHERE symbol = $1
                    ON CONFLICT DO NOTHING
                """
                result = await conn.execute(query, symbol.upper())
            else:
                query = f"""
                    INSERT INTO {to_table} (symbol, price, timestamp, open_price, high_price, low_price, close_price, volume, trade_count, vwap)
                    SELECT symbol, price, timestamp, open_price, high_price, low_price, close_price, volume, trade_count, vwap
                    FROM {from_table}
                    ON CONFLICT DO NOTHING
                """
                result = await conn.execute(query)
            
            # Parse result to get row count
            rows_affected = int(result.split()[-1])
            
            if symbol:
                print(f"✅ Migrated {rows_affected:,} rows for {symbol} from {from_source} to {to_source}")
            else:
                print(f"✅ Migrated {rows_affected:,} rows from {from_source} to {to_source}")
    
    async def compare_sources(self, source1: str, source2: str, symbol: str = None):
        """Compare data between two sources"""
        if source1 not in self.data_sources or source2 not in self.data_sources:
            print(f"❌ Invalid source. Available: {list(self.data_sources.keys())}")
            return
        
        table1 = self.data_sources[source1]
        table2 = self.data_sources[source2]
        
        print(f"\n📊 COMPARING {source1.upper()} vs {source2.upper()}")
        print("=" * 60)
        
        async with self.pool.acquire() as conn:
            if symbol:
                query1 = f"SELECT COUNT(*) FROM {table1} WHERE symbol = $1"
                query2 = f"SELECT COUNT(*) FROM {table2} WHERE symbol = $1"
                count1 = await conn.fetchval(query1, symbol.upper())
                count2 = await conn.fetchval(query2, symbol.upper())
                
                print(f"Symbol: {symbol}")
                print(f"{source1}: {count1:,} rows")
                print(f"{source2}: {count2:,} rows")
                print(f"Difference: {abs(count1 - count2):,} rows")
            else:
                # Compare overall stats
                stats1 = await conn.fetchrow(f"""
                    SELECT 
                        COUNT(*) as total_rows,
                        COUNT(DISTINCT symbol) as unique_symbols,
                        MIN(timestamp) as earliest_data,
                        MAX(timestamp) as latest_data
                    FROM {table1}
                """)
                
                stats2 = await conn.fetchrow(f"""
                    SELECT 
                        COUNT(*) as total_rows,
                        COUNT(DISTINCT symbol) as unique_symbols,
                        MIN(timestamp) as earliest_data,
                        MAX(timestamp) as latest_data
                    FROM {table2}
                """)
                
                print(f"{'Metric':<20} | {source1:<15} | {source2:<15}")
                print("-" * 60)
                print(f"{'Total Rows':<20} | {stats1['total_rows']:,:<15} | {stats2['total_rows']:,:<15}")
                print(f"{'Unique Symbols':<20} | {stats1['unique_symbols']:<15} | {stats2['unique_symbols']:<15}")
                print(f"{'Earliest Data':<20} | {str(stats1['earliest_data']):<15} | {str(stats2['earliest_data']):<15}")
                print(f"{'Latest Data':<20} | {str(stats1['latest_data']):<15} | {str(stats2['latest_data']):<15}")
    
    async def create_tables(self):
        """Create all data source tables"""
        print("🔨 Creating data source tables...")
        
        async with self.pool.acquire() as conn:
            for source, table_name in self.data_sources.items():
                try:
                    # Create table
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
                    
                    # Create hypertable (TimescaleDB)
                    try:
                        await conn.execute(f"""
                            SELECT create_hypertable('{table_name}', 'timestamp', if_not_exists => TRUE);
                        """)
                    except:
                        pass  # Ignore if not TimescaleDB
                    
                    # Create index
                    await conn.execute(f"""
                        CREATE INDEX IF NOT EXISTS idx_{table_name}_symbol_time 
                        ON {table_name} (symbol, timestamp DESC);
                    """)
                    
                    print(f"✅ Created table: {table_name}")
                    
                except Exception as e:
                    print(f"❌ Error creating {table_name}: {e}")


async def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description="Manage data source tables")
    parser.add_argument("command", choices=["list", "clean", "migrate", "compare", "create"], 
                       help="Command to execute")
    parser.add_argument("--source", help="Source data table")
    parser.add_argument("--target", help="Target data table (for migrate)")
    parser.add_argument("--symbol", help="Specific symbol to operate on")
    parser.add_argument("--confirm", action="store_true", help="Skip confirmation prompts")
    
    args = parser.parse_args()
    
    manager = DataSourceManager(DATABASE_URL)
    
    try:
        await manager.connect()
        
        if args.command == "list":
            await manager.list_tables()
        
        elif args.command == "clean":
            if not args.source:
                print("❌ --source required for clean command")
                return
            await manager.clean_table(args.source, args.confirm)
        
        elif args.command == "migrate":
            if not args.source or not args.target:
                print("❌ --source and --target required for migrate command")
                return
            await manager.migrate_data(args.source, args.target, args.symbol)
        
        elif args.command == "compare":
            if not args.source or not args.target:
                print("❌ --source and --target required for compare command")
                return
            await manager.compare_sources(args.source, args.target, args.symbol)
        
        elif args.command == "create":
            await manager.create_tables()
        
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        await manager.close()


if __name__ == "__main__":
    asyncio.run(main()) 