#!/usr/bin/env python3
"""
Symbol Management Script

Utility script to manage and test the expanded trading bot symbol system.
Provides commands to fetch symbols, test configurations, and manage the system.
"""

import asyncio
import argparse
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from tradebot.common.symbol_manager import SymbolManager
from tradebot.common.config import get_config


async def list_symbols(mode: str, limit: int = None):
    """List symbols for a given mode."""
    print(f"\n=== Listing symbols for mode: {mode} ===")
    
    manager = SymbolManager()
    symbols = await manager.initialize(mode)
    
    if limit:
        symbols = symbols[:limit]
    
    print(f"Total symbols: {len(symbols)}")
    print(f"Symbols: {', '.join(symbols)}")
    
    if manager.symbol_info:
        print(f"\nSample symbol info:")
        for symbol in symbols[:5]:
            if symbol in manager.symbol_info:
                info = manager.symbol_info[symbol]
                print(f"  {symbol}: {info['name']} ({info['exchange']})")


async def test_configuration():
    """Test current configuration."""
    print("\n=== Testing Configuration ===")
    
    config = get_config()
    config.print_configuration()
    
    # Test symbol loading
    print(f"\n=== Testing Symbol Loading ===")
    manager = SymbolManager()
    
    try:
        symbols = await manager.initialize(config.symbol_mode)
        print(f"✅ Successfully loaded {len(symbols)} symbols")
        print(f"First 10: {symbols[:10]}")
        
        if len(symbols) > config.max_websocket_symbols:
            print(f"⚠️  Note: Only {config.max_websocket_symbols} symbols will be used for WebSocket feeds")
        
    except Exception as e:
        print(f"❌ Failed to load symbols: {e}")


async def fetch_all_symbols():
    """Fetch and cache all available symbols from Alpaca."""
    print("\n=== Fetching All Symbols from Alpaca ===")
    
    manager = SymbolManager()
    
    try:
        # Force fetch from API
        all_assets = await manager.fetch_all_symbols()
        
        if all_assets:
            filtered = manager.filter_symbols(all_assets)
            print(f"✅ Fetched {len(all_assets)} total assets")
            print(f"✅ Filtered to {len(filtered)} tradeable symbols")
            print(f"✅ Cached to {manager.cache_file}")
            
            # Show sample
            print(f"\nSample symbols: {filtered[:20]}")
            
        else:
            print("❌ Failed to fetch symbols from Alpaca API")
            
    except Exception as e:
        print(f"❌ Error fetching symbols: {e}")


async def benchmark_modes():
    """Benchmark different symbol modes."""
    print("\n=== Benchmarking Symbol Modes ===")
    
    modes = ["popular", "large", "mid", "small", "all"]
    manager = SymbolManager()
    
    for mode in modes:
        try:
            import time
            start_time = time.time()
            symbols = await manager.initialize(mode)
            end_time = time.time()
            
            print(f"{mode:8s}: {len(symbols):4d} symbols in {end_time - start_time:.2f}s")
            
        except Exception as e:
            print(f"{mode:8s}: ❌ Error - {e}")


async def create_sample_configs():
    """Create sample configuration files for different use cases."""
    print("\n=== Creating Sample Configuration Files ===")
    
    configs = {
        ".env.popular": {
            "SYMBOL_MODE": "popular",
            "MAX_SYMBOLS": "100",
            "MAX_WEBSOCKET_SYMBOLS": "30",
            "description": "Popular large-cap stocks (recommended for beginners)"
        },
        ".env.all_stocks": {
            "SYMBOL_MODE": "all", 
            "MAX_SYMBOLS": "1000",
            "MAX_WEBSOCKET_SYMBOLS": "50",
            "description": "All tradeable US stocks (requires good hardware)"
        },
        ".env.large_cap": {
            "SYMBOL_MODE": "large",
            "MAX_SYMBOLS": "50", 
            "MAX_WEBSOCKET_SYMBOLS": "30",
            "description": "Large-cap stocks only (stable, liquid)"
        },
        ".env.testing": {
            "SYMBOL_MODE": "custom",
            "SYMBOLS": "AAPL,MSFT,GOOGL",
            "MAX_SYMBOLS": "10",
            "MAX_WEBSOCKET_SYMBOLS": "5",
            "description": "Minimal setup for testing"
        }
    }
    
    base_config = """# Copy this file to `.env` and customize as needed

# Trading symbols configuration
SYMBOL_MODE={symbol_mode}
{symbols_line}

# Symbol limits
MAX_SYMBOLS={max_symbols}
MAX_WEBSOCKET_SYMBOLS={max_websocket_symbols}

# Database connection
DATABASE_URL=postgresql://postgres:password@localhost:5432/tradebot
REDIS_URL=redis://localhost:6379

# Rate limiting
BACKFILL_BATCH_SIZE=10
BACKFILL_CONCURRENT_REQUESTS=3
RATE_LIMIT_DELAY=0.5

# Market data
TICK_INTERVAL=1.0
ENABLE_OHLCV=true

# Strategy
STRATEGY_ENABLED=true
SMA_SHORT_WINDOW=5
SMA_LONG_WINDOW=20

# Logging
LOG_LEVEL=INFO

# API credentials (fill in your own)
ALPACA_KEY=your_alpaca_key_here
ALPACA_SECRET=your_alpaca_secret_here
POLYGON_API_KEY=your_polygon_key_here
"""
    
    for filename, config in configs.items():
        symbols_line = f"SYMBOLS={config.get('SYMBOLS', 'AAPL,MSFT,GOOGL')}" if config["SYMBOL_MODE"] == "custom" else "# SYMBOLS=AAPL,MSFT,GOOGL  # Used when SYMBOL_MODE=custom"
        
        content = base_config.format(
            symbol_mode=config["SYMBOL_MODE"],
            symbols_line=symbols_line,
            max_symbols=config["MAX_SYMBOLS"],
            max_websocket_symbols=config["MAX_WEBSOCKET_SYMBOLS"]
        )
        
        # Add description as comment
        content = f"# {config['description']}\n" + content
        
        with open(filename, 'w') as f:
            f.write(content)
        
        print(f"✅ Created {filename} - {config['description']}")


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description="Trading Bot Symbol Management")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List symbols command
    list_parser = subparsers.add_parser('list', help='List symbols for a mode')
    list_parser.add_argument('mode', choices=['popular', 'large', 'mid', 'small', 'all', 'custom'], 
                           help='Symbol mode to list')
    list_parser.add_argument('--limit', type=int, help='Limit number of symbols to show')
    
    # Test configuration command
    subparsers.add_parser('test', help='Test current configuration')
    
    # Fetch symbols command
    subparsers.add_parser('fetch', help='Fetch all symbols from Alpaca API')
    
    # Benchmark command
    subparsers.add_parser('benchmark', help='Benchmark different symbol modes')
    
    # Create sample configs command
    subparsers.add_parser('samples', help='Create sample configuration files')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Run the appropriate async function
    if args.command == 'list':
        asyncio.run(list_symbols(args.mode, args.limit))
    elif args.command == 'test':
        asyncio.run(test_configuration())
    elif args.command == 'fetch':
        asyncio.run(fetch_all_symbols())
    elif args.command == 'benchmark':
        asyncio.run(benchmark_modes())
    elif args.command == 'samples':
        asyncio.run(create_sample_configs())


if __name__ == "__main__":
    main() 