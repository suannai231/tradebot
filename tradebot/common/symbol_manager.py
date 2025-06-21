"""
Symbol Manager Service

Fetches and manages all tradeable US stocks from Alpaca API.
Provides filtering, caching, and symbol list management capabilities.
"""

import asyncio
import logging
import os
import json
from datetime import datetime, timedelta
from typing import List, Dict, Set, Optional
from pathlib import Path

import aiohttp
import asyncpg
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("symbol_manager")

ALPACA_KEY = os.getenv("ALPACA_KEY")
ALPACA_SECRET = os.getenv("ALPACA_SECRET")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/tradebot")

# Alpaca API endpoints
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"  # Use paper for testing
ASSETS_ENDPOINT = f"{ALPACA_BASE_URL}/v2/assets"

# Symbol filtering criteria
MIN_PRICE = 1.0  # Minimum stock price to avoid penny stocks
MAX_PRICE = 1000.0  # Maximum price to avoid extreme outliers
MIN_MARKET_CAP = 100_000_000  # $100M minimum market cap (if available)

# Cache settings
CACHE_FILE = Path("symbols_cache.json")
CACHE_DURATION_HOURS = 24  # Refresh symbol list daily


class SymbolManager:
    """Manages tradeable US stock symbols with filtering and caching."""
    
    def __init__(self):
        self.all_symbols: List[Dict] = []
        self.filtered_symbols: List[str] = []
        self.symbol_info: Dict[str, Dict] = {}
        
    async def fetch_all_symbols(self) -> List[Dict]:
        """Fetch all assets from Alpaca API."""
        if not (ALPACA_KEY and ALPACA_SECRET):
            logger.warning("Alpaca credentials not found, using cached symbols")
            return await self.load_cached_symbols()
            
        headers = {
            "APCA-API-KEY-ID": ALPACA_KEY,
            "APCA-API-SECRET-KEY": ALPACA_SECRET,
        }
        
        params = {
            "status": "active",
            "asset_class": "us_equity",
            "exchange": "NASDAQ,NYSE,AMEX",  # Major US exchanges
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(ASSETS_ENDPOINT, headers=headers, params=params) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error("Failed to fetch symbols: %s %s", response.status, error_text)
                        return await self.load_cached_symbols()
                    
                    data = await response.json()
                    logger.info("Fetched %d total assets from Alpaca", len(data))
                    
                    # Cache the results
                    await self.save_cached_symbols(data)
                    return data
                    
            except Exception as e:
                logger.error("Error fetching symbols: %s", e)
                return await self.load_cached_symbols()
    
    async def load_cached_symbols(self) -> List[Dict]:
        """Load symbols from cache file."""
        if not CACHE_FILE.exists():
            logger.warning("No symbol cache found, using default symbols")
            return []
            
        try:
            with open(CACHE_FILE, 'r') as f:
                cache_data = json.load(f)
                
            # Check if cache is still valid
            cached_time = datetime.fromisoformat(cache_data.get("timestamp", "2020-01-01"))
            if datetime.now() - cached_time > timedelta(hours=CACHE_DURATION_HOURS):
                logger.info("Symbol cache expired, will refresh")
                return []
                
            symbols = cache_data.get("symbols", [])
            logger.info("Loaded %d symbols from cache", len(symbols))
            return symbols
            
        except Exception as e:
            logger.error("Error loading cached symbols: %s", e)
            return []
    
    async def save_cached_symbols(self, symbols: List[Dict]):
        """Save symbols to cache file."""
        try:
            cache_data = {
                "timestamp": datetime.now().isoformat(),
                "symbols": symbols,
                "count": len(symbols)
            }
            
            with open(CACHE_FILE, 'w') as f:
                json.dump(cache_data, f, indent=2)
                
            logger.info("Cached %d symbols to %s", len(symbols), CACHE_FILE)
            
        except Exception as e:
            logger.error("Error saving symbol cache: %s", e)
    
    def filter_symbols(self, symbols: List[Dict]) -> List[str]:
        """Filter symbols based on trading criteria."""
        filtered = []
        
        for asset in symbols:
            symbol = asset.get("symbol", "")
            name = asset.get("name", "")
            tradable = asset.get("tradable", False)
            status = asset.get("status", "")
            asset_class = asset.get("class", "")
            exchange = asset.get("exchange", "")
            
            # Basic filtering criteria
            if not all([
                symbol,
                tradable,
                status == "active",
                asset_class == "us_equity",
                exchange in ["NASDAQ", "NYSE", "AMEX"],
                len(symbol) <= 5,  # Avoid complex symbols
                not any(char in symbol for char in ['.', '-', '^']),  # Avoid special symbols
            ]):
                continue
                
            # Additional filtering
            if any([
                symbol.endswith('W'),  # Warrants
                symbol.endswith('U'),  # Units
                symbol.endswith('R'),  # Rights
                'TEST' in symbol.upper(),
                'SPAC' in name.upper(),
            ]):
                continue
                
            filtered.append(symbol)
            self.symbol_info[symbol] = {
                "name": name,
                "exchange": exchange,
                "tradable": tradable,
                "class": asset_class
            }
        
        logger.info("Filtered to %d tradeable symbols", len(filtered))
        return sorted(filtered)
    
    async def get_popular_symbols(self, limit: int = 100) -> List[str]:
        """Get most popular/liquid symbols for testing."""
        # Popular large-cap stocks for initial testing
        popular = [
            # Tech Giants
            "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "TSLA", "NVDA",
            # Financial
            "JPM", "BAC", "WFC", "GS", "MS", "C", "BRK.B",
            # Healthcare
            "JNJ", "PFE", "UNH", "ABBV", "MRK", "TMO", "ABT",
            # Consumer
            "KO", "PEP", "WMT", "HD", "MCD", "NKE", "SBUX",
            # Industrial
            "BA", "CAT", "GE", "MMM", "HON", "UPS", "RTX",
            # Energy
            "XOM", "CVX", "COP", "SLB", "EOG", "KMI",
            # Telecom
            "VZ", "T", "TMUS", "CMCSA",
            # Retail
            "AMZN", "TGT", "COST", "LOW", "TJX",
        ]
        
        # Remove duplicates and limit
        return list(dict.fromkeys(popular))[:limit]
    
    async def get_symbols_by_market_cap(self, tier: str = "large") -> List[str]:
        """Get symbols filtered by market cap tier."""
        # This would require additional API calls to get market cap data
        # For now, return popular symbols as proxy for large cap
        if tier == "large":
            return await self.get_popular_symbols(50)
        elif tier == "mid":
            return await self.get_popular_symbols(100)[50:100]
        elif tier == "small":
            return await self.get_popular_symbols(200)[100:200]
        else:
            return await self.get_popular_symbols(200)
    
    async def initialize(self, mode: str = "popular") -> List[str]:
        """Initialize symbol manager and return symbol list based on mode."""
        logger.info("Initializing symbol manager in '%s' mode", mode)
        
        if mode == "all":
            # Fetch all tradeable symbols
            all_assets = await self.fetch_all_symbols()
            self.all_symbols = all_assets
            self.filtered_symbols = self.filter_symbols(all_assets)
            return self.filtered_symbols
            
        elif mode == "popular":
            # Use curated list of popular symbols
            return await self.get_popular_symbols(100)
            
        elif mode in ["large", "mid", "small"]:
            # Use market cap tiers
            return await self.get_symbols_by_market_cap(mode)
            
        else:
            # Default to popular symbols
            logger.warning("Unknown mode '%s', using 'popular'", mode)
            return await self.get_popular_symbols(100)
    
    async def store_symbols_in_db(self, pool: asyncpg.Pool):
        """Store symbol information in database for reference."""
        if not self.symbol_info:
            return
            
        async with pool.acquire() as conn:
            # Create symbols table if not exists
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS symbols (
                    symbol TEXT PRIMARY KEY,
                    name TEXT,
                    exchange TEXT,
                    asset_class TEXT,
                    tradable BOOLEAN,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                );
            """)
            
            # Upsert symbol information
            for symbol, info in self.symbol_info.items():
                await conn.execute("""
                    INSERT INTO symbols (symbol, name, exchange, asset_class, tradable, updated_at)
                    VALUES ($1, $2, $3, $4, $5, NOW())
                    ON CONFLICT (symbol) DO UPDATE SET
                        name = EXCLUDED.name,
                        exchange = EXCLUDED.exchange,
                        asset_class = EXCLUDED.asset_class,
                        tradable = EXCLUDED.tradable,
                        updated_at = EXCLUDED.updated_at
                """, symbol, info["name"], info["exchange"], info["class"], info["tradable"])
        
        logger.info("Stored %d symbols in database", len(self.symbol_info))


async def main():
    """Test the symbol manager."""
    manager = SymbolManager()
    
    # Test different modes
    for mode in ["popular", "large", "all"]:
        print(f"\n--- Testing {mode.upper()} mode ---")
        symbols = await manager.initialize(mode)
        print(f"Got {len(symbols)} symbols")
        print(f"First 10: {symbols[:10]}")
        
        if mode == "all":
            break  # Don't test all modes if we have API limits


if __name__ == "__main__":
    asyncio.run(main()) 