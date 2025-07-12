"""
Yahoo Finance Market Data Service

Real-time market data service using Yahoo Finance API.
Provides comprehensive volume data and real-time price updates.
"""

import asyncio
import logging
import os
import aiohttp
import json
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional

from tradebot.common.bus import MessageBus
from tradebot.common.models import PriceTick
from tradebot.common.symbol_manager import SymbolManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("yahoo_market_data")

# Configuration
SYMBOL_MODE = os.getenv("SYMBOL_MODE", "custom")
LEGACY_SYMBOLS: List[str] = os.getenv("SYMBOLS", "AAPL,MSFT,AMZN,GOOG,TSLA,TNXP").split(",")
MAX_SYMBOLS = int(os.getenv("MAX_SYMBOLS", "500"))
FETCH_INTERVAL = float(os.getenv("YAHOO_FETCH_INTERVAL", "10.0"))  # Fetch every 10 seconds

# Will be populated by symbol manager
SYMBOLS: List[str] = []

# Yahoo Finance API URLs
YAHOO_BASE_URL = "https://query1.finance.yahoo.com/v8/finance/chart"
YAHOO_QUOTE_URL = "https://query1.finance.yahoo.com/v7/finance/quote"

async def send_heartbeat():
    """Send periodic heartbeat to Redis for service monitoring."""
    import redis.asyncio as redis
    redis_client = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))
    
    while True:
        try:
            await redis_client.set(
                "service:yahoo_market_data:heartbeat", 
                datetime.now(timezone.utc).isoformat()
            )
            await redis_client.set(
                "service:yahoo_market_data:symbols", 
                len(SYMBOLS)
            )
        except Exception as e:
            logger.warning(f"Failed to send heartbeat: {e}")
        
        await asyncio.sleep(30)  # Every 30 seconds


async def fetch_yahoo_data(session: aiohttp.ClientSession, symbols: List[str]) -> List[Dict]:
    """Fetch real-time data from Yahoo Finance for multiple symbols."""
    if not symbols:
        return []
    
    # Split into chunks to avoid URL length limits
    chunk_size = 50  # Yahoo Finance can handle multiple symbols
    all_data = []
    
    for i in range(0, len(symbols), chunk_size):
        chunk = symbols[i:i + chunk_size]
        symbol_string = ",".join(chunk)
        
        try:
            # Use quote API for real-time data including volume
            url = f"{YAHOO_QUOTE_URL}?symbols={symbol_string}"
            
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            
            async with session.get(url, headers=headers, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    quote_response = data.get("quoteResponse", {})
                    quotes = quote_response.get("result", [])
                    
                    for quote in quotes:
                        if quote and isinstance(quote, dict):
                            all_data.append(quote)
                            
                    logger.debug(f"Fetched {len(quotes)} quotes for chunk: {chunk[:3]}...")
                else:
                    logger.warning(f"Yahoo Finance API error {response.status} for symbols: {chunk[:3]}...")
                    
        except Exception as e:
            logger.error(f"Error fetching Yahoo data for chunk {chunk[:3]}: {e}")
            continue
            
        # Rate limiting
        await asyncio.sleep(0.1)
    
    return all_data


def parse_yahoo_quote(quote: Dict) -> Optional[PriceTick]:
    """Parse Yahoo Finance quote data into PriceTick."""
    try:
        symbol = quote.get("symbol", "").upper()
        if not symbol:
            return None
        
        # Current price data
        current_price = quote.get("regularMarketPrice")
        if current_price is None:
            return None
            
        # OHLC data
        open_price = quote.get("regularMarketOpen", current_price)
        high_price = quote.get("regularMarketDayHigh", current_price)
        low_price = quote.get("regularMarketDayLow", current_price)
        close_price = quote.get("regularMarketPreviousClose", current_price)
        
        # Volume data (this is the key improvement over Alpaca IEX!)
        volume = quote.get("regularMarketVolume", 0)
        
        # Market time - use current time if market time not available
        market_time = quote.get("regularMarketTime")
        if market_time:
            timestamp = datetime.fromtimestamp(market_time, tz=timezone.utc)
        else:
            timestamp = datetime.now(tz=timezone.utc)
        
        # Additional data
        vwap = quote.get("averageDailyVolume10Day")  # Use as proxy for VWAP
        if vwap:
            vwap = current_price  # Simplified VWAP calculation
            
        return PriceTick(
            symbol=symbol,
            price=float(current_price),
            timestamp=timestamp,
            open=float(open_price) if open_price else None,
            high=float(high_price) if high_price else None,
            low=float(low_price) if low_price else None,
            close=float(current_price),  # Use current as close for real-time
            volume=int(volume) if volume else 0,
            trade_count=None,  # Yahoo doesn't provide trade count
            vwap=float(vwap) if vwap else None
        )
        
    except (ValueError, TypeError, KeyError) as e:
        logger.warning(f"Error parsing Yahoo quote for {quote.get('symbol', 'UNKNOWN')}: {e}")
        return None


async def yahoo_data_loop(bus: MessageBus):
    """Main loop to fetch and publish Yahoo Finance data."""
    logger.info(f"Starting Yahoo Finance data loop for {len(SYMBOLS)} symbols")
    
    async with aiohttp.ClientSession() as session:
        while True:
            try:
                start_time = datetime.now()
                
                # Fetch data for all symbols
                quotes = await fetch_yahoo_data(session, SYMBOLS)
                
                if quotes:
                    # Parse and publish each quote
                    published_count = 0
                    for quote in quotes:
                        tick = parse_yahoo_quote(quote)
                        if tick:
                            await bus.publish("price.ticks", tick)
                            published_count += 1
                    
                    duration = (datetime.now() - start_time).total_seconds()
                    logger.info(f"Published {published_count}/{len(quotes)} ticks in {duration:.2f}s")
                else:
                    logger.warning("No quotes received from Yahoo Finance")
                
            except Exception as e:
                logger.error(f"Error in Yahoo data loop: {e}")
            
            # Wait for next fetch interval
            await asyncio.sleep(FETCH_INTERVAL)


async def initialize_symbols():
    """Initialize symbols based on configuration."""
    global SYMBOLS
    
    if SYMBOL_MODE == "custom":
        SYMBOLS = [s.strip().upper() for s in LEGACY_SYMBOLS if s.strip()]
        logger.info(f"Using custom symbols: {SYMBOLS}")
    else:
        # Use symbol manager to get symbols
        manager = SymbolManager()
        all_symbols = await manager.initialize(SYMBOL_MODE)
        
        # Limit symbols for Yahoo Finance (to avoid rate limiting)
        SYMBOLS = all_symbols[:MAX_SYMBOLS]
        logger.info(f"Loaded {len(SYMBOLS)} symbols in '{SYMBOL_MODE}' mode")
    
    # Ensure TNXP is included for testing
    if "TNXP" not in SYMBOLS:
        SYMBOLS.append("TNXP")
        logger.info("Added TNXP for volume testing")


async def main():
    """Main service entry point."""
    logger.info("🚀 Yahoo Finance market data service starting...")
    
    # Initialize symbols
    await initialize_symbols()
    
    if not SYMBOLS:
        logger.error("No symbols configured, exiting")
        return
    
    # Connect to message bus
    bus = MessageBus()
    await bus.connect()
    logger.info("Connected to message bus")
    
    # Start heartbeat task
    heartbeat_task = asyncio.create_task(send_heartbeat())
    logger.info("Started heartbeat task")
    
    # Test Yahoo Finance API connectivity
    try:
        async with aiohttp.ClientSession() as session:
            test_quotes = await fetch_yahoo_data(session, SYMBOLS[:5])
            if test_quotes:
                logger.info(f"✅ Yahoo Finance API test successful - {len(test_quotes)} quotes received")
                for quote in test_quotes[:3]:
                    symbol = quote.get("symbol", "?")
                    price = quote.get("regularMarketPrice", "?")
                    volume = quote.get("regularMarketVolume", "?")
                    logger.info(f"   {symbol}: ${price}, Volume: {volume:,}")
            else:
                logger.warning("⚠️ Yahoo Finance API test returned no data")
    except Exception as e:
        logger.error(f"❌ Yahoo Finance API test failed: {e}")
        return
    
    logger.info(f"✅ Yahoo Finance service ready for {len(SYMBOLS)} symbols")
    logger.info(f"📊 Fetch interval: {FETCH_INTERVAL}s")
    
    try:
        # Start main data loop
        await yahoo_data_loop(bus)
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    except Exception as e:
        logger.error(f"Service error: {e}")
    finally:
        heartbeat_task.cancel()
        await bus.close()
        logger.info("Yahoo Finance service stopped")


if __name__ == "__main__":
    asyncio.run(main()) 