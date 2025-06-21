import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from typing import List

import websockets
from dotenv import load_dotenv

from tradebot.common.bus import MessageBus
from tradebot.common.models import PriceTick
from tradebot.common.symbol_manager import SymbolManager

logger = logging.getLogger("alpaca_market_data")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

load_dotenv()

ALPACA_KEY = os.getenv("ALPACA_KEY")
ALPACA_SECRET = os.getenv("ALPACA_SECRET")

def check_credentials():
    """Check if Alpaca credentials are available."""
    if not (ALPACA_KEY and ALPACA_SECRET):
        logger.error("ALPACA_KEY and ALPACA_SECRET environment variables are required")
        logger.info("Please set your Alpaca credentials in .env file:")
        logger.info("ALPACA_KEY=your_alpaca_key_here")
        logger.info("ALPACA_SECRET=your_alpaca_secret_here")
        return False
    return True

# Configuration
SYMBOL_MODE = os.getenv("SYMBOL_MODE", "custom")
LEGACY_SYMBOLS: List[str] = os.getenv("SYMBOLS", "AAPL,MSFT,AMZN,GOOG,TSLA").split(",")
MAX_WEBSOCKET_SYMBOLS = int(os.getenv("MAX_WEBSOCKET_SYMBOLS", "30"))  # Alpaca free tier limit

# Will be populated by symbol manager
SYMBOLS: List[str] = []

WS_URL = "wss://stream.data.alpaca.markets/v2/sip"


async def alpaca_stream(bus: MessageBus):
    """Connect to Alpaca SIP WebSocket, subscribe, forward trades."""
    async with websockets.connect(WS_URL, ping_interval=20, ping_timeout=10) as ws:
        # Authenticate
        auth_msg = {"action": "auth", "key": ALPACA_KEY, "secret": ALPACA_SECRET}
        await ws.send(json.dumps(auth_msg))
        auth_resp = await ws.recv()
        auth_data = json.loads(auth_resp)
        logger.info("Alpaca auth response: %s", auth_data)
        
        # Check if auth was successful
        if not any(msg.get("T") == "success" for msg in auth_data):
            logger.error("Authentication failed: %s", auth_data)
            return

        # Subscribe to trades for selected symbols (limited by WebSocket capacity)
        websocket_symbols = SYMBOLS[:MAX_WEBSOCKET_SYMBOLS]
        if len(SYMBOLS) > MAX_WEBSOCKET_SYMBOLS:
            logger.warning("Limited to %d symbols for WebSocket (total: %d)", 
                         MAX_WEBSOCKET_SYMBOLS, len(SYMBOLS))
        
        sub_msg = {"action": "subscribe", "trades": websocket_symbols}
        await ws.send(json.dumps(sub_msg))
        logger.info("Subscribed to trades for %d symbols: %s", 
                   len(websocket_symbols), websocket_symbols[:5])

        # Listen for messages
        message_count = 0
        async for raw in ws:
            try:
                messages = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("Could not decode message: %s", raw)
                continue

            # Log all message types for debugging
            for msg in messages:
                msg_type = msg.get("T", "unknown")
                if msg_type == "t":  # trade
                    symbol = msg["S"]
                    price = msg["p"]
                    ts_ns = msg["t"]
                    timestamp = datetime.fromtimestamp(ts_ns / 1e9, tz=timezone.utc)

                    tick = PriceTick(symbol=symbol, price=price, timestamp=timestamp)
                    await bus.publish("price.ticks", tick)
                    logger.info("Trade: %s @ $%.2f", symbol, price)
                    message_count += 1
                elif msg_type == "success":
                    logger.info("Success: %s", msg.get("msg", ""))
                elif msg_type == "subscription":
                    logger.info("Subscription confirmed: %s", msg)
                else:
                    logger.debug("Other message: %s", msg)
            
            # Log activity every 100 messages
            if message_count > 0 and message_count % 100 == 0:
                logger.info("Processed %d trades so far", message_count)


async def initialize_symbols():
    """Initialize symbols based on configuration."""
    global SYMBOLS
    
    if SYMBOL_MODE == "custom":
        SYMBOLS = LEGACY_SYMBOLS
        logger.info("Using custom symbols: %s", SYMBOLS)
    else:
        # Use symbol manager to get symbols
        manager = SymbolManager()
        all_symbols = await manager.initialize(SYMBOL_MODE)
        
        # For WebSocket, we need to be more conservative with symbol count
        max_symbols = min(len(all_symbols), MAX_WEBSOCKET_SYMBOLS * 2)  # Buffer for rotation
        SYMBOLS = all_symbols[:max_symbols]
        logger.info("Loaded %d symbols in '%s' mode (WebSocket will use %d)", 
                   len(SYMBOLS), SYMBOL_MODE, min(len(SYMBOLS), MAX_WEBSOCKET_SYMBOLS))


async def main():
    if not check_credentials():
        logger.error("Cannot start Alpaca service without credentials")
        return
    
    # Initialize symbols first
    await initialize_symbols()
    
    bus = MessageBus()
    await bus.connect()

    while True:
        try:
            await alpaca_stream(bus)
        except Exception as exc:
            logger.exception("Alpaca stream crashed: %s", exc)
            logger.info("Reconnecting in 5 seconds…")
            await asyncio.sleep(5)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Alpaca market data service stopped") 