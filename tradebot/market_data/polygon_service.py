import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from typing import List

import websockets
from dotenv import load_dotenv

load_dotenv()

from tradebot.common.bus import MessageBus
from tradebot.common.models import PriceTick

logger = logging.getLogger("polygon_market_data")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
if not POLYGON_API_KEY:
    raise RuntimeError("POLYGON_API_KEY environment variable not set")

SYMBOLS: List[str] = os.getenv("SYMBOLS", "AAPL,MSFT,AMZN,GOOG,TSLA").split(",")

WS_URL = "wss://socket.polygon.io/stocks"


async def polygon_stream(bus: MessageBus):
    """Connect to Polygon WebSocket, subscribe to trades, and publish ticks."""
    async with websockets.connect(WS_URL) as ws:
        # Authenticate
        await ws.send(json.dumps({"action": "auth", "params": POLYGON_API_KEY}))
        auth_resp = await ws.recv()
        logger.info("Polygon auth response: %s", auth_resp)

        # Subscribe to trade events for symbols
        params = ",".join(f"T.{sym}" for sym in SYMBOLS)
        await ws.send(json.dumps({"action": "subscribe", "params": params}))
        logger.info("Subscribed to %s", params)

        async for raw in ws:
            try:
                messages = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("Could not decode message: %s", raw)
                continue

            # Each message is a dict with keys depending on event type
            for msg in messages:
                if msg.get("ev") != "T":
                    continue  # Ignore non-trade events
                symbol = msg["sym"]
                price = msg["p"]
                ts_ns = msg["t"]
                timestamp = datetime.fromtimestamp(ts_ns / 1e9, tz=timezone.utc)

                tick = PriceTick(symbol=symbol, price=price, timestamp=timestamp)
                await bus.publish("price.ticks", tick)
                logger.debug("Tick %s", tick)


async def main():
    bus = MessageBus()
    await bus.connect()
    while True:
        try:
            await polygon_stream(bus)
        except Exception as exc:
            logger.exception("Polygon stream crashed: %s", exc)
            logger.info("Reconnecting in 5 seconds…")
            await asyncio.sleep(5)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Polygon market data service stopped") 