import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from typing import List

import redis.asyncio as redis

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


async def send_heartbeat():
    """Send periodic heartbeat to indicate service is alive."""
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    service_name = os.getenv("SERVICE_NAME", "alpaca_market_data")
    
    try:
        redis_client = redis.from_url(redis_url)
        heartbeat_key = f"service:{service_name}:heartbeat"
        
        while True:
            try:
                await redis_client.set(heartbeat_key, datetime.now(timezone.utc).isoformat())
                logger.debug("Heartbeat sent")
                await asyncio.sleep(30)  # Send heartbeat every 30 seconds
            except Exception as e:
                logger.warning(f"Failed to send heartbeat: {e}")
                await asyncio.sleep(30)
                
    except Exception as e:
        logger.error(f"Failed to initialize heartbeat: {e}")
    finally:
        try:
            await redis_client.close()
        except:
            pass

# Configuration
SYMBOL_MODE = os.getenv("SYMBOL_MODE", "custom")
LEGACY_SYMBOLS: List[str] = os.getenv("SYMBOLS", "AAPL,MSFT,AMZN,GOOG,TSLA").split(",")
MAX_WEBSOCKET_SYMBOLS = int(os.getenv("MAX_WEBSOCKET_SYMBOLS", "5"))  # Conservative limit for stability

# Will be populated by symbol manager
SYMBOLS: List[str] = []

# Try multiple endpoints for better compatibility  
WS_ENDPOINTS = [
    "wss://stream.data.alpaca.markets/v2/iex",   # IEX feed (free tier compatible)
    "wss://stream.data.alpaca.markets/v2/sip",   # SIP feed (premium subscription required)
]

async def alpaca_stream(bus: MessageBus):
    """Connect to Alpaca WebSocket with improved stability and error handling."""
    
    # Try different endpoints for better compatibility
    for endpoint_idx, ws_url in enumerate(WS_ENDPOINTS):
        try:
            logger.info(f"Attempting connection to {ws_url} (attempt {endpoint_idx + 1}/{len(WS_ENDPOINTS)})")
            
            # Optimized connection settings for stability
            async with websockets.connect(
                ws_url,
                ping_interval=20,      # Send ping every 20 seconds
                ping_timeout=10,       # Wait 10 seconds for pong
                close_timeout=5,       # Quick close timeout
                max_size=10**6,        # 1MB max message size
                compression=None       # Disable compression for stability
            ) as ws:
                logger.info(f"Connected to {ws_url}")
                
                # Step 1: Authenticate
                auth_msg = {"action": "auth", "key": ALPACA_KEY, "secret": ALPACA_SECRET}
                await ws.send(json.dumps(auth_msg))
                
                # Wait for auth response with timeout
                try:
                    auth_resp = await asyncio.wait_for(ws.recv(), timeout=10.0)
                    auth_data = json.loads(auth_resp)
                    logger.info("Alpaca auth response: %s", auth_data)
                    
                    # Verify authentication success
                    if not any(msg.get("T") == "success" for msg in auth_data):
                        logger.error("Authentication failed: %s", auth_data)
                        continue  # Try next endpoint
                        
                except asyncio.TimeoutError:
                    logger.error("Authentication timeout")
                    continue
                
                # Step 2: Subscribe to data
                websocket_symbols = SYMBOLS[:MAX_WEBSOCKET_SYMBOLS]
                if len(SYMBOLS) > MAX_WEBSOCKET_SYMBOLS:
                    logger.warning("Limited to %d symbols for WebSocket (total: %d)", 
                                 MAX_WEBSOCKET_SYMBOLS, len(SYMBOLS))
                
                # Subscribe to trades only to stay within 30 symbol limit
                # (Alpaca free tier: 30 symbols total across all channels)
                sub_msg = {
                    "action": "subscribe", 
                    "trades": websocket_symbols
                }
                await ws.send(json.dumps(sub_msg))
                logger.info("Subscribed to trades for %d symbols: %s", 
                           len(websocket_symbols), websocket_symbols)
                
                # Wait for subscription confirmation
                try:
                    sub_resp = await asyncio.wait_for(ws.recv(), timeout=10.0)
                    sub_data = json.loads(sub_resp)
                    logger.info("Subscription response: %s", sub_data)
                except asyncio.TimeoutError:
                    logger.error("Subscription timeout")
                    continue
                
                # Step 3: Listen for messages with keep-alive
                message_count = 0
                last_message_time = datetime.now()
                last_keepalive = datetime.now()
                
                while True:
                    try:
                        # Use timeout to implement keep-alive mechanism
                        raw = await asyncio.wait_for(ws.recv(), timeout=30.0)
                        last_message_time = datetime.now()
                        
                        try:
                            messages = json.loads(raw)
                        except json.JSONDecodeError:
                            logger.warning("Could not decode message: %s", raw)
                            continue

                        # Process messages
                        for msg in messages:
                            msg_type = msg.get("T", "unknown")
                            
                            if msg_type == "t":  # trade
                                try:
                                    symbol = msg["S"]
                                    price = float(msg["p"])
                                    
                                    # Handle different timestamp formats
                                    ts_raw = msg["t"]
                                    if isinstance(ts_raw, str):
                                        # ISO 8601 format: '2025-06-23T15:50:49.004167416Z'
                                        timestamp = datetime.fromisoformat(ts_raw.replace('Z', '+00:00'))
                                    else:
                                        # Nanoseconds since epoch (integer)
                                        timestamp = datetime.fromtimestamp(int(ts_raw) / 1e9, tz=timezone.utc)

                                    tick = PriceTick(symbol=symbol, price=price, timestamp=timestamp)
                                    await bus.publish("price.ticks", tick)
                                    logger.info("Trade: %s @ $%.2f", symbol, price)
                                    message_count += 1
                                except (ValueError, TypeError, KeyError) as e:
                                    logger.warning("Failed to process trade message: %s - %s", msg, e)
                                
                            elif msg_type == "q":  # quote
                                # Just log quotes, don't flood the system
                                try:
                                    bid_price = float(msg.get("bp", 0))
                                    ask_price = float(msg.get("ap", 0))
                                    logger.debug("Quote: %s bid=%.2f ask=%.2f", msg["S"], bid_price, ask_price)
                                except (ValueError, TypeError):
                                    logger.debug("Quote: %s (could not parse prices)", msg["S"])
                                
                            elif msg_type == "success":
                                logger.info("Success: %s", msg.get("msg", ""))
                                
                            elif msg_type == "subscription":
                                logger.info("Subscription confirmed: %s", msg)
                                
                            elif msg_type == "error":
                                logger.error("Alpaca error: %s", msg)
                                
                            else:
                                logger.debug("Other message: %s", msg)
                        
                        # Log activity periodically
                        if message_count > 0 and message_count % 50 == 0:
                            logger.info("Processed %d trades so far", message_count)
                            
                    except asyncio.TimeoutError:
                        # No message received in 30 seconds - send keep-alive
                        current_time = datetime.now()
                        if (current_time - last_keepalive).total_seconds() > 60:
                            # Send a ping to keep connection alive
                            await ws.ping()
                            last_keepalive = current_time
                            logger.debug("Sent keep-alive ping")
                            
                        # Check if we've been idle too long
                        time_since_last_message = (current_time - last_message_time).total_seconds()
                        if time_since_last_message > 300:  # 5 minutes
                            logger.warning("No messages for %d seconds, reconnecting", time_since_last_message)
                            break
                            
                    except websockets.exceptions.ConnectionClosed:
                        logger.warning("WebSocket connection closed")
                        break
                        
        except Exception as e:
            logger.warning(f"Failed to connect to {ws_url}: {e}")
            if endpoint_idx == len(WS_ENDPOINTS) - 1:
                # All endpoints failed, re-raise the exception
                raise


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
    """Main service loop with improved error handling and monitoring."""
    if not check_credentials():
        logger.error("Cannot start Alpaca service without credentials")
        return
    
    # Initialize symbols first
    await initialize_symbols()
    
    # Connect to message bus
    bus = MessageBus()
    await bus.connect()
    logger.info("Connected to message bus")
    
    # Start heartbeat task
    heartbeat_task = asyncio.create_task(send_heartbeat())
    logger.info("Started heartbeat task")

    # Connection monitoring
    retry_count = 0
    max_retries = 20  # Allow more retries
    consecutive_failures = 0
    last_success_time = datetime.now()
    
    while retry_count < max_retries:
        try:
            logger.info("Starting Alpaca WebSocket connection (attempt %d/%d)", retry_count + 1, max_retries)
            
            # Track connection start time
            connection_start = datetime.now()
            
            await alpaca_stream(bus)
            
            # If we get here, connection was successful for some time
            connection_duration = (datetime.now() - connection_start).total_seconds()
            logger.info("Connection lasted %.1f seconds", connection_duration)
            
            # Reset counters if connection lasted more than 60 seconds
            if connection_duration > 60:
                retry_count = 0
                consecutive_failures = 0
                last_success_time = datetime.now()
                logger.info("Connection reset - lasted over 60 seconds")
            else:
                consecutive_failures += 1
                
        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
            break
            
        except Exception as exc:
            retry_count += 1
            consecutive_failures += 1
            
            # Calculate backoff delay based on consecutive failures
            if consecutive_failures <= 3:
                delay = 5  # Quick retry for first few attempts
            elif consecutive_failures <= 6:
                delay = 15  # Medium delay
            else:
                delay = min(30 + (consecutive_failures - 6) * 10, 120)  # Longer delay, max 2 minutes
            
            # Check if we've had no successful connections for too long
            time_since_success = (datetime.now() - last_success_time).total_seconds()
            if time_since_success > 1800:  # 30 minutes
                logger.error("No successful connection for 30 minutes, stopping service")
                break
            
            logger.warning("Alpaca stream failed (attempt %d/%d, consecutive: %d): %s", 
                         retry_count, max_retries, consecutive_failures, str(exc))
            logger.info("Reconnecting in %d seconds…", delay)
            
            try:
                await asyncio.sleep(delay)
            except KeyboardInterrupt:
                logger.info("Shutdown during retry delay")
                break
    
    if retry_count >= max_retries:
        logger.error("Max retries exceeded. Alpaca service stopping.")
    else:
        logger.info("Alpaca service stopped gracefully.")
    
    # Cleanup
    try:
        # Cancel heartbeat task
        heartbeat_task.cancel()
        try:
            await heartbeat_task
        except asyncio.CancelledError:
            pass
        logger.info("Heartbeat task stopped")
        
        await bus.close()
        logger.info("Message bus connection closed")
    except Exception as e:
        logger.warning("Error during cleanup: %s", e)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Alpaca market data service stopped") 