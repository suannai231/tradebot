"""
Trading Bot Dashboard - Main Application

Real-time web dashboard for monitoring trading bot system.
Provides live updates on market data, trading signals, and system health.
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
from contextlib import asynccontextmanager

import asyncpg
import redis.asyncio as redis
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from tradebot.common.bus import MessageBus
from tradebot.common.models import PriceTick, TradeSignal

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("dashboard")

# Configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/tradebot")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# Global state
db_pool: Optional[asyncpg.Pool] = None
redis_client: Optional[redis.Redis] = None
message_bus: Optional[MessageBus] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global db_pool, redis_client, message_bus
    
    logger.info("Starting Trading Bot Dashboard...")
    
    # Initialize database pool
    try:
        db_pool = await asyncpg.create_pool(DATABASE_URL, min_size=2, max_size=10)
        logger.info("Database pool initialized")
    except Exception as e:
        logger.error(f"Failed to initialize database pool: {e}")
    
    # Initialize Redis client
    try:
        redis_client = redis.from_url(REDIS_URL)
        await redis_client.ping()
        logger.info("Redis client initialized")
    except Exception as e:
        logger.error(f"Failed to initialize Redis client: {e}")
    
    # Initialize message bus for real-time updates
    try:
        message_bus = MessageBus()
        await message_bus.connect()
        logger.info("Message bus connected")
        
        # Start background task to listen for updates
        asyncio.create_task(listen_for_updates())
    except Exception as e:
        logger.error(f"Failed to initialize message bus: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Trading Bot Dashboard...")
    
    if db_pool:
        await db_pool.close()
    if redis_client:
        await redis_client.close()
    if message_bus:
        await message_bus.close()


# FastAPI app
app = FastAPI(title="Trading Bot Dashboard", version="1.0.0", lifespan=lifespan)

# Templates and static files
dashboard_dir = Path(__file__).parent
templates = Jinja2Templates(directory=dashboard_dir / "templates")
app.mount("/static", StaticFiles(directory=dashboard_dir / "static"), name="static")

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients"""
        if not self.active_connections:
            return
        
        message_str = json.dumps(message)
        disconnected = []
        
        for connection in self.active_connections:
            try:
                await connection.send_text(message_str)
            except Exception as e:
                logger.warning(f"Failed to send message to WebSocket: {e}")
                disconnected.append(connection)
        
        # Remove disconnected clients
        for connection in disconnected:
            self.disconnect(connection)

manager = ConnectionManager()

# Data models
class SystemStats(BaseModel):
    total_symbols: int
    active_symbols: int
    total_ticks: int
    total_signals: int
    uptime: str
    memory_usage: float
    cpu_usage: float

class MarketSummary(BaseModel):
    symbol: str
    price: float
    change: float
    change_percent: float
    volume: int
    last_update: datetime

class TradingSignal(BaseModel):
    symbol: str
    signal_type: str
    price: float
    timestamp: datetime
    strategy: str
    confidence: float

class SystemHealth(BaseModel):
    service: str
    status: str
    last_heartbeat: datetime
    error_count: int



# Background task to listen for real-time updates
async def listen_for_updates():
    """Listen for real-time updates from the trading system"""
    if not message_bus:
        return
    
    logger.info("Starting real-time update listener...")
    
    async def listen_price_ticks():
        """Listen for price tick updates"""
        try:
            async for data in message_bus.subscribe("price.ticks"):
                try:
                    if isinstance(data, dict):
                        symbol = data.get("symbol")
                        price = float(data.get("price", 0))
                        ts = data.get("timestamp") or data.get("time")
                        volume = data.get("volume", 0) or 0
                    else:
                        # Assume PriceTick-like object
                        symbol = getattr(data, "symbol", None)
                        price = float(getattr(data, "price", 0))
                        ts = getattr(data, "timestamp", None)
                        if ts:
                            ts = ts.isoformat()
                        volume = getattr(data, "volume", 0) or 0
                    await manager.broadcast({
                        "type": "price_update",
                        "data": {
                            "symbol": symbol,
                            "price": price,
                            "timestamp": ts,
                            "volume": volume
                        }
                    })
                except Exception as e:
                    logger.warning(f"Malformed tick data: {e}")
        except Exception as e:
            logger.error(f"Error in price tick listener: {e}")
    
    async def listen_trading_signals():
        """Listen for trading signal updates"""
        try:
            logger.info("Starting trading signals listener...")
            async for signal_data in message_bus.subscribe("trading.signals"):
                try:
                    logger.info(f"Received trading signal: {signal_data}")
                    # Handle signal data (already in correct format from execution service)
                    if isinstance(signal_data, dict):
                        await manager.broadcast({
                            "type": "trading_signal",
                            "data": signal_data
                        })
                        logger.info(f"Broadcasted trading signal: {signal_data['symbol']} {signal_data['signal_type']}")
                except Exception as e:
                    logger.warning(f"Failed to process trading signal: {e}")
        except Exception as e:
            logger.error(f"Error in trading signals listener: {e}")
    
    try:
        # Run both listeners concurrently
        await asyncio.gather(
            listen_price_ticks(),
            listen_trading_signals()
        )
    except Exception as e:
        logger.error(f"Error in update listener: {e}")

# Routes
@app.get("/", response_class=HTMLResponse)
async def dashboard_home(request: Request):
    """Main dashboard page"""
    response = templates.TemplateResponse("dashboard.html", {"request": request})
    # Add cache-busting headers
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

@app.get("/api/system-stats")
async def get_system_stats() -> SystemStats:
    """Get system statistics"""
    try:
        if not db_pool:
            raise Exception("Database not available")
        
        async with db_pool.acquire() as conn:
            # Get total symbols and ticks
            total_symbols = await conn.fetchval("SELECT COUNT(DISTINCT symbol) FROM price_ticks")
            total_ticks = await conn.fetchval("SELECT COUNT(*) FROM price_ticks")
            
            # Get active symbols (updated in last hour)
            one_hour_ago = datetime.now(timezone.utc) - timedelta(hours=1)
            active_symbols = await conn.fetchval(
                "SELECT COUNT(DISTINCT symbol) FROM price_ticks WHERE timestamp > $1", 
                one_hour_ago
            )
        
        # Get trading signals count
        total_signals = 0
        if redis_client:
            try:
                total_signals = await redis_client.get("trading:signals:count") or 0
                total_signals = int(total_signals)
            except:
                pass
        
        return SystemStats(
            total_symbols=total_symbols or 0,
            active_symbols=active_symbols or 0,
            total_ticks=total_ticks or 0,
            total_signals=total_signals,
            uptime="Running",
            memory_usage=0.0,  # TODO: Implement actual memory monitoring
            cpu_usage=0.0      # TODO: Implement actual CPU monitoring
        )
    
    except Exception as e:
        logger.error(f"Error getting system stats: {e}")
        return SystemStats(
            total_symbols=0,
            active_symbols=0,
            total_ticks=0,
            total_signals=0,
            uptime="Error",
            memory_usage=0.0,
            cpu_usage=0.0
        )

@app.get("/api/market-summary")
async def get_market_summary(limit: int = 20) -> List[MarketSummary]:
    """Get market summary for top symbols"""
    try:
        if not db_pool:
            return []
        
        async with db_pool.acquire() as conn:
            query = """
            WITH latest_prices AS (
                SELECT DISTINCT ON (symbol) 
                    symbol, 
                    price, 
                    timestamp,
                    volume
                FROM price_ticks 
                ORDER BY symbol, timestamp DESC
            ),
            price_changes AS (
                SELECT 
                    lp.symbol,
                    lp.price as current_price,
                    lp.timestamp,
                    lp.volume,
                    COALESCE(prev.price, lp.price) as previous_price
                FROM latest_prices lp
                LEFT JOIN (
                    SELECT DISTINCT ON (symbol) 
                        symbol, 
                        price
                    FROM price_ticks 
                    WHERE timestamp < NOW() - INTERVAL '1 hour'
                    ORDER BY symbol, timestamp DESC
                ) prev ON lp.symbol = prev.symbol
            )
            SELECT 
                symbol,
                current_price,
                (current_price - previous_price) as change,
                CASE 
                    WHEN previous_price > 0 THEN 
                        ((current_price - previous_price) / previous_price) * 100
                    ELSE 0 
                END as change_percent,
                COALESCE(volume, 0) as volume,
                timestamp
            FROM price_changes
            ORDER BY volume DESC
            LIMIT $1
            """
            
            rows = await conn.fetch(query, limit)
            
            return [
                MarketSummary(
                    symbol=row["symbol"],
                    price=float(row["current_price"]),
                    change=float(row["change"]),
                    change_percent=float(row["change_percent"]),
                    volume=row["volume"] or 0,
                    last_update=row["timestamp"]
                )
                for row in rows
            ]
    
    except Exception as e:
        logger.error(f"Error getting market summary: {e}")
        return []

@app.get("/api/recent-signals")
async def get_recent_signals(limit: int = 50) -> List[Dict[str, Any]]:
    """Get recent trading signals"""
    try:
        if not redis_client:
            return []
        
        # Get recent signals from Redis
        signals_data = await redis_client.lrange("trading:signals:recent", 0, limit - 1)
        
        signals = []
        for signal_data in signals_data:
            try:
                signal = json.loads(signal_data)
                signals.append(signal)
            except json.JSONDecodeError:
                continue
        
        return signals
    
    except Exception as e:
        logger.error(f"Error getting recent signals: {e}")
        return []

@app.get("/api/system-health")
async def get_system_health() -> List[SystemHealth]:
    """Get system health status"""
    try:
        if not redis_client:
            return []
        
        services = ["market_data", "strategy", "execution", "storage", "api"]
        health_status = []
        
        for service in services:
            try:
                # Check last heartbeat
                heartbeat_key = f"service:{service}:heartbeat"
                last_heartbeat = await redis_client.get(heartbeat_key)
                
                if last_heartbeat:
                    last_heartbeat = datetime.fromisoformat(last_heartbeat.decode())
                    time_diff = datetime.now(timezone.utc) - last_heartbeat.replace(tzinfo=timezone.utc)
                    status = "healthy" if time_diff.total_seconds() < 300 else "unhealthy"  # 5 minutes
                else:
                    last_heartbeat = datetime.now(timezone.utc)
                    status = "unknown"
                
                # Get error count
                error_key = f"service:{service}:errors"
                error_count = await redis_client.get(error_key) or 0
                error_count = int(error_count)
                
                health_status.append(SystemHealth(
                    service=service,
                    status=status,
                    last_heartbeat=last_heartbeat,
                    error_count=error_count
                ))
            
            except Exception as e:
                logger.warning(f"Error checking health for {service}: {e}")
                health_status.append(SystemHealth(
                    service=service,
                    status="error",
                    last_heartbeat=datetime.now(timezone.utc),
                    error_count=0
                ))
        
        return health_status
    
    except Exception as e:
        logger.error(f"Error getting system health: {e}")
        return []

# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await manager.connect(websocket)
    
    try:
        # Send initial data
        await websocket.send_text(json.dumps({
            "type": "connection_established",
            "data": {"message": "Connected to Trading Bot Dashboard"}
        }))
        
        # Keep connection alive by sending ping every 25 seconds
        while True:
            await asyncio.sleep(25)
            await websocket.send_text(json.dumps({
                "type": "ping",
                "data": {"timestamp": datetime.now().isoformat()}
            }))
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info") 