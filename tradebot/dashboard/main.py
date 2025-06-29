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
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
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
        await redis_client.aclose()
    # MessageBus doesn't have a close method, so we don't need to close it


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
    """Get market summary for top symbols (only recently active ones)"""
    try:
        if not db_pool:
            return []
        
        async with db_pool.acquire() as conn:
            # Only show symbols updated in the last 10 minutes (currently active)
            ten_minutes_ago = datetime.now(timezone.utc) - timedelta(minutes=10)
            query = """
            WITH latest_prices AS (
                SELECT DISTINCT ON (symbol) 
                    symbol, 
                    price, 
                    timestamp,
                    volume
                FROM price_ticks 
                WHERE timestamp > $2
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
            
            rows = await conn.fetch(query, limit, ten_minutes_ago)
            
            # If no recent data, fall back to last hour
            if not rows:
                one_hour_ago = datetime.now(timezone.utc) - timedelta(hours=1)
                rows = await conn.fetch(query, limit, one_hour_ago)
            
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
    """Get system health status for all services"""
    if not db_pool:
        return []
    
    try:
        # Get basic system stats
        async with db_pool.acquire() as conn:
            # Check database connectivity
            await conn.execute("SELECT 1")
            db_status = "healthy"
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        db_status = "error"
    
    # Mock health data for now
    health_data = [
        SystemHealth(
            service="Database",
            status=db_status,
            last_heartbeat=datetime.now(timezone.utc),
            error_count=0 if db_status == "healthy" else 1
        ),
        SystemHealth(
            service="Market Data Service",
            status="healthy",
            last_heartbeat=datetime.now(timezone.utc) - timedelta(seconds=30),
            error_count=0
        ),
        SystemHealth(
            service="Strategy Service",
            status="healthy",
            last_heartbeat=datetime.now(timezone.utc) - timedelta(seconds=45),
            error_count=0
        ),
        SystemHealth(
            service="Execution Service",
            status="healthy",
            last_heartbeat=datetime.now(timezone.utc) - timedelta(seconds=60),
            error_count=0
        )
    ]
    
    return health_data

@app.get("/api/splits/{symbol}")
async def get_stock_splits(symbol: str) -> List[Dict[str, Any]]:
    """Get stock split events for a symbol."""
    if not db_pool:
        raise HTTPException(500, "Database not available")
    
    async with db_pool.acquire() as conn:
        query = """
            SELECT symbol, split_date, split_ratio, raw_price_before, raw_price_after, adjusted_price
            FROM stock_splits 
            WHERE symbol = $1 
            ORDER BY split_date ASC
        """
        rows = await conn.fetch(query, symbol.upper())
        
        return [
            {
                "symbol": row["symbol"],
                "split_date": row["split_date"].isoformat(),
                "split_ratio": float(row["split_ratio"]),
                "raw_price_before": float(row["raw_price_before"]) if row["raw_price_before"] else None,
                "raw_price_after": float(row["raw_price_after"]) if row["raw_price_after"] else None,
                "adjusted_price": float(row["adjusted_price"]) if row["adjusted_price"] else None
            }
            for row in rows
        ]

@app.get("/api/historical-data/{symbol}")
async def get_historical_data(
    symbol: str, 
    timeframe: str = "1D",
    limit: int = 100,
    adjust_for_splits: bool = True
) -> List[Dict[str, Any]]:
    """Get historical OHLCV data for a symbol with optional split adjustment."""
    if not db_pool:
        raise HTTPException(500, "Database not available")
    
    # Default to last 2 years of data if no limit specified
    if limit == 0:
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=730)  # 2 years
    else:
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=365)  # 1 year for limited requests

    async def apply_split_adjustments(candles: List[Dict], splits: List[Dict]) -> List[Dict]:
        """Apply split adjustments to historical data for display continuity."""
        if not splits:
            return candles
        
        adjusted_candles = []
        cumulative_adjustment = 1.0
        
        # Sort splits by date (newest first) for reverse adjustment
        splits_by_date = sorted(splits, key=lambda x: x['split_date'], reverse=True)
        
        for candle in candles:
            candle_date = datetime.fromisoformat(candle['timestamp']).date()
            
            # Calculate cumulative adjustment for this candle date
            adjustment = 1.0
            for split in splits_by_date:
                split_date = datetime.fromisoformat(split['split_date']).date()
                if candle_date < split_date:
                    adjustment *= split['split_ratio']
            
            # Apply adjustment to prices (divide) and volume (multiply)
            adjusted_candle = candle.copy()
            if adjustment != 1.0:
                adjusted_candle['open'] = candle['open'] / adjustment
                adjusted_candle['high'] = candle['high'] / adjustment
                adjusted_candle['low'] = candle['low'] / adjustment
                adjusted_candle['close'] = candle['close'] / adjustment
                adjusted_candle['volume'] = int(candle['volume'] * adjustment)
            
            adjusted_candles.append(adjusted_candle)
        
        return adjusted_candles
        
    async with db_pool.acquire() as conn:
        if timeframe == "1D":
            query = """
                SELECT 
                    symbol,
                    timestamp,
                    open_price as open,
                    high_price as high,
                    low_price as low,
                    close_price as close,
                    volume
                FROM price_ticks 
                WHERE symbol = $1 
                    AND timestamp >= $2 
                    AND timestamp <= $3
                    AND open_price IS NOT NULL
                    AND high_price IS NOT NULL
                    AND low_price IS NOT NULL
                    AND close_price IS NOT NULL
                ORDER BY timestamp ASC
            """
            if limit > 0:
                query += " LIMIT $4"
                rows = await conn.fetch(query, symbol.upper(), start_date, end_date, limit)
            else:
                rows = await conn.fetch(query, symbol.upper(), start_date, end_date)
        else:
            # Weekly/Monthly data - aggregate from daily data
            if timeframe == "1W":
                interval = "week"
            else:
                interval = "month"
            
            query = f"""
                SELECT 
                    symbol,
                    date_trunc('{interval}', timestamp) as timestamp,
                    first(open_price, timestamp) as open,
                    max(high_price) as high,
                    min(low_price) as low,
                    last(close_price, timestamp) as close,
                    sum(volume) as volume
                FROM price_ticks 
                WHERE symbol = $1 
                    AND timestamp >= $2 
                    AND timestamp <= $3
                    AND open_price IS NOT NULL
                    AND high_price IS NOT NULL
                    AND low_price IS NOT NULL
                    AND close_price IS NOT NULL
                GROUP BY symbol, date_trunc('{interval}', timestamp)
                ORDER BY timestamp ASC
            """
            if limit > 0:
                query += " LIMIT $4"
                rows = await conn.fetch(query, symbol.upper(), start_date, end_date, limit)
            else:
                rows = await conn.fetch(query, symbol.upper(), start_date, end_date)
        
        # Convert to candlestick format
        candles = []
        for row in rows:
            candles.append({
                "timestamp": row["timestamp"].isoformat(),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": int(row["volume"]) if row["volume"] else 0
            })
        
        # Apply split adjustments if requested
        if adjust_for_splits and candles:
            # Get split events for this symbol
            splits_query = """
                SELECT split_date, split_ratio 
                FROM stock_splits 
                WHERE symbol = $1 
                ORDER BY split_date ASC
            """
            split_rows = await conn.fetch(splits_query, symbol.upper())
            splits = [
                {
                    "split_date": row["split_date"].isoformat(),
                    "split_ratio": float(row["split_ratio"])
                }
                for row in split_rows
            ]
            
            if splits:
                candles = await apply_split_adjustments(candles, splits)
        
        return candles

@app.get("/api/available-symbols")
async def get_available_symbols() -> List[str]:
    """Get list of available symbols"""
    if not db_pool:
        return []
    
    try:
        async with db_pool.acquire() as conn:
            query = "SELECT DISTINCT symbol FROM price_ticks ORDER BY symbol"
            rows = await conn.fetch(query)
            return [row["symbol"] for row in rows]
    except Exception as e:
        logger.error(f"Error getting available symbols: {e}")
        return []

@app.get("/api/backtest")
async def proxy_backtest(
    symbol: str,
    strategy: str = "mean_reversion",
    start: str = None,
    end: str = None
):
    """Proxy backtest requests to the API service"""
    import httpx
    
    try:
        # Forward the request to the API service
        api_url = "http://api:8000/api/backtest"  # Use Docker service name
        params = {
            "symbol": symbol,
            "strategy": strategy,
            "start": start,
            "end": end
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.get(api_url, params=params)
            response.raise_for_status()
            return response.json()
            
    except Exception as e:
        logger.error(f"Error proxying backtest request: {e}")
        return {"error": str(e)}

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