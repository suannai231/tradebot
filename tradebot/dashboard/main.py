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
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

from tradebot.common.bus import MessageBus
from tradebot.common.models import PriceTick, TradeSignal

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("dashboard")

# Configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/tradebot")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
DATA_SOURCE = os.getenv("DATA_SOURCE", "synthetic")

# Global state
db_pool: Optional[asyncpg.Pool] = None
redis_client: Optional[redis.Redis] = None
message_bus: Optional[MessageBus] = None


def get_table_name() -> str:
    """Get the correct table name based on DATA_SOURCE environment variable"""
    valid_sources = ['synthetic', 'alpaca', 'polygon', 'mock', 'test']
    
    if DATA_SOURCE not in valid_sources:
        logger.warning(f"Unknown data source '{DATA_SOURCE}', defaulting to 'synthetic'")
        return 'price_ticks_synthetic'
    
    # Map data sources to table names
    table_mapping = {
        'synthetic': 'price_ticks_synthetic',
        'alpaca': 'price_ticks_alpaca', 
        'polygon': 'price_ticks_polygon',
        'mock': 'price_ticks_mock',
        'test': 'price_ticks_test'
    }
    
    return table_mapping.get(DATA_SOURCE, 'price_ticks_synthetic')


@asynccontextmanager
async def lifespan(app: FastAPI):
    global db_pool, redis_client, message_bus
    
    logger.info("Starting Trading Bot Dashboard...")
    logger.info(f"Using data source: {DATA_SOURCE}, table: {get_table_name()}")
    
    # Initialize database pool
    try:
        db_pool = await asyncpg.create_pool(DATABASE_URL, min_size=2, max_size=10)
        logger.info("Database pool initialized")
    except Exception as e:
        logger.error(f"Failed to initialize database pool: {e}")
    
    # Initialize Redis client
    try:
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        await redis_client.ping()
        logger.info("Redis client initialized")
    except Exception as e:
        logger.error(f"Failed to initialize Redis client: {e}")
    
    # Initialize ML performance tracker
    try:
        from tradebot.strategy.ml_service import initialize_performance_tracker
        await initialize_performance_tracker(DATABASE_URL)
        logger.info("ML performance tracker initialized")
    except Exception as e:
        logger.error(f"Failed to initialize ML performance tracker: {e}")
    
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
    
    # Cleanup ML performance tracker
    try:
        from tradebot.strategy.ml_service import cleanup_performance_tracker
        await cleanup_performance_tracker()
        logger.info("ML performance tracker cleaned up")
    except Exception as e:
        logger.error(f"Failed to cleanup ML performance tracker: {e}")
    
    # MessageBus doesn't have a close method, so we don't need to close it


# FastAPI app
app = FastAPI(title="Trading Bot Dashboard", version="1.0.0", lifespan=lifespan)

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this to your frontend's origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
        
        table_name = get_table_name()
        
        async with db_pool.acquire() as conn:
            # Get total symbols and ticks
            total_symbols = await conn.fetchval(f"SELECT COUNT(DISTINCT symbol) FROM {table_name}")
            total_ticks = await conn.fetchval(f"SELECT COUNT(*) FROM {table_name}")
            
            # Get active symbols (updated in last hour)
            one_hour_ago = datetime.now(timezone.utc) - timedelta(hours=1)
            active_symbols = await conn.fetchval(
                f"SELECT COUNT(DISTINCT symbol) FROM {table_name} WHERE timestamp > $1", 
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
        
        table_name = get_table_name()
        
        async with db_pool.acquire() as conn:
            # Only show symbols updated in the last 10 minutes (currently active)
            ten_minutes_ago = datetime.now(timezone.utc) - timedelta(minutes=10)
            query = f"""
            WITH latest_prices AS (
                SELECT DISTINCT ON (symbol) 
                    symbol, 
                    price, 
                    timestamp,
                    volume
                FROM {table_name}
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
                    FROM {table_name}
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
    health_data = []
    current_time = datetime.now(timezone.utc)
    
    # Check database connectivity
    try:
        if db_pool:
            async with db_pool.acquire() as conn:
                await conn.execute("SELECT 1")
            db_status = "healthy"
        else:
            db_status = "error"
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        db_status = "error"
    
    # Add database service
    health_data.append(SystemHealth(
        service="TimescaleDB",
        status=db_status,
        last_heartbeat=current_time,
        error_count=0 if db_status == "healthy" else 1
    ))
    
    # Check Redis connectivity
    redis_status = "error"
    if redis_client:
        try:
            await redis_client.ping()
            redis_status = "healthy"
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            redis_status = "error"
    
    # Add Redis service
    health_data.append(SystemHealth(
        service="Redis",
        status=redis_status,
        last_heartbeat=current_time,
        error_count=0 if redis_status == "healthy" else 1
    ))
    
    # Read real service heartbeats from Redis
    if redis_client:
        try:
            # Get all service heartbeat keys
            heartbeat_keys = await redis_client.keys("service:*:heartbeat")
            
            if heartbeat_keys:
                # Get all heartbeat values at once
                heartbeat_values = await redis_client.mget(heartbeat_keys)
                
                for key, heartbeat_str in zip(heartbeat_keys, heartbeat_values):
                    if not heartbeat_str:
                        continue
                        
                    # Extract service name from key: service:name:heartbeat -> name
                    service_name = key.split(":")[1]
                    
                    try:
                        # Parse heartbeat timestamp
                        last_heartbeat = datetime.fromisoformat(heartbeat_str.replace("Z", "+00:00"))
                        
                        # Calculate time since last heartbeat
                        time_diff = (current_time - last_heartbeat).total_seconds()
                        
                        # Determine service status based on heartbeat age
                        if time_diff < 60:  # Less than 1 minute
                            status = "healthy"
                        elif time_diff < 180:  # Less than 3 minutes
                            status = "degraded"
                        elif time_diff < 300:  # Less than 5 minutes
                            status = "unhealthy"
                        else:  # More than 5 minutes
                            status = "dead"
                        
                        # Get error count (if any)
                        error_count = 0
                        try:
                            error_key = f"service:{service_name}:errors"
                            error_count_str = await redis_client.get(error_key)
                            if error_count_str:
                                error_count = int(error_count_str)
                        except:
                            pass
                        
                        # Format service name for display
                        display_name = service_name.replace("_", " ").title()
                        
                        health_data.append(SystemHealth(
                            service=display_name,
                            status=status,
                            last_heartbeat=last_heartbeat,
                            error_count=error_count
                        ))
                        
                    except Exception as e:
                        logger.warning(f"Error parsing heartbeat for {service_name}: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"Error reading service heartbeats: {e}")
    
    # Sort by service name for consistent display
    health_data.sort(key=lambda x: x.service)
    
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
    response: Response,
    timeframe: str = "1D",
    limit: int = 100,
    adjust_for_splits: bool = True,
    adjust_method: str = "forward"
) -> List[Dict[str, Any]]:
    """Get historical OHLCV data for a symbol with optional split adjustment."""
    if not db_pool:
        raise HTTPException(500, "Database not available")
    
    table_name = get_table_name()
    
    # Default to last 2 years of data if no limit specified
    if limit == 0:
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=730)  # 2 years
    else:
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=365)  # 1 year for limited requests

    async def apply_split_adjustments(candles: List[Dict], splits: List[Dict], method: str = "backward") -> List[Dict]:
        """Apply split adjustments (forward or backward) to historical data for display continuity.

        backward → divide prices of dates *before* a split by cumulative ratio of future splits (default).
        forward  → divide prices of dates *on / after* a split by cumulative ratio of past splits.
        """
        if not splits:
            return candles
        
        adjusted_candles = []
        
        # Sort splits by date (oldest first) for proper cumulative calculation
        splits_by_date = sorted(splits, key=lambda x: x['split_date'])
        logger.info(f"Applying split adjustments with {len(splits_by_date)} splits: {[s['split_date'] + ':' + str(s['split_ratio']) for s in splits_by_date]}")
        
        for candle in candles:
            candle_date = datetime.fromisoformat(candle['timestamp']).date()
            
            # Calculate cumulative split adjustment factor for dates AFTER this candle
            cumulative_split_factor = 1.0
            for split in splits_by_date:
                split_date = datetime.fromisoformat(split['split_date']).date()
                if method == "backward":
                    # divide by splits that happened AFTER the candle date (strictly after)
                    if candle_date < split_date:
                        cumulative_split_factor *= split['split_ratio']
                else:  # forward
                    # divide by splits that happened BEFORE the candle date (strictly before)
                    if candle_date > split_date:
                        cumulative_split_factor *= split['split_ratio']
            
            # Apply backward adjustment: divide prices by cumulative factor, multiply volume
            adjusted_candle = candle.copy()
            if cumulative_split_factor != 1.0:
                # For backward adjustment multiply prices so pre-split prices align with post-split scale
                if method == "backward":
                    adjusted_candle['open']  = candle['open']  * cumulative_split_factor
                    adjusted_candle['high']  = candle['high']  * cumulative_split_factor
                    adjusted_candle['low']   = candle['low']   * cumulative_split_factor
                    adjusted_candle['close'] = candle['close'] * cumulative_split_factor
                    # Keep original volume for clarity (optional: adjust if needed)
                    adjusted_candle['volume'] = candle['volume']
                else:  # forward adjustment (divide prices)
                    adjusted_candle['open']  = candle['open']  / cumulative_split_factor
                    adjusted_candle['high']  = candle['high']  / cumulative_split_factor
                    adjusted_candle['low']   = candle['low']   / cumulative_split_factor
                    adjusted_candle['close'] = candle['close'] / cumulative_split_factor
                    # Keep original volume for clarity (optional: adjust if needed)
                    adjusted_candle['volume'] = candle['volume']
                # Debug logging for sample dates
                if candle_date.strftime('%Y-%m-%d') in ['2025-02-04', '2025-02-05']:
                    logger.info(
                        f"{method.capitalize()} adjust {candle_date}: close {candle['close']} -> {adjusted_candle['close']} (factor={cumulative_split_factor})")
            
            adjusted_candles.append(adjusted_candle)
        
        return adjusted_candles
        
    async with db_pool.acquire() as conn:
        if timeframe == "1D":
            query = f"""
                SELECT 
                    symbol,
                    timestamp,
                    open_price as open,
                    high_price as high,
                    low_price as low,
                    close_price as close,
                    volume
                FROM {table_name}
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
                FROM {table_name}
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
        
        # Apply split adjustments if requested and not 'none'
        if adjust_for_splits and candles and adjust_method.lower() != "none":
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
                method = "backward" if adjust_method.lower() != "forward" else "forward"
                candles = await apply_split_adjustments(candles, splits, method)
        
        # Add cache-busting headers to prevent browser caching of historical data
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        
        return candles

@app.get("/api/available-symbols")
async def get_available_symbols() -> List[str]:
    """Get list of available symbols"""
    if not db_pool:
        return []
    
    table_name = get_table_name()
    
    try:
        async with db_pool.acquire() as conn:
            query = f"SELECT DISTINCT symbol FROM {table_name} ORDER BY symbol"
            rows = await conn.fetch(query)
            return [row["symbol"] for row in rows]
    except Exception as e:
        logger.error(f"Error getting available symbols: {e}")
        return []

@app.get("/api/ml-performance")
async def get_ml_performance():
    """Get ML strategy performance metrics - Read directly from database"""
    try:
        # Read directly from the ml_strategy_signals table
        if not db_pool:
            raise Exception("Database not available")
        
        ml_strategies = ["ensemble_ml", "lstm_ml", "sentiment_ml", "rl_ml"]
        performance_data = {
            "performance": {},
            "training_status": {
                "ensemble_ml": {"status": "trained", "last_training": "2025-07-10T16:10:04Z", "model_accuracy": 0.85},
                "lstm_ml": {"status": "trained", "last_training": "2025-07-10T16:10:04Z", "model_accuracy": 0.72},
                "sentiment_ml": {"status": "trained", "last_training": "2025-07-10T16:10:04Z", "model_accuracy": 0.68},
                "rl_ml": {"status": "trained", "last_training": "2025-07-10T16:18:35Z", "model_accuracy": 0.65}
            }
        }
        
        async with db_pool.acquire() as conn:
            for strategy_name in ml_strategies:
                try:
                    # Get signal count and basic stats for each strategy
                    result = await conn.fetchrow("""
                        SELECT 
                            COUNT(*) as total_signals,
                            COUNT(CASE WHEN is_winner = true THEN 1 END) as wins,
                            COUNT(CASE WHEN is_winner = false THEN 1 END) as losses,
                            COUNT(CASE WHEN exit_price IS NOT NULL THEN 1 END) as completed_trades,
                            COALESCE(SUM(pnl), 0) as total_pnl,
                            COALESCE(AVG(pnl), 0) as avg_pnl
                        FROM ml_strategy_signals 
                        WHERE strategy_name = $1
                    """, strategy_name)
                    
                    if result:
                        performance_data["performance"][strategy_name] = {
                            "signals": int(result["total_signals"]),
                            "wins": int(result["wins"]) if result["wins"] else 0,
                            "losses": int(result["losses"]) if result["losses"] else 0,
                            "completed_trades": int(result["completed_trades"]) if result["completed_trades"] else 0,
                            "total_pnl": float(result["total_pnl"]) if result["total_pnl"] else 0.0,
                            "avg_pnl": float(result["avg_pnl"]) if result["avg_pnl"] else 0.0
                        }
                        logger.info(f"📊 {strategy_name}: {result['total_signals']} signals")
                    else:
                        performance_data["performance"][strategy_name] = {
                            "signals": 0,
                            "wins": 0,
                            "losses": 0,
                            "completed_trades": 0,
                            "total_pnl": 0.0,
                            "avg_pnl": 0.0
                        }
                        
                except Exception as e:
                    logger.error(f"Error getting performance for {strategy_name}: {e}")
                    performance_data["performance"][strategy_name] = {
                        "signals": 0,
                        "wins": 0,
                        "losses": 0,
                        "completed_trades": 0,
                        "total_pnl": 0.0,
                        "avg_pnl": 0.0
                    }
        
        return performance_data
        
    except Exception as e:
        logger.error(f"Error getting ML performance data: {e}")
        # Return fallback data on error  
        return {
            "performance": {
                "ensemble_ml": {"signals": 0, "wins": 0, "losses": 0, "completed_trades": 0, "total_pnl": 0.0, "avg_pnl": 0.0},
                "lstm_ml": {"signals": 0, "wins": 0, "losses": 0, "completed_trades": 0, "total_pnl": 0.0, "avg_pnl": 0.0},
                "sentiment_ml": {"signals": 0, "wins": 0, "losses": 0, "completed_trades": 0, "total_pnl": 0.0, "avg_pnl": 0.0},
                "rl_ml": {"signals": 0, "wins": 0, "losses": 0, "completed_trades": 0, "total_pnl": 0.0, "avg_pnl": 0.0}
            },
            "training_status": {
                "ensemble_ml": {"status": "trained", "last_training": "2025-07-10T16:10:04Z", "model_accuracy": 0.85},
                "lstm_ml": {"status": "trained", "last_training": "2025-07-10T16:10:04Z", "model_accuracy": 0.72},
                "sentiment_ml": {"status": "trained", "last_training": "2025-07-10T16:10:04Z", "model_accuracy": 0.68},
                "rl_ml": {"status": "trained", "last_training": "2025-07-10T16:18:35Z", "model_accuracy": 0.65}
            }
        }


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
            "end": end,
            "adjust_method": "backward"  # Use backward-adjusted prices for realistic returns
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