import os
from datetime import datetime, timezone
from typing import List, Optional
from contextlib import asynccontextmanager

import asyncpg
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import redis.asyncio as redis
import asyncio
from tradebot.common.bus import MessageBus

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/tradebot")
SERVICE_NAME = os.getenv("SERVICE_NAME")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# Database connection pool
pool: Optional[asyncpg.Pool] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pool
    # Startup
    pool = await asyncpg.create_pool(DATABASE_URL, min_size=2, max_size=20)
    # start heartbeat if service name provided
    asyncio.get_running_loop().create_task(_heartbeat())
    # ensure MessageBus heartbeat auto-starts (shares same SERVICE_NAME)
    try:
        bus = MessageBus()
        await bus.connect()
    except Exception as e:
        print(f"MessageBus connect failed: {e}")
    
    yield
    
    # Shutdown
    if pool:
        await pool.close()


app = FastAPI(title="Tradebot Historical Data API", version="1.0.0", lifespan=lifespan)


class HistoricalTick(BaseModel):
    symbol: str
    price: float
    timestamp: datetime
    volume: int | None = None
    # OHLCV fields
    open: float | None = None
    high: float | None = None
    low: float | None = None
    close: float | None = None
    trade_count: int | None = None
    vwap: float | None = None


class HistoricalStats(BaseModel):
    symbol: str
    start_time: datetime
    end_time: datetime
    tick_count: int
    min_price: float
    max_price: float
    avg_price: float
    latest_price: float


async def _heartbeat():
    if not SERVICE_NAME:
        return
    r = redis.from_url(REDIS_URL)
    key = f"service:{SERVICE_NAME}:heartbeat"
    # set immediately then every 30 s
    while True:
        try:
            await r.set(key, datetime.now(timezone.utc).isoformat())
        except Exception:
            pass
        await asyncio.sleep(30)





@app.get("/")
async def root():
    return {"message": "Tradebot Historical Data API", "version": "1.0.0"}


@app.get("/history/{symbol}", response_model=List[HistoricalTick])
async def get_history(
    symbol: str,
    start: Optional[datetime] = Query(None, description="Start time (ISO format)"),
    end: Optional[datetime] = Query(None, description="End time (ISO format)"),
    limit: int = Query(1000, le=10000, description="Max records to return")
):
    """Get historical price ticks for a symbol."""
    if not pool:
        raise HTTPException(500, "Database not available")
    
    # Default to last 24 hours if no time range specified
    if not end:
        end = datetime.now(timezone.utc)
    if not start:
        start = datetime.fromtimestamp(end.timestamp() - 86400, timezone.utc)
    
    query = """
        SELECT symbol, price, timestamp, volume, open_price, high_price, low_price, close_price, trade_count, vwap
        FROM price_ticks
        WHERE symbol = $1 AND timestamp BETWEEN $2 AND $3
        ORDER BY timestamp DESC
        LIMIT $4
    """
    
    async with pool.acquire() as conn:
        rows = await conn.fetch(query, symbol.upper(), start, end, limit)
    
    return [
        HistoricalTick(
            symbol=row["symbol"],
            price=float(row["price"]),
            timestamp=row["timestamp"],
            volume=row["volume"],
            open=float(row["open_price"]) if row["open_price"] else None,
            high=float(row["high_price"]) if row["high_price"] else None,
            low=float(row["low_price"]) if row["low_price"] else None,
            close=float(row["close_price"]) if row["close_price"] else None,
            trade_count=row["trade_count"],
            vwap=float(row["vwap"]) if row["vwap"] else None
        )
        for row in rows
    ]


@app.get("/stats/{symbol}", response_model=HistoricalStats)
async def get_stats(
    symbol: str,
    start: Optional[datetime] = Query(None),
    end: Optional[datetime] = Query(None)
):
    """Get price statistics for a symbol over a time range."""
    if not pool:
        raise HTTPException(500, "Database not available")
    
    if not end:
        end = datetime.now(timezone.utc)
    if not start:
        start = datetime.fromtimestamp(end.timestamp() - 86400, timezone.utc)
    
    query = """
        SELECT 
            COUNT(*) as tick_count,
            MIN(price) as min_price,
            MAX(price) as max_price,
            AVG(price) as avg_price,
            (SELECT price FROM price_ticks 
             WHERE symbol = $1 AND timestamp <= $3 
             ORDER BY timestamp DESC LIMIT 1) as latest_price
        FROM price_ticks
        WHERE symbol = $1 AND timestamp BETWEEN $2 AND $3
    """
    
    async with pool.acquire() as conn:
        row = await conn.fetchrow(query, symbol.upper(), start, end)
    
    if not row or row["tick_count"] == 0:
        raise HTTPException(404, f"No data found for {symbol}")
    
    return HistoricalStats(
        symbol=symbol.upper(),
        start_time=start,
        end_time=end,
        tick_count=row["tick_count"],
        min_price=float(row["min_price"]),
        max_price=float(row["max_price"]),
        avg_price=float(row["avg_price"]),
        latest_price=float(row["latest_price"]) if row["latest_price"] else 0.0
    )


@app.get("/symbols")
async def get_symbols():
    """Get list of available symbols with OHLCV data summary."""
    if not pool:
        raise HTTPException(500, "Database not available")
    
    query = """
        SELECT DISTINCT symbol, 
               COUNT(*) as total_ticks,
               COUNT(CASE WHEN open_price IS NOT NULL THEN 1 END) as ohlcv_ticks,
               COUNT(CASE WHEN volume > 0 THEN 1 END) as volume_ticks,
               MAX(timestamp) as latest_timestamp,
               (SELECT price FROM price_ticks p2 WHERE p2.symbol = price_ticks.symbol ORDER BY timestamp DESC LIMIT 1) as latest_price
        FROM price_ticks 
        GROUP BY symbol 
        ORDER BY latest_timestamp DESC
    """
    
    async with pool.acquire() as conn:
        rows = await conn.fetch(query)
    
    return [
        {
            "symbol": row["symbol"],
            "total_ticks": row["total_ticks"],
            "ohlcv_ticks": row["ohlcv_ticks"],
            "volume_ticks": row["volume_ticks"],
            "latest_timestamp": row["latest_timestamp"],
            "latest_price": float(row["latest_price"]) if row["latest_price"] else None
        }
        for row in rows
    ]


@app.get("/ohlcv/{symbol}")
async def get_ohlcv_summary(
    symbol: str,
    start: Optional[datetime] = Query(None),
    end: Optional[datetime] = Query(None)
):
    """Get OHLCV summary statistics for a symbol."""
    if not pool:
        raise HTTPException(500, "Database not available")
    
    if not end:
        end = datetime.now(timezone.utc)
    if not start:
        start = datetime.fromtimestamp(end.timestamp() - 86400, timezone.utc)
    
    query = """
        SELECT 
            symbol,
            COUNT(*) as bar_count,
            AVG(open_price) as avg_open,
            AVG(high_price) as avg_high,
            AVG(low_price) as avg_low,
            AVG(close_price) as avg_close,
            SUM(volume) as total_volume,
            AVG(volume) as avg_volume,
            MAX(high_price) as period_high,
            MIN(low_price) as period_low
        FROM price_ticks
        WHERE symbol = $1 AND timestamp BETWEEN $2 AND $3 AND open_price IS NOT NULL
        GROUP BY symbol
    """
    
    async with pool.acquire() as conn:
        row = await conn.fetchrow(query, symbol.upper(), start, end)
    
    if not row or row["bar_count"] == 0:
        raise HTTPException(404, f"No OHLCV data found for {symbol}")
    
    return {
        "symbol": row["symbol"],
        "period_start": start,
        "period_end": end,
        "bar_count": row["bar_count"],
        "avg_open": float(row["avg_open"]) if row["avg_open"] else None,
        "avg_high": float(row["avg_high"]) if row["avg_high"] else None,
        "avg_low": float(row["avg_low"]) if row["avg_low"] else None,
        "avg_close": float(row["avg_close"]) if row["avg_close"] else None,
        "total_volume": row["total_volume"],
        "avg_volume": float(row["avg_volume"]) if row["avg_volume"] else None,
        "period_high": float(row["period_high"]) if row["period_high"] else None,
        "period_low": float(row["period_low"]) if row["period_low"] else None
    }

# Register the backtest router at the end to avoid import cycles
from tradebot.api.backtest_service import router as backtest_router
app.include_router(backtest_router)

if __name__ == "__main__":
    uvicorn.run(
        "tradebot.api.history_service:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    ) 