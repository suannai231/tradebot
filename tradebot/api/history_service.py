import os
from datetime import datetime, timezone
from typing import List, Optional

import asyncpg
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/tradebot")

app = FastAPI(title="Tradebot Historical Data API", version="1.0.0")

# Database connection pool
pool: Optional[asyncpg.Pool] = None


class HistoricalTick(BaseModel):
    symbol: str
    price: float
    timestamp: datetime
    volume: int


class HistoricalStats(BaseModel):
    symbol: str
    start_time: datetime
    end_time: datetime
    tick_count: int
    min_price: float
    max_price: float
    avg_price: float
    latest_price: float


@app.on_event("startup")
async def startup():
    global pool
    pool = await asyncpg.create_pool(DATABASE_URL, min_size=2, max_size=20)


@app.on_event("shutdown")
async def shutdown():
    if pool:
        await pool.close()


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
        SELECT symbol, price, timestamp, volume
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
            volume=row["volume"]
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
    """Get list of available symbols."""
    if not pool:
        raise HTTPException(500, "Database not available")
    
    query = """
        SELECT DISTINCT symbol, 
               COUNT(*) as tick_count,
               MAX(timestamp) as latest_timestamp
        FROM price_ticks 
        GROUP BY symbol 
        ORDER BY latest_timestamp DESC
    """
    
    async with pool.acquire() as conn:
        rows = await conn.fetch(query)
    
    return [
        {
            "symbol": row["symbol"],
            "tick_count": row["tick_count"],
            "latest_timestamp": row["latest_timestamp"]
        }
        for row in rows
    ]


if __name__ == "__main__":
    uvicorn.run(
        "tradebot.api.history_service:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    ) 