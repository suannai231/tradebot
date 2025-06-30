from __future__ import annotations
import os
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from tradebot.strategy.backtest import BacktestEngine, BacktestResult

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/tradebot")

router = APIRouter(prefix="/api", tags=["backtest"])

_engine: Optional[BacktestEngine] = None


async def _get_engine() -> BacktestEngine:
    global _engine
    if _engine is None:
        _engine = BacktestEngine(DATABASE_URL)
        await _engine.connect()
    return _engine


@router.get("/backtest")
async def run_backtest(
    symbol: str,
    strategy: str = Query("mean_reversion"),
    start: str = Query(..., description="YYYY-MM-DD"),
    end: str = Query(..., description="YYYY-MM-DD"),
    adjust_method: str = Query("forward", description="split adjustment: forward/backward/none")):
    """Run a quick back-test and return summary metrics."""
    try:
        start_dt = datetime.fromisoformat(start)
        end_dt = datetime.fromisoformat(end)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")

    if start_dt >= end_dt:
        raise HTTPException(status_code=400, detail="start must be before end")
    engine = await _get_engine()
    try:
        result: BacktestResult = await engine.run_backtest(strategy.lower(), symbol.upper(), start_dt, end_dt, adjust_method=adjust_method)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return result.to_dict()

# (No app.include_router here) 