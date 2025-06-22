from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field


class PriceTick(BaseModel):
    symbol: str = Field(..., description="Ticker symbol e.g. AAPL")
    price: float = Field(..., description="Last traded price or close price")
    timestamp: datetime = Field(..., description="Exchange timestamp in UTC")
    # Optional OHLCV fields for bar data
    open: float | None = Field(None, description="Opening price")
    high: float | None = Field(None, description="High price")
    low: float | None = Field(None, description="Low price")
    close: float | None = Field(None, description="Close price") 
    volume: int | None = Field(None, description="Volume traded")
    trade_count: int | None = Field(None, description="Number of trades")
    vwap: float | None = Field(None, description="Volume weighted average price")


class Side(str, Enum):
    buy = "BUY"
    sell = "SELL"


class Signal(BaseModel):
    symbol: str
    side: Side
    confidence: float = Field(ge=0, le=1, description="Confidence between 0 and 1")
    timestamp: datetime


class TradeSignal(BaseModel):
    """Trading signal model for dashboard compatibility"""
    symbol: str = Field(..., description="Ticker symbol e.g. AAPL")
    signal_type: str = Field(..., description="Signal type: BUY, SELL, HOLD")
    price: float = Field(..., description="Price at signal generation")
    timestamp: datetime = Field(..., description="Signal timestamp in UTC")
    strategy: str | None = Field(None, description="Strategy that generated the signal")
    confidence: float | None = Field(None, ge=0, le=1, description="Signal confidence 0-1") 