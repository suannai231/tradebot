from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field


class PriceTick(BaseModel):
    symbol: str = Field(..., description="Ticker symbol e.g. AAPL")
    price: float = Field(..., description="Last traded price")
    timestamp: datetime = Field(..., description="Exchange timestamp in UTC")


class Side(str, Enum):
    buy = "BUY"
    sell = "SELL"


class Signal(BaseModel):
    symbol: str
    side: Side
    confidence: float = Field(ge=0, le=1, description="Confidence between 0 and 1")
    timestamp: datetime 