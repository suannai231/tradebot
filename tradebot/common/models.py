from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field
from dataclasses import dataclass, field
from typing import Optional, Dict, Any


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
    price: float = Field(..., description="Price at which signal was generated")
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


@dataclass
class MLStrategySignal:
    """ML strategy signal with performance tracking"""
    id: Optional[int] = None
    strategy_name: str = ""
    symbol: str = ""
    signal_type: str = ""  # 'buy' or 'sell'
    entry_price: float = 0.0
    entry_timestamp: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_timestamp: Optional[datetime] = None
    confidence: float = 0.0
    pnl: Optional[float] = None
    is_winner: Optional[bool] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None


@dataclass
class MLPerformanceMetrics:
    """Aggregated ML strategy performance metrics"""
    strategy_name: str
    total_signals: int
    winning_signals: int
    losing_signals: int
    open_signals: int
    total_pnl: float
    avg_pnl: float
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    last_signal_time: Optional[datetime]
    model_accuracy: float
    training_status: str
    last_training_time: Optional[datetime] 


@dataclass
class TrainingJob:
    """ML Training Job Model"""
    job_id: str
    strategy_type: str  # 'ensemble', 'lstm', 'rl'
    symbol: str
    priority: int = 1
    created_at: Optional[datetime] = field(default=None)
    metadata: Optional[Dict[str, Any]] = field(default=None)
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.metadata is None:
            self.metadata = {}

@dataclass
class TrainingResult:
    """ML Training Result Model"""
    accuracy: float
    loss: float
    training_time: float
    data_points: int
    metadata: Optional[Dict[str, Any]] = field(default=None)
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class TrainingMetrics:
    """ML Training Metrics Model"""
    name: str
    value: float
    symbol: Optional[str] = None
    strategy_type: Optional[str] = None
    timestamp: Optional[datetime] = field(default=None)
    metadata: Optional[Dict[str, Any]] = field(default=None)
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.metadata is None:
            self.metadata = {} 