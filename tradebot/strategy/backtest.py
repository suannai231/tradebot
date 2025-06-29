import asyncio
import logging
import asyncpg
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import pandas as pd
import numpy as np
import math, copy

from tradebot.common.models import PriceTick, Signal, Side
from tradebot.strategy.advanced_strategies import create_strategy, StrategyConfig

logger = logging.getLogger("backtest")


@dataclass
class Trade:
    """Represents a completed trade."""
    symbol: str
    entry_time: datetime
    entry_price: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    side: Side = Side.buy
    quantity: float = 1.0
    
    @property
    def is_open(self) -> bool:
        return self.exit_time is None
    
    @property
    def pnl(self) -> Optional[float]:
        if self.exit_price is None:
            return None
        if self.side == Side.buy:
            return (self.exit_price - self.entry_price) * self.quantity
        else:
            return (self.entry_price - self.exit_price) * self.quantity
    
    @property
    def return_pct(self) -> Optional[float]:
        if self.exit_price is None:
            return None
        return (self.pnl / (self.entry_price * self.quantity)) * 100


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    strategy_name: str
    symbol: str
    start_date: datetime
    end_date: datetime
    total_trades: int
    winning_trades: int
    losing_trades: int
    total_pnl: float
    total_return_pct: float
    max_drawdown: float
    sharpe_ratio: float
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    trades: List[Trade]
    
    def to_dict(self) -> Dict[str, Any]:
        d = {
            'strategy_name': self.strategy_name,
            'symbol': self.symbol,
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat(),
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'total_pnl': self.total_pnl,
            'total_return_pct': self.total_return_pct,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': self.sharpe_ratio,
            'win_rate': self.win_rate,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'profit_factor': self.profit_factor
        }
        # replace inf / nan with None so JSON is valid
        for k, v in d.items():
            if isinstance(v, float) and (math.isinf(v) or math.isnan(v)):
                d[k] = None
        return d


class BacktestEngine:
    """Engine for backtesting trading strategies on historical data."""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.pool: Optional[asyncpg.Pool] = None
    
    async def connect(self):
        """Connect to the database."""
        self.pool = await asyncpg.create_pool(self.database_url)
    
    async def close(self):
        """Close database connection."""
        if self.pool:
            await self.pool.close()
    
    async def get_historical_data(self, symbol: str, start_date: datetime, end_date: datetime) -> List[PriceTick]:
        """Fetch historical price data from database."""
        if not self.pool:
            raise RuntimeError("Database not connected")
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT symbol, price, timestamp, open_price, high_price, low_price, close_price, volume, trade_count, vwap
                FROM price_ticks 
                WHERE symbol = $1 AND timestamp BETWEEN $2 AND $3
                ORDER BY timestamp ASC
            """, symbol, start_date, end_date)
        
        ticks = []
        for row in rows:
            tick = PriceTick(
                symbol=row['symbol'],
                price=float(row['price']),
                timestamp=row['timestamp'],
                open=float(row['open_price']) if row['open_price'] else None,
                high=float(row['high_price']) if row['high_price'] else None,
                low=float(row['low_price']) if row['low_price'] else None,
                close=float(row['close_price']) if row['close_price'] else None,
                volume=row['volume'] if row['volume'] else None,
                trade_count=row['trade_count'] if row['trade_count'] else None,
                vwap=float(row['vwap']) if row['vwap'] else None
            )
            ticks.append(tick)
        
        return ticks
    
    def calculate_metrics(self, trades: List[Trade], initial_capital: float = 10000.0) -> Dict[str, float]:
        """Calculate performance metrics from trades."""
        if not trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'total_pnl': 0.0,
                'total_return_pct': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0,
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0
            }
        
        # Basic metrics
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t.pnl and t.pnl > 0])
        losing_trades = len([t for t in trades if t.pnl and t.pnl < 0])
        
        # Calculate compound returns
        compound_return = 1.0
        for trade in trades:
            if trade.return_pct is not None:
                compound_return *= (1 + trade.return_pct / 100)
        
        total_return_pct = (compound_return - 1) * 100
        
        # Calculate total PnL based on compound returns
        total_pnl = initial_capital * (compound_return - 1)
        
        # Win rate and averages
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        wins = [t.pnl for t in trades if t.pnl and t.pnl > 0]
        losses = [t.pnl for t in trades if t.pnl and t.pnl < 0]
        
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        
        # Profit factor
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Maximum drawdown
        cumulative_return = 1.0
        peak = 1.0
        max_drawdown = 0
        
        for trade in trades:
            if trade.return_pct is not None:
                cumulative_return *= (1 + trade.return_pct / 100)
                if cumulative_return > peak:
                    peak = cumulative_return
                drawdown = (peak - cumulative_return) / peak if peak > 0 else 0
                max_drawdown = max(max_drawdown, drawdown)
        
        # Sharpe ratio (simplified - using daily returns)
        if len(trades) > 1:
            returns = [t.return_pct or 0 for t in trades]
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            sharpe_ratio = avg_return / std_return if std_return > 0 else 0
        else:
            sharpe_ratio = 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'total_pnl': total_pnl,
            'total_return_pct': total_return_pct,
            'max_drawdown': max_drawdown * 100,  # Convert to percentage
            'sharpe_ratio': sharpe_ratio,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor
        }
    
    async def run_backtest(self, 
                          strategy_type: str,
                          symbol: str,
                          start_date: datetime,
                          end_date: datetime,
                          strategy_params: Dict[str, Any] = None) -> BacktestResult:
        """Run a backtest for a specific strategy and symbol."""
        
        # Create strategy instance
        strategy = create_strategy(strategy_type, **(strategy_params or {}))
        
        # Get historical data
        ticks = await self.get_historical_data(symbol, start_date, end_date)
        if not ticks:
            raise ValueError(f"No historical data found for {symbol} between {start_date} and {end_date}")
        
        logger.info(f"Running backtest for {symbol} with {len(ticks)} ticks")
        
        # Initialize tracking variables
        trades: List[Trade] = []
        current_position: Optional[Trade] = None
        
        # Process each tick
        for tick in ticks:
            # Generate signal from strategy
            signal = strategy.on_tick(tick)
            
            if signal:
                if signal.side == Side.buy and current_position is None:
                    # Open new long position
                    current_position = Trade(
                        symbol=symbol,
                        entry_time=tick.timestamp,
                        entry_price=tick.price,
                        side=Side.buy
                    )
                    logger.debug(f"Opened long position for {symbol} at {tick.price}")
                
                elif signal.side == Side.sell and current_position is not None:
                    # Close existing position
                    current_position.exit_time = tick.timestamp
                    current_position.exit_price = tick.price
                    trades.append(current_position)
                    logger.debug(f"Closed position for {symbol} at {tick.price}, PnL: {current_position.pnl:.2f}")
                    current_position = None
        
        # Close any remaining position at the end
        if current_position is not None:
            current_position.exit_time = ticks[-1].timestamp
            current_position.exit_price = ticks[-1].price
            trades.append(current_position)
            logger.debug(f"Closed final position for {symbol} at {current_position.exit_price}")
        
        # Calculate metrics
        metrics = self.calculate_metrics(trades)
        
        return BacktestResult(
            strategy_name=strategy_type,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            trades=trades,
            **metrics
        )
    
    async def compare_strategies(self,
                               symbols: List[str],
                               start_date: datetime,
                               end_date: datetime,
                               strategies: Dict[str, Dict[str, Any]] = None) -> Dict[str, List[BacktestResult]]:
        """Compare multiple strategies across multiple symbols."""
        
        if strategies is None:
            strategies = {
                'simple_ma': {},
                'advanced': {
                    'min_composite_score': 0.5,
                    'rsi_oversold': 30,
                    'rsi_overbought': 70,
                    'short_window': 3,
                    'long_window': 10
                },
                'mean_reversion': {
                    'lookback_period': 10,
                    'z_score_threshold': -1.5
                }
            }
        
        results: Dict[str, List[BacktestResult]] = {strategy: [] for strategy in strategies}
        
        for symbol in symbols:
            logger.info(f"Testing strategies for {symbol}")
            
            for strategy_name, params in strategies.items():
                try:
                    result = await self.run_backtest(
                        strategy_name, symbol, start_date, end_date, params
                    )
                    results[strategy_name].append(result)
                    logger.info(f"{strategy_name} for {symbol}: {result.total_return_pct:.2f}% return, {result.total_trades} trades")
                except Exception as e:
                    logger.error(f"Failed to run {strategy_name} for {symbol}: {e}")
        
        return results
    
    def generate_report(self, results: Dict[str, List[BacktestResult]]) -> str:
        """Generate a comprehensive backtest report."""
        report = []
        report.append("=" * 80)
        report.append("BACKTEST RESULTS SUMMARY")
        report.append("=" * 80)
        
        for strategy_name, strategy_results in results.items():
            if not strategy_results:
                continue
            
            report.append(f"\n📊 STRATEGY: {strategy_name.upper()}")
            report.append("-" * 50)
            
            # Aggregate metrics across all symbols
            total_trades = sum(r.total_trades for r in strategy_results)
            total_pnl = sum(r.total_pnl for r in strategy_results)
            total_return = sum(r.total_return_pct for r in strategy_results)
            avg_win_rate = np.mean([r.win_rate for r in strategy_results])
            avg_sharpe = np.mean([r.sharpe_ratio for r in strategy_results])
            avg_drawdown = np.mean([r.max_drawdown for r in strategy_results])
            
            report.append(f"Total Trades: {total_trades}")
            report.append(f"Total PnL: ${total_pnl:.2f}")
            report.append(f"Total Return: {total_return:.2f}%")
            report.append(f"Avg Win Rate: {avg_win_rate:.1f}%")
            report.append(f"Avg Sharpe Ratio: {avg_sharpe:.2f}")
            report.append(f"Avg Max Drawdown: {avg_drawdown:.1f}%")
            
            # Per-symbol breakdown
            report.append(f"\nPer-Symbol Results:")
            for result in strategy_results:
                report.append(f"  {result.symbol}: {result.total_return_pct:>6.2f}% return, {result.total_trades:>3} trades")
        
        return "\n".join(report)


async def main():
    """Example usage of the backtest engine."""
    # Configuration
    database_url = "postgresql://postgres:password@localhost:5432/tradebot"
    symbols = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"]
    start_date = datetime.now(timezone.utc) - timedelta(days=365)
    end_date = datetime.now(timezone.utc) - timedelta(days=1)
    
    # Define strategies to test
    strategies = {
        'simple_ma': {},
        'advanced': {
            'min_composite_score': 0.5,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'short_window': 3,
            'long_window': 10
        },
        'mean_reversion': {
            'lookback_period': 10,
            'z_score_threshold': -1.5
        }
    }
    
    # Run backtest
    engine = BacktestEngine(database_url)
    await engine.connect()
    
    try:
        results = await engine.compare_strategies(symbols, start_date, end_date, strategies)
        report = engine.generate_report(results)
        print(report)
    finally:
        await engine.close()


if __name__ == "__main__":
    asyncio.run(main()) 