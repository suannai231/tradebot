import asyncio
import logging
from typing import Dict, List, Optional
from collections import defaultdict
from datetime import datetime, timezone, timedelta
import numpy as np
import asyncpg
import json
from dataclasses import asdict

from tradebot.common.bus import MessageBus
from tradebot.common.models import PriceTick, Signal, Side, MLStrategySignal, MLPerformanceMetrics

logger = logging.getLogger("ml_service")


# ML Strategy Service has been moved to individual strategy files
# This file now only contains the performance tracking functionality


class MLPerformanceTracker:
    """Service for tracking ML strategy performance"""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.pool: Optional[asyncpg.Pool] = None
        
    async def connect(self):
        """Connect to database and create tables if needed"""
        self.pool = await asyncpg.create_pool(self.database_url)
        await self.create_tables()
        
    async def close(self):
        """Close database connection"""
        if self.pool:
            await self.pool.close()
            
    async def create_tables(self):
        """Create ML performance tracking tables"""
        if not self.pool:
            return
            
        async with self.pool.acquire() as conn:
            # ML strategy signals table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS ml_strategy_signals (
                    id SERIAL PRIMARY KEY,
                    strategy_name VARCHAR(100) NOT NULL,
                    symbol VARCHAR(20) NOT NULL,
                    signal_type VARCHAR(10) NOT NULL,
                    entry_price DECIMAL(10, 4) NOT NULL,
                    entry_timestamp TIMESTAMPTZ NOT NULL,
                    exit_price DECIMAL(10, 4),
                    exit_timestamp TIMESTAMPTZ,
                    confidence DECIMAL(4, 3) NOT NULL,
                    pnl DECIMAL(10, 4),
                    is_winner BOOLEAN,
                    metadata JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
            """)
            
            # ML strategy training log table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS ml_training_log (
                    id SERIAL PRIMARY KEY,
                    strategy_name VARCHAR(100) NOT NULL,
                    training_timestamp TIMESTAMPTZ NOT NULL,
                    model_accuracy DECIMAL(4, 3),
                    training_duration_seconds INTEGER,
                    data_points_used INTEGER,
                    status VARCHAR(20) NOT NULL,
                    error_message TEXT,
                    metadata JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
            """)
            
            # Create indexes for performance
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_ml_signals_strategy_symbol 
                ON ml_strategy_signals(strategy_name, symbol);
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_ml_signals_timestamp 
                ON ml_strategy_signals(entry_timestamp);
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_ml_training_strategy 
                ON ml_training_log(strategy_name, training_timestamp);
            """)
            
    async def log_signal(self, strategy_name: str, symbol: str, signal: Signal) -> int:
        """Log a new ML strategy signal"""
        if not self.pool:
            return 0
            
        try:
            async with self.pool.acquire() as conn:
                signal_id = await conn.fetchval("""
                    INSERT INTO ml_strategy_signals 
                    (strategy_name, symbol, signal_type, entry_price, entry_timestamp, confidence, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    RETURNING id
                """, 
                strategy_name, 
                symbol, 
                signal.side.value, 
                signal.price, 
                signal.timestamp, 
                signal.confidence,
                json.dumps(getattr(signal, 'metadata', {}) or {})
                )
                
                logger.info(f"Logged ML signal {signal_id} for {strategy_name}: {symbol} {signal.side.value} at {signal.price}")
                return signal_id
                
        except Exception as e:
            logger.error(f"Error logging ML signal: {e}")
            return 0
            
    async def update_signal_exit(self, signal_id: int, exit_price: float, exit_timestamp: datetime):
        """Update signal with exit information and calculate PnL"""
        if not self.pool:
            return
            
        try:
            async with self.pool.acquire() as conn:
                # Get signal details
                signal_data = await conn.fetchrow("""
                    SELECT signal_type, entry_price FROM ml_strategy_signals WHERE id = $1
                """, signal_id)
                
                if not signal_data:
                    return
                    
                # Calculate PnL
                entry_price = float(signal_data['entry_price'])
                if signal_data['signal_type'] == 'buy':
                    pnl = exit_price - entry_price
                else:  # sell
                    pnl = entry_price - exit_price
                    
                is_winner = pnl > 0
                
                # Update signal
                await conn.execute("""
                    UPDATE ml_strategy_signals 
                    SET exit_price = $1, exit_timestamp = $2, pnl = $3, is_winner = $4
                    WHERE id = $5
                """, exit_price, exit_timestamp, pnl, is_winner, signal_id)
                
                logger.info(f"Updated ML signal {signal_id} exit: price={exit_price}, pnl={pnl:.2f}")
                
        except Exception as e:
            logger.error(f"Error updating ML signal exit: {e}")
            
    async def log_training_event(self, strategy_name: str, status: str, accuracy: float = None, 
                                duration: int = None, data_points: int = None, error_message: str = None):
        """Log ML model training event"""
        if not self.pool:
            return
            
        try:
            async with self.pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO ml_training_log 
                    (strategy_name, training_timestamp, model_accuracy, training_duration_seconds, 
                     data_points_used, status, error_message)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                """, 
                strategy_name, 
                datetime.now(timezone.utc), 
                accuracy, 
                duration, 
                data_points, 
                status, 
                error_message
                )
                
                logger.info(f"Logged training event for {strategy_name}: {status}")
                
        except Exception as e:
            logger.error(f"Error logging training event: {e}")
            
    async def get_strategy_performance(self, strategy_name: str, days: int = 30) -> Optional[MLPerformanceMetrics]:
        """Get performance metrics for a specific strategy"""
        if not self.pool:
            return None
            
        try:
            async with self.pool.acquire() as conn:
                # Get signal statistics
                since_date = datetime.now(timezone.utc) - timedelta(days=days)
                
                stats = await conn.fetchrow("""
                    SELECT 
                        COUNT(*) as total_signals,
                        COUNT(*) FILTER (WHERE is_winner = true) as winning_signals,
                        COUNT(*) FILTER (WHERE is_winner = false) as losing_signals,
                        COUNT(*) FILTER (WHERE exit_price IS NULL) as open_signals,
                        COALESCE(SUM(pnl), 0) as total_pnl,
                        COALESCE(AVG(pnl), 0) as avg_pnl,
                        COALESCE(AVG(pnl) FILTER (WHERE is_winner = true), 0) as avg_win,
                        COALESCE(AVG(pnl) FILTER (WHERE is_winner = false), 0) as avg_loss,
                        MAX(entry_timestamp) as last_signal_time
                    FROM ml_strategy_signals 
                    WHERE strategy_name = $1 AND entry_timestamp >= $2
                """, strategy_name, since_date)
                
                if not stats or stats['total_signals'] == 0:
                    return MLPerformanceMetrics(
                        strategy_name=strategy_name,
                        total_signals=0,
                        winning_signals=0,
                        losing_signals=0,
                        open_signals=0,
                        total_pnl=0.0,
                        avg_pnl=0.0,
                        win_rate=0.0,
                        avg_win=0.0,
                        avg_loss=0.0,
                        profit_factor=0.0,
                        last_signal_time=None,
                        model_accuracy=0.0,
                        training_status="untrained",
                        last_training_time=None
                    )
                
                # Calculate derived metrics
                total_signals = stats['total_signals']
                winning_signals = stats['winning_signals']
                losing_signals = stats['losing_signals']
                
                win_rate = (winning_signals / total_signals * 100) if total_signals > 0 else 0
                
                gross_profit = abs(float(stats['avg_win']) * winning_signals) if winning_signals > 0 else 0
                gross_loss = abs(float(stats['avg_loss']) * losing_signals) if losing_signals > 0 else 0
                profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0
                
                # Get latest training info
                training_info = await conn.fetchrow("""
                    SELECT model_accuracy, status, training_timestamp
                    FROM ml_training_log 
                    WHERE strategy_name = $1 
                    ORDER BY training_timestamp DESC 
                    LIMIT 1
                """, strategy_name)
                
                model_accuracy = float(training_info['model_accuracy']) if training_info and training_info['model_accuracy'] else 0.0
                training_status = training_info['status'] if training_info else "untrained"
                last_training_time = training_info['training_timestamp'] if training_info else None
                
                return MLPerformanceMetrics(
                    strategy_name=strategy_name,
                    total_signals=total_signals,
                    winning_signals=winning_signals,
                    losing_signals=losing_signals,
                    open_signals=stats['open_signals'],
                    total_pnl=float(stats['total_pnl']),
                    avg_pnl=float(stats['avg_pnl']),
                    win_rate=win_rate,
                    avg_win=float(stats['avg_win']),
                    avg_loss=float(stats['avg_loss']),
                    profit_factor=profit_factor,
                    last_signal_time=stats['last_signal_time'],
                    model_accuracy=model_accuracy,
                    training_status=training_status,
                    last_training_time=last_training_time
                )
                
        except Exception as e:
            logger.error(f"Error getting strategy performance: {e}")
            return None
            
    async def get_all_ml_performance(self, days: int = 30) -> Dict[str, MLPerformanceMetrics]:
        """Get performance metrics for all ML strategies"""
        ml_strategies = ["ensemble_ml", "lstm_ml", "sentiment_ml", "rl_ml"]
        results = {}
        
        for strategy in ml_strategies:
            metrics = await self.get_strategy_performance(strategy, days)
            if metrics:
                results[strategy] = metrics
                
        return results
        
    async def cleanup_old_signals(self, days: int = 90):
        """Clean up old signal data to prevent database bloat"""
        if not self.pool:
            return
            
        try:
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
            
            async with self.pool.acquire() as conn:
                deleted_signals = await conn.fetchval("""
                    DELETE FROM ml_strategy_signals 
                    WHERE entry_timestamp < $1 
                    RETURNING COUNT(*)
                """, cutoff_date)
                
                deleted_training = await conn.fetchval("""
                    DELETE FROM ml_training_log 
                    WHERE training_timestamp < $1 
                    RETURNING COUNT(*)
                """, cutoff_date)
                
                logger.info(f"Cleaned up {deleted_signals} old signals and {deleted_training} old training logs")
                
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")


# Global performance tracker instance
_performance_tracker: Optional[MLPerformanceTracker] = None

async def get_performance_tracker() -> Optional[MLPerformanceTracker]:
    """Get the global performance tracker instance"""
    return _performance_tracker

async def initialize_performance_tracker(database_url: str):
    """Initialize the global performance tracker"""
    global _performance_tracker
    _performance_tracker = MLPerformanceTracker(database_url)
    await _performance_tracker.connect()
    logger.info("ML Performance Tracker initialized")

async def cleanup_performance_tracker():
    """Cleanup the global performance tracker"""
    global _performance_tracker
    if _performance_tracker:
        await _performance_tracker.close()
        _performance_tracker = None


# Main function removed - use individual strategy classes directly
# Performance tracking is now the primary function of this module 