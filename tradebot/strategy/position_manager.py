#!/usr/bin/env python3
"""
Position Manager for ML Strategies
Handles position tracking and PnL calculation by pairing BUY/SELL signals
"""

import asyncio
import logging
from typing import Dict, Optional, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass

from tradebot.common.models import Signal, Side

logger = logging.getLogger("position_manager")

@dataclass
class OpenPosition:
    """Represents an open trading position"""
    signal_id: int
    strategy_name: str
    symbol: str
    side: Side
    entry_price: float
    entry_timestamp: datetime
    confidence: float

class PositionManager:
    """Manages open positions and handles position closing logic"""
    
    def __init__(self):
        # Track open positions: {(strategy_name, symbol): OpenPosition}
        self.open_positions: Dict[Tuple[str, str], OpenPosition] = {}
        self.position_count = 0
        self.closed_count = 0
        
    async def process_signal(self, strategy_name: str, symbol: str, signal: Signal) -> Optional[str]:
        """
        Process a new signal and handle position management
        Returns: Action taken ("opened", "closed", "ignored")
        """
        position_key = (strategy_name, symbol)
        
        try:
            # Check if we have an open position for this strategy+symbol
            open_position = self.open_positions.get(position_key)
            
            if open_position is None:
                # No open position - create new one
                await self._open_position(strategy_name, symbol, signal)
                return "opened"
            
            elif open_position.side != signal.side:
                # Opposite signal - close existing position
                await self._close_position(open_position, signal)
                return "closed"
            
            else:
                # Same side signal - ignore (no pyramiding for now)
                logger.debug(f"Ignoring {signal.side.value} signal for {symbol} - already have open {open_position.side.value} position")
                return "ignored"
                
        except Exception as e:
            logger.error(f"Error processing signal for {strategy_name} {symbol}: {e}")
            return "error"
    
    async def _open_position(self, strategy_name: str, symbol: str, signal: Signal):
        """Open a new position"""
        try:
            # Import here to avoid circular imports
            from tradebot.strategy.ml_service import get_performance_tracker, initialize_performance_tracker
            
            tracker = await get_performance_tracker()
            if not tracker:
                # Initialize tracker with Docker database URL
                database_url = "postgresql://postgres:password@localhost:5432/tradebot"
                logger.info("Initializing ML performance tracker...")
                await initialize_performance_tracker(database_url)
                tracker = await get_performance_tracker()
                
            if not tracker:
                logger.warning("No performance tracker available")
                return
            
            # Log the signal as a new position
            signal_id = await tracker.log_signal(strategy_name, symbol, signal)
            
            if signal_id > 0:
                # Store open position
                position_key = (strategy_name, symbol)
                self.open_positions[position_key] = OpenPosition(
                    signal_id=signal_id,
                    strategy_name=strategy_name,
                    symbol=symbol,
                    side=signal.side,
                    entry_price=signal.price,
                    entry_timestamp=signal.timestamp,
                    confidence=signal.confidence
                )
                
                self.position_count += 1
                logger.info(f"📈 Opened {signal.side.value} position: {strategy_name} {symbol} @ ${signal.price:.4f} (ID: {signal_id})")
            
        except Exception as e:
            logger.error(f"Error opening position: {e}")
    
    async def _close_position(self, open_position: OpenPosition, exit_signal: Signal):
        """Close an existing position"""
        try:
            # Import here to avoid circular imports
            from tradebot.strategy.ml_service import get_performance_tracker, initialize_performance_tracker
            
            tracker = await get_performance_tracker()
            if not tracker:
                # Initialize tracker with Docker database URL
                database_url = "postgresql://postgres:password@localhost:5432/tradebot"
                logger.info("Initializing ML performance tracker...")
                await initialize_performance_tracker(database_url)
                tracker = await get_performance_tracker()
                
            if not tracker:
                logger.warning("No performance tracker available")
                return
            
            # Update the signal with exit information
            await tracker.update_signal_exit(
                signal_id=open_position.signal_id,
                exit_price=exit_signal.price,
                exit_timestamp=exit_signal.timestamp
            )
            
            # Calculate PnL for logging
            if open_position.side == Side.buy:
                pnl = exit_signal.price - open_position.entry_price
            else:  # Side.sell (short position)
                pnl = open_position.entry_price - exit_signal.price
            
            # Remove from open positions
            position_key = (open_position.strategy_name, open_position.symbol)
            del self.open_positions[position_key]
            
            self.closed_count += 1
            
            # Log the trade completion
            duration = (exit_signal.timestamp - open_position.entry_timestamp).total_seconds()
            pnl_pct = (pnl / open_position.entry_price) * 100
            
            logger.info(f"💰 Closed {open_position.side.value} position: {open_position.strategy_name} {open_position.symbol}")
            logger.info(f"   Entry: ${open_position.entry_price:.4f} → Exit: ${exit_signal.price:.4f}")
            logger.info(f"   PnL: ${pnl:.4f} ({pnl_pct:+.2f}%) | Duration: {duration:.0f}s | ID: {open_position.signal_id}")
            
        except Exception as e:
            logger.error(f"Error closing position: {e}")
    
    def get_stats(self) -> Dict[str, int]:
        """Get position manager statistics"""
        return {
            "open_positions": len(self.open_positions),
            "total_opened": self.position_count,
            "total_closed": self.closed_count
        }
    
    def get_open_positions_by_strategy(self, strategy_name: str) -> Dict[str, OpenPosition]:
        """Get all open positions for a specific strategy"""
        return {
            symbol: pos for (strat, symbol), pos in self.open_positions.items() 
            if strat == strategy_name
        }
    
    async def close_all_positions_at_market(self, reason: str = "shutdown"):
        """Close all open positions at current market price (for shutdown)"""
        logger.info(f"Closing {len(self.open_positions)} open positions due to {reason}")
        
        # Create exit signals at current price for all open positions
        positions_to_close = list(self.open_positions.values())
        
        for position in positions_to_close:
            # Create opposite signal to close position
            exit_side = Side.sell if position.side == Side.buy else Side.buy
            exit_signal = Signal(
                symbol=position.symbol,
                side=exit_side,
                price=position.entry_price,  # Use entry price as fallback
                timestamp=datetime.now(timezone.utc),
                confidence=0.5
            )
            
            await self._close_position(position, exit_signal)
            logger.info(f"Force closed {position.symbol} position at entry price")

# Global position manager instance
_position_manager: Optional[PositionManager] = None

def get_position_manager() -> PositionManager:
    """Get the global position manager instance"""
    global _position_manager
    if _position_manager is None:
        _position_manager = PositionManager()
        logger.info("Position Manager initialized")
    return _position_manager

async def process_ml_signal(strategy_name: str, symbol: str, signal: Signal) -> str:
    """
    Convenience function to process ML signals through position manager
    Returns: Action taken ("opened", "closed", "ignored", "error")
    """
    manager = get_position_manager()
    return await manager.process_signal(strategy_name, symbol, signal) 