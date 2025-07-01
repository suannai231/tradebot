#!/usr/bin/env python3
"""
Enhanced Momentum Strategy Trade Analysis

Get detailed buy/sell dates and calculate returns manually for the enhanced momentum strategy.
"""

import asyncio
import sys
import os
from datetime import datetime, timezone, timedelta

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tradebot.strategy.backtest import BacktestEngine

async def analyze_enhanced_momentum_trades():
    """Analyze enhanced momentum strategy trades with detailed buy/sell dates."""
    
    # Database connection
    database_url = "postgresql://postgres:password@localhost:5432/tradebot"
    engine = BacktestEngine(database_url)
    await engine.connect()
    
    # Test parameters
    symbol = "TNXP"  # Focus on TNXP as it's been discussed
    end_date = datetime.now(timezone.utc) - timedelta(days=1)
    start_date = end_date - timedelta(days=365)  # Last 1 year for comprehensive analysis
    
    # Momentum Breakout strategy parameters
    strategy_name = "momentum_breakout"
    strategy_params = {
        "lookback": 15,
        "volume_multiplier": 2.0,
        "support_window": 10
    }
    
    print(f"🔍 {strategy_name.upper()} STRATEGY TRADE ANALYSIS")
    print("=" * 80)
    print(f"Symbol: {symbol}")
    print(f"Date Range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Strategy Parameters: {strategy_params}")
    print("=" * 80)
    
    try:
        # Run backtest
        result = await engine.run_backtest(
            strategy_name, 
            symbol, 
            start_date, 
            end_date, 
            strategy_params,
            adjust_method="backward"  # Use backward adjustment for realistic returns
        )
        
        if not result or not result.trades:
            print("❌ No trades found for the enhanced momentum strategy")
            return
        
        print(f"\n📊 BACKTEST SUMMARY")
        print("-" * 50)
        print(f"Total Trades: {result.total_trades}")
        print(f"Winning Trades: {result.winning_trades}")
        print(f"Losing Trades: {result.losing_trades}")
        print(f"Win Rate: {result.win_rate:.1f}%")
        print(f"Total Return: {result.total_return_pct:.2f}%")
        print(f"Total PnL: ${result.total_pnl:.2f}")
        print(f"Max Drawdown: {result.max_drawdown:.1f}%")
        print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
        
        print(f"\n📈 DETAILED TRADE BREAKDOWN")
        print("-" * 80)
        print(f"{'Trade':<6} {'Entry Date':<12} {'Entry Price':<12} {'Exit Date':<12} {'Exit Price':<12} {'Return %':<10} {'Return $':<10} {'Duration':<10}")
        print("-" * 80)
        
        # Manual calculation variables
        total_manual_return = 0.0
        total_manual_pnl = 0.0
        compound_return = 1.0
        
        for i, trade in enumerate(result.trades, 1):
            # Calculate individual trade metrics
            entry_price = trade.entry_price
            exit_price = trade.exit_price
            entry_date = trade.entry_time.strftime('%Y-%m-%d')
            exit_date = trade.exit_time.strftime('%Y-%m-%d')
            
            # Manual return calculation
            trade_return_pct = ((exit_price / entry_price) - 1) * 100
            trade_return_dollar = exit_price - entry_price
            duration_days = (trade.exit_time - trade.entry_time).days
            
            # Update compound return
            compound_return *= (1 + trade_return_pct / 100)
            
            # Accumulate totals
            total_manual_pnl += trade_return_dollar
            total_manual_return = (compound_return - 1) * 100
            
            # Format output
            return_color = "🟢" if trade_return_pct > 0 else "🔴" if trade_return_pct < 0 else "⚪"
            print(f"{return_color} {i:<4} {entry_date:<12} ${entry_price:<11.4f} {exit_date:<12} ${exit_price:<11.4f} {trade_return_pct:>+8.2f}% ${trade_return_dollar:>+8.4f} {duration_days:>8}d")
        
        print("-" * 80)
        print(f"📊 MANUAL CALCULATION SUMMARY")
        print("-" * 50)
        print(f"Total Manual PnL: ${total_manual_pnl:.4f}")
        print(f"Compound Return: {total_manual_return:.2f}%")
        print(f"Simple Sum of Returns: {sum(((t.exit_price / t.entry_price) - 1) * 100 for t in result.trades):.2f}%")
        
        print(f"\n🔍 VERIFICATION")
        print("-" * 50)
        print(f"Backtest Engine Return: {result.total_return_pct:.2f}%")
        print(f"Manual Compound Return: {total_manual_return:.2f}%")
        print(f"Difference: {abs(result.total_return_pct - total_manual_return):.4f}%")
        
        if abs(result.total_return_pct - total_manual_return) < 0.01:
            print("✅ Manual calculation matches backtest engine!")
        else:
            print("⚠️  Manual calculation differs from backtest engine")
        
        # Show individual trade analysis
        print(f"\n📋 TRADE ANALYSIS")
        print("-" * 50)
        
        winning_trades = [t for t in result.trades if t.pnl and t.pnl > 0]
        losing_trades = [t for t in result.trades if t.pnl and t.pnl < 0]
        
        if winning_trades:
            avg_win = sum(t.pnl for t in winning_trades) / len(winning_trades)
            max_win = max(t.pnl for t in winning_trades)
            print(f"Average Win: ${avg_win:.4f}")
            print(f"Max Win: ${max_win:.4f}")
        
        if losing_trades:
            avg_loss = sum(t.pnl for t in losing_trades) / len(losing_trades)
            max_loss = min(t.pnl for t in losing_trades)
            print(f"Average Loss: ${avg_loss:.4f}")
            print(f"Max Loss: ${max_loss:.4f}")
        
        # Duration analysis
        durations = [(t.exit_time - t.entry_time).days for t in result.trades]
        avg_duration = sum(durations) / len(durations)
        print(f"Average Trade Duration: {avg_duration:.1f} days")
        print(f"Shortest Trade: {min(durations)} days")
        print(f"Longest Trade: {max(durations)} days")
        
    except Exception as e:
        print(f"❌ Error running backtest: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await engine.close()

if __name__ == "__main__":
    asyncio.run(analyze_enhanced_momentum_trades()) 