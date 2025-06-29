#!/usr/bin/env python3
"""
High Return Strategy Backtest Runner

Test enhanced strategies designed to achieve >35% returns.
"""

import asyncio
import sys
import os
from datetime import datetime, timezone, timedelta

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from tradebot.strategy.backtest import BacktestEngine

async def test_high_return_strategies():
    """Test high-return strategies on different symbols."""
    
    # Database connection
    database_url = "postgresql://postgres:password@localhost:5432/tradebot"
    engine = BacktestEngine(database_url)
    await engine.connect()
    
    # Test symbols (mix of stable and volatile)
    symbols = ["MSFT", "TNXP", "AAPL", "TSLA"]
    
    # High-return strategies with parameters
    strategies = {
        "aggressive_mean_reversion": {
            "lookback_period": 15,
            "z_score_threshold": -1.5,
            "stop_loss": 0.015,
            "take_profit": 0.05
        },
        "enhanced_momentum": {
            "lookback": 15,
            "volume_multiplier": 2.0,
            "momentum_threshold": 0.03,
            "stop_loss": 0.02
        },
        "momentum_breakout": {
            "lookback": 15,
            "volume_multiplier": 2.0,
            "support_window": 10
        }
    }
    
    # Date range (last 1 year)
    end_date = datetime.now(timezone.utc) - timedelta(days=1)
    start_date = end_date - timedelta(days=365)
    
    print("🚀 HIGH RETURN STRATEGY BACKTEST")
    print("=" * 60)
    print(f"Date Range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Target: >35% returns")
    print("=" * 60)
    
    results = {}
    
    for symbol in symbols:
        print(f"\n📊 Testing {symbol}")
        print("-" * 40)
        
        symbol_results = {}
        
        for strategy_name, params in strategies.items():
            try:
                # Run backtest with parameters
                result = await engine.run_backtest(strategy_name, symbol, start_date, end_date, params)
                
                if result:
                    total_return = result.total_return_pct
                    total_trades = result.total_trades
                    win_rate = result.win_rate
                    
                    symbol_results[strategy_name] = {
                        'return': total_return,
                        'trades': total_trades,
                        'win_rate': win_rate
                    }
                    
                    # Highlight high returns
                    if total_return > 35:
                        print(f"🎯 {strategy_name}: {total_return:+.2f}% ({total_trades} trades, {win_rate:.1f}% win rate) - TARGET ACHIEVED!")
                    elif total_return > 20:
                        print(f"✅ {strategy_name}: {total_return:+.2f}% ({total_trades} trades, {win_rate:.1f}% win rate) - Good")
                    elif total_return > 10:
                        print(f"📈 {strategy_name}: {total_return:+.2f}% ({total_trades} trades, {win_rate:.1f}% win rate) - Decent")
                    else:
                        print(f"📉 {strategy_name}: {total_return:+.2f}% ({total_trades} trades, {win_rate:.1f}% win rate) - Poor")
                else:
                    print(f"❌ {strategy_name}: Failed to run")
                    
            except Exception as e:
                print(f"❌ {strategy_name}: Error - {e}")
        
        results[symbol] = symbol_results
    
    await engine.close()
    
    # Summary
    print(f"\n" + "=" * 60)
    print("📊 HIGH RETURN STRATEGY SUMMARY")
    print("=" * 60)
    
    for symbol, symbol_results in results.items():
        print(f"\n{symbol}:")
        for strategy_name, result in symbol_results.items():
            return_val = result['return']
            trades = result['trades']
            win_rate = result['win_rate']
            
            if return_val > 35:
                print(f"  🎯 {strategy_name}: {return_val:+.2f}% ({trades} trades)")
            elif return_val > 20:
                print(f"  ✅ {strategy_name}: {return_val:+.2f}% ({trades} trades)")
            else:
                print(f"  📉 {strategy_name}: {return_val:+.2f}% ({trades} trades)")
    
    # Find best performers
    print(f"\n🏆 BEST PERFORMERS (>35% target):")
    high_performers = []
    
    for symbol, symbol_results in results.items():
        for strategy_name, result in symbol_results.items():
            if result['return'] > 35:
                high_performers.append({
                    'symbol': symbol,
                    'strategy': strategy_name,
                    'return': result['return'],
                    'trades': result['trades']
                })
    
    if high_performers:
        high_performers.sort(key=lambda x: x['return'], reverse=True)
        for i, performer in enumerate(high_performers, 1):
            print(f"  {i}. {performer['symbol']} - {performer['strategy']}: {performer['return']:+.2f}% ({performer['trades']} trades)")
    else:
        print("  No strategies achieved >35% target")
    
    print(f"\n💡 RECOMMENDATIONS:")
    print(f"• Focus on volatile stocks (like TNXP) for momentum strategies")
    print(f"• Use tighter stop-losses and take-profits for better risk management")
    print(f"• Consider longer timeframes for trend-following strategies")
    print(f"• Test during high-volatility market periods")

async def main():
    """Main execution function."""
    await test_high_return_strategies()

if __name__ == "__main__":
    asyncio.run(main()) 