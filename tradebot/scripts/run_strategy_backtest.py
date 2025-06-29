#!/usr/bin/env python3
"""
Strategy Backtest Runner

Run backtests on historical data to evaluate different buy signal strategies.
"""

import asyncio
import argparse
import logging
import os
import sys
from datetime import datetime, timezone, timedelta
from typing import List

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from tradebot.strategy.backtest import BacktestEngine
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("strategy_backtest")


def parse_date(date_str: str) -> datetime:
    """Parse date string in YYYY-MM-DD format."""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid date format: {date_str}. Use YYYY-MM-DD")


async def main():
    parser = argparse.ArgumentParser(description="Run strategy backtests on historical data")
    parser.add_argument("--symbols", nargs="+", default=["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"],
                       help="Symbols to test (default: AAPL MSFT GOOG AMZN TSLA)")
    parser.add_argument("--start-date", type=parse_date, 
                       default=(datetime.now(timezone.utc) - timedelta(days=365)).strftime("%Y-%m-%d"),
                       help="Start date for backtest (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=parse_date,
                       default=(datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d"),
                       help="End date for backtest (YYYY-MM-DD)")
    parser.add_argument("--strategies", nargs="+", 
                       choices=["simple_ma", "advanced", "mean_reversion", "low_volume", "momentum_breakout", "volatility_mean_reversion", "gap_trading", "multi_timeframe", "risk_managed"],
                       default=["simple_ma", "advanced", "mean_reversion", "low_volume", "momentum_breakout", "volatility_mean_reversion", "gap_trading", "multi_timeframe", "risk_managed"],
                       help="Strategies to test")
    parser.add_argument("--database-url", 
                       default=os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/tradebot"),
                       help="Database connection URL")
    parser.add_argument("--output", help="Output file for results (optional)")
    
    args = parser.parse_args()
    
    # Strategy configurations
    strategies = {
        "simple_ma": {},
        "advanced": {
            'min_composite_score': 0.5,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'short_window': 3,
            'long_window': 10
        },
        "mean_reversion": {
            'lookback_period': 10,
            'z_score_threshold': -1.5
        },
        "low_volume": {
            'short_window': 3,
            'long_window': 8,
            'rsi_period': 7
        },
        "momentum_breakout": {
            'lookback': 20,
            'volume_multiplier': 1.5,
            'support_window': 10
        },
        "volatility_mean_reversion": {
            'lookback': 20,
            'std_multiplier': 2.5,
            'rsi_period': 14
        },
        "gap_trading": {
            'gap_threshold': 0.05,
            'volume_threshold': 1.2
        },
        "multi_timeframe": {
            'daily_lookback': 20,
            'weekly_lookback': 5
        },
        "risk_managed": {
            'base_strategy_type': 'mean_reversion',
            'stop_loss': 0.02,
            'max_drawdown': 0.10
        }
    }
    
    # Filter strategies based on command line arguments
    strategies_to_test = {name: config for name, config in strategies.items() if name in args.strategies}
    
    print("🚀 STRATEGY BACKTEST RUNNER")
    print("=" * 60)
    print(f"Symbols: {', '.join(args.symbols)}")
    print(f"Date Range: {args.start_date.strftime('%Y-%m-%d')} to {args.end_date.strftime('%Y-%m-%d')}")
    print(f"Strategies: {', '.join(strategies_to_test.keys())}")
    print(f"Database: {args.database_url}")
    print()
    
    # Initialize backtest engine
    engine = BacktestEngine(args.database_url)
    await engine.connect()
    
    try:
        # Run backtests
        results = await engine.compare_strategies(
            args.symbols, 
            args.start_date, 
            args.end_date, 
            strategies_to_test
        )
        
        # Generate and display report
        report = engine.generate_report(results)
        print(report)
        
        # Save results to file if requested
        if args.output:
            with open(args.output, 'w') as f:
                f.write(report)
            print(f"\n📄 Results saved to: {args.output}")
        
        # Detailed analysis
        print("\n" + "=" * 60)
        print("📈 DETAILED ANALYSIS")
        print("=" * 60)
        
        for strategy_name, strategy_results in results.items():
            if not strategy_results:
                continue
                
            print(f"\n🎯 {strategy_name.upper()} STRATEGY")
            print("-" * 40)
            
            # Find best and worst performing symbols
            sorted_results = sorted(strategy_results, key=lambda x: x.total_return_pct, reverse=True)
            
            if sorted_results:
                best = sorted_results[0]
                worst = sorted_results[-1]
                
                print(f"🏆 Best Performer: {best.symbol}")
                print(f"   Return: {best.total_return_pct:.2f}%")
                print(f"   Trades: {best.total_trades}")
                print(f"   Win Rate: {best.win_rate:.1f}%")
                print(f"   Sharpe: {best.sharpe_ratio:.2f}")
                
                print(f"📉 Worst Performer: {worst.symbol}")
                print(f"   Return: {worst.total_return_pct:.2f}%")
                print(f"   Trades: {worst.total_trades}")
                print(f"   Win Rate: {worst.win_rate:.1f}%")
                print(f"   Sharpe: {worst.sharpe_ratio:.2f}")
                
                # Strategy consistency
                positive_returns = [r for r in strategy_results if r.total_return_pct > 0]
                consistency = (len(positive_returns) / len(strategy_results)) * 100
                print(f"📊 Consistency: {consistency:.1f}% of symbols profitable")
        
        # Strategy comparison
        print("\n" + "=" * 60)
        print("🏁 STRATEGY COMPARISON")
        print("=" * 60)
        
        comparison_data = []
        for strategy_name, strategy_results in results.items():
            if strategy_results:
                avg_return = sum(r.total_return_pct for r in strategy_results) / len(strategy_results)
                avg_trades = sum(r.total_trades for r in strategy_results) / len(strategy_results)
                avg_win_rate = sum(r.win_rate for r in strategy_results) / len(strategy_results)
                avg_sharpe = sum(r.sharpe_ratio for r in strategy_results) / len(strategy_results)
                
                comparison_data.append({
                    'name': strategy_name,
                    'avg_return': avg_return,
                    'avg_trades': avg_trades,
                    'avg_win_rate': avg_win_rate,
                    'avg_sharpe': avg_sharpe
                })
        
        # Sort by average return
        comparison_data.sort(key=lambda x: x['avg_return'], reverse=True)
        
        print(f"{'Strategy':<15} {'Avg Return':<12} {'Avg Trades':<12} {'Win Rate':<10} {'Sharpe':<8}")
        print("-" * 60)
        
        for data in comparison_data:
            print(f"{data['name']:<15} {data['avg_return']:<12.2f}% {data['avg_trades']:<12.1f} "
                  f"{data['avg_win_rate']:<10.1f}% {data['avg_sharpe']:<8.2f}")
        
        # Recommendations
        if comparison_data:
            best_strategy = comparison_data[0]
            print(f"\n🎯 RECOMMENDATION:")
            print(f"Best overall strategy: {best_strategy['name']}")
            print(f"Average return: {best_strategy['avg_return']:.2f}%")
            print(f"Average win rate: {best_strategy['avg_win_rate']:.1f}%")
            
            if best_strategy['avg_sharpe'] > 1.0:
                print("✅ Good risk-adjusted returns (Sharpe > 1.0)")
            elif best_strategy['avg_sharpe'] > 0.5:
                print("⚠️  Moderate risk-adjusted returns (Sharpe > 0.5)")
            else:
                print("❌ Poor risk-adjusted returns (Sharpe < 0.5)")
    
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        return 1
    finally:
        await engine.close()
    
    return 0


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⚠️  Backtest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Backtest failed: {e}")
        sys.exit(1) 