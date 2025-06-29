#!/usr/bin/env python3
"""
Trade Analysis Tool

Analyze trade performance and patterns from backtest results.
"""

import asyncio
import asyncpg
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
import sys
import os
import argparse

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from tradebot.strategy.backtest import BacktestEngine
from tradebot.strategy.advanced_strategies import create_strategy

async def analyze_symbol_trades(symbol: str, start_date: datetime, end_date: datetime):
    """Analyze trades and price movements for a symbol."""
    
    # Database connection
    database_url = "postgresql://postgres:password@localhost:5432/tradebot"
    engine = BacktestEngine(database_url)
    await engine.connect()
    
    try:
        # Get historical data
        ticks = await engine.get_historical_data(symbol, start_date, end_date)
        if not ticks:
            print(f"No data found for {symbol}")
            return
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame([
            {
                'timestamp': tick.timestamp,
                'price': tick.price,
                'open': tick.open,
                'high': tick.high,
                'low': tick.low,
                'close': tick.close,
                'volume': tick.volume
            }
            for tick in ticks
        ])
        
        print(f"\n📊 ANALYSIS FOR {symbol}")
        print("=" * 60)
        
        # Basic price statistics
        print(f"Data Period: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"Total Days: {len(df)}")
        print(f"Price Range: ${df['price'].min():.4f} - ${df['price'].max():.4f}")
        print(f"Average Price: ${df['price'].mean():.4f}")
        print(f"Price Volatility: {df['price'].std():.4f}")
        
        # Calculate daily returns
        df['daily_return'] = df['price'].pct_change()
        df['cumulative_return'] = (1 + df['daily_return']).cumprod() - 1
        
        print(f"\n📈 PRICE MOVEMENT ANALYSIS")
        print(f"Total Price Change: {((df['price'].iloc[-1] / df['price'].iloc[0]) - 1) * 100:.2f}%")
        print(f"Best Day: {df['daily_return'].max() * 100:.2f}%")
        print(f"Worst Day: {df['daily_return'].min() * 100:.2f}%")
        print(f"Average Daily Return: {df['daily_return'].mean() * 100:.2f}%")
        print(f"Daily Return Std: {df['daily_return'].std() * 100:.2f}%")
        
        # Analyze strategy trades
        strategies = {
            'simple_ma': create_strategy("simple_ma"),
            'advanced': create_strategy("advanced", min_composite_score=0.5),
            'mean_reversion': create_strategy("mean_reversion", lookback_period=10),
            'low_volume': create_strategy("low_volume")
        }
        
        print(f"\n🔍 STRATEGY TRADE ANALYSIS")
        print("-" * 60)
        
        for strategy_name, strategy in strategies.items():
            print(f"\n{strategy_name.upper()}:")
            
            # Simulate trades
            trades = []
            current_position = None
            
            for tick in ticks:
                signal = strategy.on_tick(tick)
                
                if signal:
                    if signal.side.value == "BUY" and current_position is None:
                        current_position = {
                            'entry_time': tick.timestamp,
                            'entry_price': tick.price,
                            'entry_index': len(trades)
                        }
                    elif signal.side.value == "SELL" and current_position is not None:
                        trade = {
                            'entry_time': current_position['entry_time'],
                            'entry_price': current_position['entry_price'],
                            'exit_time': tick.timestamp,
                            'exit_price': tick.price,
                            'return_pct': ((tick.price / current_position['entry_price']) - 1) * 100,
                            'return_abs': tick.price - current_position['entry_price'],
                            'duration_days': (tick.timestamp - current_position['entry_time']).days
                        }
                        trades.append(trade)
                        current_position = None
            
            if trades:
                df_trades = pd.DataFrame(trades)
                print(f"  Total Trades: {len(trades)}")
                print(f"  Total Return: {df_trades['return_pct'].sum():.2f}%")
                print(f"  Average Return per Trade: {df_trades['return_pct'].mean():.2f}%")
                print(f"  Win Rate: {(df_trades['return_pct'] > 0).mean() * 100:.1f}%")
                print(f"  Average Duration: {df_trades['duration_days'].mean():.1f} days")
                print(f"  Best Trade: {df_trades['return_pct'].max():.2f}%")
                print(f"  Worst Trade: {df_trades['return_pct'].min():.2f}%")
                
                # Show individual trades
                print(f"  Individual Trades:")
                for i, trade in enumerate(trades):
                    print(f"    Trade {i+1}: {trade['entry_price']:.4f} → {trade['exit_price']:.4f} ({trade['return_pct']:+.2f}%)")
            else:
                print(f"  No trades generated")
        
        # Market trend analysis
        print(f"\n📊 MARKET TREND ANALYSIS")
        print("-" * 60)
        
        # Calculate moving averages
        df['sma_20'] = df['price'].rolling(20).mean()
        df['sma_50'] = df['price'].rolling(50).mean()
        
        # Trend periods
        df['trend'] = np.where(df['sma_20'] > df['sma_50'], 'Uptrend', 'Downtrend')
        
        trend_counts = df['trend'].value_counts()
        print(f"Uptrend Days: {trend_counts.get('Uptrend', 0)}")
        print(f"Downtrend Days: {trend_counts.get('Downtrend', 0)}")
        
        # Volatility periods
        df['volatility'] = df['daily_return'].rolling(20).std()
        high_vol_days = (df['volatility'] > df['volatility'].quantile(0.8)).sum()
        print(f"High Volatility Days: {high_vol_days}")
        
        # Price momentum
        df['momentum_5'] = df['price'].pct_change(5)
        df['momentum_20'] = df['price'].pct_change(20)
        
        positive_momentum_5 = (df['momentum_5'] > 0).sum()
        positive_momentum_20 = (df['momentum_20'] > 0).sum()
        
        print(f"Positive 5-day Momentum: {positive_momentum_5}/{len(df)} days")
        print(f"Positive 20-day Momentum: {positive_momentum_20}/{len(df)} days")
        
    finally:
        await engine.close()

async def main():
    """Run the analysis."""
    symbol = "TNXP"
    end_date = datetime.now(timezone.utc) - timedelta(days=1)
    start_date = end_date - timedelta(days=365)
    
    await analyze_symbol_trades(symbol, start_date, end_date)

if __name__ == "__main__":
    asyncio.run(main()) 