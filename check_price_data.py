#!/usr/bin/env python3
"""
Price Data Verification Script

Check actual price data around specific dates to verify trade accuracy.
"""

import asyncio
import sys
import os
from datetime import datetime, timezone, timedelta

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from tradebot.strategy.backtest import BacktestEngine

async def check_price_data(symbol: str, target_date: datetime, days_around: int = 5):
    """Check price data around a specific date."""
    
    # Database connection
    database_url = "postgresql://postgres:password@localhost:5432/tradebot"
    engine = BacktestEngine(database_url)
    await engine.connect()
    
    try:
        # Calculate date range
        start_date = target_date - timedelta(days=days_around)
        end_date = target_date + timedelta(days=days_around)
        
        # Get historical data
        ticks = await engine.get_historical_data(symbol, start_date, end_date)
        
        if not ticks:
            print(f"No data found for {symbol} around {target_date.strftime('%Y-%m-%d')}")
            return
        
        print(f"\n📊 PRICE DATA FOR {symbol}")
        print(f"Date Range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        print(f"Target Date: {target_date.strftime('%Y-%m-%d')}")
        print("=" * 80)
        
        # Find the closest tick to target date
        target_timestamp = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
        closest_tick = None
        min_diff = float('inf')
        
        for tick in ticks:
            diff = abs((tick.timestamp - target_timestamp).total_seconds())
            if diff < min_diff:
                min_diff = diff
                closest_tick = tick
        
        print(f"🔍 CLOSEST TICK TO TARGET DATE:")
        if closest_tick:
            print(f"Date: {closest_tick.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}")
            print(f"Price: ${closest_tick.price:.4f}")
            print(f"Open: ${closest_tick.open:.4f}" if closest_tick.open else "Open: N/A")
            print(f"High: ${closest_tick.high:.4f}" if closest_tick.high else "High: N/A")
            print(f"Low: ${closest_tick.low:.4f}" if closest_tick.low else "Low: N/A")
            print(f"Close: ${closest_tick.close:.4f}" if closest_tick.close else "Close: N/A")
            print(f"Volume: {closest_tick.volume}" if closest_tick.volume else "Volume: N/A")
        
        print(f"\n📈 ALL TICKS IN RANGE:")
        print("-" * 80)
        print(f"{'Date':<12} {'Time':<8} {'Price':<10} {'Open':<10} {'High':<10} {'Low':<10} {'Close':<10} {'Volume':<10}")
        print("-" * 80)
        
        for tick in sorted(ticks, key=lambda x: x.timestamp):
            date_str = tick.timestamp.strftime('%Y-%m-%d')
            time_str = tick.timestamp.strftime('%H:%M:%S')
            price_str = f"${tick.price:.4f}"
            open_str = f"${tick.open:.4f}" if tick.open else "N/A"
            high_str = f"${tick.high:.4f}" if tick.high else "N/A"
            low_str = f"${tick.low:.4f}" if tick.low else "N/A"
            close_str = f"${tick.close:.4f}" if tick.close else "Close: N/A"
            volume_str = str(tick.volume) if tick.volume else "N/A"
            
            print(f"{date_str:<12} {time_str:<8} {price_str:<10} {open_str:<10} {high_str:<10} {low_str:<10} {close_str:<10} {volume_str:<10}")
        
        # Check for data quality issues
        print(f"\n🔍 DATA QUALITY CHECK:")
        print("-" * 80)
        
        # Check for missing OHLCV data
        missing_ohlcv = sum(1 for tick in ticks if not all([tick.open, tick.high, tick.low, tick.close, tick.volume]))
        total_ticks = len(ticks)
        
        print(f"Total ticks: {total_ticks}")
        print(f"Ticks with missing OHLCV: {missing_ohlcv}")
        print(f"Data completeness: {((total_ticks - missing_ohlcv) / total_ticks) * 100:.1f}%")
        
        # Check for price anomalies
        prices = [tick.price for tick in ticks]
        if prices:
            min_price = min(prices)
            max_price = max(prices)
            avg_price = sum(prices) / len(prices)
            
            print(f"Price range: ${min_price:.4f} - ${max_price:.4f}")
            print(f"Average price: ${avg_price:.4f}")
            
            # Check for extreme price changes
            price_changes = []
            for i in range(1, len(prices)):
                change = abs((prices[i] - prices[i-1]) / prices[i-1]) * 100
                price_changes.append(change)
            
            if price_changes:
                max_change = max(price_changes)
                avg_change = sum(price_changes) / len(price_changes)
                print(f"Max daily change: {max_change:.2f}%")
                print(f"Average daily change: {avg_change:.2f}%")
        
    finally:
        await engine.close()

async def check_stock_split_period(symbol: str):
    """Check for potential stock split or data anomaly around the suspicious period."""
    
    # Database connection
    database_url = "postgresql://postgres:password@localhost:5432/tradebot"
    engine = BacktestEngine(database_url)
    await engine.connect()
    
    try:
        # Check a wider period around the suspicious jump
        start_date = datetime(2025, 1, 15, tzinfo=timezone.utc)
        end_date = datetime(2025, 2, 15, tzinfo=timezone.utc)
        
        ticks = await engine.get_historical_data(symbol, start_date, end_date)
        
        if not ticks:
            print(f"No data found for {symbol} in the specified period")
            return
        
        print(f"\n🔍 STOCK SPLIT/ANOMALY ANALYSIS FOR {symbol}")
        print(f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        print("=" * 100)
        
        # Sort ticks by timestamp
        sorted_ticks = sorted(ticks, key=lambda x: x.timestamp)
        
        print(f"{'Date':<12} {'Time':<8} {'Price':<10} {'Volume':<12} {'Daily Change':<15} {'Change %':<12}")
        print("-" * 100)
        
        prev_price = None
        for tick in sorted_ticks:
            date_str = tick.timestamp.strftime('%Y-%m-%d')
            time_str = tick.timestamp.strftime('%H:%M:%S')
            price_str = f"${tick.price:.4f}"
            volume_str = f"{tick.volume:,}" if tick.volume else "N/A"
            
            if prev_price is not None:
                change = tick.price - prev_price
                change_pct = (change / prev_price) * 100
                change_str = f"${change:+.4f}"
                change_pct_str = f"{change_pct:+.2f}%"
            else:
                change_str = "N/A"
                change_pct_str = "N/A"
            
            print(f"{date_str:<12} {time_str:<8} {price_str:<10} {volume_str:<12} {change_str:<15} {change_pct_str:<12}")
            
            prev_price = tick.price
        
        # Analyze for potential splits
        print(f"\n🔍 SPLIT ANALYSIS:")
        print("-" * 100)
        
        # Look for price jumps that could indicate splits
        for i in range(1, len(sorted_ticks)):
            prev_tick = sorted_ticks[i-1]
            curr_tick = sorted_ticks[i]
            
            price_change = abs((curr_tick.price - prev_tick.price) / prev_tick.price)
            
            if price_change > 5.0:  # More than 500% change
                print(f"⚠️  SUSPICIOUS PRICE JUMP DETECTED:")
                print(f"   From: {prev_tick.timestamp.strftime('%Y-%m-%d')} ${prev_tick.price:.4f}")
                print(f"   To:   {curr_tick.timestamp.strftime('%Y-%m-%d')} ${curr_tick.price:.4f}")
                print(f"   Change: {price_change*100:.2f}%")
                print(f"   Volume change: {prev_tick.volume:,} → {curr_tick.volume:,}")
                
                # Check if this looks like a reverse split
                if curr_tick.price > prev_tick.price * 10 and curr_tick.volume < prev_tick.volume:
                    print(f"   🔄 LIKELY REVERSE STOCK SPLIT")
                    split_ratio = curr_tick.price / prev_tick.price
                    print(f"   Estimated ratio: 1:{split_ratio:.0f}")
                elif prev_tick.price > curr_tick.price * 10 and curr_tick.volume > prev_tick.volume:
                    print(f"   🔄 LIKELY FORWARD STOCK SPLIT")
                    split_ratio = prev_tick.price / curr_tick.price
                    print(f"   Estimated ratio: {split_ratio:.0f}:1")
                else:
                    print(f"   ❓ UNUSUAL PRICE MOVEMENT - POSSIBLE DATA ERROR")
                print()
        
    finally:
        await engine.close()

async def main():
    """Check price data for specific dates mentioned in trades."""
    
    # Dates to check from the trade analysis
    dates_to_check = [
        ("2024-08-12", "Advanced strategy buy signal"),
        ("2024-07-08", "Low volume strategy buy signal"),
        ("2025-01-24", "Simple MA trade #3 buy"),
        ("2025-02-03", "Mean reversion trade #5 buy"),
        ("2025-02-05", "Mean reversion trade #5 sell"),
    ]
    
    symbol = "TNXP"
    
    for date_str, description in dates_to_check:
        print(f"\n{'='*80}")
        print(f"CHECKING: {description}")
        print(f"{'='*80}")
        
        target_date = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        await check_price_data(symbol, target_date, days_around=3)
    
    # Check for stock split or data anomaly
    print(f"\n{'='*100}")
    print(f"STOCK SPLIT/ANOMALY ANALYSIS")
    print(f"{'='*100}")
    await check_stock_split_period(symbol)

if __name__ == "__main__":
    asyncio.run(main()) 