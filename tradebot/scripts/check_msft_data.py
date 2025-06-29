#!/usr/bin/env python3
"""
MSFT Data Analysis Script

Analyze MSFT historical data to understand market context and strategy performance.
"""

import asyncio
import asyncpg
import os
import sys
from datetime import datetime, timezone, timedelta
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

async def analyze_msft_data():
    """Analyze MSFT historical data."""
    database_url = "postgresql://postgres:password@localhost:5432/tradebot"
    
    print("📊 MSFT HISTORICAL DATA ANALYSIS")
    print("=" * 60)
    
    try:
        pool = await asyncpg.create_pool(database_url)
        
        async with pool.acquire() as conn:
            # Get all MSFT data
            rows = await conn.fetch("""
                SELECT timestamp, price, volume, open_price, high_price, low_price, close_price
                FROM price_ticks 
                WHERE symbol = 'MSFT'
                ORDER BY timestamp ASC
            """)
            
            if not rows:
                print("❌ No MSFT data found in database")
                return
            
            print(f"📈 Found {len(rows)} MSFT data points")
            
            # Convert to lists for analysis
            timestamps = [row['timestamp'] for row in rows]
            prices = [float(row['price']) for row in rows]
            volumes = [row['volume'] if row['volume'] else 0 for row in rows]
            
            # Basic statistics
            start_price = prices[0]
            end_price = prices[-1]
            min_price = min(prices)
            max_price = max(prices)
            
            print(f"\n💰 PRICE ANALYSIS")
            print(f"Start Price: ${start_price:.2f}")
            print(f"End Price: ${end_price:.2f}")
            print(f"Min Price: ${min_price:.2f}")
            print(f"Max Price: ${max_price:.2f}")
            print(f"Total Return: {((end_price/start_price)-1)*100:+.2f}%")
            print(f"Price Range: ${max_price - min_price:.2f}")
            print(f"Volatility: {np.std(prices)/np.mean(prices)*100:.2f}%")
            
            # Date range
            start_date = timestamps[0]
            end_date = timestamps[-1]
            duration = end_date - start_date
            
            print(f"\n📅 TIME PERIOD")
            print(f"Start Date: {start_date.strftime('%Y-%m-%d')}")
            print(f"End Date: {end_date.strftime('%Y-%m-%d')}")
            print(f"Duration: {duration.days} days")
            
            # Monthly returns
            print(f"\n📊 MONTHLY RETURNS")
            monthly_returns = []
            current_month = start_date.replace(day=1)
            
            while current_month <= end_date:
                month_end = (current_month.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
                
                # Find prices for this month
                month_prices = [p for i, p in enumerate(prices) if timestamps[i].month == current_month.month and timestamps[i].year == current_month.year]
                
                if len(month_prices) >= 2:
                    month_return = ((month_prices[-1] / month_prices[0]) - 1) * 100
                    monthly_returns.append(month_return)
                    print(f"{current_month.strftime('%Y-%m')}: {month_return:+.2f}%")
                
                current_month = (current_month.replace(day=28) + timedelta(days=4)).replace(day=1)
            
            # Volatility analysis
            print(f"\n📈 VOLATILITY ANALYSIS")
            daily_returns = []
            for i in range(1, len(prices)):
                daily_return = (prices[i] / prices[i-1] - 1) * 100
                daily_returns.append(daily_return)
            
            print(f"Average Daily Return: {np.mean(daily_returns):+.3f}%")
            print(f"Daily Return Std Dev: {np.std(daily_returns):.3f}%")
            print(f"Max Daily Gain: {max(daily_returns):+.2f}%")
            print(f"Max Daily Loss: {min(daily_returns):+.2f}%")
            
            # Trend analysis
            print(f"\n📊 TREND ANALYSIS")
            
            # Calculate moving averages
            ma_20 = []
            ma_50 = []
            
            for i in range(len(prices)):
                if i >= 19:
                    ma_20.append(np.mean(prices[i-19:i+1]))
                else:
                    ma_20.append(None)
                
                if i >= 49:
                    ma_50.append(np.mean(prices[i-49:i+1]))
                else:
                    ma_50.append(None)
            
            # Count trend changes
            trend_changes = 0
            current_trend = None
            
            for i in range(50, len(prices)):
                if ma_20[i] and ma_50[i]:
                    if ma_20[i] > ma_50[i]:
                        if current_trend != 'up':
                            if current_trend is not None:
                                trend_changes += 1
                            current_trend = 'up'
                    else:
                        if current_trend != 'down':
                            if current_trend is not None:
                                trend_changes += 1
                            current_trend = 'down'
            
            print(f"Trend Changes (20/50 MA): {trend_changes}")
            
            # Support and resistance levels
            print(f"\n🎯 SUPPORT & RESISTANCE")
            
            # Find local minima and maxima
            local_minima = []
            local_maxima = []
            
            for i in range(5, len(prices)-5):
                if all(prices[i] <= prices[j] for j in range(i-5, i+6)):
                    local_minima.append(prices[i])
                if all(prices[i] >= prices[j] for j in range(i-5, i+6)):
                    local_maxima.append(prices[i])
            
            if local_minima:
                print(f"Key Support Levels: ${min(local_minima):.2f}, ${np.percentile(local_minima, 25):.2f}, ${np.median(local_minima):.2f}")
            if local_maxima:
                print(f"Key Resistance Levels: ${np.median(local_maxima):.2f}, ${np.percentile(local_maxima, 75):.2f}, ${max(local_maxima):.2f}")
            
            # Strategy context
            print(f"\n🎯 STRATEGY CONTEXT")
            print(f"• MSFT showed a {((end_price/start_price)-1)*100:+.1f}% return over {duration.days} days")
            print(f"• Average daily volatility: {np.std(daily_returns):.2f}%")
            print(f"• Price range: ${min_price:.2f} - ${max_price:.2f}")
            print(f"• Trend changes: {trend_changes} (indicates {trend_changes} major trend shifts)")
            
            if ((end_price/start_price)-1)*100 < 20:
                print(f"• Low overall return ({((end_price/start_price)-1)*100:+.1f}%) limits strategy upside")
            if np.std(daily_returns) < 2:
                print(f"• Low volatility ({np.std(daily_returns):.2f}%) reduces mean reversion opportunities")
            if trend_changes < 3:
                print(f"• Few trend changes ({trend_changes}) limit momentum strategy opportunities")
            
            print(f"\n💡 RECOMMENDATIONS")
            print(f"• Consider longer timeframes for better trend capture")
            print(f"• Adjust strategy parameters for lower volatility environment")
            print(f"• Focus on mean reversion during consolidation periods")
            print(f"• Use tighter stop-losses due to lower volatility")
        
        await pool.close()
        
    except Exception as e:
        print(f"❌ Error analyzing MSFT data: {e}")

async def main():
    """Main execution function."""
    await analyze_msft_data()

if __name__ == "__main__":
    asyncio.run(main()) 