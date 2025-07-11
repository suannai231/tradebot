#!/usr/bin/env python3
"""
Dashboard Demo Script

Generates sample trading data to showcase the real-time dashboard functionality.
Run this alongside the dashboard to see live updates.
"""

import asyncio
import json
import random
import time
from datetime import datetime, timezone
from typing import List

import redis.asyncio as redis
from tradebot.common.models import PriceTick, TradeSignal

class DashboardDemo:
    def __init__(self):
        self.redis_client = None
        self.symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX", "AMD", "INTC"]
        self.prices = {
            "AAPL": 220.50,
            "MSFT": 430.25,
            "GOOGL": 175.80,
            "AMZN": 185.40,
            "TSLA": 245.60,
            "NVDA": 875.30,
            "META": 520.15,
            "NFLX": 485.75,
            "AMD": 155.90,
            "INTC": 32.45
        }
        self.signal_count = 0
    
    async def connect(self):
        """Connect to Redis"""
        self.redis_client = redis.from_url("redis://localhost:6379")
        await self.redis_client.ping()
        print("✅ Connected to Redis")
    
    async def generate_price_tick(self, symbol: str) -> PriceTick:
        """Generate a realistic price tick"""
        current_price = self.prices[symbol]
        
        # Random price movement (-2% to +2%)
        change_percent = random.uniform(-0.02, 0.02)
        new_price = current_price * (1 + change_percent)
        
        # Update stored price
        self.prices[symbol] = new_price
        
        # Generate OHLCV data
        high = new_price * random.uniform(1.0, 1.005)
        low = new_price * random.uniform(0.995, 1.0)
        volume = random.randint(10000, 500000)
        
        return PriceTick(
            symbol=symbol,
            price=new_price,
            timestamp=datetime.now(timezone.utc),
            open=current_price,
            high=high,
            low=low,
            close=new_price,
            volume=volume,
            trade_count=random.randint(100, 1000),
            vwap=random.uniform(low, high)
        )
    
    async def generate_trading_signal(self, symbol: str) -> TradeSignal:
        """Generate a random trading signal"""
        signal_types = ["BUY", "SELL", "HOLD"]
        signal_type = random.choice(signal_types)
        
        self.signal_count += 1
        
        signal = TradeSignal(
            symbol=symbol,
            signal_type=signal_type,
            price=self.prices[symbol],
            timestamp=datetime.now(timezone.utc)
        )
        
        # Store signal in Redis for API access
        signal_data = {
            "symbol": symbol,
            "signal_type": signal_type,
            "price": self.prices[symbol],
            "timestamp": signal.timestamp.isoformat(),
            "strategy": "demo_strategy",
            "confidence": random.uniform(0.6, 0.95)
        }
        
        await self.redis_client.lpush("trading:signals:recent", json.dumps(signal_data))
        await self.redis_client.ltrim("trading:signals:recent", 0, 49)  # Keep last 50
        await self.redis_client.set("trading:signals:count", self.signal_count)
        
        return signal
    
    async def update_service_heartbeats(self):
        """Update service heartbeat timestamps"""
        services = ["market_data", "strategy", "execution", "storage", "api"]
        
        for service in services:
            heartbeat_key = f"service:{service}:heartbeat"
            await self.redis_client.set(heartbeat_key, datetime.now(timezone.utc).isoformat())
            
            # Randomly add some errors to demonstrate monitoring
            if random.random() < 0.1:  # 10% chance
                error_key = f"service:{service}:errors"
                current_errors = await self.redis_client.get(error_key) or 0
                await self.redis_client.set(error_key, int(current_errors) + 1)
    
    async def publish_price_tick(self, tick: PriceTick):
        """Publish price tick to Redis stream using proper MessageBus format"""
        # Use the MessageBus format to ensure compatibility with all subscribers
        tick_data = tick.model_dump()
        
        # Convert to JSON string and wrap in MessageBus format
        import json
        wrapped_data = {"data": json.dumps(tick_data, default=str)}
        
        await self.redis_client.xadd("price.ticks", wrapped_data)
    
    async def publish_trading_signal(self, signal: TradeSignal):
        """Publish trading signal to Redis stream"""
        signal_data = {
            "symbol": signal.symbol,
            "signal_type": signal.signal_type,
            "price": str(signal.price),
            "timestamp": signal.timestamp.isoformat()
        }
        
        await self.redis_client.xadd("trading.signals", signal_data)
    
    async def run_demo(self, duration_minutes: int = 30):
        """Run the dashboard demo"""
        print(f"🚀 Starting Dashboard Demo for {duration_minutes} minutes...")
        print("📊 Dashboard URL: http://localhost:8001")
        print("=" * 60)
        
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        tick_count = 0
        signal_count = 0
        
        while time.time() < end_time:
            try:
                # Generate price ticks for random symbols (2-5 symbols per iteration)
                active_symbols = random.sample(self.symbols, random.randint(2, 5))
                
                for symbol in active_symbols:
                    tick = await self.generate_price_tick(symbol)
                    await self.publish_price_tick(tick)
                    tick_count += 1
                
                # Occasionally generate trading signals
                if random.random() < 0.3:  # 30% chance
                    signal_symbol = random.choice(self.symbols)
                    signal = await self.generate_trading_signal(signal_symbol)
                    await self.publish_trading_signal(signal)
                    signal_count += 1
                    print(f"📈 {signal.signal_type} signal for {signal.symbol} at ${signal.price:.2f}")
                
                # Update service heartbeats every 10 iterations
                if tick_count % 10 == 0:
                    await self.update_service_heartbeats()
                
                # Status update every 50 ticks
                if tick_count % 50 == 0:
                    elapsed = (time.time() - start_time) / 60
                    print(f"📊 Generated {tick_count} ticks, {signal_count} signals in {elapsed:.1f} minutes")
                
                # Wait before next iteration (1-3 seconds)
                await asyncio.sleep(random.uniform(1.0, 3.0))
                
            except KeyboardInterrupt:
                print("\n⏹️  Demo stopped by user")
                break
            except Exception as e:
                print(f"❌ Error in demo: {e}")
                await asyncio.sleep(1)
        
        print(f"\n✅ Demo completed! Generated {tick_count} ticks and {signal_count} signals")
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.redis_client:
            await self.redis_client.close()

async def main():
    """Main demo function"""
    demo = DashboardDemo()
    
    try:
        await demo.connect()
        await demo.run_demo(duration_minutes=30)  # Run for 30 minutes
    finally:
        await demo.cleanup()

if __name__ == "__main__":
    print("🎛️  Trading Bot Dashboard Demo")
    print("This script generates sample data to showcase the dashboard.")
    print("Make sure Redis is running and start the dashboard first!")
    print()
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Demo stopped. Thanks for trying the dashboard!") 