#!/usr/bin/env python3
"""
Test script for ML performance tracking system
"""

import asyncio
import os
import sys
from datetime import datetime, timezone
from dotenv import load_dotenv

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tradebot.common.models import PriceTick, Signal, Side
from tradebot.strategy.ml_service import initialize_performance_tracker, get_performance_tracker, cleanup_performance_tracker

# Load environment variables
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://tradebot:tradebot123@localhost:5432/tradebot")

async def test_ml_performance_tracking():
    """Test the ML performance tracking system"""
    print("🧪 Testing ML Performance Tracking System")
    print("=" * 50)
    
    try:
        # Initialize the performance tracker
        print("1. Initializing performance tracker...")
        await initialize_performance_tracker(DATABASE_URL)
        tracker = await get_performance_tracker()
        
        if not tracker:
            print("❌ Failed to initialize performance tracker")
            return
        
        print("✅ Performance tracker initialized")
        
        # Test signal logging
        print("\n2. Testing signal logging...")
        
        # Create test signals
        test_signals = [
            Signal(
                symbol="AAPL",
                side=Side.buy,
                price=150.25,
                timestamp=datetime.now(timezone.utc),
                confidence=0.75,
                strategy="ensemble_ml",
                metadata={"test": True, "model_accuracy": 0.68}
            ),
            Signal(
                symbol="AAPL",
                side=Side.sell,
                price=152.50,
                timestamp=datetime.now(timezone.utc),
                confidence=0.82,
                strategy="ensemble_ml",
                metadata={"test": True, "exit_reason": "take_profit"}
            ),
            Signal(
                symbol="TSLA",
                side=Side.buy,
                price=220.75,
                timestamp=datetime.now(timezone.utc),
                confidence=0.65,
                strategy="lstm_ml",
                metadata={"test": True, "predicted_price": 225.00}
            )
        ]
        
        signal_ids = []
        for signal in test_signals:
            signal_id = await tracker.log_signal(signal.strategy, signal.symbol, signal)
            signal_ids.append(signal_id)
            print(f"   📝 Logged signal {signal_id}: {signal.symbol} {signal.side.value} at {signal.price}")
        
        # Test training event logging
        print("\n3. Testing training event logging...")
        await tracker.log_training_event(
            "ensemble_ml", 
            "completed", 
            accuracy=0.68, 
            duration=120, 
            data_points=1000
        )
        await tracker.log_training_event(
            "lstm_ml", 
            "completed", 
            accuracy=0.62, 
            duration=180, 
            data_points=800
        )
        print("   🎯 Logged training events")
        
        # Test signal exit updates
        print("\n4. Testing signal exit updates...")
        if signal_ids:
            # Update the first signal with an exit
            await tracker.update_signal_exit(
                signal_ids[0], 
                exit_price=151.80, 
                exit_timestamp=datetime.now(timezone.utc)
            )
            print(f"   🔄 Updated signal {signal_ids[0]} with exit price 151.80")
        
        # Test performance retrieval
        print("\n5. Testing performance retrieval...")
        
        # Get performance for specific strategy
        ensemble_performance = await tracker.get_strategy_performance("ensemble_ml")
        if ensemble_performance:
            print(f"   📊 Ensemble ML Performance:")
            print(f"      - Total signals: {ensemble_performance.total_signals}")
            print(f"      - Winning signals: {ensemble_performance.winning_signals}")
            print(f"      - Win rate: {ensemble_performance.win_rate:.1f}%")
            print(f"      - Total PnL: ${ensemble_performance.total_pnl:.2f}")
            print(f"      - Training status: {ensemble_performance.training_status}")
        
        # Get all ML performance
        all_performance = await tracker.get_all_ml_performance()
        print(f"\n   📈 All ML Strategy Performance:")
        for strategy_name, metrics in all_performance.items():
            print(f"      {strategy_name}: {metrics.total_signals} signals, "
                  f"{metrics.win_rate:.1f}% win rate, ${metrics.total_pnl:.2f} PnL")
        
        print("\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        print("\n6. Cleaning up...")
        await cleanup_performance_tracker()
        print("   🧹 Cleanup completed")

async def test_dashboard_integration():
    """Test the dashboard ML performance endpoint"""
    print("\n🌐 Testing Dashboard Integration")
    print("=" * 50)
    
    try:
        import httpx
        
        # Test the ML performance endpoint
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:8001/api/ml-performance")
            
            if response.status_code == 200:
                data = response.json()
                print("✅ Dashboard ML performance endpoint working")
                print("📊 Performance data structure:")
                
                if "performance" in data:
                    for strategy, metrics in data["performance"].items():
                        print(f"   {strategy}: {metrics['signals']} signals, "
                              f"{metrics['wins']} wins, ${metrics['total_pnl']:.2f} PnL")
                
                if "training_status" in data:
                    print("🎯 Training status:")
                    for strategy, status in data["training_status"].items():
                        print(f"   {strategy}: {status['status']}, "
                              f"accuracy: {status['model_accuracy']:.2f}")
            else:
                print(f"❌ Dashboard endpoint failed: {response.status_code}")
                print(f"   Response: {response.text}")
    
    except Exception as e:
        print(f"⚠️  Dashboard test failed (this is expected if dashboard is not running): {e}")

if __name__ == "__main__":
    print("🚀 Starting ML Performance Tracking Tests")
    print("=" * 60)
    
    asyncio.run(test_ml_performance_tracking())
    
    # Test dashboard integration (optional)
    try:
        asyncio.run(test_dashboard_integration())
    except:
        print("⚠️  Skipping dashboard integration test (dashboard not running)")
    
    print("\n🎉 Test suite completed!") 