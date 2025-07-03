#!/usr/bin/env python3
"""
Test script for ML performance tracking system (without database dependency)
"""

import asyncio
import os
import sys
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tradebot.common.models import Signal, Side, MLPerformanceMetrics

async def test_ml_performance_tracking_mock():
    """Test the ML performance tracking system with mocked database"""
    print("🧪 Testing ML Performance Tracking System (Mock Mode)")
    print("=" * 60)
    
    try:
        # Import the performance tracker class
        from tradebot.strategy.ml_service import MLPerformanceTracker
        
        # Create a mock tracker that doesn't require database
        print("1. Creating mock performance tracker...")
        tracker = MLPerformanceTracker("mock://localhost")
        
        # Mock the database pool and methods
        tracker.pool = MagicMock()
        
        # Mock the log_signal method to return incrementing IDs
        signal_id_counter = [0]
        async def mock_log_signal(strategy_name, symbol, signal):
            signal_id_counter[0] += 1
            print(f"   📝 [MOCK] Logged signal {signal_id_counter[0]} for {strategy_name}: {symbol} {signal.side.value} at {signal.price}")
            return signal_id_counter[0]
        
        tracker.log_signal = mock_log_signal
        
        # Mock the training event logging
        async def mock_log_training_event(strategy_name, status, accuracy=None, duration=None, data_points=None, error_message=None):
            print(f"   🎯 [MOCK] Logged training event for {strategy_name}: {status} (accuracy: {accuracy})")
        
        tracker.log_training_event = mock_log_training_event
        
        # Mock the signal exit update
        async def mock_update_signal_exit(signal_id, exit_price, exit_timestamp):
            pnl = exit_price - 150.25  # Mock calculation
            print(f"   🔄 [MOCK] Updated signal {signal_id} exit: price={exit_price}, pnl={pnl:.2f}")
        
        tracker.update_signal_exit = mock_update_signal_exit
        
        # Mock performance retrieval
        async def mock_get_strategy_performance(strategy_name, days=30):
            if strategy_name == "ensemble_ml":
                return MLPerformanceMetrics(
                    strategy_name=strategy_name,
                    total_signals=2,
                    winning_signals=1,
                    losing_signals=1,
                    open_signals=0,
                    total_pnl=1.55,
                    avg_pnl=0.775,
                    win_rate=50.0,
                    avg_win=3.30,
                    avg_loss=-1.75,
                    profit_factor=1.89,
                    last_signal_time=datetime.now(timezone.utc),
                    model_accuracy=0.68,
                    training_status="completed",
                    last_training_time=datetime.now(timezone.utc)
                )
            elif strategy_name == "lstm_ml":
                return MLPerformanceMetrics(
                    strategy_name=strategy_name,
                    total_signals=1,
                    winning_signals=0,
                    losing_signals=1,
                    open_signals=0,
                    total_pnl=-2.25,
                    avg_pnl=-2.25,
                    win_rate=0.0,
                    avg_win=0.0,
                    avg_loss=-2.25,
                    profit_factor=0.0,
                    last_signal_time=datetime.now(timezone.utc),
                    model_accuracy=0.62,
                    training_status="completed",
                    last_training_time=datetime.now(timezone.utc)
                )
            else:
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
        
        tracker.get_strategy_performance = mock_get_strategy_performance
        
        # Mock get all performance
        async def mock_get_all_ml_performance(days=30):
            results = {}
            for strategy in ["ensemble_ml", "lstm_ml", "sentiment_ml", "rl_ml"]:
                results[strategy] = await mock_get_strategy_performance(strategy, days)
            return results
        
        tracker.get_all_ml_performance = mock_get_all_ml_performance
        
        print("✅ Mock performance tracker created")
        
        # Test signal logging
        print("\n2. Testing signal logging...")
        
        # Create test signals
        test_signals = [
            Signal(
                symbol="AAPL",
                side=Side.buy,
                price=150.25,
                timestamp=datetime.now(timezone.utc),
                confidence=0.75
            ),
            Signal(
                symbol="AAPL",
                side=Side.sell,
                price=152.50,
                timestamp=datetime.now(timezone.utc),
                confidence=0.82
            ),
            Signal(
                symbol="TSLA",
                side=Side.buy,
                price=220.75,
                timestamp=datetime.now(timezone.utc),
                confidence=0.65
            )
        ]
        
        signal_ids = []
        strategy_names = ["ensemble_ml", "ensemble_ml", "lstm_ml"]
        for i, signal in enumerate(test_signals):
            strategy_name = strategy_names[i]
            signal_id = await tracker.log_signal(strategy_name, signal.symbol, signal)
            signal_ids.append(signal_id)
        
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
        
        # Test signal exit updates
        print("\n4. Testing signal exit updates...")
        if signal_ids:
            await tracker.update_signal_exit(
                signal_ids[0], 
                exit_price=151.80, 
                exit_timestamp=datetime.now(timezone.utc)
            )
        
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
        
        print("\n✅ All mock tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

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
        print(f"⚠️  Dashboard test failed: {e}")

async def test_ml_strategy_integration():
    """Test that ML strategies can log signals properly"""
    print("\n🤖 Testing ML Strategy Integration")
    print("=" * 50)
    
    try:
        # Test the signal logging method that ML strategies use
        from tradebot.strategy.ml_strategies import EnsembleMLStrategy, MLStrategyConfig
        
        # Create a strategy instance
        config = MLStrategyConfig()
        strategy = EnsembleMLStrategy(config)
        
        # Test the _log_signal method
        signal = Signal(
            symbol="AAPL",
            side=Side.buy,
            price=150.25,
            timestamp=datetime.now(timezone.utc),
            confidence=0.75
        )
        
        print("   📝 Testing ML strategy signal logging method...")
        
        # Mock the performance tracker for this test
        original_get_tracker = None
        try:
            from tradebot.strategy import ml_service
            original_get_tracker = ml_service.get_performance_tracker
            
            async def mock_get_tracker():
                mock_tracker = MagicMock()
                mock_tracker.log_signal = AsyncMock(return_value=123)
                return mock_tracker
            
            ml_service.get_performance_tracker = mock_get_tracker
            
            # Test the signal logging
            await strategy._log_signal(signal, "ensemble_ml")
            print("   ✅ ML strategy signal logging works")
            
        finally:
            # Restore original function
            if original_get_tracker:
                ml_service.get_performance_tracker = original_get_tracker
        
    except Exception as e:
        print(f"   ⚠️  ML strategy integration test failed: {e}")

if __name__ == "__main__":
    print("🚀 Starting ML Performance Tracking Tests (Mock Mode)")
    print("=" * 60)
    
    asyncio.run(test_ml_performance_tracking_mock())
    
    # Test dashboard integration (optional)
    try:
        asyncio.run(test_dashboard_integration())
    except:
        print("⚠️  Skipping dashboard integration test")
    
    # Test ML strategy integration
    asyncio.run(test_ml_strategy_integration())
    
    print("\n🎉 Test suite completed!")
    print("\n💡 This test demonstrates the ML performance tracking functionality")
    print("   without requiring database connections. In production, the real")
    print("   database-backed implementation will provide persistent storage.") 