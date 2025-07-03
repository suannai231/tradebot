#!/usr/bin/env python3
"""
Test script for Machine Learning Trading Strategies
"""

import asyncio
import logging
from datetime import datetime, timezone
import numpy as np

from tradebot.common.models import PriceTick, Signal, Side
from tradebot.strategy.advanced_strategies import create_strategy
from tradebot.strategy.ml_strategies import create_ml_strategy, MLStrategyConfig
from tradebot.strategy.rl_strategies import create_rl_strategy, RLStrategyConfig

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("test_ml_strategies")


def generate_test_data(symbol: str = "AAPL", num_ticks: int = 200) -> list:
    """Generate synthetic test data"""
    base_price = 150.0
    prices = []
    
    for i in range(num_ticks):
        # Add some trend and noise
        trend = np.sin(i * 0.1) * 5  # Cyclical trend
        noise = np.random.normal(0, 1)  # Random noise
        price = base_price + trend + noise + i * 0.1  # Slight upward trend
        
        tick = PriceTick(
            symbol=symbol,
            price=price,
            timestamp=datetime.now(timezone.utc),
            volume=1000 + np.random.randint(-200, 200)
        )
        prices.append(tick)
    
    return prices


def test_ensemble_strategy():
    """Test Ensemble ML Strategy"""
    logger.info("Testing Ensemble ML Strategy...")
    
    try:
        # Create strategy
        strategy = create_strategy("ensemble_ml", 
            lookback_period=30,
            confidence_threshold=0.5,
            retrain_frequency=50
        )
        
        # Generate test data
        test_data = generate_test_data("AAPL", 100)
        
        # Process ticks
        signals = []
        for tick in test_data:
            signal = strategy.on_tick(tick)
            if signal:
                signals.append(signal)
                logger.info(f"Ensemble signal: {signal.symbol} {signal.side} @ {signal.price:.2f}")
        
        logger.info(f"Ensemble strategy generated {len(signals)} signals")
        return len(signals) > 0
        
    except Exception as e:
        logger.error(f"Ensemble strategy test failed: {e}")
        return False


def test_lstm_strategy():
    """Test LSTM ML Strategy"""
    logger.info("Testing LSTM ML Strategy...")
    
    try:
        # Create strategy
        strategy = create_strategy("lstm_ml",
            lookback_period=30,
            prediction_horizon=3,
            confidence_threshold=0.5
        )
        
        # Generate test data
        test_data = generate_test_data("MSFT", 100)
        
        # Process ticks
        signals = []
        for tick in test_data:
            signal = strategy.on_tick(tick)
            if signal:
                signals.append(signal)
                logger.info(f"LSTM signal: {signal.symbol} {signal.side} @ {signal.price:.2f}")
        
        logger.info(f"LSTM strategy generated {len(signals)} signals")
        return len(signals) > 0
        
    except Exception as e:
        logger.error(f"LSTM strategy test failed: {e}")
        return False


async def test_sentiment_strategy():
    """Test Sentiment ML Strategy"""
    logger.info("Testing Sentiment ML Strategy...")
    
    try:
        # Create strategy
        strategy = create_strategy("sentiment_ml",
            use_sentiment=True,
            confidence_threshold=0.5,
            lookback_period=20
        )
        
        # Generate test data
        test_data = generate_test_data("GOOGL", 100)
        
        # Process ticks
        signals = []
        for tick in test_data:
            signal = await strategy.on_tick(tick)
            if signal:
                signals.append(signal)
                logger.info(f"Sentiment signal: {signal.symbol} {signal.side} @ {signal.price:.2f}")
        
        logger.info(f"Sentiment strategy generated {len(signals)} signals")
        return len(signals) > 0
        
    except Exception as e:
        logger.error(f"Sentiment strategy test failed: {e}")
        return False


def test_rl_strategy():
    """Test Reinforcement Learning Strategy"""
    logger.info("Testing RL Strategy...")
    
    try:
        # Create strategy
        strategy = create_strategy("rl_ml",
            initial_balance=10000.0,
            max_position_size=0.1,
            lookback_period=30
        )
        
        # Generate test data
        test_data = generate_test_data("TSLA", 100)
        
        # Process ticks
        signals = []
        for tick in test_data:
            signal = strategy.on_tick(tick)
            if signal:
                signals.append(signal)
                logger.info(f"RL signal: {signal.symbol} {signal.side} @ {signal.price:.2f}")
        
        logger.info(f"RL strategy generated {len(signals)} signals")
        return len(signals) > 0
        
    except Exception as e:
        logger.error(f"RL strategy test failed: {e}")
        return False


def test_ml_service():
    """Test ML Strategy Service"""
    logger.info("Testing ML Strategy Service...")
    
    try:
        from tradebot.strategy.ml_service import MLStrategyService
        
        # Create service
        service = MLStrategyService()
        
        # Generate test data
        test_data = generate_test_data("AMZN", 50)
        
        # Process ticks
        signals = []
        for tick in test_data:
            signal = asyncio.run(service.process_tick(tick))
            if signal:
                signals.append(signal)
                logger.info(f"ML Service signal: {signal.symbol} {signal.side} @ {signal.price:.2f}")
        
        # Get performance summary
        summary = service.get_performance_summary()
        logger.info(f"Performance Summary: {summary}")
        
        logger.info(f"ML Service generated {len(signals)} signals")
        return len(signals) > 0
        
    except Exception as e:
        logger.error(f"ML Service test failed: {e}")
        return False


def test_feature_engineering():
    """Test Feature Engineering"""
    logger.info("Testing Feature Engineering...")
    
    try:
        from tradebot.strategy.ml_strategies import FeatureEngineer, MLStrategyConfig
        
        # Create feature engineer
        config = MLStrategyConfig()
        feature_engineer = FeatureEngineer(config)
        
        # Generate test data
        prices = [150.0 + i * 0.1 + np.random.normal(0, 1) for i in range(100)]
        volumes = [1000 + np.random.randint(-200, 200) for _ in range(100)]
        
        # Create features
        features = feature_engineer.create_features(prices, volumes)
        
        logger.info(f"Generated {features.shape[1]} features from {len(prices)} data points")
        logger.info(f"Feature names: {feature_engineer.feature_names[:5]}...")  # Show first 5
        
        return features.shape[0] > 0 and features.shape[1] > 0
        
    except Exception as e:
        logger.error(f"Feature engineering test failed: {e}")
        return False


async def run_all_tests():
    """Run all ML strategy tests"""
    logger.info("Starting ML Strategy Tests...")
    
    results = {}
    
    # Test individual strategies
    results["ensemble"] = test_ensemble_strategy()
    results["lstm"] = test_lstm_strategy()
    results["sentiment"] = await test_sentiment_strategy()
    results["rl"] = test_rl_strategy()
    
    # Test ML service
    results["ml_service"] = test_ml_service()
    
    # Test feature engineering
    results["feature_engineering"] = test_feature_engineering()
    
    # Print results
    logger.info("\n" + "="*50)
    logger.info("ML STRATEGY TEST RESULTS")
    logger.info("="*50)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"{test_name:20} {status}")
    
    # Summary
    passed_count = sum(results.values())
    total_count = len(results)
    
    logger.info("="*50)
    logger.info(f"Overall: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        logger.info("🎉 All ML strategies are working correctly!")
    else:
        logger.warning("⚠️  Some tests failed. Check the logs above for details.")
    
    return passed_count == total_count


if __name__ == "__main__":
    try:
        success = asyncio.run(run_all_tests())
        exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("Tests interrupted by user")
        exit(1)
    except Exception as e:
        logger.error(f"Test suite failed: {e}")
        exit(1) 