#!/usr/bin/env python3
"""
Test script for Dashboard ML Strategy Integration
"""

import asyncio
import logging
import requests
import json
from datetime import datetime, timezone

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("test_dashboard_ml")


def test_dashboard_api():
    """Test dashboard API endpoints"""
    base_url = "http://localhost:8000"
    
    logger.info("Testing Dashboard API endpoints...")
    
    # Test system stats
    try:
        response = requests.get(f"{base_url}/api/system-stats")
        if response.status_code == 200:
            data = response.json()
            logger.info(f"✅ System stats: {data}")
        else:
            logger.error(f"❌ System stats failed: {response.status_code}")
    except Exception as e:
        logger.error(f"❌ System stats error: {e}")
    
    # Test ML performance endpoint
    try:
        response = requests.get(f"{base_url}/api/ml-performance")
        if response.status_code == 200:
            data = response.json()
            logger.info(f"✅ ML Performance data: {json.dumps(data, indent=2)}")
        else:
            logger.error(f"❌ ML Performance failed: {response.status_code}")
    except Exception as e:
        logger.error(f"❌ ML Performance error: {e}")
    
    # Test backtest with ML strategies
    ml_strategies = ["ensemble_ml", "lstm_ml", "sentiment_ml", "rl_ml"]
    
    for strategy in ml_strategies:
        try:
            params = {
                "symbol": "AAPL",
                "strategy": strategy,
                "start": "2024-01-01",
                "end": "2024-01-15"
            }
            response = requests.get(f"{base_url}/api/backtest", params=params)
            if response.status_code == 200:
                data = response.json()
                logger.info(f"✅ Backtest {strategy}: {data.get('total_return_pct', 'N/A')}% return")
            else:
                logger.error(f"❌ Backtest {strategy} failed: {response.status_code}")
        except Exception as e:
            logger.error(f"❌ Backtest {strategy} error: {e}")


def test_ml_strategies():
    """Test ML strategy creation and functionality"""
    logger.info("Testing ML Strategy creation...")
    
    try:
        from tradebot.strategy.advanced_strategies import create_strategy
        
        # Test each ML strategy
        ml_strategies = [
            ("ensemble_ml", {"lookback_period": 30, "confidence_threshold": 0.6}),
            ("lstm_ml", {"lookback_period": 30, "prediction_horizon": 5}),
            ("sentiment_ml", {"use_sentiment": True, "confidence_threshold": 0.6}),
            ("rl_ml", {"initial_balance": 10000.0, "max_position_size": 0.1})
        ]
        
        for strategy_name, config in ml_strategies:
            try:
                strategy = create_strategy(strategy_name, **config)
                logger.info(f"✅ Created {strategy_name} strategy successfully")
                
                # Test strategy has required methods
                if hasattr(strategy, 'on_tick'):
                    logger.info(f"✅ {strategy_name} has on_tick method")
                else:
                    logger.error(f"❌ {strategy_name} missing on_tick method")
                    
            except Exception as e:
                logger.error(f"❌ Failed to create {strategy_name}: {e}")
                
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")


def test_ml_service():
    """Test ML service functionality"""
    logger.info("Testing ML Service...")
    
    try:
        from tradebot.strategy.ml_service import MLStrategyService
        from tradebot.common.models import PriceTick
        
        # Create ML service
        service = MLStrategyService()
        logger.info("✅ ML Service created successfully")
        
        # Test with mock tick
        mock_tick = PriceTick(
            symbol="AAPL",
            price=150.0,
            timestamp=datetime.now(timezone.utc),
            volume=1000
        )
        
        # Test processing tick
        try:
            signal = asyncio.run(service.process_tick(mock_tick))
            logger.info(f"✅ Processed tick, signal: {signal}")
        except Exception as e:
            logger.error(f"❌ Error processing tick: {e}")
        
        # Test performance summary
        try:
            summary = service.get_performance_summary()
            logger.info(f"✅ Performance summary: {summary}")
        except Exception as e:
            logger.error(f"❌ Error getting performance summary: {e}")
            
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")


def test_dashboard_html():
    """Test dashboard HTML loads correctly"""
    logger.info("Testing Dashboard HTML...")
    
    try:
        response = requests.get("http://localhost:8000/")
        if response.status_code == 200:
            html_content = response.text
            
            # Check for ML strategy elements
            ml_indicators = [
                "🤖 Ensemble ML",
                "🧠 LSTM Deep Learning", 
                "📰 Sentiment Analysis",
                "🎯 Reinforcement Learning",
                "Machine Learning Strategy Performance",
                "ml-performance-chart",
                "bt-strategy-selector"
            ]
            
            found_indicators = []
            for indicator in ml_indicators:
                if indicator in html_content:
                    found_indicators.append(indicator)
                    logger.info(f"✅ Found ML indicator: {indicator}")
                else:
                    logger.warning(f"⚠️  Missing ML indicator: {indicator}")
            
            logger.info(f"✅ Dashboard HTML loaded with {len(found_indicators)}/{len(ml_indicators)} ML indicators")
            
        else:
            logger.error(f"❌ Dashboard HTML failed: {response.status_code}")
            
    except Exception as e:
        logger.error(f"❌ Dashboard HTML error: {e}")


def main():
    """Run all tests"""
    logger.info("Starting Dashboard ML Integration Tests...")
    
    # Test ML strategies
    test_ml_strategies()
    
    # Test ML service
    test_ml_service()
    
    # Test dashboard API (if running)
    try:
        test_dashboard_api()
    except Exception as e:
        logger.warning(f"⚠️  Dashboard API test skipped (dashboard may not be running): {e}")
    
    # Test dashboard HTML (if running)
    try:
        test_dashboard_html()
    except Exception as e:
        logger.warning(f"⚠️  Dashboard HTML test skipped (dashboard may not be running): {e}")
    
    logger.info("Dashboard ML Integration Tests completed!")


if __name__ == "__main__":
    main() 