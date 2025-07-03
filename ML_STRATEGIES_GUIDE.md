# 🤖 Machine Learning Trading Strategies Guide

This guide covers all the machine learning strategies implemented in your trading system, from ensemble learning to reinforcement learning.

## 📊 Overview

Your trading system now includes four advanced machine learning strategies that can be used individually or combined for optimal performance:

1. **Ensemble Learning Strategy** - Combines multiple ML models for robust predictions
2. **LSTM Time Series Strategy** - Deep learning for sequence-based price forecasting
3. **Sentiment Analysis Strategy** - Market sentiment-driven trading
4. **Reinforcement Learning Strategy** - Adaptive trading using PPO algorithm

## 🚀 Quick Start

### Installation

All ML dependencies are included in `requirements.txt`. Install them with:

```bash
pip install -r requirements.txt
```

### Running ML Strategies

#### Individual Strategy
```python
from tradebot.strategy.advanced_strategies import create_strategy

# Ensemble ML Strategy
ensemble_strategy = create_strategy("ensemble_ml", 
    lookback_period=50,
    confidence_threshold=0.6
)

# LSTM Strategy
lstm_strategy = create_strategy("lstm_ml",
    lookback_period=50,
    prediction_horizon=5
)

# Sentiment Strategy
sentiment_strategy = create_strategy("sentiment_ml",
    use_sentiment=True,
    confidence_threshold=0.6
)

# Reinforcement Learning Strategy
rl_strategy = create_strategy("rl_ml",
    initial_balance=10000.0,
    max_position_size=0.1
)
```

#### Combined ML Service
```python
# Run all ML strategies together
python -m tradebot.strategy.ml_service
```

## 🧠 Strategy Details

### 1. Ensemble Learning Strategy (`ensemble_ml`)

**What it does:** Combines Random Forest, XGBoost, and LightGBM models using weighted voting.

**Key Features:**
- **Multiple Models:** Random Forest, XGBoost, LightGBM
- **Feature Engineering:** 50+ technical indicators
- **Ensemble Voting:** Soft voting for robust predictions
- **Auto-retraining:** Retrains every 1000 ticks

**Configuration:**
```python
MLStrategyConfig(
    lookback_period=50,          # Historical data window
    prediction_horizon=5,        # Future prediction steps
    min_data_points=100,         # Minimum data for training
    retrain_frequency=1000,      # Retrain every N ticks
    confidence_threshold=0.6,    # Minimum confidence for signals
    use_technical_indicators=True,
    stop_loss=0.02,              # 2% stop loss
    take_profit=0.05             # 5% take profit
)
```

**Best For:** Stable, reliable predictions with good risk management.

### 2. LSTM Time Series Strategy (`lstm_ml`)

**What it does:** Uses Long Short-Term Memory networks for sequence-based price prediction.

**Key Features:**
- **Deep Learning:** LSTM neural networks
- **Sequence Modeling:** Captures temporal dependencies
- **Feature Engineering:** Technical indicators + price sequences
- **Adaptive Learning:** Continuous model updates

**Architecture:**
```
Input Layer → LSTM(50) → Dropout(0.2) → LSTM(50) → Dropout(0.2) → Dense(25) → Dense(1)
```

**Configuration:**
```python
MLStrategyConfig(
    lookback_period=50,          # Sequence length
    prediction_horizon=5,        # Prediction steps ahead
    min_data_points=100,         # Minimum training data
    retrain_frequency=1000,      # Retrain frequency
    confidence_threshold=0.6     # Signal confidence
)
```

**Best For:** Capturing complex temporal patterns and trends.

### 3. Sentiment Analysis Strategy (`sentiment_ml`)

**What it does:** Combines market sentiment with technical analysis for trading decisions.

**Key Features:**
- **News Sentiment:** Real-time sentiment analysis
- **Social Media:** Twitter/Reddit sentiment (simulated)
- **Technical Integration:** Combines with RSI, moving averages
- **Async Processing:** Non-blocking sentiment updates

**Sentiment Sources:**
- News headlines and articles
- Social media posts
- Earnings call transcripts
- Options flow sentiment

**Configuration:**
```python
MLStrategyConfig(
    use_sentiment=True,          # Enable sentiment analysis
    use_technical_indicators=True,
    confidence_threshold=0.6,
    lookback_period=20           # Shorter for sentiment
)
```

**Best For:** Event-driven trading and market sentiment shifts.

### 4. Reinforcement Learning Strategy (`rl_ml`)

**What it does:** Uses PPO (Proximal Policy Optimization) for adaptive trading decisions.

**Key Features:**
- **Custom Environment:** Realistic trading simulation
- **PPO Algorithm:** Stable policy gradient method
- **Risk Management:** Built-in stop-loss and position sizing
- **Continuous Learning:** Adapts to market changes

**Environment Features:**
- **Action Space:** Continuous [-1, 1] for buy/sell/hold
- **Observation Space:** 20-dimensional state vector
- **Reward Function:** Risk-adjusted returns + Sharpe ratio
- **Risk Constraints:** Max drawdown, position limits

**Configuration:**
```python
RLStrategyConfig(
    initial_balance=10000.0,     # Starting capital
    max_position_size=0.1,       # Max 10% per position
    transaction_cost=0.001,      # 0.1% per trade
    learning_rate=0.0003,        # PPO learning rate
    batch_size=64,               # Training batch size
    n_steps=2048,                # PPO steps per update
    gamma=0.99                   # Discount factor
)
```

**Best For:** Adaptive trading in changing market conditions.

## 🔧 Advanced Configuration

### Feature Engineering

All ML strategies use advanced feature engineering:

```python
# Technical Indicators
- SMA/EMA (5, 20 periods)
- RSI, Stochastic, Williams %R
- MACD, Bollinger Bands
- ATR, Volume indicators
- Price action patterns

# Lagged Features
- Price lags (1, 2, 3, 5 periods)
- Volume lags
- Return calculations

# Market Microstructure
- Bid-ask spreads
- Order flow
- Volume profiles
```

### Model Training

**Training Frequency:**
- Ensemble/LSTM: Every 1000 ticks
- RL: Every 5000 ticks
- Sentiment: Every 30 minutes

**Data Requirements:**
- Minimum 100 data points
- Lookback period: 50 ticks
- Feature scaling: StandardScaler/MinMaxScaler

### Risk Management

**Built-in Protections:**
- Stop-loss: 2% default
- Take-profit: 5% default
- Position sizing: 10% max per trade
- Drawdown limits: 20% max

## 📈 Performance Monitoring

### Strategy Performance Tracking

```python
# Get performance summary
summary = ml_service.get_performance_summary()

# Example output:
{
    "ensemble": {
        "total_signals": 150,
        "win_rate": 65.3,
        "total_pnl": 1250.50,
        "avg_pnl_per_signal": 8.34
    },
    "lstm": {
        "total_signals": 120,
        "win_rate": 58.3,
        "total_pnl": 890.25,
        "avg_pnl_per_signal": 7.42
    }
    # ... other strategies
}
```

### Signal Aggregation

The ML service combines signals from all strategies:

```python
# Strategy weights
strategy_weights = {
    "ensemble": 0.3,    # 30% weight
    "lstm": 0.25,       # 25% weight
    "sentiment": 0.25,  # 25% weight
    "rl": 0.2          # 20% weight
}

# Consensus thresholds
buy_threshold = 0.6    # 60% consensus for buy
sell_threshold = 0.6   # 60% consensus for sell
```

## 🎯 Usage Examples

### Backtesting ML Strategies

```python
from tradebot.strategy.backtest import BacktestEngine
from tradebot.strategy.advanced_strategies import create_strategy

# Create ML strategy
strategy = create_strategy("ensemble_ml", 
    lookback_period=50,
    confidence_threshold=0.6
)

# Run backtest
backtest = BacktestEngine()
trades = backtest.run_backtest(strategy, historical_data)
metrics = backtest.calculate_metrics(trades)

print(f"Total Return: {metrics['total_return_pct']:.2f}%")
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {metrics['max_drawdown']:.2f}%")
```

### Real-time Trading

```python
import asyncio
from tradebot.strategy.ml_service import MLStrategyService

async def run_ml_trading():
    ml_service = MLStrategyService()
    
    # Connect to message bus
    bus = MessageBus()
    await bus.connect()
    
    # Subscribe to price ticks
    async for msg in bus.subscribe("price.ticks"):
        tick = PriceTick(**msg)
        signal = await ml_service.process_tick(tick)
        
        if signal:
            await bus.publish("orders.new", signal.model_dump())
            print(f"ML Signal: {signal.symbol} {signal.side} @ {signal.price}")

# Run the service
asyncio.run(run_ml_trading())
```

### Custom Strategy Configuration

```python
# High-frequency trading setup
hf_config = MLStrategyConfig(
    lookback_period=20,          # Shorter lookback
    prediction_horizon=2,        # Near-term predictions
    retrain_frequency=500,       # More frequent retraining
    confidence_threshold=0.7,    # Higher confidence
    stop_loss=0.01,              # Tighter stop-loss
    take_profit=0.03             # Smaller take-profit
)

# Conservative setup
conservative_config = MLStrategyConfig(
    lookback_period=100,         # Longer lookback
    prediction_horizon=10,       # Longer-term predictions
    retrain_frequency=2000,      # Less frequent retraining
    confidence_threshold=0.8,    # Very high confidence
    stop_loss=0.03,              # Wider stop-loss
    take_profit=0.08             # Larger take-profit
)
```

## 🔍 Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# Install missing dependencies
pip install scikit-learn xgboost lightgbm tensorflow torch stable-baselines3 transformers
```

**2. Memory Issues**
```python
# Reduce model complexity
config = MLStrategyConfig(
    lookback_period=30,          # Shorter lookback
    min_data_points=50,          # Less data
    retrain_frequency=2000       # Less frequent training
)
```

**3. Training Time**
```python
# Use smaller models for faster training
config = MLStrategyConfig(
    retrain_frequency=5000,      # Train less often
    min_data_points=200          # More data before training
)
```

### Performance Optimization

**1. Feature Selection**
```python
# Disable unused features
config = MLStrategyConfig(
    use_sentiment=False,         # Disable sentiment
    use_market_data=False        # Disable market data
)
```

**2. Model Tuning**
```python
# Adjust model parameters
ensemble_strategy = create_strategy("ensemble_ml",
    confidence_threshold=0.7,    # Higher threshold
    retrain_frequency=2000       # Less frequent training
)
```

## 📚 Further Reading

### Machine Learning Resources
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)
- [Stable-Baselines3 Guide](https://stable-baselines3.readthedocs.io/)
- [Transformers Library](https://huggingface.co/transformers/)

### Trading Strategy Resources
- [Technical Analysis Library](https://technical-analysis-library-in-python.readthedocs.io/)
- [Quantitative Trading Strategies](https://www.quantstart.com/)
- [Machine Learning for Trading](https://ml4trading.io/)

## 🤝 Contributing

To add new ML strategies:

1. Create strategy class in `ml_strategies.py`
2. Implement `on_tick()` method
3. Add to factory function in `advanced_strategies.py`
4. Update this documentation

## 📞 Support

For questions or issues:
1. Check the troubleshooting section
2. Review the code examples
3. Test with different configurations
4. Monitor performance metrics

---

**Happy Trading with Machine Learning! 🚀📈** 