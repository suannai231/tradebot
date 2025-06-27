# 📈 Buy Time Strategies Guide

This guide covers comprehensive strategies for finding optimal buy times based on historical data analysis.

## 🎯 Overview

Finding the right time to buy is crucial for successful trading. This guide presents multiple approaches ranging from simple technical indicators to advanced machine learning techniques, all designed to work with your existing trading system.

## 📊 Strategy Categories

### 1. **Technical Analysis Strategies**

#### **Moving Average Crossovers**
- **Simple Moving Average (SMA)**: Your current implementation
- **Exponential Moving Average (EMA)**: More responsive to recent price changes
- **Multiple Timeframe Analysis**: Combine daily and hourly signals

```python
# Golden Cross: Short MA crosses above Long MA
if short_ma > long_ma and previous_short_ma <= previous_long_ma:
    return Signal(side=Side.buy, ...)
```

#### **Momentum Indicators**
- **RSI (Relative Strength Index)**: Identifies oversold conditions
- **MACD (Moving Average Convergence Divergence)**: Trend and momentum
- **Stochastic Oscillator**: Momentum and overbought/oversold levels

```python
# RSI Buy Signal: Price oversold (RSI < 30)
if rsi < 30 and rsi_trending_up:
    return Signal(side=Side.buy, ...)
```

#### **Volume-Based Strategies**
- **Volume Price Trend (VPT)**: Volume-weighted price momentum
- **On-Balance Volume (OBV)**: Cumulative volume indicator
- **Volume Rate of Change**: Volume momentum

```python
# High volume with price increase
if volume > avg_volume * 1.5 and price > previous_price:
    return Signal(side=Side.buy, ...)
```

### 2. **Pattern Recognition Strategies**

#### **Candlestick Patterns**
- **Bullish Engulfing**: Strong reversal signal
- **Hammer**: Potential reversal at support
- **Morning Star**: Three-candle reversal pattern
- **Doji**: Indecision, potential reversal

#### **Chart Patterns**
- **Support and Resistance Levels**: Key price levels
- **Double Bottom**: W-shaped reversal pattern
- **Cup and Handle**: Continuation pattern
- **Flag and Pennant**: Continuation patterns

### 3. **Statistical Strategies**

#### **Mean Reversion**
- **Bollinger Bands**: Price relative to moving average
- **Z-Score Analysis**: Statistical deviation from mean
- **Pivot Points**: Key support/resistance levels

```python
# Price at lower Bollinger Band
if current_price <= lower_band:
    return Signal(side=Side.buy, ...)
```

#### **Volatility-Based**
- **Average True Range (ATR)**: Volatility measurement
- **Volatility Breakouts**: Price breaking out of range
- **Volatility Contraction**: Low volatility before breakout

### 4. **Machine Learning Strategies**

#### **Feature Engineering**
- Technical indicators as features
- Price action patterns
- Volume characteristics
- Market microstructure data

#### **Model Types**
- **Random Forest**: Ensemble learning
- **Gradient Boosting**: Sequential learning
- **Neural Networks**: Deep learning
- **Support Vector Machines**: Classification

## 🚀 Implementation in Your System

### **Available Strategies**

Your system now includes four main strategy types:

#### **1. Simple Moving Average (simple_ma)**
```python
# Basic SMA crossover strategy
strategy = create_strategy("simple_ma")
```

#### **2. Advanced Multi-Factor (advanced)**
```python
# Combines multiple technical indicators
strategy = create_strategy("advanced", 
    min_composite_score=0.5,  # Reduced threshold for better signal generation
    rsi_oversold=30,
    rsi_overbought=70,
    short_window=3,  # Optimized for limited data
    long_window=10   # Optimized for limited data
)
```

#### **3. Mean Reversion (mean_reversion)**
```python
# Statistical mean reversion strategy
strategy = create_strategy("mean_reversion",
    lookback_period=10,  # Reduced from 20 for better performance
    z_score_threshold=-1.5  # Less strict threshold
)
```

#### **4. Low Volume Strategy (low_volume)**
```python
# Optimized for low-volume stocks and limited data
strategy = create_strategy("low_volume",
    short_window=3,
    long_window=8,
    rsi_period=7
)
```

### **Running Strategies**

#### **Live Trading**
```bash
# Use advanced strategy in live trading
python -m tradebot.strategy.service --strategy advanced
```

#### **Backtesting**
```bash
# Test strategies on historical data (now with 1 year of data)
python run_strategy_backtest.py --strategies advanced mean_reversion low_volume --symbols AAPL MSFT GOOG

# Custom date range
python run_strategy_backtest.py --start-date 2024-01-01 --end-date 2024-12-31

# Save results to file
python run_strategy_backtest.py --output results.txt
```

## 📈 Advanced Strategy Details

### **Composite Scoring System**

The advanced strategy uses a weighted scoring system:

| Factor | Weight | Description |
|--------|--------|-------------|
| MA Crossover | 25% | Short vs long moving average |
| RSI | 20% | Relative strength index |
| Bollinger Bands | 20% | Price position relative to bands |
| Support Levels | 15% | Proximity to support |
| Volume | 10% | Volume analysis |
| Patterns | 10% | Candlestick patterns |

### **Signal Generation Logic**

```python
# Buy Signal Conditions
if composite_score >= 0.7 and not currently_long:
    return Signal(side=Side.buy, confidence=composite_score)

# Sell Signal Conditions  
if composite_score < 0.3 and currently_long:
    return Signal(side=Side.sell, confidence=1.0 - composite_score)
```

## 🔧 Configuration Options

### **Strategy Parameters**

#### **Advanced Strategy**
```python
config = StrategyConfig(
    # Moving Average settings
    short_window=5,
    long_window=20,
    
    # RSI settings
    rsi_period=14,
    rsi_oversold=30,
    rsi_overbought=70,
    
    # Bollinger Bands
    bb_period=20,
    bb_std_dev=2.0,
    
    # Volume settings
    volume_period=20,
    volume_threshold=1.5,
    
    # Support/Resistance
    support_window=20,
    support_threshold=0.02,
    
    # Signal generation
    min_composite_score=0.7
)
```

#### **Mean Reversion Strategy**
```python
config = {
    'lookback_period': 20,      # Period for mean calculation
    'z_score_threshold': -2.0   # Standard deviations below mean
}
```

## 📊 Performance Metrics

### **Key Metrics Tracked**

- **Total Return**: Overall percentage gain/loss
- **Win Rate**: Percentage of profitable trades
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Profit Factor**: Gross profit / Gross loss
- **Average Win/Loss**: Average profit and loss per trade

### **Interpreting Results**

| Metric | Good | Excellent |
|--------|------|-----------|
| Win Rate | >50% | >60% |
| Sharpe Ratio | >0.5 | >1.0 |
| Max Drawdown | <20% | <10% |
| Profit Factor | >1.5 | >2.0 |

## 🎯 Strategy Selection Guide

### **Market Conditions**

#### **Trending Markets**
- **Recommended**: Moving Average strategies
- **Avoid**: Mean reversion strategies
- **Focus**: Trend following indicators

#### **Ranging Markets**
- **Recommended**: Mean reversion strategies
- **Avoid**: Trend following strategies
- **Focus**: Support/resistance levels

#### **Volatile Markets**
- **Recommended**: Volatility-based strategies
- **Avoid**: Simple moving averages
- **Focus**: Risk management

### **Time Horizons**

#### **Short-term (Day Trading)**
- Use shorter timeframes (1-5 minute)
- Focus on momentum indicators
- High frequency signals

#### **Medium-term (Swing Trading)**
- Use daily timeframes
- Combine multiple indicators
- Moderate signal frequency

#### **Long-term (Position Trading)**
- Use weekly/monthly timeframes
- Focus on trend indicators
- Low signal frequency

## 🔍 Strategy Optimization

### **Parameter Optimization**

```bash
# Test different RSI thresholds
python run_strategy_backtest.py --strategies advanced \
    --custom-params '{"rsi_oversold": 25, "rsi_overbought": 75}'

# Test different moving average periods
python run_strategy_backtest.py --strategies advanced \
    --custom-params '{"short_window": 3, "long_window": 15}'
```

### **Walk-Forward Analysis**

1. **Training Period**: Use first 60% of data
2. **Validation Period**: Use next 20% of data
3. **Testing Period**: Use final 20% of data

### **Cross-Validation**

- Test on multiple symbols
- Test on different time periods
- Test on different market conditions

## ⚠️ Risk Management

### **Position Sizing**

```python
# Risk per trade (1-2% of capital)
risk_per_trade = 0.02
position_size = (capital * risk_per_trade) / stop_loss_distance
```

### **Stop Losses**

- **Fixed Percentage**: 2-5% below entry
- **ATR-based**: 2-3 ATR below entry
- **Support-based**: Below key support level

### **Take Profits**

- **Risk-Reward Ratio**: 2:1 or 3:1
- **Technical Levels**: At resistance levels
- **Trailing Stops**: Dynamic profit protection

## 📚 Best Practices

### **Data Quality**
- Use clean, reliable data sources
- Handle missing data appropriately
- Account for corporate actions

### **Overfitting Prevention**
- Use out-of-sample testing
- Limit parameter complexity
- Regular strategy re-evaluation

### **Transaction Costs**
- Include commission costs in backtests
- Account for slippage
- Consider bid-ask spreads

### **Market Regime Awareness**
- Adapt strategies to market conditions
- Monitor correlation changes
- Regular performance review

## 🚀 Quick Start Examples

### **Basic Usage**
```bash
# Test all strategies on major stocks
python run_strategy_backtest.py --symbols AAPL MSFT GOOG AMZN TSLA

# Test specific strategy
python run_strategy_backtest.py --strategies advanced

# Custom time period
python run_strategy_backtest.py --start-date 2023-01-01 --end-date 2023-12-31
```

### **Advanced Usage**
```bash
# Compare strategies with custom parameters
python run_strategy_backtest.py \
    --strategies advanced mean_reversion \
    --symbols AAPL MSFT \
    --output comparison_results.txt

# Test on specific market conditions
python run_strategy_backtest.py \
    --start-date 2020-03-01 --end-date 2020-12-31 \
    --strategies advanced
```

## 📈 Next Steps

1. **Run Initial Backtests**: Test strategies on your data
2. **Analyze Results**: Identify best-performing strategies
3. **Optimize Parameters**: Fine-tune for your specific needs
4. **Paper Trading**: Test in simulated environment
5. **Live Trading**: Start with small positions
6. **Monitor Performance**: Regular review and adjustment

## 🔗 Related Files

- `tradebot/strategy/advanced_strategies.py` - Strategy implementations
- `tradebot/strategy/backtest.py` - Backtesting framework
- `run_strategy_backtest.py` - Command-line backtest runner
- `tradebot/strategy/service.py` - Live trading service

## 📞 Support

For questions or issues:
1. Check the backtest results for strategy performance
2. Review the configuration parameters
3. Examine the historical data quality
4. Consider market conditions during testing periods

---

*Remember: Past performance does not guarantee future results. Always use proper risk management and test thoroughly before live trading.* 