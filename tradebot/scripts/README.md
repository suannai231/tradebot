# Trading Bot Scripts

This directory contains analysis and backtest scripts for the trading bot system.

## 📊 Backtest Scripts

### `run_strategy_backtest.py` - **Main Strategy Comparison Tool**
**Purpose**: Compare multiple strategies across multiple symbols with comprehensive reporting.

```bash
# Run all strategies on default symbols
python tradebot/scripts/run_strategy_backtest.py

# Test specific strategies and symbols
python tradebot/scripts/run_strategy_backtest.py \
  --symbols AAPL MSFT TSLA \
  --strategies mean_reversion advanced momentum_breakout \
  --start-date 2023-01-01 \
  --end-date 2023-12-31

# Save results to file
python tradebot/scripts/run_strategy_backtest.py --output results.txt
```

**Features**:
- ✅ Multi-strategy comparison
- ✅ Strategy ranking and recommendations  
- ✅ Consistency analysis
- ✅ Comprehensive reporting
- ✅ Command-line interface

---

### `test_high_returns.py` - **High-Return Strategy Tester**
**Purpose**: Test aggressive strategies designed to achieve >35% returns.

```bash
# Test high-return strategies
python tradebot/scripts/test_high_returns.py
```

**Features**:
- 🎯 Focuses on >35% return target
- 🚀 Tests aggressive strategies with tight risk management
- 📈 Performance rating system
- 💡 Strategy recommendations

**Strategies Tested**:
- `aggressive_mean_reversion` - Tight Z-score thresholds with stop-loss/take-profit
- `enhanced_momentum` - Volume-confirmed momentum with risk management
- `momentum_breakout` - Breakout detection with volume confirmation

---

## 🔍 Analysis Scripts

### `check_trade_times.py` - **General Trade Analysis**
**Purpose**: Detailed analysis of individual trades for any strategy.

```bash
# Analyze all strategies for a symbol
python tradebot/scripts/check_trade_times.py --symbol AAPL

# Analyze specific date range
python tradebot/scripts/check_trade_times.py \
  --symbol AAPL \
  --start 2023-01-01 \
  --end 2023-12-31
```

**Features**:
- 📋 Individual trade breakdowns
- ⏰ Entry/exit times and prices
- 📊 Market context analysis
- 💡 Strategy-specific insights

---

### `check_high_return_trade_times.py` - **High-Return Trade Analysis**
**Purpose**: Detailed trade analysis specifically for high-return strategies.

```bash
# Analyze all high-return strategies
python tradebot/scripts/check_high_return_trade_times.py --symbol TNXP

# Analyze specific strategy
python tradebot/scripts/check_high_return_trade_times.py \
  --symbol TNXP \
  --strategy aggressive_mean_reversion
```

**Features**:
- 🎯 High-return performance rating
- 🚀 Target achievement analysis (>35%)
- 📈 Strategy vs buy-and-hold comparison
- 💡 Volatility and market context

---

### `analyze_trades.py` - **Trade Performance Analysis**
**Purpose**: Analyze trade performance patterns and statistics.

```bash
python tradebot/scripts/analyze_trades.py
```

**Features**:
- 📊 Trade performance statistics
- 📈 Pattern analysis
- 🔍 Performance breakdowns

---

### `check_price_data.py` - **Price Data Validation**
**Purpose**: Validate and analyze historical price data quality.

```bash
python tradebot/scripts/check_price_data.py
```

**Features**:
- 📊 Data quality checks
- 📈 Price data statistics
- 🔍 Missing data detection

---

### `check_msft_data.py` - **MSFT-Specific Data Analysis**
**Purpose**: Specialized analysis for MSFT price data and patterns.

```bash
python tradebot/scripts/check_msft_data.py
```

**Features**:
- 📊 MSFT-specific analysis
- 📈 Price pattern detection
- 🔍 Data validation

---

## 🎯 Usage Recommendations

### **For Strategy Development**:
1. **Start with**: `run_strategy_backtest.py` - Get overall strategy comparison
2. **Deep dive with**: `check_trade_times.py` - Understand individual trades
3. **Optimize with**: Analysis results to tune parameters

### **For High-Return Focus**:
1. **Test targets**: `test_high_returns.py` - Check if >35% achievable
2. **Analyze details**: `check_high_return_trade_times.py` - Understand how
3. **Optimize**: Adjust aggressive parameters based on results

### **For Production**:
1. **Validate**: Use `run_strategy_backtest.py` for comprehensive testing
2. **Monitor**: Regular analysis with trade detail scripts
3. **Improve**: Continuous optimization based on analysis

---

## 📋 Available Strategies

### **Standard Strategies**:
- `simple_ma` - Simple moving average crossover
- `advanced` - Multi-factor composite strategy
- `mean_reversion` - Statistical mean reversion
- `low_volume` - Optimized for low-volume stocks
- `momentum_breakout` - Momentum breakout detection
- `volatility_mean_reversion` - Volatility-based mean reversion
- `gap_trading` - Gap trading strategy
- `multi_timeframe` - Multi-timeframe analysis
- `risk_managed` - Risk-managed wrapper

### **High-Return Strategies**:
- `aggressive_mean_reversion` - Aggressive mean reversion with tight risk management
- `enhanced_momentum` - Enhanced momentum with volume confirmation
- `multi_timeframe_momentum` - Multi-timeframe momentum strategy

---

## 🛠️ Requirements

- PostgreSQL database with historical price data
- Python environment with required packages
- Proper environment variables (DATABASE_URL)

## 📁 File Organization

```
tradebot/scripts/
├── README.md                           # This file
├── __init__.py                         # Package initialization
├── run_strategy_backtest.py           # Main strategy comparison
├── test_high_returns.py               # High-return strategy tester
├── check_trade_times.py               # General trade analysis
├── check_high_return_trade_times.py   # High-return trade analysis
├── analyze_trades.py                  # Trade performance analysis
├── check_price_data.py                # Price data validation
└── check_msft_data.py                 # MSFT-specific data analysis
``` 