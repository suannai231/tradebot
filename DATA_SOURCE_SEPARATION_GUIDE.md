# Data Source Separation Guide

## Overview

The trading bot system now uses **separate tables** for different data sources instead of mixing real and mock data in a single table. This approach provides better data integrity, cleaner testing, and easier maintenance.

## 📊 Table Structure

### Data Source Tables

| Data Source | Table Name | Purpose |
|-------------|------------|---------|
| `synthetic` | `price_ticks_synthetic` | Generated mock data for development |
| `alpaca` | `price_ticks_alpaca` | Real market data from Alpaca API |
| `polygon` | `price_ticks_polygon` | Real market data from Polygon API |
| `mock` | `price_ticks_mock` | Test data for unit tests |
| `test` | `price_ticks_test` | Integration test data |

### Why Separate Tables?

✅ **Data Integrity**: No risk of mixing real and mock data  
✅ **Clean Testing**: Guaranteed pure datasets for backtesting  
✅ **Easy Cleanup**: Drop entire mock tables without affecting production  
✅ **Clear Audit Trail**: Know exactly where each data point came from  
✅ **Performance**: Smaller tables, faster queries  
✅ **Compliance**: Better for financial data regulations  

## 🚀 Usage

### Docker Compose Profiles

The system automatically uses the correct table based on the data source:

```bash
# Development with synthetic data
docker compose --profile default up -d
# Uses: price_ticks_synthetic

# Production with real Alpaca data  
docker compose --profile alpaca up -d
# Uses: price_ticks_alpaca

# Production with real Polygon data
docker compose --profile polygon up -d
# Uses: price_ticks_polygon
```

### Environment Variables

Control which table is used with the `DATA_SOURCE` environment variable:

```bash
# .env file
DATA_SOURCE=synthetic  # Default
DATA_SOURCE=alpaca     # Real Alpaca data
DATA_SOURCE=polygon    # Real Polygon data
DATA_SOURCE=mock       # Test data
DATA_SOURCE=test       # Integration tests
```

### Manual Override

You can override the data source for specific services:

```bash
# Run dashboard with specific data source
DATA_SOURCE=alpaca python -m tradebot.dashboard.main

# Run backtest with test data
DATA_SOURCE=test python -m tradebot.scripts.run_strategy_backtest
```

## 🔧 Management Tools

### Data Source Manager

Use the `manage_data_sources.py` utility to manage tables:

```bash
# List all data source tables and their stats
python manage_data_sources.py list

# Create all data source tables
python manage_data_sources.py create

# Clean a specific table
python manage_data_sources.py clean --source synthetic

# Migrate data between sources
python manage_data_sources.py migrate --source synthetic --target test --symbol AAPL

# Compare data between sources
python manage_data_sources.py compare --source alpaca --target synthetic
```

### Example Output

```
📊 DATA SOURCE TABLES
============================================================
✅ synthetic    | price_ticks_synthetic     | 45,231 rows | 12 symbols
   synthetic    |                           | 2025-01-15 to 2025-01-20
✅ alpaca       | price_ticks_alpaca        | 1,234,567 rows | 500 symbols  
   alpaca       |                           | 2024-01-01 to 2025-01-20
⚪ polygon      | price_ticks_polygon       | Empty
❌ mock         | price_ticks_mock          | Not created
❌ test         | price_ticks_test          | Not created
```

## 📈 Database Schema

Each table has identical structure:

```sql
CREATE TABLE price_ticks_[source] (
    symbol VARCHAR(10) NOT NULL,
    price DECIMAL(10,4) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    open_price DECIMAL(10,4),
    high_price DECIMAL(10,4),
    low_price DECIMAL(10,4),
    close_price DECIMAL(10,4),
    volume BIGINT DEFAULT 0,
    trade_count INTEGER DEFAULT 0,
    vwap DECIMAL(10,4)
);

-- TimescaleDB hypertable for time-series optimization
SELECT create_hypertable('price_ticks_[source]', 'timestamp');

-- Index for fast queries
CREATE INDEX idx_price_ticks_[source]_symbol_time 
ON price_ticks_[source] (symbol, timestamp DESC);
```

## 🧪 Testing Workflows

### Development Testing

```bash
# Start with synthetic data
docker compose --profile default up -d

# Run strategies against synthetic data
python -m tradebot.strategy.service

# Dashboard shows synthetic data
open http://localhost:8001
```

### Production Testing

```bash
# Test with real data in a separate table
DATA_SOURCE=test docker compose up -d

# Migrate some real data to test table
python manage_data_sources.py migrate --source alpaca --target test --symbol AAPL

# Run backtests on test data
DATA_SOURCE=test python -m tradebot.scripts.run_strategy_backtest
```

### A/B Testing

```bash
# Compare strategy performance on different data sources
python manage_data_sources.py compare --source alpaca --target synthetic

# Run same strategy on both sources
DATA_SOURCE=alpaca python backtest_strategy.py > results_real.json
DATA_SOURCE=synthetic python backtest_strategy.py > results_synthetic.json
```

## 🔄 Migration Scenarios

### Scenario 1: Development to Production

```bash
# 1. Start with synthetic data during development
DATA_SOURCE=synthetic

# 2. Switch to real data for production
DATA_SOURCE=alpaca

# 3. Keep synthetic data for testing
python manage_data_sources.py clean --source synthetic
```

### Scenario 2: Data Source Switch

```bash
# 1. Currently using Alpaca
DATA_SOURCE=alpaca

# 2. Want to switch to Polygon
DATA_SOURCE=polygon

# 3. Compare data quality first
python manage_data_sources.py compare --source alpaca --target polygon
```

### Scenario 3: Backup & Restore

```bash
# 1. Backup production data to test table
python manage_data_sources.py migrate --source alpaca --target test

# 2. Clean production table
python manage_data_sources.py clean --source alpaca --confirm

# 3. Restore from backup
python manage_data_sources.py migrate --source test --target alpaca
```

## 🛠️ Code Integration

### Service Implementation

Services automatically use the correct table:

```python
# tradebot/storage/timeseries_service.py
DATA_SOURCE = os.getenv('DATA_SOURCE', 'synthetic')
table_name = f'price_ticks_{DATA_SOURCE}'

# Insert data into correct table
await conn.execute(f"""
    INSERT INTO {table_name} (symbol, price, timestamp, ...)
    VALUES ($1, $2, $3, ...)
""", tick.symbol, tick.price, tick.timestamp, ...)
```

### Dashboard Queries

The dashboard automatically queries the correct table:

```python
# tradebot/dashboard/main.py
def get_table_name() -> str:
    data_source = os.getenv('DATA_SOURCE', 'synthetic')
    return f'price_ticks_{data_source}'

@app.get("/api/market-summary")
async def get_market_summary():
    table_name = get_table_name()
    query = f"SELECT * FROM {table_name} WHERE ..."
```

## 🔍 Monitoring & Troubleshooting

### Common Issues

**1. Wrong data source showing in dashboard**
```bash
# Check which data source is configured
echo $DATA_SOURCE

# Verify table has data
python manage_data_sources.py list
```

**2. Empty dashboard**
```bash
# Check if table exists and has data
python manage_data_sources.py list

# Check if services are using correct table
docker compose logs dashboard
```

**3. Data inconsistency**
```bash
# Compare data between sources
python manage_data_sources.py compare --source alpaca --target synthetic

# Check for data gaps
python manage_data_sources.py list
```

### Health Checks

```bash
# Check all tables
python manage_data_sources.py list

# Check specific table
python manage_data_sources.py compare --source alpaca --target alpaca

# Verify table structure
psql -d tradebot -c "\d price_ticks_alpaca"
```

## 📋 Best Practices

### ✅ DO:
- Use separate environments for different data sources
- Clean test data regularly
- Monitor data source consistency
- Use descriptive table names
- Keep schemas identical across sources

### ❌ DON'T:
- Mix different data sources in the same table
- Use production data for development without consent
- Forget to clean test data
- Assume data consistency between sources
- Skip table creation steps

## 🚧 Future Enhancements

### Planned Features
- [ ] Automatic data source detection
- [ ] Data quality validation
- [ ] Cross-source data synchronization
- [ ] Automated testing pipelines
- [ ] Data archival policies

### Configuration Options
- [ ] Custom table naming schemes
- [ ] Data retention policies
- [ ] Automatic cleanup schedules
- [ ] Data source health monitoring

## 💡 Tips & Tricks

### Quick Commands

```bash
# Quick status check
python manage_data_sources.py list

# Quick cleanup
python manage_data_sources.py clean --source synthetic --confirm

# Quick comparison
python manage_data_sources.py compare --source alpaca --target synthetic

# Quick migration
python manage_data_sources.py migrate --source alpaca --target test --symbol AAPL
```

### Environment Setup

```bash
# Development environment
export DATA_SOURCE=synthetic
docker compose --profile default up -d

# Production environment  
export DATA_SOURCE=alpaca
docker compose --profile alpaca up -d

# Testing environment
export DATA_SOURCE=test
docker compose up -d
```

## 📞 Support

For issues or questions:
1. Check the troubleshooting section above
2. Run `python manage_data_sources.py list` to verify table status
3. Check service logs: `docker compose logs [service_name]`
4. Verify environment variables: `echo $DATA_SOURCE`

---

**Remember**: This separation approach provides better data integrity and cleaner testing workflows. Always verify which data source you're using before making important trading decisions! 