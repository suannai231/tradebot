# 🚀 Backfill Performance Optimization Guide

## Current Performance Bottlenecks

### 1. **Rate Limiting (Biggest Impact)**
- **Current**: 3 concurrent requests, 0.5s delay = ~6 requests/second
- **Alpaca Limit**: 200 requests/minute = ~3.33 requests/second
- **Issue**: We're being too conservative

### 2. **Batch Processing**
- **Current**: 10 symbols per batch, 5s delay between batches
- **Issue**: Too much idle time between batches

### 3. **Database Operations**
- **Current**: Small connection pool (2-10 connections)
- **Issue**: Database becomes bottleneck with high concurrency

## 🎯 Optimization Strategies

### **Level 1: Conservative Speedup (2-3x faster)**
```bash
# Update your .env file with these settings:
BACKFILL_BATCH_SIZE=20
BACKFILL_CONCURRENT_REQUESTS=6
RATE_LIMIT_DELAY=0.3
```

### **Level 2: Aggressive Speedup (5-7x faster)**
```bash
# For users with stable internet and good hardware:
BACKFILL_BATCH_SIZE=30
BACKFILL_CONCURRENT_REQUESTS=10
RATE_LIMIT_DELAY=0.2
DB_POOL_MIN_SIZE=5
DB_POOL_MAX_SIZE=20
```

### **Level 3: Maximum Performance (10x+ faster)**
```bash
# Use the fast_backfill.py script with these settings:
BACKFILL_BATCH_SIZE=50
BACKFILL_CONCURRENT_REQUESTS=15
RATE_LIMIT_DELAY=0.1
DB_POOL_MIN_SIZE=10
DB_POOL_MAX_SIZE=30
LOG_LEVEL=WARNING  # Reduce logging overhead
```

## 📊 Performance Comparison

| Method | Symbols/Batch | Concurrent | Delay | Est. Speed | Time for 1000 symbols |
|--------|---------------|------------|-------|------------|----------------------|
| **Current** | 10 | 3 | 0.5s | ~2 sym/sec | ~8 minutes |
| **Level 1** | 20 | 6 | 0.3s | ~6 sym/sec | ~3 minutes |
| **Level 2** | 30 | 10 | 0.2s | ~12 sym/sec | ~1.5 minutes |
| **Level 3** | 50 | 15 | 0.1s | ~25 sym/sec | ~40 seconds |

## 🛠 Implementation Methods

### **Method 1: Update Environment Variables**
```bash
# Edit your .env file
cp .env .env.backup  # Backup current settings
nano .env

# Add optimized settings:
BACKFILL_BATCH_SIZE=25
BACKFILL_CONCURRENT_REQUESTS=10
RATE_LIMIT_DELAY=0.2

# Run normal backfill
python -m tradebot.backfill.alpaca_historical
```

### **Method 2: Use Fast Backfill Script**
```bash
# Run the optimized script
python fast_backfill.py

# Or with custom settings
BACKFILL_BATCH_SIZE=30 BACKFILL_CONCURRENT_REQUESTS=12 python fast_backfill.py
```

### **Method 3: Docker with Optimized Settings**
```bash
# Run via Docker with performance settings
docker-compose run --rm \
  -e BACKFILL_BATCH_SIZE=25 \
  -e BACKFILL_CONCURRENT_REQUESTS=10 \
  -e RATE_LIMIT_DELAY=0.2 \
  market_data python -m tradebot.backfill.alpaca_historical
```

## ⚠️ **Important Considerations**

### **API Rate Limits**
- **Alpaca Free**: 200 requests/minute
- **Don't exceed**: ~3 requests/second sustained
- **Monitor**: Watch for 429 errors

### **System Resources**
- **Memory**: Higher concurrency = more memory usage
- **CPU**: More concurrent requests = higher CPU load
- **Network**: Ensure stable internet connection

### **Database Performance**
- **Connections**: Increase pool size for high concurrency
- **Disk I/O**: SSD recommended for TimescaleDB
- **Memory**: Ensure sufficient RAM for database cache

## 🔧 **Advanced Optimizations**

### **1. Database Tuning**
```sql
-- Optimize TimescaleDB for bulk inserts
ALTER SYSTEM SET max_connections = 200;
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET work_mem = '4MB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';
```

### **2. Network Optimization**
```python
# Use connection pooling and keep-alive
connector = aiohttp.TCPConnector(
    limit=100,              # Total connection limit
    limit_per_host=20,      # Per-host limit
    ttl_dns_cache=300,      # DNS cache
    use_dns_cache=True,
    keepalive_timeout=30    # Keep connections alive
)
```

### **3. Batch Database Inserts**
```python
# Insert multiple symbols' data in single transaction
async with pool.acquire() as conn:
    async with conn.transaction():
        await conn.executemany(insert_query, all_data)
```

## 🎯 **Recommended Settings by Use Case**

### **Development/Testing**
```bash
BACKFILL_BATCH_SIZE=15
BACKFILL_CONCURRENT_REQUESTS=5
RATE_LIMIT_DELAY=0.4
```

### **Production (Stable)**
```bash
BACKFILL_BATCH_SIZE=25
BACKFILL_CONCURRENT_REQUESTS=8
RATE_LIMIT_DELAY=0.25
```

### **High-Performance (Risk of rate limiting)**
```bash
BACKFILL_BATCH_SIZE=40
BACKFILL_CONCURRENT_REQUESTS=12
RATE_LIMIT_DELAY=0.15
```

## 📈 **Monitoring Performance**

### **Watch for These Metrics**
- **Success Rate**: Should be >95%
- **429 Errors**: Should be <5%
- **Database Errors**: Should be 0%
- **Memory Usage**: Monitor for leaks

### **Logging Commands**
```bash
# Monitor backfill progress
tail -f backfill.log | grep "Batch.*complete"

# Check for rate limit errors
tail -f backfill.log | grep "429\|rate limit"

# Monitor database performance
docker exec trade-timescaledb-1 psql -U postgres -d tradebot -c "
SELECT COUNT(*) as total_records, 
       COUNT(DISTINCT symbol) as symbols,
       MAX(timestamp) as latest 
FROM price_ticks;"
```

## 🚀 **Quick Start Commands**

### **Fast Backfill (Recommended)**
```bash
# Set optimized environment variables
export BACKFILL_BATCH_SIZE=25
export BACKFILL_CONCURRENT_REQUESTS=10
export RATE_LIMIT_DELAY=0.2

# Run fast backfill script
python fast_backfill.py
```

### **Ultra-Fast Backfill (Advanced)**
```bash
# Maximum performance settings
export BACKFILL_BATCH_SIZE=50
export BACKFILL_CONCURRENT_REQUESTS=15
export RATE_LIMIT_DELAY=0.1
export LOG_LEVEL=WARNING

# Run with performance monitoring
time python fast_backfill.py
```

## 🎉 **Expected Results**

With optimized settings, you should see:
- **5-10x faster** backfill completion
- **Higher throughput**: 15-25 symbols/second
- **Better resource utilization**
- **Reduced total time**: 1000 symbols in 1-2 minutes

Remember to monitor for rate limiting and adjust settings based on your system's capabilities! 