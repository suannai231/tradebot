# ML Training Service Configuration Guide

The ML Training Service provides scheduled, automated training for all ML strategies in the TradingBot system. This guide covers configuration, deployment, and monitoring.

## 🚀 Quick Start

### 1. Start with Default Configuration
```bash
# Start the ML training service with default settings
docker compose --profile ml-training up -d
```

### 2. Start with Custom Configuration
```bash
# Create .env file with custom settings
cp .env.example .env

# Edit .env with your preferred settings
# Then start the service
docker compose --profile ml-training up -d
```

### 3. Start Everything Together
```bash
# Start core system + ML training
docker compose --profile default --profile ml-training up -d
```

## ⚙️ Configuration Options

### Training Schedule Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `TRAINING_CRON_SCHEDULE` | `0 2,14,22 * * *` | Cron schedule for training runs |

#### Schedule Examples:
```bash
# Every 6 hours
TRAINING_CRON_SCHEDULE=0 */6 * * *

# Business hours only (9 AM, 1 PM, 5 PM on weekdays)
TRAINING_CRON_SCHEDULE=0 9,13,17 * * 1-5

# Once daily at 2 AM
TRAINING_CRON_SCHEDULE=0 2 * * *

# Every 2 hours during market hours (9 AM - 4 PM EST)
TRAINING_CRON_SCHEDULE=0 9-16/2 * * 1-5
```

### Performance Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `TRAINING_BATCH_SIZE` | `1000` | Number of data points per training session |
| `MAX_CONCURRENT_TRAINING_JOBS` | `3` | Maximum parallel training jobs |
| `TRAINING_JOB_TIMEOUT` | `3600` | Job timeout in seconds (1 hour) |
| `TRAINING_RETRY_ATTEMPTS` | `3` | Retry attempts for failed jobs |
| `MIN_TRAINING_DATA_POINTS` | `100` | Minimum data points required |
| `MODEL_RETENTION_DAYS` | `30` | Model retention period |

### Example .env Configuration
```bash
# ML Training Service Configuration
TRAINING_CRON_SCHEDULE=0 2,14,22 * * *
TRAINING_BATCH_SIZE=1000
MAX_CONCURRENT_TRAINING_JOBS=3
TRAINING_JOB_TIMEOUT=3600
TRAINING_RETRY_ATTEMPTS=3
MIN_TRAINING_DATA_POINTS=100
MODEL_RETENTION_DAYS=30

# Performance tuning
ML_TRAINING_DEBUG=false
LOG_LEVEL=INFO
```

## 🏗️ Architecture Overview

### Components
1. **Scheduler Worker**: Handles cron-based scheduling
2. **Queue Worker**: Processes training jobs from Redis queue
3. **Heartbeat Worker**: Sends health status updates
4. **Metrics Worker**: Updates performance metrics

### Data Flow
```
Cron Schedule → Training Jobs → Redis Queue → Training Workers → Model Storage
     ↓              ↓              ↓              ↓              ↓
Database Tracking → Job Status → Queue Management → Model Training → File System
```

### Supported Strategies
- **Ensemble ML**: Global model trained on all symbols
- **LSTM**: Per-symbol LSTM models for time series prediction
- **Reinforcement Learning**: Per-symbol RL environments

## 📊 Monitoring & Metrics

### Health Monitoring
The service provides comprehensive health monitoring through:
- **Redis Heartbeat**: Updates every 30 seconds
- **Database Tracking**: All jobs tracked in `ml_training_jobs` table
- **Model Registry**: Active models tracked in `ml_model_registry` table
- **Performance Metrics**: Stored in `ml_training_metrics` table

### Key Metrics
- Jobs queued, completed, failed
- Total training time
- Active models count
- Model accuracy trends
- Last training run timestamp

### Viewing Metrics
```bash
# Check service health
docker compose exec redis redis-cli GET service:ml_training:heartbeat

# View training job status
docker compose exec timescaledb psql -U postgres -d tradebot -c "
SELECT job_id, strategy_type, symbol, status, created_at, completed_at 
FROM ml_training_jobs 
ORDER BY created_at DESC LIMIT 10;"

# View active models
docker compose exec timescaledb psql -U postgres -d tradebot -c "
SELECT model_id, strategy_type, symbol, created_at, is_active 
FROM ml_model_registry 
WHERE is_active = TRUE 
ORDER BY created_at DESC;"
```

## 💾 Model Persistence

### File Storage
Models are stored in the `models/` directory with the following structure:
```
models/
├── ensemble/
│   ├── AAPL/
│   │   ├── model_20240101_120000.pkl
│   │   └── model_20240101_140000.pkl
│   └── MSFT/
│       └── model_20240101_120000.pkl
├── lstm/
│   ├── AAPL/
│   │   └── model_20240101_120000.pkl
│   └── MSFT/
│       └── model_20240101_120000.pkl
└── rl/
    ├── AAPL/
    │   └── model_20240101_120000.pkl
    └── MSFT/
        └── model_20240101_120000.pkl
```

### Model Format
Each model file contains:
```python
{
    'strategy_type': 'ensemble',
    'symbol': 'AAPL',
    'timestamp': '20240101_120000',
    'model_state': {
        'ensemble': <trained_model>,
        'scaler': <fitted_scaler>
    }
}
```

### Cleanup Policy
- Models older than `MODEL_RETENTION_DAYS` are automatically deleted
- Only inactive models are cleaned up
- Database entries are also cleaned up during maintenance

## 🔧 Troubleshooting

### Common Issues

#### 1. Service Won't Start
```bash
# Check logs
docker compose logs ml_training

# Common fixes:
# - Ensure Redis and TimescaleDB are healthy
# - Check .env file configuration
# - Verify database permissions
```

#### 2. No Training Jobs Running
```bash
# Check queue status
docker compose exec redis redis-cli LLEN ml_training_queue

# Check active jobs
docker compose exec timescaledb psql -U postgres -d tradebot -c "
SELECT COUNT(*) FROM ml_training_jobs WHERE status = 'running';"

# Common fixes:
# - Verify cron schedule format
# - Check minimum data requirements
# - Review training logs
```

#### 3. Training Jobs Failing
```bash
# Check error messages
docker compose exec timescaledb psql -U postgres -d tradebot -c "
SELECT job_id, strategy_type, symbol, error_message 
FROM ml_training_jobs 
WHERE status = 'failed' 
ORDER BY created_at DESC LIMIT 5;"

# Common fixes:
# - Increase TRAINING_JOB_TIMEOUT
# - Reduce TRAINING_BATCH_SIZE
# - Check data availability
```

#### 4. Models Not Persisting
```bash
# Check model directory
ls -la models/

# Check database registry
docker compose exec timescaledb psql -U postgres -d tradebot -c "
SELECT COUNT(*) FROM ml_model_registry WHERE is_active = TRUE;"

# Common fixes:
# - Verify volume mount in docker-compose.yml
# - Check file permissions
# - Review model saving logs
```

### Debug Mode
Enable detailed logging:
```bash
# In .env file
ML_TRAINING_DEBUG=true
LOG_LEVEL=DEBUG

# Restart service
docker compose restart ml_training
```

## 🚦 Production Deployment

### Recommended Settings
```bash
# Production .env configuration
TRAINING_CRON_SCHEDULE=0 2,14,22 * * *  # 3 times daily
TRAINING_BATCH_SIZE=2000                 # Larger batches
MAX_CONCURRENT_TRAINING_JOBS=5           # More parallelism
TRAINING_JOB_TIMEOUT=7200               # 2 hour timeout
MIN_TRAINING_DATA_POINTS=500            # More data required
MODEL_RETENTION_DAYS=90                 # Longer retention
```

### Scaling Considerations
- **Memory**: Each training job requires ~500MB-1GB RAM
- **CPU**: Training is CPU-intensive, consider multi-core instances
- **Storage**: Models require ~10-50MB per symbol per strategy
- **Network**: Minimal network usage (Redis queue only)

### Monitoring Setup
1. **Health Checks**: Monitor Redis heartbeat key
2. **Metrics Collection**: Export metrics to monitoring system
3. **Alerting**: Set up alerts for failed training jobs
4. **Log Aggregation**: Centralize training service logs

## 🔄 Integration with Existing System

### Immediate Training (Tick-based)
The system also supports immediate training triggers in strategy `on_tick` methods:
```python
# In strategy on_tick method
if self.tick_count[symbol] % self.config.retrain_frequency == 0:
    self.train_model(symbol)
```

### Hybrid Approach
You can run both:
- **Scheduled Training**: For regular model updates
- **Immediate Training**: For real-time adaptation

### Dashboard Integration
The ML training service integrates with the dashboard for:
- Training job status display
- Model performance metrics
- Training schedule visualization
- Health monitoring

## 📈 Performance Optimization

### Batch Processing
- Increase `TRAINING_BATCH_SIZE` for better throughput
- Balance with memory usage and timeout limits

### Parallel Processing
- Adjust `MAX_CONCURRENT_TRAINING_JOBS` based on system resources
- Consider CPU cores and memory availability

### Data Optimization
- Ensure proper database indexing
- Use appropriate `MIN_TRAINING_DATA_POINTS`
- Consider data preprocessing optimization

### Model Optimization
- Implement incremental learning where possible
- Use model checkpointing for long training sessions
- Consider model compression for storage efficiency

---

For additional support, check the main README.md or create an issue in the repository. 