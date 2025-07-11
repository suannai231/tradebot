# TradingBot - Microservices Trading System

A real-time trading bot system built with Python, featuring microservices architecture, live market data processing, and a web-based dashboard for monitoring.

![Dashboard](https://img.shields.io/badge/Dashboard-Live%20Updates-brightgreen)
![Architecture](https://img.shields.io/badge/Architecture-Microservices-blue)
![Database](https://img.shields.io/badge/Database-TimescaleDB-orange)

## ⚡ Quick Start

```bash
# Clone the repository
git clone <your-repo-url>
cd Trade

# Start the full system with Docker Compose
docker compose up -d --build

# Open the live dashboard
open http://localhost:8001

# (Optional) Generate demo data for dashboard
python demo_dashboard.py

# Choose your market data source:
docker compose --profile default up -d       # Synthetic data (default)
docker compose --profile alpaca up -d        # Real Alpaca data (requires ALPACA_KEY/SECRET)
docker compose --profile polygon up -d       # Real Polygon data (requires POLYGON_API_KEY)
```

The first build takes ~2-3 minutes. Subsequent starts are instant.

> **Note**: The system comes with working defaults and will start immediately. For custom configuration, copy `.env.example` to `.env` and modify as needed.

---

## 🏗️ Architecture

The system follows a microservices architecture with the following components:

### Infrastructure Services
| Service      | Image            | Port | Purpose                           |
|--------------|------------------|------|-----------------------------------|
| `redis`      | redis:7-alpine   | 6379 | Message bus & system heartbeats  |
| `timescaledb`| timescaledb      | 5432 | Time-series data storage         |

### Core Services
| Service      | Purpose                                    |
|--------------|--------------------------------------------|
| `market_data`| Synthetic tick generation (default)       |
| `strategy`   | Trading signal generation                  |
| `execution`  | Trade execution (paper trading)           |
| `storage`    | Writes market data to TimescaleDB         |
| `api`        | REST API for historical data (port 8000)  |
| `dashboard`  | Web UI & real-time monitoring (port 8001) |

### Market Data Services (Choose One)
| Service                | Data Source | Requirements            | Usage                    |
|------------------------|-------------|-------------------------|--------------------------|
| `market_data`          | Synthetic   | None                    | `docker compose --profile default up -d`  |
| `alpaca_market_data`   | Alpaca API  | ALPACA_KEY/SECRET       | `docker compose --profile alpaca up -d` |
| `polygon_market_data`  | Polygon API | POLYGON_API_KEY         | `docker compose --profile polygon up -d` |

**Note**: Each profile starts only one market data service, preventing conflicts.

### Communication
- **Message Bus**: Redis Streams for real-time data flow
- **Data Storage**: TimescaleDB for OHLCV market data (separate tables per data source)
- **Real-time Updates**: WebSocket connections for live dashboard

### Data Source Separation
The system uses **separate tables** for different data sources to ensure data integrity:

| Data Source | Table | Purpose |
|-------------|-------|---------|
| `synthetic` | `price_ticks_synthetic` | Development/testing |
| `alpaca` | `price_ticks_alpaca` | Real Alpaca data |
| `polygon` | `price_ticks_polygon` | Real Polygon data |

**Benefits**: No data contamination, clean backtests, easy cleanup, better compliance.

---

## ⚙️ Configuration

### Environment Variables

The system reads configuration from environment variables. Key settings:

| Variable                    | Default                                    | Description                                |
|-----------------------------|--------------------------------------------|--------------------------------------------|
| `DATABASE_URL`              | `postgresql://postgres:password@localhost:5432/tradebot` | TimescaleDB connection |
| `REDIS_URL`                 | `redis://localhost:6379`                  | Redis connection for message bus           |
| `SYMBOL_MODE`               | `custom`                                   | Symbol selection mode                      |
| `SYMBOLS`                   | `AAPL,MSFT,AMZN,GOOG,TSLA`               | Custom symbol list (when mode=custom)     |
| `MAX_SYMBOLS`               | `500`                                      | Maximum symbols to process                 |
| `MAX_WEBSOCKET_SYMBOLS`     | `30`                                       | WebSocket subscription limit               |
| `ALPACA_KEY`                | -                                          | Alpaca API key (optional)                  |
| `ALPACA_SECRET`             | -                                          | Alpaca API secret (optional)               |
| `POLYGON_API_KEY`           | -                                          | Polygon API key (optional)                 |

### Symbol Modes

| Mode       | Description                              | Count    |
|------------|------------------------------------------|----------|
| `custom`   | User-defined symbol list                | Variable |
| `popular`  | Popular large-cap stocks                 | ~100     |
| `large`    | Large-cap stocks                         | ~50      |
| `mid`      | Mid-cap stocks                           | ~50      |
| `small`    | Small-cap stocks                         | ~100     |
| `all`      | All tradeable US stocks                  | Up to MAX_SYMBOLS |

### Creating Configuration

Create a `.env` file in the project root for custom settings:

```bash
# Symbol configuration
SYMBOL_MODE=popular
MAX_SYMBOLS=100
MAX_WEBSOCKET_SYMBOLS=30

# Database connections
DATABASE_URL=postgresql://postgres:password@localhost:5432/tradebot
REDIS_URL=redis://localhost:6379

# API credentials (optional - for live data)
ALPACA_KEY=your_key_here
ALPACA_SECRET=your_secret_here

# Performance tuning
BACKFILL_BATCH_SIZE=10
BACKFILL_CONCURRENT_REQUESTS=3
RATE_LIMIT_DELAY=0.5
```

---

## 🛠️ CLI Tools

### Symbol Management
```bash
# List available symbols for different modes
python manage_symbols.py list popular --limit 20
python manage_symbols.py list all --limit 50

# Test current configuration
python manage_symbols.py test

# Fetch fresh symbol data from Alpaca
python manage_symbols.py fetch

# Benchmark different symbol modes
python manage_symbols.py benchmark

# Generate sample configuration files
python manage_symbols.py samples
```

### Data Source Management
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

### Historical Data Backfill
```bash
# Standard backfill
python fast_backfill.py

# Benchmark different performance settings
python benchmark_backfill.py
```

### Dashboard & Demo
```bash
# Run dashboard standalone (without Docker)
python run_dashboard.py

# Generate demo data for dashboard testing
python demo_dashboard.py
```

---

## 📊 Dashboard Features

The web dashboard (http://localhost:8001) provides:

### Real-time Monitoring
- **Live price updates** via WebSocket
- **Trading signals** as they're generated
- **System health** with service heartbeats
- **Market summary** with top movers

### System Health Panel
The dashboard automatically detects and monitors all running services. Services are color-coded based on status:
- 🟢 **Healthy**: Recent heartbeat (<30s), no errors
- 🟡 **Degraded**: Stale heartbeat (30-60s)
- 🔴 **Unhealthy**: Very stale heartbeat (1-5 minutes)
- ⚫ **Dead**: No heartbeat (>5 minutes)
- 🔶 **Error**: Active error count > 0

The active market data service is clearly labeled:
- **"market data"** = Synthetic data
- **"alpaca market data"** = Real Alpaca data  
- **"polygon market data"** = Real Polygon data

### Reset Error States
```bash
# Reset specific service errors
docker compose exec redis redis-cli DEL service:market_data:errors

# Reset all service errors
docker compose exec redis redis-cli DEL service:*:errors
```

---

## 🚀 Performance Optimization

### Backfill Performance
The system includes optimized backfill configurations:

| Configuration    | Batch Size | Concurrent | Rate Delay | Performance |
|------------------|------------|------------|------------|-------------|
| Conservative     | 10         | 3          | 0.5s       | Baseline    |
| Balanced         | 15         | 6          | 0.3s       | ~2x faster  |
| Aggressive       | 20         | 8          | 0.2s       | ~3x faster  |
| High Performance | 25         | 10         | 0.15s      | ~4x faster  |

Benchmark different settings:
```bash
python benchmark_backfill.py
```

### Database Optimization
- TimescaleDB hypertables for efficient time-series queries
- Connection pooling for concurrent access
- Batch inserts for high-throughput writes

---

## 🔧 Development

### Project Structure
```
tradebot/
├── common/          # Shared utilities and models
│   ├── bus.py       # Redis message bus
│   ├── config.py    # Configuration management
│   ├── models.py    # Data models (PriceTick, TradeSignal)
│   └── symbol_manager.py  # Symbol loading and filtering
├── market_data/     # Market data services
├── strategy/        # Trading strategy engine
├── execution/       # Trade execution service
├── storage/         # Database storage service
├── api/             # REST API service
└── dashboard/       # Web dashboard
    ├── main.py      # FastAPI application
    ├── static/      # CSS/JS assets
    └── templates/   # HTML templates
```

### Running Individual Services
```bash
# Run a specific service
python -m tradebot.dashboard.main
python -m tradebot.api.history_service
python -m tradebot.market_data.service
```

### Dependencies
- **Python 3.8+** with asyncio support
- **Redis 7+** for message streaming
- **TimescaleDB/PostgreSQL** for time-series data
- **FastAPI** for web services
- **Docker & Docker Compose** for orchestration

---

## 📈 Data Flow

```
Market Data → Redis Streams → Strategy Engine → Execution Engine
     ↓              ↓              ↓               ↓
Storage Service → TimescaleDB → API Service → Dashboard
```

1. **Market Data Service** generates or fetches real-time price ticks
2. **Redis Streams** distribute ticks to all interested services
3. **Strategy Service** analyzes ticks and generates trading signals
4. **Execution Service** processes signals (paper trading)
5. **Storage Service** persists all data to TimescaleDB
6. **API Service** provides historical data access
7. **Dashboard** displays real-time updates via WebSocket

---

## 🔍 Monitoring & Debugging

### View Service Logs
```bash
# All services
docker compose logs -f

# Specific service
docker compose logs -f dashboard
docker compose logs -f market_data
```

### Check System Health
```bash
# Redis connection
docker compose exec redis redis-cli ping

# Database connection
docker compose exec timescaledb psql -U postgres -d tradebot -c "SELECT NOW();"

# Service heartbeats
docker compose exec redis redis-cli KEYS "service:*:heartbeat"
```

### Performance Monitoring
- Dashboard shows real-time system statistics
- Service heartbeats tracked every 30 seconds
- Error counts monitored per service
- WebSocket connection status displayed

---

Happy Trading! 🚀

For issues or questions, check the logs or create an issue in the repository. 