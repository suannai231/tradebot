# Trading Bot System

## Overview

A comprehensive **event-driven trading bot system** for US equities built with Python, featuring real-time data processing, technical analysis, and automated trading capabilities. The system follows a microservices architecture with full OHLCV data support and persistent storage.

## 🏗️ Architecture

The system consists of 7 microservices communicating via Redis Streams:

```
┌─────────────────┐  price.ticks   ┌─────────────┐  orders.new   ┌──────────────┐
│   Market Data   │ ──────────────▶│  Strategy   │ ─────────────▶│  Execution   │
│   Services      │                │   Engine    │               │   Service    │
└─────────────────┘                └─────────────┘               └──────────────┘
         │                                                               │
         ▼                                                               ▼
┌─────────────────┐                ┌─────────────┐               ┌──────────────┐
│  Storage        │                │ TimescaleDB │               │ Trade Logs   │
│  Service        │ ──────────────▶│ Database    │◀──────────────│ & Analytics  │
└─────────────────┘                └─────────────┘               └──────────────┘
                                           │
                                           ▼
                                   ┌─────────────┐
                                   │ REST API    │
                                   │ Service     │
                                   └─────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- Python 3.11+ (for development)

### 1. Clone and Setup
```bash
git clone <repository-url>
cd Trade
cp env.example .env  # Edit with your API keys
```

### 2. Start All Services
```bash
# Start entire system with Docker Compose
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f
```

### 3. Access the System
- **API Documentation**: http://localhost:8000/docs
- **Historical Data**: http://localhost:8000/history/AAPL
- **Symbols Overview**: http://localhost:8000/symbols
- **OHLCV Analytics**: http://localhost:8000/ohlcv/AAPL

## 📊 Services Overview

| Service | Purpose | Port | Status |
|---------|---------|------|--------|
| **market_data** | Mock data generation with realistic OHLCV | - | ✅ Active |
| **alpaca_service** | Live data from Alpaca WebSocket | - | ⚠️ Requires API keys |
| **polygon_service** | Live data from Polygon.io | - | ⚠️ Requires API keys |
| **strategy** | Moving average crossover signals | - | ✅ Active |
| **execution** | Trade execution (placeholder) | - | ✅ Active |
| **storage** | TimescaleDB data persistence | - | ✅ Active |
| **api** | REST API for historical data | 8000 | ✅ Active |
| **redis** | Message bus | 6379 | ✅ Active |
| **timescaledb** | Time-series database | 5432 | ✅ Active |

## 📁 Project Structure

```
Trade/
├── tradebot/                    # Main Python package
│   ├── common/                  # Shared components
│   │   ├── models.py           # Data models (PriceTick, Signal)
│   │   └── bus.py              # Redis message bus
│   ├── market_data/            # Data ingestion services
│   │   ├── service.py          # Mock data generator
│   │   ├── alpaca_service.py   # Live Alpaca WebSocket
│   │   └── polygon_service.py  # Live Polygon.io WebSocket
│   ├── strategy/               # Trading logic
│   │   └── service.py          # Moving average strategy
│   ├── execution/              # Trade execution
│   │   └── service.py          # Order management (placeholder)
│   ├── storage/                # Data persistence
│   │   └── timeseries_service.py # TimescaleDB storage
│   ├── api/                    # REST API
│   │   └── history_service.py  # FastAPI endpoints
│   └── backfill/               # Historical data
│       └── alpaca_historical.py # Alpaca data backfill
├── docker-compose.yml          # Service orchestration
├── Dockerfile                  # Container definition
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables
└── README.md                   # This file
```

## 🔧 Configuration

### Quick Start Configurations

Use the management script to create sample configurations:

```bash
python manage_symbols.py samples
```

This creates several `.env` files for different use cases:
- `.env.popular` - Popular large-cap stocks (~100 symbols)
- `.env.all_stocks` - All tradeable US stocks (~1000+ symbols)  
- `.env.large_cap` - Large-cap stocks only (~50 symbols)
- `.env.testing` - Minimal setup for testing (~3 symbols)

Copy your preferred configuration:
```bash
cp .env.popular .env
# Or manually copy env.example and customize
```

### Symbol Configuration Options

The system now supports **all US stocks** with multiple modes:

```bash
# Symbol mode options
SYMBOL_MODE=popular     # Popular large-cap stocks (~100)
SYMBOL_MODE=all         # All tradeable US stocks (1000+)
SYMBOL_MODE=large       # Large-cap stocks (~50)
SYMBOL_MODE=mid         # Mid-cap stocks (~50)  
SYMBOL_MODE=small       # Small-cap stocks (~100)
SYMBOL_MODE=custom      # Use custom SYMBOLS list

# Custom symbols (used when SYMBOL_MODE=custom)
SYMBOLS=AAPL,MSFT,AMZN,GOOG,TSLA

# Limits to prevent system overload
MAX_SYMBOLS=500                    # Total symbols to process
MAX_WEBSOCKET_SYMBOLS=30           # WebSocket subscriptions (API limit)
```

### Environment Variables (.env)
```bash
# Database connection
DATABASE_URL=postgresql://postgres:password@localhost:5432/tradebot

# Redis connection  
REDIS_URL=redis://localhost:6379

# Symbol configuration (see above for modes)
SYMBOL_MODE=popular
MAX_SYMBOLS=500
MAX_WEBSOCKET_SYMBOLS=30

# Rate limiting
BACKFILL_BATCH_SIZE=10
BACKFILL_CONCURRENT_REQUESTS=3
RATE_LIMIT_DELAY=0.5

# Alpaca API credentials
ALPACA_KEY=your_alpaca_key_here
ALPACA_SECRET=your_alpaca_secret_here

# Polygon API key (optional)
POLYGON_API_KEY=your_polygon_key_here
```

## 📈 Data Features

### OHLCV Data Support
Full market data including:
- **Open, High, Low, Close** prices
- **Volume** and **Trade Count**
- **VWAP** (Volume Weighted Average Price)
- **Real-time timestamps**

### Data Sources
1. **Mock Data**: Realistic price simulation (~$220 AAPL, ~$430 MSFT)
2. **Alpaca**: Live WebSocket feed (free tier with paper trading)
3. **Polygon.io**: Professional market data (requires subscription)

### Storage
- **TimescaleDB**: Optimized time-series database
- **Hypertables**: Automatic partitioning and compression
- **Indexes**: Optimized for symbol and time-based queries
- **Persistent**: Data survives container restarts

## 🔌 API Endpoints

### Historical Data
```bash
# Get recent OHLCV data
GET /history/{symbol}?start=2024-01-01&end=2024-12-31&limit=1000

# Example response
{
  "symbol": "AAPL",
  "price": 220.50,
  "timestamp": "2024-06-21T10:30:00Z",
  "open": 220.10,
  "high": 221.00,
  "low": 219.80,
  "close": 220.50,
  "volume": 1500000,
  "trade_count": 12500,
  "vwap": 220.35
}
```

### Analytics
```bash
# Symbol overview
GET /symbols

# OHLCV statistics
GET /ohlcv/{symbol}

# Price statistics
GET /stats/{symbol}
```

## 🤖 Trading Strategy

### Current Implementation
- **Moving Average Crossover**: 5-tick vs 20-tick simple moving average
- **Signal Generation**: BUY when short MA > long MA, SELL when opposite
- **Confidence Scoring**: Based on difference between moving averages

### Example Signals
```json
{
  "symbol": "AAPL",
  "side": "BUY",
  "confidence": 0.068,
  "timestamp": "2024-06-21T10:30:00Z"
}
```

## 📊 Historical Data

### Backfill from Alpaca
```bash
# Download 90 days of historical data
python -m tradebot.backfill.alpaca_historical

# Or with Docker
docker-compose run --rm market_data python -m tradebot.backfill.alpaca_historical
```

### Data Coverage
- **90 days** of daily OHLCV bars
- **Free tier compatible** (uses IEX feed)
- **Automatic retry** logic for API failures
- **Batch processing** with rate limiting
- **All US stocks** support with intelligent filtering

## 🎯 Symbol Management

### Management Commands
```bash
# Test your current configuration
python manage_symbols.py test

# List symbols for different modes
python manage_symbols.py list popular --limit 20
python manage_symbols.py list all --limit 50

# Fetch all symbols from Alpaca API
python manage_symbols.py fetch

# Benchmark different modes
python manage_symbols.py benchmark

# Create sample configurations
python manage_symbols.py samples
```

### Symbol Filtering
The system automatically filters symbols to include only:
- ✅ **Active** and **tradeable** stocks
- ✅ **Major exchanges**: NYSE, NASDAQ, AMEX
- ✅ **Standard symbols**: Excludes warrants, rights, units
- ✅ **Reasonable price range**: Filters penny stocks and extreme outliers
- ❌ **Excludes**: SPACs, test symbols, complex derivatives

### Performance Considerations
- **WebSocket Limits**: Alpaca free tier supports 30 concurrent symbols
- **Rate Limiting**: Built-in delays and batch processing
- **Memory Management**: Configurable symbol limits
- **Caching**: Daily symbol list caching for performance

## 🔄 Development Workflow

### Local Development
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run individual services
python -m tradebot.market_data.service
python -m tradebot.strategy.service
python -m tradebot.execution.service
```

### VS Code Debugging
Pre-configured launch configurations available in `.vscode/launch.json`:
- Individual service debugging
- Compound debugging (multiple services)
- Environment variable support

### Service Management
```bash
# Start specific services
docker-compose up -d redis timescaledb api

# Scale services
docker-compose up -d --scale market_data=2

# View service logs
docker-compose logs -f strategy execution

# Restart services
docker-compose restart market_data

# Stop all services
docker-compose down
```

## 🧪 Testing

### Service Health Checks
```bash
# Test Redis
docker exec trade-redis-1 redis-cli ping

# Test TimescaleDB
docker exec trade-timescaledb-1 pg_isready -U postgres

# Test API
curl http://localhost:8000/

# Check data flow
curl http://localhost:8000/symbols
```

### Data Verification
```bash
# Check stored data count
docker exec trade-timescaledb-1 psql -U postgres -d tradebot -c "SELECT COUNT(*) FROM price_ticks;"

# View recent data
curl "http://localhost:8000/history/AAPL?limit=5" | python -m json.tool
```

## 🚀 Deployment

### Production Considerations
1. **Environment Variables**: Use secure secret management
2. **Database**: Configure persistent volumes and backups
3. **Monitoring**: Add health checks and alerting
4. **Scaling**: Use Docker Swarm or Kubernetes
5. **Security**: Configure firewalls and access controls

### Docker Compose Production
```bash
# Production deployment
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# With resource limits and restart policies
docker-compose up -d --scale strategy=2 --scale execution=2
```

## 🔮 Future Enhancements

### Immediate Next Steps
- [ ] **Fidelity Integration**: Replace execution placeholder with real broker API
- [ ] **Advanced Strategies**: RSI, MACD, Bollinger Bands
- [ ] **Risk Management**: Position sizing, stop-loss, portfolio limits
- [ ] **More Symbols**: Expand from 5 to 100+ stocks

### Advanced Features
- [ ] **Machine Learning**: Price prediction models
- [ ] **Options Trading**: Derivatives support
- [ ] **Crypto Support**: Bitcoin and altcoin trading
- [ ] **Web Dashboard**: Real-time monitoring UI
- [ ] **Backtesting**: Historical strategy validation
- [ ] **Paper Trading**: Risk-free strategy testing

## 📚 Technology Stack

- **Python 3.11+**: Core application language
- **Redis Streams**: Event-driven messaging
- **TimescaleDB**: Time-series database (PostgreSQL extension)
- **FastAPI**: Modern REST API framework
- **Docker Compose**: Service orchestration
- **Pydantic**: Data validation and serialization
- **asyncio**: Asynchronous programming
- **WebSockets**: Real-time market data
- **VS Code**: Development environment with debugging

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes and test thoroughly
4. Submit a pull request with detailed description

## 📄 License

This project is for educational and research purposes. Please ensure compliance with your broker's API terms and applicable financial regulations.

---

**Happy Trading! 🚀📈** 