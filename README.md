# Trade Bot Project

## Overview

This repository contains the scaffold of a US–equities trading bot written in **Python**.  The design follows an event-driven, micro-services architecture with three main services:

1. **market_data** – streams real-time prices into a message bus.
2. **strategy** – subscribes to prices, runs technical-analysis rules, and emits trading signals.
3. **execution** – routes orders to a brokerage adapter (placeholder for Fidelity for now).

A lightweight Redis Streams instance acts as the internal message bus.  A Postgres/TimescaleDB container is planned for historical tick storage but is not included in the first commit.

```
┌─────────────┐   price.ticks   ┌────────────┐   orders.new   ┌──────────────┐
│ market_data │ ─────────────▶ │  strategy  │ ─────────────▶ │  execution   │
└─────────────┘                └────────────┘                └──────────────┘
```

## Quick start (development)

1. **Install Python 3.10+**
2. Clone the repo and create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

3. **Run Redis**
   The easiest way is Docker:

```bash
docker run -p 6379:6379 --name redis -d redis:7-alpine
```

4. **Start services in separate terminals:**

```bash
python -m tradebot.market_data.service
python -m tradebot.strategy.service
python -m tradebot.execution.service
```

Each service logs to the console.  Market-data is mocked with random prices until you wire up a real data provider (Polygon, IEX Cloud, etc.).

## Project layout

```
tradebot/             # Python package with all services
    common/           # Shared helpers (message bus, models, utils)
    market_data/      # Ingestion service
    strategy/         # Strategy / TA logic
    execution/        # Order manager / broker adapter
requirements.txt      # Python dependencies
README.md             # This file
```

## Next steps

* Swap the mock data source with a real WebSocket client.
* Persist ticks to TimescaleDB.
* Implement technical indicators with TA-Lib or pandas-ta.
* Fill out the Fidelity (or alternate broker) adapter in execution service.
* Add Docker Compose & Kubernetes manifests for full orchestration.

Happy hacking! 

### Live market data via Polygon.io

If you have a Polygon.io API key you can stream real trades:

```bash
export POLYGON_API_KEY=your_key_here
export SYMBOLS=AAPL,MSFT,TSLA   # optional symbol filter
python -m tradebot.market_data.polygon_service
```

The rest of the stack consumes the same `price.ticks` stream, so no changes
are needed to the strategy or execution services. 

### Live market data via Alpaca (free)

1. Sign up for a free *Paper Trading* account at https://alpaca.markets
2. In the dashboard → *API Keys* → *Generate New Key*.
3. Export the credentials and run the service:

```bash
export ALPACA_KEY=AK6N...      # key ID
export ALPACA_SECRET=3bWq7...  # secret key
export SYMBOLS=AAPL,MSFT,TSLA  # optional symbol list
python -m tradebot.market_data.alpaca_service
```

This streams full-SIP trades in real time at no cost. 

## Environment variables

Create a `.env` file (copy `env.example`) and fill in credentials.  All
services load it automatically via `python-dotenv`. 

## Historical Data Storage

The system now includes full historical data capabilities:

### TimescaleDB Storage
All price ticks are automatically stored in a TimescaleDB (PostgreSQL) database:

```bash
# Start with Docker Compose (includes TimescaleDB)
docker compose up

# Or run storage service manually
python -m tradebot.storage.timeseries_service
```

### Historical Data API
Query historical data via REST API at http://localhost:8000:

```bash
# Get recent ticks for AAPL
curl "http://localhost:8000/history/AAPL?limit=100"

# Get price statistics
curl "http://localhost:8000/stats/AAPL"

# List all available symbols
curl "http://localhost:8000/symbols"
```

### Backfill Historical Data
Download months of historical bars from Alpaca:

```bash
python -m tradebot.backfill.alpaca_historical
```

This fetches 90 days of daily bars + 7 days of hourly bars for all symbols. 