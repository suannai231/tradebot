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