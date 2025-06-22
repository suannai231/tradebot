# Trade-Bot ‑ Multi-service Real-time Trading Demo

![Dashboard Screenshot](docs/dashboard.png)

## 1  Quick start

```bash
# clone & enter repo
git clone <your fork/url>
cd Trade

# copy defaults and (optionally) add API keys
cp .env.example .env   # <- edit ALPACA_KEY etc. if you have them

# build & launch the full stack
docker compose up -d --build

# open the live dashboard
open http://localhost:8001
```
The first build takes ≈2-3 minutes. Subsequent `docker compose up` starts instantly.

---

## 2  Configuration
All services read their settings from two env-files loaded by **docker-compose**:

| file              | committed | purpose                                    |
|-------------------|-----------|--------------------------------------------|
| `.env.example`    | **yes**   | canonical defaults for every variable      |
| `.env`            | **no**    | developer overrides & private API keys     |

Key variables:

| Variable                     | Description                                               |
|------------------------------|-----------------------------------------------------------|
| `REDIS_URL`                  | Redis connection used by message-bus & heartbeats         |
| `DATABASE_URL`               | TimescaleDB connection URL                                |
| `SYMBOL_MODE`                | `all\|large\|mid\|small\|popular\|custom`              |
| `SYMBOLS`                    | Comma list when `SYMBOL_MODE=custom`                      |
| `MAX_SYMBOLS`                | Hard cap for data generator                               |
| `MAX_WEBSOCKET_SYMBOLS`      | Concurrent Alpaca stream subscriptions                    |
| `ALPACA_KEY / SECRET`        | (optional) live data keys                                 |
| `POLYGON_API_KEY`            | (optional) live data key                                  |
| `RATE_LIMIT_SLEEP`           | Seconds between historical-data HTTP calls                |
| `CONCURRENT_REQUESTS`        | Threads for `fast_backfill.py`                            |

> **Tip**   Any change to `.env` requires `docker compose up -d` to propagate.

---

## 3  Services (docker-compose)

| Name          | Image / Build | Port | Purpose                                   |
|---------------|--------------|------|-------------------------------------------|
| `redis`       | redis:7       | 6379 | Message bus (Redis Streams) + heartbeats  |
| `timescaledb` | timescaledb   | 5432 | Time-series storage (OHLCV ticks)         |
| `market_data` | local build   | —    | Real-time tick generator (mock / Alpaca)  |
| `strategy`    | local build   | —    | Signal engine                             |
| `execution`   | local build   | —    | Paper-trading executor                    |
| `storage`     | local build   | —    | Writes ticks to TimescaleDB               |
| `api`         | local build   | 8000 | Historical REST API                       |
| `dashboard`   | local build   | 8001 | Live Web UI (REST + WebSocket)            |

The **dashboard** directory is bind-mounted, so HTML/JS edits appear after a simple browser refresh.

---

## 4  System-Health panel
Every container publishes a heartbeat once every 30 s to Redis:

```
service:<SERVICE_NAME>:heartbeat   → ISO-8601 timestamp
service:<SERVICE_NAME>:errors      → optional error counter
```

The dashboard colour codes each row:

| Status      | Condition                                   |
|-------------|---------------------------------------------|
| healthy     | heartbeat < 5 min ago & error_count == 0     |
| unhealthy   | heartbeat ≥ 5 min                           |
| error       | `service:*:errors` > 0                      |
| unknown     | no heartbeat key found                      |

You can reset a stuck error state with:

```bash
docker compose exec redis redis-cli DEL service:<name>:errors
```

---

## 5  CLI utilities (optional)

All helper scripts live in project root:

| Script                  | Description                              |
|-------------------------|------------------------------------------|
| `manage_symbols.py`     | List / filter Alpaca symbol universe     |
| `benchmark_backfill.py` | Measure historical back-fill performance |
| `fast_backfill.py`      | High-concurrency back-fill               |
| `run_dashboard.py`      | Run dashboard outside Docker             |
| `demo_dashboard.py`     | Generate fake ticks for demo             |

Run any script with `python <script> --help` for options.

---

Happy trading 🚀 