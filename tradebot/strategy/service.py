import asyncio
import logging
from collections import deque, defaultdict
from statistics import mean
from datetime import datetime, timezone

from tradebot.common.bus import MessageBus
from tradebot.common.models import PriceTick, Signal, Side

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("strategy")

SHORT_WINDOW = 5  # in ticks
LONG_WINDOW = 20


class MovingAverageStrategy:
    """Very naive SMA crossover strategy per symbol."""

    def __init__(self):
        self.prices: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=LONG_WINDOW))
        self.last_side: dict[str, Side | None] = defaultdict(lambda: None)

    def on_tick(self, tick: PriceTick) -> Signal | None:
        dq = self.prices[tick.symbol]
        dq.append(tick.price)

        if len(dq) < LONG_WINDOW:
            return None

        short_ma = mean(list(dq)[-SHORT_WINDOW:])
        long_ma = mean(dq)

        if short_ma > long_ma and self.last_side[tick.symbol] != Side.buy:
            self.last_side[tick.symbol] = Side.buy
            return Signal(symbol=tick.symbol, side=Side.buy, confidence=abs(short_ma - long_ma) / long_ma, timestamp=tick.timestamp)
        elif short_ma < long_ma and self.last_side[tick.symbol] != Side.sell:
            self.last_side[tick.symbol] = Side.sell
            return Signal(symbol=tick.symbol, side=Side.sell, confidence=abs(short_ma - long_ma) / long_ma, timestamp=tick.timestamp)
        return None


async def main():
    bus = MessageBus()
    await bus.connect()

    strat = MovingAverageStrategy()

    async for msg in bus.subscribe("price.ticks", last_id="$", block_ms=1000):
        try:
            tick = PriceTick(**msg)
        except Exception as e:
            logger.warning("Malformed tick %s: %s", msg, e)
            continue

        signal = strat.on_tick(tick)
        if signal:
            await bus.publish("orders.new", signal.dict())
            logger.info("Generated signal %s", signal.json())


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Strategy service stopped") 