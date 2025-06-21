import asyncio
import logging
from datetime import datetime, timezone

from tradebot.common.bus import MessageBus
from tradebot.common.models import Signal

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("execution")


async def execute_order(signal: Signal):
    """Placeholder order execution – replace with Fidelity API call."""
    # TODO: Implement Fidelity brokerage integration here.
    logger.info("EXECUTE %s %s @ %s (confidence=%.2f)" % (
        signal.side.value,
        signal.symbol,
        signal.timestamp.strftime("%H:%M:%S"),
        signal.confidence,
    ))
    # Simulate network latency
    await asyncio.sleep(0.1)


async def main():
    bus = MessageBus()
    await bus.connect()

    async for msg in bus.subscribe("orders.new", last_id="$", block_ms=1000):
        try:
            signal = Signal(**msg)
        except Exception as e:
            logger.warning("Malformed signal %s: %s", msg, e)
            continue

        await execute_order(signal)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Execution service stopped") 