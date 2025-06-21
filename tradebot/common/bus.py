import asyncio
import json
import os
from typing import AsyncIterator, Dict, Any

import redis.asyncio as redis

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")


class MessageBus:
    """Lightweight wrapper around Redis Streams for pub-sub style messaging."""

    def __init__(self):
        self._redis: redis.Redis | None = None

    async def connect(self) -> None:
        if self._redis is None:
            self._redis = redis.from_url(REDIS_URL, decode_responses=True)

    # ---------------------------------------------------------------------
    # Publish
    # ---------------------------------------------------------------------
    async def publish(self, stream: str, message: Dict[str, Any] | Any) -> None:
        """Publish *message* to *stream*.

        *message* can be a plain dict or a Pydantic model. Datetime objects are
        automatically converted to ISO-8601 strings via ``default=str``.
        """
        if self._redis is None:
            await self.connect()
        assert self._redis is not None

        if hasattr(message, "model_dump"):
            # Pydantic v2 model
            payload = message.model_dump()
        elif hasattr(message, "dict"):
            # Pydantic v1 model or similar
            payload = message.dict()
        else:
            payload = message

        await self._redis.xadd(stream, {"data": json.dumps(payload, default=str)})

    # ---------------------------------------------------------------------
    # Subscribe
    # ---------------------------------------------------------------------
    async def subscribe(
        self,
        stream: str,
        last_id: str = "$",
        block_ms: int = 0,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Yield messages from *stream* as they arrive.

        Args:
            stream: Redis stream key, e.g. "price.ticks".
            last_id: Start reading after this entry ID.  Use "$" to read only new messages.
            block_ms: Timeout for XREAD in milliseconds. 0 = no block.
        """
        if self._redis is None:
            await self.connect()
        assert self._redis is not None

        while True:
            response = await self._redis.xread({stream: last_id}, block=block_ms, count=1)
            if not response:
                # No message within timeout
                await asyncio.sleep(0.01)
                continue

            _, entries = response[0]
            entry_id, data = entries[0]
            last_id = entry_id
            yield json.loads(data["data"]) 