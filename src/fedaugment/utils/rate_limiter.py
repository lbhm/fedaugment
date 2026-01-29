import asyncio
import time
from collections import deque


class RateLimiter:
    def __init__(self, max_requests_per_minute: int, max_cost_per_minute: float) -> None:
        """Create a rate limiter with specified rate and cost constraints.

        Args:
            max_requests_per_minute: Maximum number of requests allowed per minute.
            max_cost_per_minute: Max cumulative cost allowed in a 60-second window.
        """
        self._max_requests = max_requests_per_minute
        self._max_cost = max_cost_per_minute

        self._request_log: deque[tuple[float, float]] = deque()  # (timestamp, cost)
        self._total_cost = 0.0

    def _prune(self, now: float) -> None:
        """Remove requests older than 60 seconds."""
        while self._request_log and self._request_log[0][0] <= now - 60:
            _, old_cost = self._request_log.popleft()
            self._total_cost -= old_cost

    def _can_proceed(self, now: float, cost: float) -> bool:
        """Check if a request can proceed based on the current state."""
        self._prune(now)
        return (
            len(self._request_log) < self._max_requests
            and (self._total_cost + cost) <= self._max_cost
        )

    def allow_request(self, cost: float = 1.0) -> bool:
        """Returns True if the request is allowed under the rate and cost limits."""
        now = time.monotonic()
        if self._can_proceed(now, cost):
            self._request_log.append((now, cost))
            self._total_cost += cost
            return True
        return False

    async def wait(self, cost: float = 1.0) -> None:
        """Block until a request with the given cost is allowed."""
        while True:
            now = time.monotonic()
            if self._can_proceed(now, cost):
                self._request_log.append((now, cost))
                self._total_cost += cost
                return
            await asyncio.sleep(0.1)
