"""In-process per-client attempt counter for `POST /api/ask`.

Why this exists: the per-IP rate limit counts ROWS in `live_questions` (`store.count_recent` /
`count_today`), and since the dedupe landed an ask no longer always creates a row — a repeat of an
answered question, or one that joins a twin already in line, is served with no insert at all. That
made the limit bypassable by re-sending the same wording, and every such ask still costs one
outbound Turnstile verification. Counting attempts here, before anything else, closes both.

Memory-only and per-process on purpose: it protects the current process from a burst, and a restart
losing the counters is harmless (the row-based limit in `store.py` is the durable half; the limiter
takes `max()` of the two). Bounded by `max_clients`, oldest client evicted first, so a crowd of
unique IPs cannot grow it without limit.
"""

from __future__ import annotations

import time
from collections import OrderedDict, deque


class AttemptCounter:
    def __init__(self, max_clients: int = 50_000, retain_s: float = 86_400.0) -> None:
        self._by_client: OrderedDict[str, deque[float]] = OrderedDict()
        self._max_clients = max_clients
        self._retain_s = retain_s

    def record(self, client: str, now: float | None = None) -> None:
        stamp = time.monotonic() if now is None else now
        stamps = self._by_client.get(client)
        if stamps is None:
            stamps = deque()
            self._by_client[client] = stamps
        stamps.append(stamp)
        self._prune(stamps, stamp)
        self._by_client.move_to_end(client)
        while len(self._by_client) > self._max_clients:
            self._by_client.popitem(last=False)

    def count(self, client: str, window_s: float, now: float | None = None) -> int:
        stamp = time.monotonic() if now is None else now
        stamps = self._by_client.get(client)
        if not stamps:
            return 0
        self._prune(stamps, stamp)
        cutoff = stamp - window_s
        # `stamps` is ordered oldest-first, so a reverse walk stops at the first stamp outside it.
        n = 0
        for value in reversed(stamps):
            if value < cutoff:
                break
            n += 1
        return n

    def _prune(self, stamps: deque[float], now: float) -> None:
        cutoff = now - self._retain_s
        while stamps and stamps[0] < cutoff:
            stamps.popleft()

    def clients(self) -> int:
        return len(self._by_client)
