"""SSE event diffing + fan-out for `GET /api/events` (scaling task brief, 2026-09-19 rewrite).

`LiveEventTracker`/`ChatEventTracker` are pure polling logic against anything shaped like
`store.LiveStore` (duck-typed, so both are unit-testable with a tiny in-memory fake, no SQLite
needed) -- they compute WHAT changed since the last poll, nothing about WHO gets told.

`Broadcaster` is the process-wide fan-out this scaling pass adds: ONE asyncio task polls the store
every `LIVE_POLL_INTERVAL_S` (started once, in `api.py`'s app `lifespan`, not per connection) and
pushes each event, serialized ONCE, into every subscriber's own bounded `asyncio.Queue`. Before
this rewrite, `api.py`'s `GET /api/events` route ran its OWN `LiveEventTracker`/poll loop per open
connection -- O(connections x queue) store queries per second, and a per-question `queued` event
meant `LiveEventTracker.watching` grew with the live queue too. A subscriber's own work is now a
single `await queue.get()` per event (O(1)), independent of how many other tabs are open or how
long the queue is.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from dataclasses import dataclass, field

from bioreservoir.live import card, pipeline, summary


@dataclass
class LiveEventTracker:
    """Diffs `live_questions` since the last poll into `thinking`/`answered`/`queue` events (task
    brief: no more per-question `queued` broadcast -- "the asker gets its position from the ask
    response + `queue` ticks"). Two monotonic id cursors (`thinking_since`/`answered_since` are
    each "rows changed since this id", index-backed in `store.py`) replace the old design's per-id
    `watching` dict, which cost O(watched ids) per poll and existed only to know when to emit the
    now-removed `queued` event.
    """

    last_thinking_id: int = 0
    last_answered_id: int = 0
    last_queue_snapshot: tuple[int, int, float] | None = None
    queue_heartbeat_s: float = 30.0  # task brief: emit a `queue` tick at least this often regardless of change
    _last_queue_emit_monotonic: float = field(default=0.0, repr=False, compare=False)

    def poll(self, live_store) -> list[tuple[str, dict]]:
        events: list[tuple[str, dict]] = []

        for row in live_store.thinking_since(self.last_thinking_id):
            self.last_thinking_id = max(self.last_thinking_id, row["id"])
            events.append((
                "thinking",
                {
                    "id": row["id"],
                    "question": row["question"],
                    "started_at": row["started_at"],
                    "embargoed": card.question_embargoed(row["question"]),
                    "yes_side": pipeline.yes_side_for(row["question"]),
                },
            ))

        for row in live_store.answered_since(self.last_answered_id):
            self.last_answered_id = max(self.last_answered_id, row["id"])
            events.append((
                "answered",
                summary.answer_summary(json.loads(row["answer_json"]), likes=row["likes"] or 0),
            ))

        snapshot = (live_store.queue_length(), live_store.claimed_total(), live_store.avg_cycle_s())
        now = time.monotonic()
        changed = snapshot != self.last_queue_snapshot
        due_for_heartbeat = (now - self._last_queue_emit_monotonic) >= self.queue_heartbeat_s
        if changed or due_for_heartbeat:
            events.append((
                "queue",
                {"queue_length": snapshot[0], "claimed_total": snapshot[1], "avg_cycle_s": snapshot[2]},
            ))
            self.last_queue_snapshot = snapshot
            self._last_queue_emit_monotonic = now

        return events


def _chat_payload(row) -> dict:
    return {
        "id": row["id"],
        "nickname": row["nickname"],
        "text": row["text"],
        "created_at": row["created_at"],
    }


@dataclass
class ChatEventTracker:
    """SSE event diffing for the live chat, polled from the SAME broadcaster loop as
    `LiveEventTracker` above (task brief: "do not open a second stream"). Duck-typed against
    anything shaped like `store.LiveStore` (`.chat_state`/`.chat_messages_after`/
    `.chat_deleted_after`/`.chat_restored_after`).

    Three event kinds, all diffed against what this tracker has already emitted so nothing repeats
    across polls:
      - `chat_state`: emitted once immediately on the FIRST poll (so a freshly-connected client
        learns the current slow-mode interval / soft-pause state without waiting for a change),
        then again only when the state actually changes.
      - `chat`: a brand-new, not-yet-deleted message, OR a previously-deleted message an operator
        just restored (`chat_admin restore`) -- reusing the same event name/shape for a restore is
        deliberate: the frontend does not need a fourth event type, it just re-shows/re-inserts by
        id, the same handling a genuinely new message already needs.
      - `chat_delete`: a message (admin-deleted or auto-hidden by the report threshold) that this
        tracker had not yet announced as deleted.
    """

    last_id: int = 0
    last_deleted_at: str = ""
    last_restored_at: str = ""
    last_state: dict | None = None

    def poll(self, live_store) -> list[tuple[str, dict]]:
        out: list[tuple[str, dict]] = []

        state = live_store.chat_state()
        if state != self.last_state:
            out.append(("chat_state", state))
            self.last_state = state

        for row in live_store.chat_messages_after(self.last_id):
            self.last_id = max(self.last_id, row["id"])
            if not row["deleted"]:
                out.append(("chat", _chat_payload(row)))

        for row in live_store.chat_deleted_after(self.last_deleted_at):
            if row["deleted_at"] and row["deleted_at"] > self.last_deleted_at:
                self.last_deleted_at = row["deleted_at"]
            out.append(("chat_delete", {"id": row["id"]}))

        for row in live_store.chat_restored_after(self.last_restored_at):
            if row["restored_at"] and row["restored_at"] > self.last_restored_at:
                self.last_restored_at = row["restored_at"]
            out.append(("chat", _chat_payload(row)))

        return out


class Broadcaster:
    """ONE poll loop per process, fanned out to N subscribers (task brief). Constructed once in
    `api.py`'s app `lifespan` (started there too -- "in the app lifespan" per the task brief's
    allowed options, simpler than a lazy-first-subscriber start since it needs no race protection
    around task creation) and stopped cleanly on shutdown.

    `chat_tracker=None` (the default) means chat is not wired up at all for this process (hard
    kill switch off, or a required secret missing -- `api.py`'s own `_chat_available` check, which
    this module does not need to know about: the caller decides, matching this codebase's existing
    store-owns-the-query/caller-owns-the-business-rule split) -- the broadcaster simply never polls
    or emits any chat event in that case, same as the old per-connection code's `chat_tracker =
    None` branch.

    Slow-subscriber policy (task brief: "a slow client must never block the others"): each
    subscriber gets a BOUNDED `asyncio.Queue`; a full queue drops its OLDEST pending event to make
    room for the new one, rather than dropping the new event or dropping the subscriber outright.
    Chosen over "drop the subscriber" because a single missed `thinking`/`queue` tick is
    self-healing (the next tick supersedes it, and `answered` is idempotent by id on the frontend)
    -- disconnecting a client whose tab was merely backgrounded for a few seconds would be a worse
    experience than it silently catching up on the next tick. Chosen over "drop the new event"
    because the newest snapshot is always the most useful one to a client that reconnects/catches
    up (an old `queue` tick's numbers are stale the moment a newer one exists).
    """

    def __init__(
        self,
        live_store,
        poll_interval_s: float,
        chat_tracker: ChatEventTracker | None = None,
        queue_maxsize: int = 64,
    ) -> None:
        self._store = live_store
        self._poll_interval_s = poll_interval_s
        self._tracker = LiveEventTracker()
        self._chat_tracker = chat_tracker
        self._queue_maxsize = queue_maxsize
        self._subscribers: dict[int, asyncio.Queue] = {}
        self._next_id = 0
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    async def _run(self) -> None:
        while True:
            for name, payload in self._tracker.poll(self._store):
                self._publish(name, payload)
            if self._chat_tracker is not None:
                for name, payload in self._chat_tracker.poll(self._store):
                    self._publish(name, payload)
            await asyncio.sleep(self._poll_interval_s)

    def _publish(self, name: str, payload: dict) -> None:
        data = json.dumps(payload)  # serialized ONCE per event, not once per subscriber
        for q in list(self._subscribers.values()):
            self._put_dropping_oldest(q, (name, data))

    @staticmethod
    def _put_dropping_oldest(q: asyncio.Queue, item: tuple[str, str]) -> None:
        try:
            q.put_nowait(item)
        except asyncio.QueueFull:
            with contextlib.suppress(asyncio.QueueEmpty):
                q.get_nowait()
            with contextlib.suppress(asyncio.QueueFull):
                q.put_nowait(item)

    def subscribe(self) -> tuple[int, asyncio.Queue]:
        """Registers a new subscriber and returns `(subscriber_id, queue)` -- `api.py`'s SSE route
        awaits `queue.get()` in a loop and calls `unsubscribe(subscriber_id)` on disconnect. Task
        brief: "a fresh connection gets nothing retroactive... except it is fine to send one
        `queue` tick and one `chat_state` on connect" -- seeded here from whatever this
        broadcaster's trackers already know, so a subscriber never has to wait a full poll interval
        for its very first render."""
        sid = self._next_id
        self._next_id += 1
        q: asyncio.Queue = asyncio.Queue(maxsize=self._queue_maxsize)
        self._subscribers[sid] = q
        if self._tracker.last_queue_snapshot is not None:
            ql, ct, ac = self._tracker.last_queue_snapshot
            q.put_nowait(("queue", json.dumps({"queue_length": ql, "claimed_total": ct, "avg_cycle_s": ac})))
        if self._chat_tracker is not None and self._chat_tracker.last_state is not None:
            q.put_nowait(("chat_state", json.dumps(self._chat_tracker.last_state)))
        return sid, q

    def unsubscribe(self, subscriber_id: int) -> None:
        self._subscribers.pop(subscriber_id, None)

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)
