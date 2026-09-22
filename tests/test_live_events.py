"""events.py: `LiveEventTracker`/`ChatEventTracker` (SSE event diffing against a duck-typed store)
and `Broadcaster` (the process-wide fan-out, scaling task brief 2026-09-19).
"""

from __future__ import annotations

import asyncio
import json

import pytest

from bioreservoir.live import pipeline
from bioreservoir.live.events import Broadcaster, ChatEventTracker, LiveEventTracker


class FakeRow(dict):
    """dict with attribute-free `row["key"]` access, matching `sqlite3.Row`'s subscript API."""


class FakeStore:
    """Duck-typed against everything `LiveEventTracker`/`Broadcaster` need:
    `.thinking_since`/`.answered_since`/`.queue_length`/`.claimed_total`/`.avg_cycle_s` -- the same
    shape `store.LiveStore` exposes (see that module's own docstrings for each method's exact
    contract)."""

    def __init__(self):
        self.rows: dict[int, FakeRow] = {}
        self._claimed_total = 0
        self._avg_cycle_s = 60.0

    def add(self, id_, question, status="queued", started_at=None, answer=None, likes=0):
        # `likes` mirrors the real column (store.py `_migrate_columns`): the broadcast carries the
        # count, so a double that lacks it would hide exactly that bug.
        row = FakeRow(id=id_, question=question, status=status, started_at=started_at, likes=likes)
        if answer is not None:
            row["answer_json"] = json.dumps(answer)
        self.rows[id_] = row
        return row

    def claim(self, id_, started_at="t0"):
        self.rows[id_]["status"] = "thinking"
        self.rows[id_]["started_at"] = started_at
        self._claimed_total += 1

    def answer(self, id_, answer):
        self.rows[id_]["status"] = "answered"
        self.rows[id_]["answer_json"] = json.dumps(answer)

    def reject_thinking(self, id_):
        """Simulates `worker.mark_failed`: leaves `started_at` in place but the row is never
        `answered` -- `answered_since` must never pick this up (store.py's own distinction)."""
        self.rows[id_]["status"] = "rejected"

    def thinking_since(self, last_id):
        return sorted(
            (r for i, r in self.rows.items() if i > last_id and r["started_at"] is not None),
            key=lambda r: r["id"],
        )

    def answered_since(self, last_id):
        return sorted(
            (r for i, r in self.rows.items() if i > last_id and r["status"] == "answered"),
            key=lambda r: r["id"],
        )

    def queue_length(self):
        return sum(1 for r in self.rows.values() if r["status"] in ("queued", "thinking"))

    def claimed_total(self):
        return self._claimed_total

    def avg_cycle_s(self):
        return self._avg_cycle_s


def test_poll_emits_nothing_for_a_brand_new_queued_row():
    """Task brief: no more per-question `queued` broadcast -- only the FIRST `queue` tick (the
    heartbeat, since nothing has been emitted yet) comes out of a freshly-queued row."""
    store = FakeStore()
    store.add(1, "Will it rain?")
    tracker = LiveEventTracker()
    events = tracker.poll(store)
    assert [e[0] for e in events] == ["queue"]
    assert events[0][1] == {"queue_length": 1, "claimed_total": 0, "avg_cycle_s": 60.0}


def test_poll_emits_thinking_when_a_row_is_claimed():
    store = FakeStore()
    store.add(1, "Will it rain?")
    tracker = LiveEventTracker()
    tracker.poll(store)
    store.claim(1, started_at="2026-09-19T00:00:00+00:00")
    events = tracker.poll(store)
    thinking = [e for e in events if e[0] == "thinking"]
    assert thinking == [
        ("thinking", {
            "id": 1, "question": "Will it rain?",
            "started_at": "2026-09-19T00:00:00+00:00", "embargoed": False,
            # Known from the wording alone: the page labels both options while the fly decides.
            "yes_side": pipeline.yes_side_for("Will it rain?"),
        })
    ]


def test_poll_does_not_flag_an_election_question_while_thinking():
    """The embargo is off (2026-09-21) — an election question is thinking in public like any
    other."""
    store = FakeStore()
    store.add(1, "Will the incumbent win the election?")
    store.claim(1)
    thinking = next(e for e in LiveEventTracker().poll(store) if e[0] == "thinking")
    assert thinking[1]["embargoed"] is False


def test_poll_flags_an_election_question_as_embargoed_while_thinking(monkeypatch):
    """Mechanism cover for whenever the embargo is switched back on."""
    from bioreservoir.live import config as live_config

    monkeypatch.setattr(live_config, "ELECTION_CARD_EMBARGO_ENABLED", True)
    store = FakeStore()
    store.add(1, "Will the incumbent win the election?")
    store.claim(1)
    tracker = LiveEventTracker()
    events = tracker.poll(store)
    thinking = next(e for e in events if e[0] == "thinking")
    assert thinking[1]["embargoed"] is True


def test_poll_does_not_re_emit_the_same_thinking_row_twice():
    store = FakeStore()
    store.add(1, "Will it rain?")
    store.claim(1)
    tracker = LiveEventTracker()
    tracker.poll(store)
    events = tracker.poll(store)
    assert [e for e in events if e[0] == "thinking"] == []


def test_poll_emits_answered_with_the_summary_payload_not_the_full_answer():
    store = FakeStore()
    store.add(1, "Will it rain?", likes=4)
    store.claim(1)
    tracker = LiveEventTracker()
    tracker.poll(store)
    answer = {
        "id": 1, "question": "Will it rain?", "answer": "yes", "lateral_bias": 0.2,
        "lab": {"yes_side": "left"}, "answered_at": "2026-09-19T00:00:00+00:00",
    }
    store.answer(1, answer)
    events = tracker.poll(store)
    answered = [e for e in events if e[0] == "answered"]
    assert len(answered) == 1
    payload = answered[0][1]
    assert payload["id"] == 1
    assert payload["answer"] == "yes"
    assert "lab" not in payload  # AnswerSummary -- no spike data (task brief)
    assert set(payload) == {
        "id", "question", "answer", "embargoed", "yes_side", "lateral_bias", "turn_strength", "answered_at",
        "likes",
    }
    assert payload["likes"] == 4  # the row's real count, not a hardcoded zero


def test_poll_does_not_re_emit_the_same_answered_row_twice():
    store = FakeStore()
    store.add(1, "Will it rain?")
    store.claim(1)
    store.answer(1, {"id": 1, "question": "Will it rain?", "answer": "yes", "lateral_bias": 0.1})
    tracker = LiveEventTracker()
    tracker.poll(store)
    events = tracker.poll(store)
    assert [e for e in events if e[0] == "answered"] == []


def test_poll_never_treats_a_mark_failed_row_as_answered():
    """`mark_failed` moves a `thinking` row to `rejected` -- it must still get its ONE `thinking`
    event (it WAS claimed), but never an `answered` one."""
    store = FakeStore()
    store.add(1, "spam???")
    store.claim(1)
    store.reject_thinking(1)
    tracker = LiveEventTracker()
    events = tracker.poll(store)
    assert any(e[0] == "thinking" for e in events)
    assert not any(e[0] == "answered" for e in events)


def test_poll_emits_a_queue_tick_when_the_snapshot_changes():
    store = FakeStore()
    tracker = LiveEventTracker()
    tracker.poll(store)  # first poll always emits (heartbeat)
    store.add(1, "Will it rain?")
    events = tracker.poll(store)
    assert [e[0] for e in events] == ["queue"]
    assert events[0][1]["queue_length"] == 1


def test_poll_does_not_re_emit_an_unchanged_queue_snapshot_before_the_heartbeat():
    store = FakeStore()
    tracker = LiveEventTracker(queue_heartbeat_s=9999.0)
    tracker.poll(store)
    assert tracker.poll(store) == []


def test_poll_emits_a_queue_heartbeat_even_with_no_change():
    store = FakeStore()
    tracker = LiveEventTracker(queue_heartbeat_s=0.0)  # always "due"
    tracker.poll(store)
    events = tracker.poll(store)
    assert [e[0] for e in events] == ["queue"]


# -- ChatEventTracker ---------------------------------------------------------------------------


class FakeChatStore:
    def __init__(self, enabled=True, slowmode_s=4.0):
        self.messages: dict[int, FakeRow] = {}
        self._state = {"enabled": enabled, "slowmode_s": slowmode_s}

    def add(self, id_, nickname, text, created_at="t0"):
        self.messages[id_] = FakeRow(
            id=id_, nickname=nickname, text=text, created_at=created_at,
            deleted=0, deleted_at=None, restored_at=None,
        )
        return self.messages[id_]

    def chat_state(self):
        return dict(self._state)

    def set_state(self, **kwargs):
        self._state.update(kwargs)

    def chat_messages_after(self, last_id):
        return sorted((r for i, r in self.messages.items() if i > last_id), key=lambda r: r["id"])

    def soft_delete(self, id_, deleted_at):
        self.messages[id_]["deleted"] = 1
        self.messages[id_]["deleted_at"] = deleted_at

    def restore(self, id_, restored_at):
        self.messages[id_]["deleted"] = 0
        self.messages[id_]["deleted_at"] = None
        self.messages[id_]["restored_at"] = restored_at

    def chat_deleted_after(self, since):
        return sorted(
            (r for r in self.messages.values() if r["deleted"] and r["deleted_at"] and r["deleted_at"] > since),
            key=lambda r: r["deleted_at"],
        )

    def chat_restored_after(self, since):
        return sorted(
            (r for r in self.messages.values() if not r["deleted"] and r["restored_at"] and r["restored_at"] > since),
            key=lambda r: r["restored_at"],
        )


def test_chat_tracker_emits_state_on_the_first_poll_even_with_no_messages():
    store = FakeChatStore()
    tracker = ChatEventTracker()
    events = tracker.poll(store)
    assert events == [("chat_state", {"enabled": True, "slowmode_s": 4.0})]


def test_chat_tracker_does_not_re_emit_unchanged_state():
    store = FakeChatStore()
    tracker = ChatEventTracker()
    tracker.poll(store)
    assert tracker.poll(store) == []


def test_chat_tracker_emits_state_again_on_change():
    store = FakeChatStore()
    tracker = ChatEventTracker()
    tracker.poll(store)
    store.set_state(slowmode_s=8.0)
    events = tracker.poll(store)
    assert events == [("chat_state", {"enabled": True, "slowmode_s": 8.0})]


def test_chat_tracker_emits_chat_for_a_new_message():
    store = FakeChatStore()
    store.add(1, "fly-1", "hello")
    tracker = ChatEventTracker()
    events = tracker.poll(store)
    chat_events = [e for e in events if e[0] == "chat"]
    assert chat_events == [("chat", {"id": 1, "nickname": "fly-1", "text": "hello", "created_at": "t0"})]
    assert tracker.last_id == 1


def test_chat_tracker_does_not_re_emit_the_same_new_message_twice():
    store = FakeChatStore()
    store.add(1, "fly-1", "hello")
    tracker = ChatEventTracker()
    tracker.poll(store)
    events = tracker.poll(store)
    assert [e for e in events if e[0] == "chat"] == []


def test_chat_tracker_emits_multiple_new_messages_in_order():
    store = FakeChatStore()
    store.add(1, "fly-1", "one")
    store.add(2, "fly-2", "two")
    tracker = ChatEventTracker()
    events = tracker.poll(store)
    chat_events = [e for e in events if e[0] == "chat"]
    assert [e[1]["text"] for e in chat_events] == ["one", "two"]


def test_chat_tracker_never_emits_chat_for_a_message_deleted_before_first_seen():
    store = FakeChatStore()
    store.add(1, "fly-1", "gone before anyone saw it")
    store.soft_delete(1, "t1")
    tracker = ChatEventTracker()
    events = tracker.poll(store)
    assert [e for e in events if e[0] == "chat"] == []


def test_chat_tracker_emits_chat_delete_on_deletion():
    store = FakeChatStore()
    store.add(1, "fly-1", "will be deleted")
    tracker = ChatEventTracker()
    tracker.poll(store)
    store.soft_delete(1, "t1")
    events = tracker.poll(store)
    assert ("chat_delete", {"id": 1}) in events


def test_chat_tracker_does_not_re_emit_the_same_deletion_twice():
    store = FakeChatStore()
    store.add(1, "fly-1", "will be deleted")
    tracker = ChatEventTracker()
    tracker.poll(store)
    store.soft_delete(1, "t1")
    tracker.poll(store)
    events = tracker.poll(store)
    assert [e for e in events if e[0] == "chat_delete"] == []


def test_chat_tracker_emits_chat_again_on_restore():
    store = FakeChatStore()
    store.add(1, "fly-1", "restored message")
    tracker = ChatEventTracker()
    tracker.poll(store)
    store.soft_delete(1, "t1")
    tracker.poll(store)
    store.restore(1, "t2")
    events = tracker.poll(store)
    chat_events = [e for e in events if e[0] == "chat"]
    assert chat_events == [("chat", {"id": 1, "nickname": "fly-1", "text": "restored message", "created_at": "t0"})]


def test_chat_tracker_seeded_to_now_does_not_replay_old_deletions_or_restores_f6():
    """F6 (2026-09-18 security review): a fresh SSE connection must never replay deleted/hidden
    history. Reproduced against the OLD default (`last_deleted_at`/`last_restored_at = ""`, "the
    beginning of time"): a tracker constructed after a `chat_admin clear` replayed every
    `chat_delete`/restore ever issued. api.py's fix seeds both cursors to "now" at connect time --
    this test proves a tracker seeded that way sees nothing from before its own construction."""
    store = FakeChatStore()
    store.add(1, "fly-1", "old message")
    store.soft_delete(1, "2020-01-01T00:00:00")
    store.add(2, "fly-2", "old restored message")
    store.soft_delete(2, "2020-01-01T00:00:01")
    store.restore(2, "2020-01-01T00:00:02")

    # Unseeded (the pre-fix default) DOES replay the old activity -- documents the bug shape.
    buggy_tracker = ChatEventTracker(last_id=2)
    buggy_events = buggy_tracker.poll(store)
    assert ("chat_delete", {"id": 1}) in buggy_events
    assert any(e[0] == "chat" and e[1]["id"] == 2 for e in buggy_events)

    # Seeded to "now" (after all the old activity, mirroring api.py's live_events()) sees none of
    # it -- only a "chat_state" event on the very first poll.
    fixed_tracker = ChatEventTracker(
        last_id=2, last_deleted_at="2030-01-01T00:00:00", last_restored_at="2030-01-01T00:00:00"
    )
    fixed_events = fixed_tracker.poll(store)
    assert ("chat_delete", {"id": 1}) not in fixed_events
    assert not any(e[0] == "chat" and e[1]["id"] == 2 for e in fixed_events)
    assert [e[0] for e in fixed_events] == ["chat_state"]


# -- Broadcaster (task brief: one store poll per tick, fanned out to N subscribers) --------------


async def _drain(q: asyncio.Queue, n: int, timeout: float = 2.0) -> list[tuple[str, str]]:
    items = []
    for _ in range(n):
        items.append(await asyncio.wait_for(q.get(), timeout=timeout))
    return items


@pytest.mark.anyio
async def test_broadcaster_fans_out_one_poll_to_every_subscriber():
    """A single `LiveEventTracker.poll` call per tick must reach every subscriber -- not one poll
    per subscriber (the whole point of the scaling rewrite)."""
    store = FakeStore()
    store.add(1, "Will it rain?")
    b = Broadcaster(store, poll_interval_s=0.01)
    _, q1 = b.subscribe()
    _, q2 = b.subscribe()
    b.start()
    try:
        e1 = await _drain(q1, 1)
        e2 = await _drain(q2, 1)
        assert e1 == e2  # identical serialized payload delivered to both
        assert e1[0][0] == "queue"
    finally:
        await b.stop()


@pytest.mark.anyio
async def test_broadcaster_new_subscriber_gets_no_backlog_before_it_joined():
    store = FakeStore()
    b = Broadcaster(store, poll_interval_s=0.01)
    b.start()
    try:
        # Let a few ticks pass before the second subscriber ever joins.
        await asyncio.sleep(0.05)
        _, late = b.subscribe()
        assert late.qsize() <= 1  # at most the one seeded "queue" tick task brief allows, no backlog
    finally:
        await b.stop()


@pytest.mark.anyio
async def test_broadcaster_subscribe_seeds_the_last_known_queue_snapshot_immediately():
    """Task brief: "it is fine to send one `queue` tick... on connect" -- a subscriber joining
    AFTER the broadcaster already has state should not have to wait a full poll interval."""
    store = FakeStore()
    store.add(1, "Will it rain?")
    b = Broadcaster(store, poll_interval_s=999.0)  # long enough that only the seed can arrive
    b.start()
    try:
        await asyncio.sleep(0.02)  # let the first poll happen once
        _, q = b.subscribe()
        name, data = await asyncio.wait_for(q.get(), timeout=1.0)
        assert name == "queue"
        assert json.loads(data)["queue_length"] == 1
    finally:
        await b.stop()


@pytest.mark.anyio
async def test_broadcaster_unsubscribe_removes_the_client_from_fan_out():
    store = FakeStore()
    b = Broadcaster(store, poll_interval_s=0.01)
    sid, q = b.subscribe()
    b.start()
    try:
        await asyncio.sleep(0.03)
        b.unsubscribe(sid)
        assert b.subscriber_count == 0
        # Draining whatever arrived before unsubscribing, then confirming no NEW growth after.
        while not q.empty():
            q.get_nowait()
        await asyncio.sleep(0.05)
        assert q.qsize() == 0  # nothing more delivered to an unsubscribed queue
    finally:
        await b.stop()


def test_broadcaster_slow_subscriber_drops_oldest_and_never_blocks_publish():
    """A full subscriber queue must not block delivery to anyone else (task brief: "a slow client
    must never block the others") -- `_publish` drops that subscriber's OLDEST pending item to make
    room rather than awaiting a blocked `put`. White-box (`_publish` called directly, no real poll
    loop/timing) for a deterministic, non-flaky assertion on exactly which items survive."""
    store = FakeStore()
    b = Broadcaster(store, poll_interval_s=0.01, queue_maxsize=2)
    _, slow_q = b.subscribe()  # never drained
    _, fast_q = b.subscribe()  # drained below, after every publish
    for i in range(10):
        b._publish("queue", {"i": i})
        assert fast_q.get_nowait()[0] == "queue"  # the other subscriber never falls behind

    assert slow_q.qsize() == 2  # bounded at queue_maxsize regardless of how many were published
    survivors = [json.loads(slow_q.get_nowait()[1])["i"] for _ in range(2)]
    assert survivors == [8, 9]  # the two MOST RECENT events -- oldest dropped along the way


@pytest.mark.anyio
async def test_broadcaster_chat_tracker_none_emits_no_chat_events():
    store = FakeStore()
    b = Broadcaster(store, poll_interval_s=0.01, chat_tracker=None)
    _, q = b.subscribe()
    b.start()
    try:
        events = await _drain(q, 1)
        assert all(name != "chat_state" for name, _ in events)
    finally:
        await b.stop()


@pytest.mark.anyio
async def test_broadcaster_stop_is_idempotent_and_cancels_the_poll_task():
    store = FakeStore()
    b = Broadcaster(store, poll_interval_s=0.01)
    b.start()
    await asyncio.sleep(0.02)
    await b.stop()
    await b.stop()  # second call must not raise
