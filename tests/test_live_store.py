"""store.py: SQLite queue + answer store — FIFO order/position, rate-limit counters, atomic
`claim_next`, daily-salted IP hashing, and the rejected-question retention purge.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from bioreservoir.live.store import LiveStore


@pytest.fixture
def live_store(tmp_path):
    s = LiveStore(path=tmp_path / "live.sqlite")
    yield s
    s.close()


def test_enqueue_assigns_fifo_ids_and_position(live_store):
    id1 = live_store.enqueue("Will it rain?", ip_hash="h1")
    id2 = live_store.enqueue("Is the sky blue?", ip_hash="h1")
    assert id2 > id1
    assert live_store.position(id1) == 1
    assert live_store.position(id2) == 2


def test_position_is_none_once_claimed(live_store):
    id1 = live_store.enqueue("Will it rain?", ip_hash="h1")
    live_store.claim_next()
    assert live_store.position(id1) is None


def test_position_is_none_for_unknown_id(live_store):
    assert live_store.position(999) is None


def test_claim_next_is_fifo_and_sets_thinking(live_store):
    id1 = live_store.enqueue("Q1?", ip_hash="h1")
    live_store.enqueue("Q2?", ip_hash="h1")
    claimed = live_store.claim_next()
    assert claimed["id"] == id1
    assert claimed["status"] == "thinking"
    assert claimed["started_at"] is not None


def test_claim_next_returns_none_when_queue_empty(live_store):
    assert live_store.claim_next() is None


def test_record_answer_sets_status_and_payload(live_store):
    id1 = live_store.enqueue("Q1?", ip_hash="h1")
    live_store.claim_next()
    live_store.record_answer(id1, {"id": id1, "answer": "yes"})
    row = live_store.get(id1)
    assert row["status"] == "answered"
    assert row["answered_at"] is not None
    assert '"answer": "yes"' in row["answer_json"]


def test_mark_failed_sets_rejected_terminal_status(live_store):
    id1 = live_store.enqueue("Q1?", ip_hash="h1")
    live_store.claim_next()
    live_store.mark_failed(id1, "boom")
    row = live_store.get(id1)
    assert row["status"] == "rejected"
    assert row["reason"] == "spam"
    assert row["message"] == "boom"


def test_queue_length_counts_only_questions_still_waiting(live_store):
    """The one being simulated is shown separately on the live stage ("deciding now"), so counting
    it as waiting too showed one extra person in line."""
    id1 = live_store.enqueue("Q1?", ip_hash="h1")
    live_store.enqueue("Q2?", ip_hash="h1")
    assert live_store.queue_length() == 2
    live_store.claim_next()  # id1 -> thinking
    assert live_store.queue_length() == 1
    live_store.record_answer(id1, {"id": id1, "answer": "yes"})
    assert live_store.queue_length() == 1


def _answer_all(live_store, questions):
    ids = [live_store.enqueue(q, ip_hash="h1") for q in questions]
    for i in ids:
        live_store.claim_next()
        live_store.record_answer(i, {"id": i, "answer": "yes"})
    return ids


def test_answered_page_returns_newest_first(live_store):
    ids = _answer_all(live_store, [f"Q{i}?" for i in range(3)])
    page = live_store.answered_page("recent", limit=10)
    assert [row["id"] for row in page] == list(reversed(ids))
    assert live_store.answered_count() == 3


def test_answered_page_respects_limit_and_offset(live_store):
    ids = _answer_all(live_store, [f"Q{i}?" for i in range(5)])
    assert [row["id"] for row in live_store.answered_page("recent", limit=2)] == list(reversed(ids))[:2]
    assert [row["id"] for row in live_store.answered_page("recent", limit=2, offset=2)] == list(reversed(ids))[2:4]


def test_answered_page_top_is_most_liked_first_and_likes_toggle(live_store):
    """The "most liked" tab, and the rule that one client can only ever add one like to a
    question: liking twice from the same client takes the like back instead of counting twice."""
    a, b, c = _answer_all(live_store, ["Q a?", "Q b?", "Q c?"])
    assert live_store.toggle_like(b, "client-1") == (1, True)
    assert live_store.toggle_like(b, "client-2") == (2, True)
    assert live_store.toggle_like(c, "client-1") == (1, True)
    # Same client, same question: the like is taken back, not doubled.
    assert live_store.toggle_like(b, "client-2") == (1, False)
    top = live_store.answered_page("top", limit=10)
    # Equal likes tie-break by id DESC — exactly, not "one of two orders": paging depends on it.
    assert [row["id"] for row in top] == [c, b, a]
    assert {row["id"]: row["likes"] for row in top} == {a: 0, b: 1, c: 1}
    # Ties break by id, newest first, so paging can never repeat or skip a row.
    assert [row["id"] for row in live_store.answered_page("top", limit=2)] == [c, b]
    assert live_store.toggle_like(99999, "client-1") is None


# -- rate-limit counters -----------------------------------------------------------------------


def test_count_recent_only_counts_within_window(live_store):
    now = datetime.now(UTC)
    live_store.enqueue("old", ip_hash="h1", now=now - timedelta(seconds=120))
    live_store.enqueue("new", ip_hash="h1", now=now)
    assert live_store.count_recent("h1", window_seconds=60, now=now) == 1
    assert live_store.count_recent("h1", window_seconds=200, now=now) == 2


def test_count_recent_is_per_ip_hash(live_store):
    # Distinct text: same-wording, still-active (queued) rows from two different askers are now
    # collapsed into one row by the `question_key` dedupe/race guard (store.py's
    # `idx_live_questions_active_key`) -- this test only cares about `count_recent` being scoped
    # per `ip_hash`, not about question-text collisions, so it must not incidentally exercise that.
    now = datetime.now(UTC)
    live_store.enqueue("q1", ip_hash="h1", now=now)
    live_store.enqueue("q2", ip_hash="h2", now=now)
    assert live_store.count_recent("h1", window_seconds=60, now=now) == 1


def test_count_today_resets_at_utc_midnight(live_store):
    now = datetime.now(UTC).replace(hour=12)
    yesterday = now - timedelta(days=1)
    live_store.enqueue("q1", ip_hash="h1", now=yesterday)
    live_store.enqueue("q2", ip_hash="h1", now=now)
    assert live_store.count_today("h1", now=now) == 1


def test_count_queued_ignores_answered_and_rejected(live_store):
    id1 = live_store.enqueue("q1", ip_hash="h1")
    live_store.enqueue("q2", ip_hash="h1")
    assert live_store.count_queued("h1") == 2
    live_store.claim_next()
    live_store.record_answer(id1, {"id": id1})
    assert live_store.count_queued("h1") == 1


# -- IP hashing ------------------------------------------------------------------------------------


def test_hash_ip_is_stable_within_the_same_day(live_store):
    now = datetime.now(UTC)
    a = live_store.hash_ip("1.2.3.4", now=now)
    b = live_store.hash_ip("1.2.3.4", now=now)
    assert a == b


def test_hash_ip_changes_across_utc_days(live_store):
    day1 = datetime(2026, 9, 18, 12, tzinfo=UTC)
    day2 = datetime(2026, 9, 19, 12, tzinfo=UTC)
    a = live_store.hash_ip("1.2.3.4", now=day1)
    b = live_store.hash_ip("1.2.3.4", now=day2)
    assert a != b


def test_hash_ip_never_stores_the_raw_ip(live_store, tmp_path):
    live_store.hash_ip("203.0.113.42")
    live_store.enqueue("some question", ip_hash=live_store.hash_ip("203.0.113.42"))
    raw_bytes = (tmp_path / "live.sqlite").read_bytes()
    assert b"203.0.113.42" not in raw_bytes


# -- retention purge --------------------------------------------------------------------------------


def test_purge_expired_blanks_old_rejected_question_text(live_store):
    now = datetime.now(UTC)
    old = now - timedelta(days=10)
    id1 = live_store.reject("sensitive question", ip_hash="h1", reason="spam", message="nope", now=old)
    n = live_store.purge_expired(retention_days=7, now=now)
    assert n == 1
    row = live_store.get(id1)
    assert row["question"] == ""
    assert row["message"] is None


def test_purge_expired_leaves_recent_rejected_alone(live_store):
    now = datetime.now(UTC)
    id1 = live_store.reject("recent question", ip_hash="h1", reason="spam", message="nope", now=now)
    n = live_store.purge_expired(retention_days=7, now=now)
    assert n == 0
    assert live_store.get(id1)["question"] == "recent question"


def test_purge_expired_never_touches_answered_rows(live_store):
    now = datetime.now(UTC)
    old = now - timedelta(days=10)
    id1 = live_store.enqueue("kept forever", ip_hash="h1", now=old)
    live_store.claim_next()
    live_store.record_answer(id1, {"id": id1, "answer": "yes"}, now=old)
    live_store.purge_expired(retention_days=7, now=now)
    assert live_store.get(id1)["question"] == "kept forever"


def test_purge_expired_also_blanks_question_key(live_store):
    """The `question_key` column is a normalized near-copy of the question text (not a hash) --
    it must be blanked on the same schedule, or the retention rule is defeated in substance."""
    now = datetime.now(UTC)
    old = now - timedelta(days=10)
    id1 = live_store.reject("sensitive question", ip_hash="h1", reason="spam", message="nope", now=old)
    live_store.purge_expired(retention_days=7, now=now)
    assert live_store.get(id1)["question_key"] is None


# -- ask dedupe: question_key, twins, and the concurrent-ask race guard --------------------------


def test_enqueue_computes_question_key_when_not_given(live_store):
    id1 = live_store.enqueue("  Will IT rain tomorrow?  ", ip_hash="h1")
    assert live_store.get(id1)["question_key"] == "will it rain tomorrow"


def test_active_twin_finds_a_queued_row_by_normalized_key(live_store):
    id1 = live_store.enqueue("Will it rain tomorrow?", ip_hash="h1")
    twin = live_store.active_twin("will it rain tomorrow")
    assert twin["id"] == id1


def test_active_twin_finds_a_thinking_row_too(live_store):
    id1 = live_store.enqueue("Will it rain tomorrow?", ip_hash="h1")
    live_store.claim_next()
    twin = live_store.active_twin("will it rain tomorrow")
    assert twin["id"] == id1
    assert twin["status"] == "thinking"


def test_active_twin_is_none_once_answered(live_store):
    id1 = live_store.enqueue("Will it rain tomorrow?", ip_hash="h1")
    live_store.claim_next()
    live_store.record_answer(id1, {"id": id1, "answer": "yes"})
    assert live_store.active_twin("will it rain tomorrow") is None


def test_answered_twin_finds_the_answered_row_by_key(live_store):
    id1 = live_store.enqueue("Will it rain tomorrow?", ip_hash="h1")
    live_store.claim_next()
    live_store.record_answer(id1, {"id": id1, "answer": "yes"})
    twin = live_store.answered_twin("will it rain tomorrow")
    assert twin["id"] == id1


def test_answered_twin_is_none_for_an_unanswered_key(live_store):
    live_store.enqueue("Will it rain tomorrow?", ip_hash="h1")
    assert live_store.answered_twin("will it rain tomorrow") is None


def test_rejected_twin_finds_the_most_recent_rejection_by_key(live_store):
    live_store.reject("Will it rain tomorrow?", ip_hash="h1", reason="spam", message="nope")
    twin = live_store.rejected_twin("will it rain tomorrow")
    assert twin["reason"] == "spam"


def test_rejected_twin_is_none_after_its_text_has_been_purged(live_store):
    now = datetime.now(UTC)
    old = now - timedelta(days=10)
    live_store.reject("Will it rain tomorrow?", ip_hash="h1", reason="spam", message="nope", now=old)
    live_store.purge_expired(retention_days=7, now=now)
    assert live_store.rejected_twin("will it rain tomorrow") is None


def test_enqueue_raises_on_a_concurrent_duplicate_active_question(live_store):
    """The unique partial index (`idx_live_questions_active_key`) is the race guard task brief
    asks for: two identical asks that both pass every earlier check must not both create a row."""
    from bioreservoir.live.store import DuplicateActiveQuestion

    live_store.enqueue("Will it rain tomorrow?", ip_hash="h1")
    with pytest.raises(DuplicateActiveQuestion):
        live_store.enqueue("will it rain tomorrow?", ip_hash="h2")  # same key, different casing


def test_enqueue_allows_reusing_a_key_once_the_earlier_row_is_terminal(live_store):
    id1 = live_store.enqueue("Will it rain tomorrow?", ip_hash="h1")
    live_store.claim_next()
    live_store.record_answer(id1, {"id": id1, "answer": "yes"})
    id2 = live_store.enqueue("Will it rain tomorrow?", ip_hash="h2")  # same key, but id1 is terminal
    assert id2 != id1


def test_question_key_migration_backfills_an_existing_db_without_the_column(tmp_path):
    """Simulates a DB created before `question_key` existed (bare `INSERT` with no such column) --
    the next `LiveStore(...)` open must add the column AND backfill every existing row, not just
    rows inserted after the migration."""
    import sqlite3

    path = tmp_path / "legacy.sqlite"
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE live_questions (id INTEGER PRIMARY KEY AUTOINCREMENT, question TEXT NOT NULL, "
        "status TEXT NOT NULL, ip_hash TEXT NOT NULL, reason TEXT, message TEXT, created_at TEXT NOT "
        "NULL, started_at TEXT, answered_at TEXT, answer_json TEXT)"
    )
    conn.execute(
        "INSERT INTO live_questions (question, status, ip_hash, created_at) "
        "VALUES ('Will it rain tomorrow?', 'queued', 'h1', '2026-01-01T00:00:00+00:00')"
    )
    conn.commit()
    conn.close()

    s = LiveStore(path=path)
    try:
        row = s._conn.execute("SELECT question_key FROM live_questions").fetchone()
        assert row["question_key"] == "will it rain tomorrow"
        assert s.active_twin("will it rain tomorrow") is not None
    finally:
        s.close()


# -- claimed_total / avg_cycle_s (task brief: the client's live position math) --------------------


def test_claimed_total_counts_every_claim_ever_and_never_decreases(live_store):
    assert live_store.claimed_total() == 0
    id1 = live_store.enqueue("Q1?", ip_hash="h1")
    live_store.enqueue("Q2?", ip_hash="h1")
    live_store.claim_next()
    assert live_store.claimed_total() == 1
    live_store.claim_next()
    assert live_store.claimed_total() == 2
    live_store.record_answer(id1, {"id": id1, "answer": "yes"})
    assert live_store.claimed_total() == 2  # answering doesn't change the claim count


def test_claimed_total_counts_a_row_marked_failed_too(live_store):
    id1 = live_store.enqueue("Q1?", ip_hash="h1")
    live_store.claim_next()
    live_store.mark_failed(id1, "boom")
    assert live_store.claimed_total() == 1


def test_avg_cycle_s_falls_back_when_no_answers_exist_yet(live_store):
    assert live_store.avg_cycle_s(fallback=60.0) == 60.0


def test_avg_cycle_s_is_the_median_of_the_last_n_cycle_durations(live_store):
    now = datetime.now(UTC)
    for i, secs in enumerate((10, 20, 90)):
        id_ = live_store.enqueue(f"Q{i}?", ip_hash="h1", now=now)
        live_store.claim_next(now=now)
        live_store.record_answer(id_, {"id": id_, "answer": "yes"}, now=now + timedelta(seconds=secs))
    assert live_store.avg_cycle_s() == 20.0  # median of [10, 20, 90]


def test_current_thinking_and_latest_answered(live_store):
    assert live_store.current_thinking() is None
    assert live_store.latest_answered() is None
    id1 = live_store.enqueue("Q1?", ip_hash="h1")
    live_store.claim_next()
    thinking = live_store.current_thinking()
    assert thinking["id"] == id1
    live_store.record_answer(id1, {"id": id1, "answer": "yes"})
    assert live_store.current_thinking() is None
    assert live_store.latest_answered()["id"] == id1


def test_thinking_since_and_answered_since_do_not_include_mark_failed_as_answered(live_store):
    id1 = live_store.enqueue("Q1?", ip_hash="h1")
    live_store.claim_next()
    live_store.mark_failed(id1, "boom")
    assert [r["id"] for r in live_store.thinking_since(0)] == [id1]
    assert live_store.answered_since(0) == []


# -- chat: writes/reads -----------------------------------------------------------------------


def test_chat_send_and_get(live_store):
    id1 = live_store.chat_send("fly-1234", "hello everyone", ip_hash="h1")
    row = live_store.chat_get(id1)
    assert row["nickname"] == "fly-1234"
    assert row["text"] == "hello everyone"
    assert row["deleted"] == 0


def test_chat_recent_is_oldest_first_and_excludes_deleted(live_store):
    id1 = live_store.chat_send("a", "first", ip_hash="h1")
    live_store.chat_send("b", "second", ip_hash="h2")
    id3 = live_store.chat_send("c", "third", ip_hash="h3")
    live_store.chat_delete(id3)
    recent = live_store.chat_recent(limit=10)
    assert [r["text"] for r in recent] == ["first", "second"]
    assert id1 == recent[0]["id"]


def test_chat_recent_respects_limit_keeping_the_newest(live_store):
    for i in range(5):
        live_store.chat_send("a", f"msg{i}", ip_hash="h1")
    recent = live_store.chat_recent(limit=2)
    assert [r["text"] for r in recent] == ["msg3", "msg4"]


def test_chat_recent_include_deleted_shows_everything(live_store):
    id1 = live_store.chat_send("a", "kept", ip_hash="h1")
    live_store.chat_delete(id1)
    assert live_store.chat_recent(limit=10) == []
    all_rows = live_store.chat_recent(limit=10, include_deleted=True)
    assert [r["text"] for r in all_rows] == ["kept"]


# -- chat: delete / restore / clear ------------------------------------------------------------


def test_chat_delete_marks_deleted_with_admin_reason(live_store):
    id1 = live_store.chat_send("a", "bye", ip_hash="h1")
    assert live_store.chat_delete(id1) is True
    row = live_store.chat_get(id1)
    assert row["deleted"] == 1
    assert row["deleted_reason"] == "admin"
    assert row["deleted_at"] is not None


def test_chat_delete_is_idempotent(live_store):
    id1 = live_store.chat_send("a", "bye", ip_hash="h1")
    live_store.chat_delete(id1)
    assert live_store.chat_delete(id1) is False


def test_chat_delete_unknown_id_returns_false(live_store):
    assert live_store.chat_delete(999999) is False


def test_chat_restore_undoes_a_delete(live_store):
    id1 = live_store.chat_send("a", "back again", ip_hash="h1")
    live_store.chat_delete(id1)
    assert live_store.chat_restore(id1) is True
    row = live_store.chat_get(id1)
    assert row["deleted"] == 0
    assert row["deleted_at"] is None
    assert row["deleted_reason"] is None
    assert row["restored_at"] is not None


def test_chat_restore_clears_report_tally(live_store):
    id1 = live_store.chat_send("a", "reported then restored", ip_hash="h1")
    live_store.chat_report(id1, "r1")
    live_store.chat_report(id1, "r2")
    live_store.chat_auto_hide(id1)
    live_store.chat_restore(id1)
    # Same two reporters reporting again after restore should not instantly re-trigger with a
    # stale tally -- the count must start over from these fresh reports.
    assert live_store.chat_report(id1, "r1") == 1


def test_chat_restore_on_a_non_deleted_message_returns_false(live_store):
    id1 = live_store.chat_send("a", "never deleted", ip_hash="h1")
    assert live_store.chat_restore(id1) is False


def test_chat_clear_soft_deletes_only_active_messages(live_store):
    id1 = live_store.chat_send("a", "one", ip_hash="h1")
    id2 = live_store.chat_send("b", "two", ip_hash="h2")
    live_store.chat_delete(id1)
    n = live_store.chat_clear()
    assert n == 1  # id1 was already deleted, only id2 gets touched
    assert live_store.chat_get(id2)["deleted"] == 1


# -- chat: bans ---------------------------------------------------------------------------------


def test_chat_ban_and_is_banned(live_store):
    assert live_store.chat_is_banned("h1") is False
    live_store.chat_ban("h1")
    assert live_store.chat_is_banned("h1") is True


def test_chat_ban_is_idempotent(live_store):
    live_store.chat_ban("h1")
    live_store.chat_ban("h1")  # must not raise
    assert live_store.chat_is_banned("h1") is True


# -- chat: reports + auto-hide -------------------------------------------------------------------


def test_chat_report_counts_distinct_reporters(live_store):
    id1 = live_store.chat_send("a", "reportable", ip_hash="h1")
    assert live_store.chat_report(id1, "r1") == 1
    assert live_store.chat_report(id1, "r2") == 2


def test_chat_report_from_the_same_client_does_not_double_count(live_store):
    id1 = live_store.chat_send("a", "reportable", ip_hash="h1")
    live_store.chat_report(id1, "r1")
    assert live_store.chat_report(id1, "r1") == 1


def test_chat_auto_hide_tags_reported_reason(live_store):
    id1 = live_store.chat_send("a", "reportable", ip_hash="h1")
    assert live_store.chat_auto_hide(id1) is True
    row = live_store.chat_get(id1)
    assert row["deleted"] == 1
    assert row["deleted_reason"] == "reported"


def test_chat_reported_lists_only_auto_hidden_messages(live_store):
    id1 = live_store.chat_send("a", "auto-hidden", ip_hash="h1")
    id2 = live_store.chat_send("b", "admin-deleted", ip_hash="h2")
    live_store.chat_auto_hide(id1)
    live_store.chat_delete(id2)
    reported = live_store.chat_reported()
    assert [r["id"] for r in reported] == [id1]


# -- chat: rate limiting + duplicate lookup ------------------------------------------------------


def test_chat_count_recent_only_within_window(live_store):
    now = datetime.now(UTC)
    live_store.chat_send("a", "old", ip_hash="h1", now=now - timedelta(seconds=30))
    live_store.chat_send("a", "new", ip_hash="h1", now=now)
    assert live_store.chat_count_recent("h1", window_seconds=10, now=now) == 1
    assert live_store.chat_count_recent("h1", window_seconds=60, now=now) == 2


def test_chat_last_text_returns_most_recent_message(live_store):
    live_store.chat_send("a", "first", ip_hash="h1")
    live_store.chat_send("a", "second", ip_hash="h1")
    assert live_store.chat_last_text("h1") == "second"


def test_chat_last_text_is_none_when_no_history(live_store):
    assert live_store.chat_last_text("h1") is None


# -- chat: event-tracker feeder queries -----------------------------------------------------------


def test_chat_messages_after_returns_rows_in_order(live_store):
    id1 = live_store.chat_send("a", "one", ip_hash="h1")
    id2 = live_store.chat_send("b", "two", ip_hash="h2")
    rows = live_store.chat_messages_after(0)
    assert [r["id"] for r in rows] == [id1, id2]


def test_chat_deleted_after_only_returns_rows_deleted_since(live_store):
    id1 = live_store.chat_send("a", "one", ip_hash="h1")
    live_store.chat_delete(id1)
    high_water = live_store.chat_get(id1)["deleted_at"]
    assert live_store.chat_deleted_after(high_water) == []
    assert len(live_store.chat_deleted_after("")) == 1


def test_chat_restored_after_only_returns_rows_restored_since(live_store):
    id1 = live_store.chat_send("a", "one", ip_hash="h1")
    live_store.chat_delete(id1)
    live_store.chat_restore(id1)
    assert len(live_store.chat_restored_after("")) == 1


# -- chat: runtime state (slow-mode + soft kill switch) --------------------------------------------


def test_chat_state_has_sane_defaults(live_store):
    state = live_store.chat_state()
    assert state["enabled"] is True
    assert state["slowmode_s"] == pytest.approx(4.0)


def test_chat_set_enabled_toggles_state(live_store):
    live_store.chat_set_enabled(False)
    assert live_store.chat_state()["enabled"] is False
    live_store.chat_set_enabled(True)
    assert live_store.chat_state()["enabled"] is True


def test_chat_set_slowmode_updates_interval(live_store):
    live_store.chat_set_slowmode(8.0)
    assert live_store.chat_state()["slowmode_s"] == pytest.approx(8.0)


# -- chat: atomic attempt reservation (F1/F2, 2026-09-18 security review) -----------------------


def test_chat_reserve_attempt_allows_the_first_attempt(live_store):
    assert live_store.chat_reserve_attempt("h1", slowmode_s=4.0) is True


def test_chat_reserve_attempt_rejects_a_second_attempt_within_slowmode(live_store):
    now = datetime.now(UTC)
    assert live_store.chat_reserve_attempt("h1", slowmode_s=4.0, now=now) is True
    assert live_store.chat_reserve_attempt("h1", slowmode_s=4.0, now=now) is False


def test_chat_reserve_attempt_allows_again_once_slowmode_elapses(live_store):
    now = datetime.now(UTC)
    assert live_store.chat_reserve_attempt("h1", slowmode_s=4.0, now=now) is True
    later = now + timedelta(seconds=5)
    assert live_store.chat_reserve_attempt("h1", slowmode_s=4.0, now=later) is True


def test_chat_reserve_attempt_is_per_client(live_store):
    now = datetime.now(UTC)
    assert live_store.chat_reserve_attempt("h1", slowmode_s=4.0, now=now) is True
    assert live_store.chat_reserve_attempt("h2", slowmode_s=4.0, now=now) is True


def test_chat_reserve_attempt_enforces_the_window_cap_even_outside_slowmode(live_store, monkeypatch):
    """F2: every attempt counts, so config.CHAT_RATE_PER_WINDOW attempts spread out (one every
    5s, well past a 4s slowmode) still trips the 10-minute cap on the next one."""
    from bioreservoir.live import config as live_config

    monkeypatch.setattr(live_config, "CHAT_RATE_PER_WINDOW", 3)
    base = datetime.now(UTC)
    for i in range(3):
        assert live_store.chat_reserve_attempt("h1", slowmode_s=1.0, now=base + timedelta(seconds=i * 5)) is True
    assert live_store.chat_reserve_attempt("h1", slowmode_s=1.0, now=base + timedelta(seconds=20)) is False


def test_chat_reserve_attempt_is_atomic_under_real_concurrency(tmp_path):
    """F1's core claim: a concurrent burst against the SAME client must not all observe an
    under-threshold count before any of them commits. Real OS threads, each with its OWN
    LiveStore/sqlite3 connection against the SAME file (the strongest reproduction -- this would
    also hold across multiple processes/workers, not just multiple asyncio coroutines on one
    event loop)."""
    import threading

    db_path = tmp_path / "concurrent.sqlite"
    results: list[bool] = []
    errors: list[BaseException] = []
    results_lock = threading.Lock()
    n_threads = 40
    # A timeout, and daemon threads below: one thread failing before the barrier used to leave
    # the other 39 waiting forever, and non-daemon threads kept pytest from exiting at all -- a
    # silent hang of the whole suite on a busy host, not a test failure (2026-09-19 review).
    barrier = threading.Barrier(n_threads, timeout=30)

    def worker():
        try:
            s = LiveStore(path=db_path)
            try:
                barrier.wait()  # maximize actual overlap
                ok = s.chat_reserve_attempt("shared-client", slowmode_s=4.0)
            finally:
                s.close()
            with results_lock:
                results.append(ok)
        except BaseException as exc:  # noqa: BLE001 -- surfaced by the assertion below
            with results_lock:
                errors.append(exc)

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert not errors, errors[:3]
    assert len(results) == n_threads
    # Slow-mode allows exactly one success in this window -- the pre-fix check-then-act race
    # reproduced 40/40 successes here.
    assert sum(results) == 1


# -- chat: site-wide LLM circuit breaker (F2) ----------------------------------------------------


def test_chat_reserve_llm_call_allows_up_to_the_cap(live_store, monkeypatch):
    from bioreservoir.live import config as live_config

    monkeypatch.setattr(live_config, "CHAT_LLM_MAX_PER_MINUTE", 3)
    now = datetime.now(UTC)
    for _ in range(3):
        assert live_store.chat_reserve_llm_call(now=now) is True
    assert live_store.chat_reserve_llm_call(now=now) is False


def test_chat_reserve_llm_call_is_site_wide_not_per_client(live_store, monkeypatch):
    """Unlike chat_reserve_attempt (per ip_hash), this breaker has no client argument at all --
    it protects the shared LLM budget from ANY combination of clients."""
    from bioreservoir.live import config as live_config

    monkeypatch.setattr(live_config, "CHAT_LLM_MAX_PER_MINUTE", 1)
    now = datetime.now(UTC)
    assert live_store.chat_reserve_llm_call(now=now) is True
    assert live_store.chat_reserve_llm_call(now=now) is False


def test_chat_reserve_llm_call_resets_after_the_window(live_store, monkeypatch):
    from bioreservoir.live import config as live_config

    monkeypatch.setattr(live_config, "CHAT_LLM_MAX_PER_MINUTE", 1)
    now = datetime.now(UTC)
    assert live_store.chat_reserve_llm_call(now=now) is True
    later = now + timedelta(seconds=61)
    assert live_store.chat_reserve_llm_call(now=later) is True


# -- chat: report eligibility helpers (F3) --------------------------------------------------------


def test_chat_has_accepted_message_false_before_any_post(live_store):
    assert live_store.chat_has_accepted_message("h1") is False


def test_chat_has_accepted_message_true_after_a_post(live_store):
    live_store.chat_send("a", "hello", ip_hash="h1")
    assert live_store.chat_has_accepted_message("h1") is True


def test_chat_has_accepted_message_true_even_if_later_deleted(live_store):
    id1 = live_store.chat_send("a", "hello", ip_hash="h1")
    live_store.chat_delete(id1)
    assert live_store.chat_has_accepted_message("h1") is True


def test_chat_report_count_reflects_distinct_reporters(live_store):
    id1 = live_store.chat_send("a", "reportable", ip_hash="h1")
    assert live_store.chat_report_count(id1) == 0
    live_store.chat_report(id1, "group-a")
    live_store.chat_report(id1, "group-b")
    assert live_store.chat_report_count(id1) == 2


# -- chat: retention purge (F4 identity purge + 7-day soft-deleted text purge) -------------------


def test_chat_purge_expired_blanks_old_soft_deleted_text(live_store):
    now = datetime.now(UTC)
    old = now - timedelta(days=10)
    id1 = live_store.chat_send("a", "sensitive text", ip_hash="h1", now=old)
    live_store.chat_delete(id1, now=old)
    result = live_store.chat_purge_expired(now=now)
    assert result["text_blanked"] == 1
    assert live_store.chat_get(id1)["text"] == ""


def test_chat_purge_expired_leaves_recently_deleted_text_alone(live_store):
    now = datetime.now(UTC)
    id1 = live_store.chat_send("a", "recent text", ip_hash="h1", now=now)
    live_store.chat_delete(id1, now=now)
    result = live_store.chat_purge_expired(now=now)
    assert result["text_blanked"] == 0
    assert live_store.chat_get(id1)["text"] == "recent text"


def test_chat_purge_expired_never_touches_active_message_text(live_store):
    now = datetime.now(UTC)
    old = now - timedelta(days=10)
    id1 = live_store.chat_send("a", "still active", ip_hash="h1", now=old)
    live_store.chat_purge_expired(now=now)
    assert live_store.chat_get(id1)["text"] == "still active"


def test_chat_purge_expired_blanks_old_author_identity(live_store):
    now = datetime.now(UTC)
    old = now - timedelta(days=40)
    id1 = live_store.chat_send("a", "old message", ip_hash="h1", now=old)
    result = live_store.chat_purge_expired(now=now)
    assert result["identity_blanked"] == 1
    assert live_store.chat_get(id1)["ip_hash"] == ""


def test_chat_purge_expired_keeps_identity_for_a_banned_author(live_store):
    """F4: "purge after 30 days unless banned" -- a banned author's linkage must survive so the
    ban stays enforceable/auditable."""
    now = datetime.now(UTC)
    old = now - timedelta(days=40)
    id1 = live_store.chat_send("a", "old message from a troll", ip_hash="h1", now=old)
    live_store.chat_ban("h1")
    result = live_store.chat_purge_expired(now=now)
    assert result["identity_blanked"] == 0
    assert live_store.chat_get(id1)["ip_hash"] == "h1"


def test_chat_purge_expired_leaves_recent_identity_alone(live_store):
    now = datetime.now(UTC)
    id1 = live_store.chat_send("a", "recent message", ip_hash="h1", now=now)
    result = live_store.chat_purge_expired(now=now)
    assert result["identity_blanked"] == 0
    assert live_store.chat_get(id1)["ip_hash"] == "h1"


# -- chat: attempts/llm_calls purge (round-2 security review, "never purged") --------------------


def test_chat_purge_expired_drops_old_attempts(live_store):
    now = datetime.now(UTC)
    old = now - timedelta(days=2)
    live_store.chat_reserve_attempt("h1", slowmode_s=0.0, now=old)
    result = live_store.chat_purge_expired(now=now)
    assert result["attempts_purged"] == 1


def test_chat_purge_expired_leaves_recent_attempts_alone(live_store):
    now = datetime.now(UTC)
    live_store.chat_reserve_attempt("h1", slowmode_s=0.0, now=now)
    result = live_store.chat_purge_expired(now=now)
    assert result["attempts_purged"] == 0


def test_chat_purge_expired_drops_old_llm_calls(live_store):
    now = datetime.now(UTC)
    old = now - timedelta(days=2)
    live_store.chat_reserve_llm_call(now=old)
    result = live_store.chat_purge_expired(now=now)
    assert result["llm_calls_purged"] == 1


def test_chat_purge_expired_attempts_retention_is_configurable(live_store):
    now = datetime.now(UTC)
    live_store.chat_reserve_attempt("h1", slowmode_s=0.0, now=now - timedelta(hours=2))
    result = live_store.chat_purge_expired(attempts_retention_days=0.05, now=now)  # ~72 min
    assert result["attempts_purged"] == 1


# -- chat: chat_reserve_attempt generalized for namespaced reuse (R1/R2) -------------------------


def test_chat_reserve_attempt_accepts_a_custom_window_and_cap(live_store):
    now = datetime.now(UTC)
    for i in range(3):
        assert live_store.chat_reserve_attempt(
            "report:aid1", slowmode_s=0.0, window_s=600.0, max_per_window=3, now=now + timedelta(seconds=i)
        ) is True
    assert live_store.chat_reserve_attempt(
        "report:aid1", slowmode_s=0.0, window_s=600.0, max_per_window=3, now=now + timedelta(seconds=10)
    ) is False


def test_chat_reserve_attempt_namespaced_keys_do_not_collide_with_plain_ones(live_store):
    """A "report:{author_id}" reservation must not share a rate-limit bucket with that same
    author_id's ordinary POST /api/chat attempts -- they are DIFFERENT actions with different
    caps (R1/R2's whole point in reusing this table via a namespaced key)."""
    now = datetime.now(UTC)
    assert live_store.chat_reserve_attempt("aid1", slowmode_s=4.0, now=now) is True
    # A namespaced report-quota reservation for the SAME author, same instant, must still succeed.
    assert live_store.chat_reserve_attempt("report:aid1", slowmode_s=0.0, now=now) is True


def test_chat_reserve_attempt_default_window_and_cap_match_post_message_thresholds(live_store, monkeypatch):
    """Calling with no window_s/max_per_window override must behave exactly as before (the plain
    POST /api/chat rate limiter), so this generalization is backward compatible."""
    from bioreservoir.live import config as live_config

    monkeypatch.setattr(live_config, "CHAT_RATE_PER_WINDOW", 2)
    now = datetime.now(UTC)
    assert live_store.chat_reserve_attempt("h1", slowmode_s=0.0, now=now) is True
    assert live_store.chat_reserve_attempt("h1", slowmode_s=0.0, now=now + timedelta(seconds=1)) is True
    assert live_store.chat_reserve_attempt("h1", slowmode_s=0.0, now=now + timedelta(seconds=2)) is False


# -- chat: site-wide auto-hide cap (R1) -----------------------------------------------------------


def test_chat_auto_hide_count_recent_counts_reported_hides_only(live_store):
    now = datetime.now(UTC)
    id1 = live_store.chat_send("a", "one", ip_hash="h1", now=now)
    id2 = live_store.chat_send("b", "two", ip_hash="h2", now=now)
    live_store.chat_auto_hide(id1, now=now)
    live_store.chat_delete(id2, now=now)  # admin delete, not a report -- must not count
    assert live_store.chat_auto_hide_count_recent(600.0, now=now) == 1


def test_chat_auto_hide_count_recent_ignores_old_hides(live_store):
    now = datetime.now(UTC)
    old = now - timedelta(seconds=700)
    id1 = live_store.chat_send("a", "one", ip_hash="h1", now=old)
    live_store.chat_auto_hide(id1, now=old)
    assert live_store.chat_auto_hide_count_recent(600.0, now=now) == 0


# -- game: guesses ("which one is the real fly?") -------------------------------------------------


def test_record_guess_and_get_guess_round_trip(live_store):
    assert live_store.get_guess(1, "ip-a") is None
    inserted = live_store.record_guess(1, "ip-a", "B", correct=True, counted=True)
    assert inserted is True
    row = live_store.get_guess(1, "ip-a")
    assert row["pick"] == "B"
    assert row["correct"] == 1
    assert row["counted"] == 1


def test_record_guess_one_per_answer_per_client(live_store):
    live_store.record_guess(1, "ip-a", "B", correct=True, counted=True)
    # A second guess attempt for the SAME (answer, client) is a silent no-op -- the stored row
    # (pick="B", correct=True) is untouched, even though this call claims a different pick/result.
    inserted_again = live_store.record_guess(1, "ip-a", "A", correct=False, counted=True)
    assert inserted_again is False
    row = live_store.get_guess(1, "ip-a")
    assert row["pick"] == "B"
    assert row["correct"] == 1


def test_record_guess_is_independent_per_client(live_store):
    live_store.record_guess(1, "ip-a", "A", correct=True, counted=True)
    live_store.record_guess(1, "ip-b", "C", correct=False, counted=False)
    assert live_store.get_guess(1, "ip-a")["pick"] == "A"
    assert live_store.get_guess(1, "ip-b")["pick"] == "C"


def test_record_guess_is_independent_per_answer(live_store):
    live_store.record_guess(1, "ip-a", "A", correct=True, counted=True)
    live_store.record_guess(2, "ip-a", "C", correct=False, counted=True)
    assert live_store.get_guess(1, "ip-a")["pick"] == "A"
    assert live_store.get_guess(2, "ip-a")["pick"] == "C"


def test_guess_stats_empty_store_is_all_zeros(live_store):
    assert live_store.guess_stats() == {"total": 0, "correct": 0}


def test_guess_stats_counts_total_and_correct(live_store):
    live_store.record_guess(1, "ip-a", "A", correct=True, counted=True)
    live_store.record_guess(1, "ip-b", "B", correct=False, counted=True)
    live_store.record_guess(2, "ip-a", "C", correct=True, counted=True)
    assert live_store.guess_stats() == {"total": 3, "correct": 2}


def test_answered_questions_before_returns_only_earlier_answered_rows(live_store):
    """R2(a) (round-2 logic review): `api.py`'s first-answer-per-question-key check reads this."""
    id1 = live_store.enqueue("Will it rain?", ip_hash="h1")
    live_store.claim_next()
    live_store.record_answer(id1, {"id": id1, "answer": "yes"})

    live_store.enqueue("Is the sky blue?", ip_hash="h2")  # queued but never answered
    id3 = live_store.enqueue("Will it rain again?", ip_hash="h3")
    live_store.claim_next()  # claims id2 (FIFO) -> "thinking", never answered
    live_store.claim_next()  # claims id3 -> "thinking"
    live_store.record_answer(id3, {"id": id3, "answer": "no"})

    before_id3 = live_store.answered_questions_before(id3)
    assert [(r["id"], r["question"]) for r in before_id3] == [(id1, "Will it rain?")]
    # id2 (never answered) is excluded even though its id is smaller than id3's.
    assert live_store.answered_questions_before(id1) == []  # nothing answered before the first one


def test_guess_stats_excludes_uncounted_guesses(live_store):
    """F2: a guess that isn't the question's own asker (or arrives outside the count window) is
    still stored (for idempotency) but must never move the published site-wide percentage."""
    live_store.record_guess(1, "ip-asker", "A", correct=True, counted=True)
    live_store.record_guess(1, "ip-stranger", "B", correct=True, counted=False)
    live_store.record_guess(2, "ip-stranger2", "C", correct=False, counted=False)
    assert live_store.guess_stats() == {"total": 1, "correct": 1}
    # Both rows are still there (idempotency, auditability) -- just excluded from the stat.
    assert live_store.get_guess(1, "ip-stranger") is not None
    assert live_store.get_guess(2, "ip-stranger2") is not None
