"""api.py: FastAPI route contract (task brief §1) via `TestClient` — one throwaway `LIVE_DB` per
test (env var set before the app's `lifespan` opens its `LiveStore`, torn down after).
"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("LIVE_DB", str(tmp_path / "live.sqlite"))
    monkeypatch.setenv("LIVE_ENV", "dev")
    # F8 (2026-09-18 security review): chat has NO default secret for any of these three -- set
    # them here so the existing/new chat tests exercise the normal (secrets-configured) path;
    # test_chat_post_503_when_a_required_secret_is_missing below unsets each individually.
    monkeypatch.setenv("CHAT_SESSION_SECRET", "test-session-secret")
    monkeypatch.setenv("CHAT_ID_PEPPER", "test-id-pepper")
    monkeypatch.setenv("CHAT_LLM_API_KEY", "test-chat-llm-key")
    # Force-reimport so `config.LIVE_DB`/`LIVE_ENV` (module-level constants, read once at import)
    # pick up this test's env vars — every other `bioreservoir.live.*` module that imported
    # `config` already holds a reference to the SAME module object, so reloading it in place is
    # enough (no need to reload every submodule too).
    import importlib

    import bioreservoir.live.config as live_config

    importlib.reload(live_config)

    import bioreservoir.live.api as live_api

    importlib.reload(live_api)

    with TestClient(live_api.app) as c:
        yield c


def test_healthz(client):
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_ask_accepts_a_clean_yes_no_question(client):
    resp = client.post("/api/ask", json={"question": "Will it rain tomorrow?", "turnstile_token": "x"})
    assert resp.status_code == 202
    body = resp.json()
    assert body["status"] == "queued"
    assert body["id"] > 0
    assert body["position"] == 1


def test_ask_rejects_a_non_yes_no_question(client):
    resp = client.post("/api/ask", json={"question": "What is your favorite color?", "turnstile_token": "x"})
    assert resp.status_code == 422
    body = resp.json()
    assert body["status"] == "rejected"
    assert body["reason"] == "not_yes_no"
    assert "message" in body
    assert "id" not in body  # rejected attempts never get a client-visible id


def test_ask_rejects_voting_procedure_questions(client):
    resp = client.post("/api/ask", json={"question": "Can I vote by SMS?", "turnstile_token": "x"})
    assert resp.status_code == 422
    assert resp.json()["reason"] == "voting_procedure"


def test_ask_rejects_too_long_questions(client):
    resp = client.post("/api/ask", json={"question": "Will it rain? " * 20, "turnstile_token": "x"})
    assert resp.status_code == 422
    assert resp.json()["reason"] == "too_long"


def test_ask_enforces_per_minute_rate_limit(client):
    for _ in range(3):
        r = client.post("/api/ask", json={"question": "Will it rain tomorrow?", "turnstile_token": "x"})
        assert r.status_code == 202
    r = client.post("/api/ask", json={"question": "Is the sky blue today?", "turnstile_token": "x"})
    assert r.status_code == 422
    assert r.json()["reason"] == "rate_limited"


def test_get_answers_queued_includes_position(client):
    r = client.post("/api/ask", json={"question": "Will it rain tomorrow?", "turnstile_token": "x"})
    id_ = r.json()["id"]
    r2 = client.get(f"/api/answers/{id_}")
    assert r2.status_code == 200
    body = r2.json()
    assert body["status"] == "queued"
    assert body["position"] == 1
    assert body["question"] == "Will it rain tomorrow?"


def test_a_waiting_question_already_knows_which_side_means_yes(client):
    """The live stage puts both answer labels on their real sides while the fly is still
    deciding (operator, 2026-09-19), so the side has to travel with the QUESTION -- on the
    queued/thinking lookup and on `/api/now` -- not with the answer. That it is the same coin
    the finished answer records is `test_live_pipeline.py`'s
    `test_yes_side_for_is_the_same_coin_the_answer_records`."""
    question = "Will it rain tomorrow?"
    r = client.post("/api/ask", json={"question": question, "turnstile_token": "x"})
    id_ = r.json()["id"]
    queued = client.get(f"/api/answers/{id_}").json()
    assert queued["status"] == "queued"
    assert queued["yes_side"] in ("left", "right")
    assert client.get(f"/api/answers/{id_}").json()["yes_side"] == queued["yes_side"]  # stable

    # Once the worker claims it, the same side comes with the "what is happening now" snapshot
    # and does not change mid-run. The claim goes through a second LiveStore on the same file,
    # exactly like the real worker process (the app's own connection belongs to its thread).
    from bioreservoir.live import config as live_config
    from bioreservoir.live import store as live_store

    assert live_store.LiveStore(path=live_config.LIVE_DB).claim_next() is not None
    thinking = client.get("/api/now").json()["thinking"]
    assert thinking["id"] == int(id_)
    assert thinking["yes_side"] == queued["yes_side"]
    assert client.get(f"/api/answers/{id_}").json() == {
        **{k: v for k, v in thinking.items() if k != "embargoed"},
        "status": "thinking",
        "yes_side": queued["yes_side"],
    }


def _answer_one(question: str, client) -> int:
    """Ask `question` and mark it answered through a second store, like the worker does."""
    from bioreservoir.live import config as live_config
    from bioreservoir.live import store as live_store

    id_ = client.post("/api/ask", json={"question": question, "turnstile_token": "x"}).json()["id"]
    with live_store.LiveStore(path=live_config.LIVE_DB) as s:
        s.claim_next()
        s.record_answer(int(id_), {"id": int(id_), "question": question, "answer": "yes"})
    return int(id_)


def test_liking_a_question_toggles_and_never_double_counts(client, monkeypatch):
    """Operator, 2026-09-19: visitors like questions and the most-liked ones get their own tab.
    The same client tapping twice must take the like back, not count two."""
    from bioreservoir.live import config as live_config

    monkeypatch.setattr(live_config, "LIKE_SLOWMODE_S", 0.0)
    id_ = _answer_one("Will it rain tomorrow?", client)

    assert client.post(f"/api/answers/{id_}/like").json() == {"id": id_, "likes": 1, "liked": True}
    assert client.post(f"/api/answers/{id_}/like").json() == {"id": id_, "likes": 0, "liked": False}
    assert client.post(f"/api/answers/{id_}/like").json()["likes"] == 1
    # The count travels with the question everywhere it is listed.
    assert client.get("/api/feed").json()["recent"][0]["likes"] == 1
    assert client.get("/api/questions").json()["items"][0]["likes"] == 1
    assert client.post("/api/answers/999999/like").status_code == 404


def test_questions_list_pages_and_sorts_by_likes(client, monkeypatch):
    from bioreservoir.live import config as live_config

    monkeypatch.setattr(live_config, "LIKE_SLOWMODE_S", 0.0)
    monkeypatch.setattr(live_config, "RATE_LIMIT_PER_MINUTE", 100)
    first = _answer_one("Will it rain tomorrow?", client)
    second = _answer_one("Will the sun rise on Tuesday?", client)
    third = _answer_one("Will coffee stay expensive?", client)
    client.post(f"/api/answers/{second}/like")

    recent = client.get("/api/questions?limit=2").json()
    assert [i["id"] for i in recent["items"]] == [third, second]
    assert recent["total"] == 3 and recent["has_more"] is True
    page2 = client.get("/api/questions?limit=2&offset=2").json()
    assert [i["id"] for i in page2["items"]] == [first] and page2["has_more"] is False

    top = client.get("/api/questions?sort=top").json()
    assert [i["id"] for i in top["items"]] == [second, third, first]
    assert client.get("/api/questions?sort=sideways").status_code == 422


def test_an_election_question_is_answered_in_public_like_any_other(client):
    """The embargo is off (2026-09-21): a visitor's election question gets the same treatment as
    "will it rain" -- verdict on the page, in the archive, to everyone."""
    id_ = _answer_one("Will the Senate flip in the midterms?", client)
    body = client.get(f"/api/answers/{id_}", headers={"cf-connecting-ip": "203.0.113.9"}).json()
    assert body["embargoed"] is False
    assert body["answer"]["answer"] in ("yes", "no")
    listed = [i for i in client.get("/api/questions").json()["items"] if i["id"] == id_]
    assert listed and listed[0]["answer"] in ("yes", "no")


def test_an_embargoed_verdict_goes_to_the_asker_only(client, monkeypatch):
    """Ids are public (the feed, the questions archive), so holding one must not be proof of being
    the asker: before this, anyone could read a sealed election verdict straight off
    `/api/answers/{id}` while every page carefully hid it (2026-09-19 external review).

    The embargo ships OFF; this keeps the mechanism covered for whenever it is switched back on."""
    from bioreservoir.live import config as live_config

    monkeypatch.setattr(live_config, "ELECTION_CARD_EMBARGO_ENABLED", True)
    question = "Will the Senate flip in the midterms?"
    id_ = _answer_one(question, client)

    asker = client.get(f"/api/answers/{id_}").json()
    assert asker["embargoed"] is True
    assert asker["answer"]["answer"] == "yes"  # same IP as the ask: this is the asker

    other = client.get(f"/api/answers/{id_}", headers={"cf-connecting-ip": "203.0.113.9"})
    assert other.json() == {"id": id_, "question": question, "status": "answered", "embargoed": True}
    assert other.headers["cache-control"] == "no-store"  # never cache a per-caller answer
    # The public list still shows the question, just not the verdict.
    listed = [i for i in client.get("/api/questions").json()["items"] if i["id"] == id_]
    assert listed and listed[0]["answer"] is None and listed[0]["embargoed"] is True


def test_get_answers_404_for_unknown_id(client):
    r = client.get("/api/answers/999999")
    assert r.status_code == 404


def test_get_answers_answered_includes_full_answer(client):
    r = client.post("/api/ask", json={"question": "Will it rain tomorrow?", "turnstile_token": "x"})
    id_ = r.json()["id"]

    from bioreservoir.live import config
    from bioreservoir.live.store import LiveStore

    with LiveStore(path=config.LIVE_DB) as s:
        s.claim_next()
        s.record_answer(id_, {"id": id_, "question": "Will it rain tomorrow?", "answer": "yes", "confidence": 0.8})

    r2 = client.get(f"/api/answers/{id_}")
    assert r2.status_code == 200
    body = r2.json()
    assert body["status"] == "answered"
    assert body["answer"]["answer"] == "yes"


def test_feed_reports_queue_length_and_recent(client):
    r = client.post("/api/ask", json={"question": "Will it rain tomorrow?", "turnstile_token": "x"})
    id_ = r.json()["id"]

    from bioreservoir.live import config
    from bioreservoir.live.store import LiveStore

    with LiveStore(path=config.LIVE_DB) as s:
        s.claim_next()
        s.record_answer(id_, {"id": id_, "answer": "yes"})

    r2 = client.get("/api/feed")
    assert r2.status_code == 200
    body = r2.json()
    assert body["queue_length"] == 0
    # The feed carries SUMMARIES now (no spike data): fanning a ~150 KB answer out to every open
    # tab is what this contract change removed (see live/summary.py).
    assert body["recent"] == [
        {
            "id": id_,
            "question": "",
            "answer": "yes",
            "embargoed": False,
            "yes_side": None,
            "lateral_bias": None,
            "turn_strength": 0.0,
            "answered_at": None,
            "likes": 0,
        }
    ]


def test_card_404_without_an_answer(client):
    r = client.get("/api/card/999999.png")
    assert r.status_code == 404


def test_card_returns_a_png_once_answered(client):
    r = client.post("/api/ask", json={"question": "Will it rain tomorrow?", "turnstile_token": "x"})
    id_ = r.json()["id"]

    from bioreservoir.live import config
    from bioreservoir.live.store import LiveStore

    with LiveStore(path=config.LIVE_DB) as s:
        s.claim_next()
        s.record_answer(
            id_,
            {
                "id": id_, "question": "Will it rain tomorrow?", "answer": "yes", "confidence": 0.8,
                "lateral_bias": 0.1, "states": {"appetite": None, "fear": None, "backoff": None, "courtship": None, "arousal": 0.3},
                "frames": None, "brain": "malecns", "sim_ms": 250, "n_trials": 3, "answered_at": "2026-09-18T00:00:00Z",
            },
        )

    r2 = client.get(f"/api/card/{id_}.png")
    assert r2.status_code == 200
    assert r2.headers["content-type"] == "image/png"
    assert r2.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_share_page_has_opengraph_tags(client):
    r = client.post("/api/ask", json={"question": "Will it rain tomorrow?", "turnstile_token": "x"})
    id_ = r.json()["id"]

    from bioreservoir.live import config
    from bioreservoir.live.store import LiveStore

    with LiveStore(path=config.LIVE_DB) as s:
        s.claim_next()
        s.record_answer(
            id_,
            {
                "id": id_, "question": "Will it rain tomorrow?", "answer": "yes", "confidence": 0.8,
                "lateral_bias": 0.1, "states": {"appetite": None, "fear": None, "backoff": None, "courtship": None, "arousal": 0.3},
                "frames": None, "brain": "malecns", "sim_ms": 250, "n_trials": 3, "answered_at": "2026-09-18T00:00:00Z",
            },
        )

    r2 = client.get(f"/a/{id_}")
    assert r2.status_code == 200
    assert "og:image" in r2.text
    assert f"/api/card/{id_}.png" in r2.text
    assert "twitter:card" in r2.text


def test_stats_empty_store_is_all_zeros(client):
    r = client.get("/api/stats")
    assert r.status_code == 200
    body = r.json()
    assert body["answered"] == 0
    assert body["yes"] == 0
    assert body["total_spikes"] == 0
    assert body["state_counts"] == {"appetite": 0, "fear": 0, "backoff": 0, "courtship": 0, "arousal": 0}
    assert body["since"] is None
    assert body["guesses"] == {"total": 0, "correct": 0}


def test_stats_reflects_an_answered_question(client):
    r = client.post("/api/ask", json={"question": "Will it rain tomorrow?", "turnstile_token": "x"})
    id_ = r.json()["id"]

    from bioreservoir.live import config
    from bioreservoir.live.store import LiveStore

    with LiveStore(path=config.LIVE_DB) as s:
        s.claim_next()
        s.record_answer(
            id_,
            {
                "id": id_, "question": "Will it rain tomorrow?", "answer": "yes", "confidence": 0.8,
                "lateral_bias": 0.1, "states": {"appetite": None, "fear": None, "backoff": None, "courtship": None, "arousal": 0.3},
                "frames": None, "brain": "malecns", "sim_ms": 250, "n_trials": 3, "answered_at": "2026-09-18T00:00:00Z",
                "lab": {"total_spikes": 5000, "active_fraction": 0.06, "provenance": {"n_neurons": 165122, "dt_ms": 0.1}},
            },
        )

    r2 = client.get("/api/stats")
    body = r2.json()
    assert body["answered"] == 1
    assert body["yes"] == 1
    assert body["total_spikes"] == 5000
    assert body["state_counts"]["arousal"] == 1
    assert body["since"] is not None


# -- game: POST /api/guess -------------------------------------------------------------------------


def _record_game_answer(id_, order, answered_at=None):
    """`order` (task brief's A/B/C slot order) determines which of "A"/"B"/"C" is the real brain
    for this fixture answer. `answered_at`, if given, is a `datetime` -- passed through to
    `record_answer`'s own `now=` so it lands in the `live_questions.answered_at` SQL COLUMN
    `api.py`'s `_within_guess_count_window` actually reads (NOT the `answered_at` key inside the
    Answer JSON payload, which is a display-only field `record_answer` never touches). Defaults to
    "now" so a `counted` guess naturally falls inside `config.GUESS_COUNT_WINDOW_S` in tests that
    expect one."""
    from datetime import UTC, datetime

    from bioreservoir.live import config
    from bioreservoir.live.store import LiveStore

    contenders = {
        "real": {"answer": "yes", "corrected_bias": 0.2, "decisiveness": 0.7},
        "random_graph": {"answer": "no", "corrected_bias": -0.05, "decisiveness": 0.55},
        "no_brain": {"answer": "yes", "corrected_bias": 0.02, "decisiveness": 0.51},
    }
    now = answered_at or datetime.now(UTC)
    with LiveStore(path=config.LIVE_DB) as s:
        s.claim_next()
        s.record_answer(
            id_,
            {
                "id": id_, "question": "Will it rain tomorrow?", "answer": "yes", "confidence": 0.8,
                "lateral_bias": 0.2, "states": {"appetite": None, "fear": None, "backoff": None, "courtship": None, "arousal": 0.3},
                "frames": None, "brain": "malecns", "sim_ms": 250, "n_trials": 3,
                "answered_at": now.isoformat(),
                "game": {"order": order, "contenders": contenders},
            },
            now=now,
        )


def test_guess_correct_pick_returns_correct_true_and_the_real_slot(client):
    r = client.post("/api/ask", json={"question": "Will it rain tomorrow?", "turnstile_token": "x"})
    id_ = r.json()["id"]
    _record_game_answer(id_, order=["random_graph", "real", "no_brain"])  # real is slot B

    r2 = client.post("/api/guess", json={"id": id_, "pick": "B"})
    assert r2.status_code == 200
    assert r2.json() == {"correct": True, "real": "B", "counted": True, "pick": "B", "repeat": False}


def test_guess_incorrect_pick_returns_correct_false_and_the_real_slot(client):
    r = client.post("/api/ask", json={"question": "Will it rain tomorrow?", "turnstile_token": "x"})
    id_ = r.json()["id"]
    _record_game_answer(id_, order=["random_graph", "real", "no_brain"])  # real is slot B

    r2 = client.post("/api/guess", json={"id": id_, "pick": "A"})
    assert r2.status_code == 200
    assert r2.json() == {"correct": False, "real": "B", "counted": True, "pick": "A", "repeat": False}


def test_guess_unknown_id_404s(client):
    r = client.post("/api/guess", json={"id": 999999, "pick": "A"})
    assert r.status_code == 404


def test_guess_on_an_answer_with_no_game_404s(client):
    r = client.post("/api/ask", json={"question": "Will it rain tomorrow?", "turnstile_token": "x"})
    id_ = r.json()["id"]

    from bioreservoir.live import config
    from bioreservoir.live.store import LiveStore

    with LiveStore(path=config.LIVE_DB) as s:
        s.claim_next()
        s.record_answer(id_, {"id": id_, "question": "Will it rain tomorrow?", "answer": "yes", "confidence": 0.8})

    r2 = client.post("/api/guess", json={"id": id_, "pick": "A"})
    assert r2.status_code == 404


def test_guess_rejects_an_invalid_pick(client):
    r = client.post("/api/ask", json={"question": "Will it rain tomorrow?", "turnstile_token": "x"})
    id_ = r.json()["id"]
    _record_game_answer(id_, order=["real", "random_graph", "no_brain"])

    r2 = client.post("/api/guess", json={"id": id_, "pick": "D"})
    assert r2.status_code == 422


def test_guess_is_idempotent_per_client_returning_the_stored_pick_not_the_new_one(client):
    """F6: a repeat guess must not re-score against the new pick -- the response's own `pick`
    field must be the STORED one, so a frontend rendering `res.pick` never mismarks which slot
    the visitor originally chose."""
    r = client.post("/api/ask", json={"question": "Will it rain tomorrow?", "turnstile_token": "x"})
    id_ = r.json()["id"]
    _record_game_answer(id_, order=["random_graph", "real", "no_brain"])  # real is slot B

    first = client.post("/api/guess", json={"id": id_, "pick": "B"})
    assert first.json() == {"correct": True, "real": "B", "counted": True, "pick": "B", "repeat": False}

    # Same client (TestClient reuses one connection/IP), a different pick this time -- must
    # return the SAME result the first guess got, not re-score against the new pick, and flag
    # `repeat: True` so the frontend knows not to double-count its local tally either.
    second = client.post("/api/guess", json={"id": id_, "pick": "A"})
    assert second.json() == {"correct": True, "real": "B", "counted": True, "pick": "B", "repeat": True}


def test_guess_different_clients_score_independently(client):
    r = client.post("/api/ask", json={"question": "Will it rain tomorrow?", "turnstile_token": "x"})
    id_ = r.json()["id"]
    _record_game_answer(id_, order=["random_graph", "real", "no_brain"])  # real is slot B

    ok = client.post("/api/guess", json={"id": id_, "pick": "B"}, headers={"cf-connecting-ip": "1.2.3.4"})
    wrong = client.post("/api/guess", json={"id": id_, "pick": "A"}, headers={"cf-connecting-ip": "5.6.7.8"})
    assert ok.json()["correct"] is True
    assert wrong.json()["correct"] is False
    # Neither of these IPs asked the question (the asker used no CF header at all) -- neither
    # guess counts toward the site-wide stat (F2), even though both are legitimately scored.
    assert ok.json()["counted"] is False
    assert wrong.json()["counted"] is False


def test_guess_updates_site_wide_stats_only_for_the_asker(client):
    """F2: only the asker's own guess on their own question counts toward `GET /api/stats`'s
    `guesses` -- a stranger's guess (share link, browsing the feed) is scored and stored, but
    never moves the published percentage."""
    r = client.post("/api/ask", json={"question": "Will it rain tomorrow?", "turnstile_token": "x"})
    id_ = r.json()["id"]
    _record_game_answer(id_, order=["random_graph", "real", "no_brain"])  # real is slot B

    # The asker (SAME client, no CF header, matching /api/ask above) guesses correctly.
    asker = client.post("/api/guess", json={"id": id_, "pick": "B"})
    assert asker.json()["counted"] is True
    # A stranger (different CF-Connecting-IP) also guesses, on the SAME question.
    stranger = client.post("/api/guess", json={"id": id_, "pick": "A"}, headers={"cf-connecting-ip": "9.9.9.9"})
    assert stranger.json()["counted"] is False

    stats_body = client.get("/api/stats").json()
    assert stats_body["guesses"] == {"total": 1, "correct": 1}  # only the asker's guess counted


def test_guess_outside_the_count_window_is_not_counted_even_for_the_asker(client):
    from datetime import UTC, datetime, timedelta

    from bioreservoir.live import config as live_config

    r = client.post("/api/ask", json={"question": "Will it rain tomorrow?", "turnstile_token": "x"})
    id_ = r.json()["id"]
    stale = datetime.now(UTC) - timedelta(seconds=live_config.GUESS_COUNT_WINDOW_S + 60)
    _record_game_answer(id_, order=["random_graph", "real", "no_brain"], answered_at=stale)

    resp = client.post("/api/guess", json={"id": id_, "pick": "B"})
    assert resp.json()["correct"] is True
    assert resp.json()["counted"] is False  # too long after the answer, even though it's the asker


def test_guess_rate_limited_per_client(client, monkeypatch):
    import bioreservoir.live.api as live_api

    # `client` (this file's fixture) reloads bioreservoir.live.config/api at setup, but reload
    # mutates the SAME module object in place -- `live_api.config` and a fresh import of
    # `bioreservoir.live.config` are one and the same object, so a single patch is enough.
    monkeypatch.setattr(live_api.config, "GUESS_SLOWMODE_S", 3600.0)  # 1/hour -- the 2nd call must trip it

    r = client.post("/api/ask", json={"question": "Will it rain tomorrow?", "turnstile_token": "x"})
    id_ = r.json()["id"]
    _record_game_answer(id_, order=["random_graph", "real", "no_brain"])

    first = client.post("/api/guess", json={"id": id_, "pick": "B"})
    assert first.status_code == 200
    r2 = client.post("/api/ask", json={"question": "Is the sky blue?", "turnstile_token": "x"})
    id_2 = r2.json()["id"]
    _record_game_answer(id_2, order=["random_graph", "real", "no_brain"])
    second = client.post("/api/guess", json={"id": id_2, "pick": "A"})
    # Second call within the slowmode window from the same client (a DIFFERENT question, so this
    # is genuinely the rate limit firing, not the per-id idempotency path) -- 429.
    assert second.status_code == 429


def test_guess_not_counted_if_this_answer_is_not_the_first_for_its_question_key(client):
    """R2(a) (round-2 logic review): the example chips (or any popular/repeated wording) let a
    visitor guess a question whose answer another asker already made public -- a guess only
    counts if THIS answer is the FIRST answered occurrence of its normalized question text
    (`pipeline.question_key`), regardless of who asked it or when."""
    first_ask = client.post("/api/ask", json={"question": "Will it rain tomorrow?", "turnstile_token": "x"})
    first_id = first_ask.json()["id"]
    _record_game_answer(first_id, order=["random_graph", "real", "no_brain"])  # real is slot B

    # A second asker asks the SAME wording (case/whitespace-different, still the same
    # question_key) -- their own answer is a real, separate row, but NOT the first for this key.
    second_ask = client.post(
        "/api/ask",
        json={"question": "  WILL it rain tomorrow?  ", "turnstile_token": "x"},
        headers={"cf-connecting-ip": "8.8.8.8"},
    )
    second_id = second_ask.json()["id"]
    _record_game_answer(second_id, order=["real", "random_graph", "no_brain"])  # real is slot A

    # The second asker guesses their OWN question, promptly -- would be counted by every other
    # rule (own question, within the window), but the first-answer rule still excludes it.
    resp = client.post("/api/guess", json={"id": second_id, "pick": "A"}, headers={"cf-connecting-ip": "8.8.8.8"})
    assert resp.json()["correct"] is True
    assert resp.json()["counted"] is False

    # The FIRST asker's own guess on the FIRST id is unaffected -- still counted.
    resp_first = client.post("/api/guess", json={"id": first_id, "pick": "B"})
    assert resp_first.json()["counted"] is True


def test_guess_skip_consumes_the_slot_so_a_later_real_guess_is_an_uncounted_repeat(client):
    """R2(b) (round-2 logic review): skip is recorded server-side as a NEVER-counted guess too --
    a later real guess attempt by the SAME client on the SAME id (e.g. via the share link, with
    the answer now known from having skipped) is treated as an idempotent repeat of the skip, not
    a fresh countable guess."""
    r = client.post("/api/ask", json={"question": "Will it rain tomorrow?", "turnstile_token": "x"})
    id_ = r.json()["id"]
    _record_game_answer(id_, order=["random_graph", "real", "no_brain"])  # real is slot B

    skip_resp = client.post("/api/guess", json={"id": id_, "pick": "skip"})
    assert skip_resp.status_code == 200
    assert skip_resp.json() == {"correct": False, "real": "B", "counted": False, "pick": "skip", "repeat": False}

    # A later "real" guess attempt from the SAME client is a repeat of the SKIP, not a fresh guess.
    later = client.post("/api/guess", json={"id": id_, "pick": "B"})
    assert later.json() == {"correct": False, "real": "B", "counted": False, "pick": "skip", "repeat": True}

    # It never counted, so the site-wide stat is untouched by either call.
    assert client.get("/api/stats").json()["guesses"] == {"total": 0, "correct": 0}


def test_guess_skip_from_a_different_client_does_not_consume_that_id_for_others(client):
    r = client.post("/api/ask", json={"question": "Will it rain tomorrow?", "turnstile_token": "x"})
    id_ = r.json()["id"]
    _record_game_answer(id_, order=["random_graph", "real", "no_brain"])  # real is slot B

    client.post("/api/guess", json={"id": id_, "pick": "skip"}, headers={"cf-connecting-ip": "1.1.1.1"})
    # A DIFFERENT client's guess on the same id is independent -- not a repeat of the first
    # client's skip, and can still be counted on its own merits (own question, within window,
    # first-for-its-key) if it happens to be the asker -- here it's a stranger, so still False,
    # but for the right reason (not the asker), not because of the other client's skip.
    resp = client.post("/api/guess", json={"id": id_, "pick": "B"}, headers={"cf-connecting-ip": "2.2.2.2"})
    assert resp.json()["repeat"] is False
    assert resp.json()["correct"] is True
    assert resp.json()["counted"] is False  # not the asker


def test_validation_endpoint_returns_the_documented_shape(client):
    r = client.get("/api/validation")
    assert r.status_code == 200
    body = r.json()
    for key in ("model", "brain_passport", "calibration", "lateral_validation", "handedness_b0", "stamps"):
        assert key in body
    assert body["brain_passport"]["malecns"]["n_neurons"] == 165122
    assert body["lateral_validation"]["malecns"]["passed"] is True
    assert body["lateral_validation"]["banc"]["passed"] is False
    assert any(s["manifest_file"].endswith(".sha256") for s in body["stamps"])


# `GET /api/events` itself is an infinite SSE stream (poll loop + `sse-starlette` heartbeat) —
# round-tripping it through `TestClient`'s sync transport reliably hangs in this environment (the
# generator never completes, and `TestClient`'s portal-thread streaming does not reliably surface
# partial chunks the way a real ASGI server does). The event SHAPES this route emits
# (`queued`/`thinking`/`answered`, exact payloads) are already covered end-to-end, without any
# HTTP/ASGI layer, by `tests/test_live_events.py`'s `LiveEventTracker` tests — this route is a
# thin wrapper (`events.LiveEventTracker.poll` + `EventSourceResponse`) with no extra logic of its
# own worth re-testing here. Same reasoning applies to `events.ChatEventTracker`, exercised by
# `tests/test_live_events.py`'s own `ChatEventTracker` tests below.


# -- live chat -------------------------------------------------------------------------------------


async def _llm_ok(text, http_client=None):
    return "ok"


def _first_message(client, monkeypatch, text="hello everyone"):
    """Posts a first chat message (content-accepting LLM mocked) and returns the parsed body,
    which always includes a fresh `chat_token` (Turnstile dev-bypasses under LIVE_ENV=dev)."""
    from bioreservoir.live import moderation

    monkeypatch.setattr(moderation, "classify_chat_with_llm", _llm_ok)
    r = client.post("/api/chat", json={"text": text, "turnstile_token": "x"})
    return r


def _set_slowmode(zero_ok=True):
    """Reaches directly into the current LIVE_DB to zero out the slow-mode interval, so a test can
    send several messages back to back without tripping the rate limiter it isn't testing."""
    from bioreservoir.live import config as live_config
    from bioreservoir.live.store import LiveStore

    with LiveStore(path=live_config.LIVE_DB) as s:
        s.chat_set_slowmode(0.0 if zero_ok else s.chat_state()["slowmode_s"])


def test_chat_post_accepts_a_clean_first_message_and_issues_a_token(client, monkeypatch):
    r = _first_message(client, monkeypatch, "what a cool project")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["text"] == "what a cool project"
    assert body["nickname"].startswith("fly-")
    assert body.get("chat_token")


def test_chat_post_accepts_a_custom_nickname(client, monkeypatch):
    from bioreservoir.live import moderation

    monkeypatch.setattr(moderation, "classify_chat_with_llm", _llm_ok)
    r = client.post(
        "/api/chat", json={"text": "hi", "nickname": "fly_fan_1", "turnstile_token": "x"}
    )
    assert r.status_code == 200
    assert r.json()["nickname"] == "fly_fan_1"


def test_chat_post_rejects_bad_nickname_format(client, monkeypatch):
    from bioreservoir.live import moderation

    monkeypatch.setattr(moderation, "classify_chat_with_llm", _llm_ok)
    r = client.post("/api/chat", json={"text": "hi", "nickname": "a", "turnstile_token": "x"})
    assert r.status_code == 422
    assert r.json()["reason"] == "nickname_invalid"
    # Turnstile already succeeded before the nickname was checked -- the token still comes back
    # so the client isn't forced through a second captcha on its retry.
    assert "chat_token" in r.json()


def test_chat_post_second_message_uses_token_without_turnstile(client, monkeypatch):
    from bioreservoir.live import moderation

    first = _first_message(client, monkeypatch, "first message")
    token = first.json()["chat_token"]
    _set_slowmode()

    monkeypatch.setattr(moderation, "verify_turnstile", _boom_if_called)
    r = client.post("/api/chat", json={"text": "second message", "chat_token": token})
    assert r.status_code == 200
    assert "chat_token" not in r.json()  # no NEW token issued -- the existing one is still valid


async def _boom_if_called(*args, **kwargs):
    raise AssertionError("verify_turnstile should not be called when a valid chat_token is given")


def test_chat_post_rejects_without_turnstile_or_token(client, monkeypatch):
    from bioreservoir.live import moderation

    async def fail_turnstile(token, remote_ip=None, http_client=None):
        return False

    monkeypatch.setattr(moderation, "verify_turnstile", fail_turnstile)
    r = client.post("/api/chat", json={"text": "hello"})
    assert r.status_code == 422
    assert r.json()["reason"] == "captcha"


def test_chat_post_rejects_content_but_still_returns_a_fresh_token(client, monkeypatch):
    r = _first_message(client, monkeypatch, "where do I vote in this election?")
    assert r.status_code == 422
    body = r.json()
    assert body["reason"] == "voting_procedure"
    assert body.get("chat_token")


def test_chat_post_rate_limited_within_the_slowmode_interval(client, monkeypatch):
    first = _first_message(client, monkeypatch, "message one")
    token = first.json()["chat_token"]
    r = client.post("/api/chat", json={"text": "message two", "chat_token": token})
    assert r.status_code == 422
    assert r.json()["reason"] == "rate_limited"


def test_chat_post_rejects_duplicate_message(client, monkeypatch):
    first = _first_message(client, monkeypatch, "same text twice")
    token = first.json()["chat_token"]
    _set_slowmode()
    r = client.post("/api/chat", json={"text": "same text twice", "chat_token": token})
    assert r.status_code == 422
    assert r.json()["reason"] == "duplicate"


def test_chat_post_rejects_a_banned_client(client, monkeypatch):
    from bioreservoir.live import config as live_config
    from bioreservoir.live.store import LiveStore

    first = _first_message(client, monkeypatch, "hello")
    id_ = first.json()["id"]
    token = first.json()["chat_token"]
    with LiveStore(path=live_config.LIVE_DB) as s:
        s.chat_ban(s.chat_get(id_)["ip_hash"])
    _set_slowmode()
    r = client.post("/api/chat", json={"text": "let me back in", "chat_token": token})
    assert r.status_code == 422
    assert r.json()["reason"] == "banned"


def test_chat_post_rejects_while_soft_paused(client, monkeypatch):
    from bioreservoir.live import config as live_config
    from bioreservoir.live.store import LiveStore

    with LiveStore(path=live_config.LIVE_DB) as s:
        s.chat_set_enabled(False)
    r = _first_message(client, monkeypatch, "anyone home?")
    assert r.status_code == 422
    assert r.json()["reason"] == "chat_paused"


def test_chat_get_still_works_while_soft_paused(client, monkeypatch):
    """Soft pause (chat_admin off) must not 503 GET /api/chat -- the frontend needs the endpoint
    alive to keep polling/SSE-listening for the chat_state flip back to enabled."""
    from bioreservoir.live import config as live_config
    from bioreservoir.live.store import LiveStore

    with LiveStore(path=live_config.LIVE_DB) as s:
        s.chat_set_enabled(False)
    r = client.get("/api/chat")
    assert r.status_code == 200
    assert r.json()["state"]["enabled"] is False


def test_chat_get_returns_recent_messages_and_state(client, monkeypatch):
    _first_message(client, monkeypatch, "message A")
    r = client.get("/api/chat")
    assert r.status_code == 200
    body = r.json()
    assert body["messages"][0]["text"] == "message A"
    assert body["state"] == {"enabled": True, "slowmode_s": pytest.approx(4.0)}


def test_chat_get_excludes_deleted_messages(client, monkeypatch):
    from bioreservoir.live import config as live_config
    from bioreservoir.live.store import LiveStore

    first = _first_message(client, monkeypatch, "will be deleted")
    id_ = first.json()["id"]
    with LiveStore(path=live_config.LIVE_DB) as s:
        s.chat_delete(id_)
    r = client.get("/api/chat")
    assert r.json()["messages"] == []


# -- kill switch (hard, env-level) -----------------------------------------------------------------


def test_chat_post_503_when_hard_kill_switch_off(tmp_path, monkeypatch):
    import importlib

    monkeypatch.setenv("LIVE_DB", str(tmp_path / "live.sqlite"))
    monkeypatch.setenv("LIVE_ENV", "dev")
    monkeypatch.setenv("LIVE_CHAT_ENABLED", "0")
    import bioreservoir.live.config as live_config

    importlib.reload(live_config)
    import bioreservoir.live.api as live_api

    importlib.reload(live_api)
    with TestClient(live_api.app) as c:
        assert c.post("/api/chat", json={"text": "hi"}).status_code == 503
        assert c.get("/api/chat").status_code == 503


# -- report / auto-hide -----------------------------------------------------------------------------


def test_chat_report_requires_a_valid_token(client, monkeypatch):
    first = _first_message(client, monkeypatch, "reportable message")
    id_ = first.json()["id"]
    r = client.post(f"/api/chat/{id_}/report", json={"chat_token": "not-a-real-token"})
    assert r.status_code == 422
    assert r.json()["reason"] == "captcha"


def test_chat_report_404s_for_an_unknown_message(client, monkeypatch):
    first = _first_message(client, monkeypatch, "seed message")
    token = first.json()["chat_token"]
    r = client.post("/api/chat/999999/report", json={"chat_token": token})
    assert r.status_code == 404


def _accepted_message_from(client, monkeypatch, ip: str, text: str) -> dict:
    """POSTs one accepted chat message as if from `ip` (via CF-Connecting-IP) and returns the
    parsed response body -- used to give a synthetic "reporter" identity both a fresh chat_token
    AND >=1 accepted message (F3's second eligibility condition)."""
    from bioreservoir.live import moderation

    monkeypatch.setattr(moderation, "classify_chat_with_llm", _llm_ok)
    r = client.post(
        "/api/chat",
        json={"text": text, "turnstile_token": "x"},
        headers={"CF-Connecting-IP": ip},
    )
    assert r.status_code == 200, r.text
    return r.json()


def _eligible_reporter_token(ip: str) -> str:
    """Mints a chat_token directly (bypassing the HTTP round trip through Turnstile) that is
    already old enough to satisfy F3's minimum-session-age rule, bound to `ip`'s author_id."""
    from bioreservoir.live import chat as chat_mod
    from bioreservoir.live import config as live_config

    aid = chat_mod.author_id(ip, live_config.CHAT_ID_PEPPER)
    old_now = time.time() - live_config.CHAT_REPORT_MIN_SESSION_AGE_S - 10
    return chat_mod.issue_chat_token(
        live_config.CHAT_SESSION_SECRET, aid, live_config.CHAT_TOKEN_TTL_S, now=old_now
    )


def test_chat_report_is_idempotent_per_client(client, monkeypatch):
    _accepted_message_from(client, monkeypatch, "9.9.9.1", "seed for the reporter's own history")
    posted = _accepted_message_from(client, monkeypatch, "9.9.9.2", "reportable message")
    id_ = posted["id"]
    token = _eligible_reporter_token("9.9.9.1")
    r1 = client.post(f"/api/chat/{id_}/report", json={"chat_token": token}, headers={"CF-Connecting-IP": "9.9.9.1"})
    r2 = client.post(f"/api/chat/{id_}/report", json={"chat_token": token}, headers={"CF-Connecting-IP": "9.9.9.1"})
    assert r1.json()["reports"] == 1
    assert r2.json()["reports"] == 1  # same reporter -- does not double count


def test_chat_report_auto_hides_at_the_threshold_and_broadcasts_via_chat_delete(client, monkeypatch):
    """The Nth DISTINCT eligible reporter (each with their own IP/author_id, an old-enough
    session, and >=1 accepted message -- F3) crosses `config.CHAT_REPORT_THRESHOLD` and
    auto-hides the message -- which shows up to every connected SSE client as a normal
    `chat_delete` event (events.ChatEventTracker), no special-cased event type needed."""
    from bioreservoir.live import config as live_config

    posted = _accepted_message_from(client, monkeypatch, "9.9.0.1", "reportable message")
    id_ = posted["id"]

    r = None
    for i in range(live_config.CHAT_REPORT_THRESHOLD):
        reporter_ip = f"10.{i}.0.1"  # a distinct IPv4 /24 per reporter
        _accepted_message_from(client, monkeypatch, reporter_ip, f"hi, I'm reporter {i}")
        token = _eligible_reporter_token(reporter_ip)
        r = client.post(
            f"/api/chat/{id_}/report",
            json={"chat_token": token},
            headers={"CF-Connecting-IP": reporter_ip},
        )
        assert r.json()["reports"] == i + 1
        if i + 1 < live_config.CHAT_REPORT_THRESHOLD:
            assert r.json()["auto_hidden"] is False

    assert r.json()["auto_hidden"] is True

    remaining = client.get("/api/chat").json()["messages"]
    assert all(m["id"] != id_ for m in remaining)


# -- F3 additional: bans / token-identity binding / eligibility ----------------------------------


def test_chat_report_rejects_a_banned_reporter(client, monkeypatch):
    from bioreservoir.live import chat as chat_mod
    from bioreservoir.live import config as live_config
    from bioreservoir.live.store import LiveStore

    posted = _accepted_message_from(client, monkeypatch, "9.9.9.3", "reportable message")
    _accepted_message_from(client, monkeypatch, "9.9.9.4", "seed for the banned reporter")
    token = _eligible_reporter_token("9.9.9.4")

    aid = chat_mod.author_id("9.9.9.4", live_config.CHAT_ID_PEPPER)
    with LiveStore(path=live_config.LIVE_DB) as s:
        s.chat_ban(aid)

    r = client.post(
        f"/api/chat/{posted['id']}/report",
        json={"chat_token": token},
        headers={"CF-Connecting-IP": "9.9.9.4"},
    )
    assert r.status_code == 422
    assert r.json()["reason"] == "banned"


def test_chat_report_rejects_a_token_presented_from_a_different_ip(client, monkeypatch):
    """F3's core fix: a chat_token is bound to the author_id that received it -- presenting it
    from a different IP (a different /64 or a different IPv4 address entirely) must fail."""
    posted = _accepted_message_from(client, monkeypatch, "9.9.9.5", "reportable message")
    _accepted_message_from(client, monkeypatch, "9.9.9.6", "seed for reporter A")
    token_for_a = _eligible_reporter_token("9.9.9.6")

    r = client.post(
        f"/api/chat/{posted['id']}/report",
        json={"chat_token": token_for_a},
        headers={"CF-Connecting-IP": "9.9.9.7"},  # a DIFFERENT ip than the token was minted for
    )
    assert r.status_code == 422
    assert r.json()["reason"] == "captcha"


def test_chat_report_from_a_too_young_session_does_not_count(client, monkeypatch):
    posted = _accepted_message_from(client, monkeypatch, "9.9.9.8", "reportable message")
    reporter = _accepted_message_from(client, monkeypatch, "9.9.9.9", "seed for a fresh reporter")
    fresh_token = reporter["chat_token"]  # just issued -- session age ~0s

    r = client.post(
        f"/api/chat/{posted['id']}/report",
        json={"chat_token": fresh_token},
        headers={"CF-Connecting-IP": "9.9.9.9"},
    )
    assert r.status_code == 200
    assert r.json()["reports"] == 0
    assert r.json()["auto_hidden"] is False

    # Sanity: the SAME reporter, once their session is old enough, DOES count.
    old_token = _eligible_reporter_token("9.9.9.9")
    r2 = client.post(
        f"/api/chat/{posted['id']}/report",
        json={"chat_token": old_token},
        headers={"CF-Connecting-IP": "9.9.9.9"},
    )
    assert r2.json()["reports"] == 1


def test_chat_report_from_a_client_with_no_accepted_message_does_not_count(client, monkeypatch):
    """An old-enough token minted for an author_id that has never posted a real message must not
    count toward auto-hide (F3's second eligibility condition)."""
    posted = _accepted_message_from(client, monkeypatch, "9.9.9.10", "reportable message")
    token = _eligible_reporter_token("9.9.9.11")  # never posted anything as this author

    r = client.post(
        f"/api/chat/{posted['id']}/report",
        json={"chat_token": token},
        headers={"CF-Connecting-IP": "9.9.9.11"},
    )
    assert r.status_code == 200
    assert r.json()["reports"] == 0


def test_chat_report_reporters_in_the_same_ipv4_slash_24_count_once(client, monkeypatch):
    posted = _accepted_message_from(client, monkeypatch, "9.9.9.12", "reportable message")
    _accepted_message_from(client, monkeypatch, "203.0.113.1", "seed A")
    _accepted_message_from(client, monkeypatch, "203.0.113.2", "seed B")
    token_a = _eligible_reporter_token("203.0.113.1")
    token_b = _eligible_reporter_token("203.0.113.2")

    r1 = client.post(
        f"/api/chat/{posted['id']}/report", json={"chat_token": token_a},
        headers={"CF-Connecting-IP": "203.0.113.1"},
    )
    r2 = client.post(
        f"/api/chat/{posted['id']}/report", json={"chat_token": token_b},
        headers={"CF-Connecting-IP": "203.0.113.2"},
    )
    assert r1.json()["reports"] == 1
    assert r2.json()["reports"] == 1  # same /24 as reporter A -- does not add a second count


# -- F5: reserved nicknames rejected end-to-end ----------------------------------------------------


def test_chat_post_rejects_a_reserved_nickname(client, monkeypatch):
    from bioreservoir.live import moderation

    monkeypatch.setattr(moderation, "classify_chat_with_llm", _llm_ok)
    r = client.post(
        "/api/chat", json={"text": "hi", "nickname": "BioReservoir", "turnstile_token": "x"}
    )
    assert r.status_code == 422
    assert r.json()["reason"] == "impersonation"


def test_chat_post_rejects_a_user_chosen_server_handle_shape(client, monkeypatch):
    from bioreservoir.live import moderation

    monkeypatch.setattr(moderation, "classify_chat_with_llm", _llm_ok)
    r = client.post(
        "/api/chat", json={"text": "hi", "nickname": "fly-4821", "turnstile_token": "x"}
    )
    assert r.status_code == 422
    assert r.json()["reason"] == "impersonation"


# -- F8: missing chat secrets -> 503, logged once at startup ---------------------------------------


@pytest.mark.parametrize("missing_var", ["CHAT_SESSION_SECRET", "CHAT_ID_PEPPER", "CHAT_LLM_API_KEY"])
def test_chat_503s_when_a_required_secret_is_missing(tmp_path, monkeypatch, missing_var, caplog):
    import importlib
    import logging

    monkeypatch.setenv("LIVE_DB", str(tmp_path / "live.sqlite"))
    monkeypatch.setenv("LIVE_ENV", "dev")
    all_secrets = {"CHAT_SESSION_SECRET": "s", "CHAT_ID_PEPPER": "p", "CHAT_LLM_API_KEY": "k"}
    for name, value in all_secrets.items():
        if name == missing_var:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)

    import bioreservoir.live.config as live_config

    importlib.reload(live_config)
    import bioreservoir.live.api as live_api

    importlib.reload(live_api)

    with caplog.at_level(logging.WARNING, logger="bioreservoir.live.api"), TestClient(live_api.app) as c:
        assert c.post("/api/chat", json={"text": "hi"}).status_code == 503
        assert c.get("/api/chat").status_code == 503
        assert c.post("/api/chat/1/report", json={"chat_token": "x"}).status_code == 503
    assert any(missing_var in record.message for record in caplog.records)


def test_chat_available_when_all_three_secrets_present(client, monkeypatch):
    """Sanity check that the `client` fixture's defaults (all three secrets set) actually work --
    guards against the F8 tests above accidentally passing for the wrong reason."""
    r = _first_message(client, monkeypatch, "hello, secrets are configured")
    assert r.status_code == 200


# -- F1: concurrent posts through the real HTTP layer -----------------------------------------------


def test_chat_post_concurrent_burst_only_one_succeeds(client, monkeypatch):
    """Reproduces the security review's F1 exploit shape directly: a single authenticated client
    fires many POST /api/chat requests at once. Before the fix, 40/40 were accepted despite
    slow-mode; after the fix (store.py's chat_reserve_attempt, reserved atomically before any
    await), only ONE of a concurrent burst may succeed."""
    import concurrent.futures

    from bioreservoir.live import moderation

    monkeypatch.setattr(moderation, "classify_chat_with_llm", _llm_ok)
    first = client.post(
        "/api/chat",
        json={"text": "first message", "turnstile_token": "x"},
        headers={"CF-Connecting-IP": "9.9.9.20"},
    )
    assert first.status_code == 200
    token = first.json()["chat_token"]

    def fire(i: int):
        return client.post(
            "/api/chat",
            json={"text": f"burst message {i}", "chat_token": token},
            headers={"CF-Connecting-IP": "9.9.9.20"},
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=40) as pool:
        responses = list(pool.map(fire, range(40)))

    accepted = [r for r in responses if r.status_code == 200]
    assert len(accepted) <= 1, f"expected at most 1 acceptance from a concurrent burst, got {len(accepted)}"


# -- F2: rejected/failed attempts still consume the rate limit --------------------------------------


def test_chat_post_rejected_attempts_still_count_toward_the_window_cap(client, monkeypatch):
    """F2: "every attempt counts toward the rate limits: accepted, rejected and failed" -- content
    that always gets rejected by the deterministic rule layer (so no LLM call is even needed)
    still occupies rate-limit slots, eventually tripping the window cap."""
    from bioreservoir.live import config as live_config
    from bioreservoir.live.store import LiveStore

    with LiveStore(path=live_config.LIVE_DB) as s:
        s.chat_set_slowmode(0.0)  # isolate the 10-minute window cap from the per-message slowmode

    r = None
    for _ in range(live_config.CHAT_RATE_PER_WINDOW):
        r = client.post(
            "/api/chat",
            json={"text": "where do I vote in this election?", "turnstile_token": "x"},
            headers={"CF-Connecting-IP": "9.9.9.21"},
        )
        assert r.status_code == 422
        assert r.json()["reason"] == "voting_procedure"

    r = client.post(
        "/api/chat",
        json={"text": "where do I vote in this election?", "turnstile_token": "x"},
        headers={"CF-Connecting-IP": "9.9.9.21"},
    )
    assert r.status_code == 422
    assert r.json()["reason"] == "rate_limited"


def test_chat_post_llm_circuit_breaker_trips_chat_busy(client, monkeypatch):
    """F2's site-wide breaker: independent of any one client's identity -- tripped by requests
    from DIFFERENT IPs, then the next LLM-needing message from yet another IP gets "chat_busy"."""
    from bioreservoir.live import config as live_config
    from bioreservoir.live import moderation
    from bioreservoir.live.store import LiveStore

    monkeypatch.setattr(live_config, "CHAT_LLM_MAX_PER_MINUTE", 2)
    monkeypatch.setattr(moderation, "classify_chat_with_llm", _llm_ok)
    with LiveStore(path=live_config.LIVE_DB) as s:
        s.chat_set_slowmode(0.0)

    for i in range(2):
        r = client.post(
            "/api/chat",
            json={"text": f"ordinary message {i}", "turnstile_token": "x"},
            headers={"CF-Connecting-IP": f"9.9.9.{30 + i}"},
        )
        assert r.status_code == 200, r.text

    r = client.post(
        "/api/chat",
        json={"text": "one more ordinary message", "turnstile_token": "x"},
        headers={"CF-Connecting-IP": "9.9.9.40"},
    )
    assert r.status_code == 422
    assert r.json()["reason"] == "chat_busy"


def test_chat_llm_breaker_default_raised_to_200():
    """R2 (round-2 security review): 30/min tripped under 35 ordinary concurrent chatters."""
    from bioreservoir.live import config as live_config

    assert live_config.CHAT_LLM_MAX_PER_MINUTE == 200


def test_chat_post_per_author_llm_quota_trips_before_the_global_breaker(client, monkeypatch):
    """R2: a smaller per-author LLM quota, checked BEFORE the site-wide breaker, so a handful of
    identities can't keep it permanently tripped for everyone else."""
    from bioreservoir.live import config as live_config
    from bioreservoir.live import moderation
    from bioreservoir.live.store import LiveStore

    monkeypatch.setattr(live_config, "CHAT_LLM_PER_AUTHOR_PER_MINUTE", 2)
    monkeypatch.setattr(live_config, "CHAT_LLM_MAX_PER_MINUTE", 1000)  # global breaker has plenty of room
    monkeypatch.setattr(moderation, "classify_chat_with_llm", _llm_ok)
    with LiveStore(path=live_config.LIVE_DB) as s:
        s.chat_set_slowmode(0.0)

    for i in range(2):
        r = client.post(
            "/api/chat",
            json={"text": f"ordinary message {i}", "turnstile_token": "x"},
            headers={"CF-Connecting-IP": "172.21.0.1"},
        )
        assert r.status_code == 200, r.text

    r = client.post(
        "/api/chat",
        json={"text": "one too many from the same author", "turnstile_token": "x"},
        headers={"CF-Connecting-IP": "172.21.0.1"},
    )
    assert r.status_code == 422
    assert r.json()["reason"] == "chat_busy"


def test_chat_post_per_author_llm_quota_does_not_affect_other_authors(client, monkeypatch):
    from bioreservoir.live import config as live_config
    from bioreservoir.live import moderation
    from bioreservoir.live.store import LiveStore

    monkeypatch.setattr(live_config, "CHAT_LLM_PER_AUTHOR_PER_MINUTE", 1)
    monkeypatch.setattr(moderation, "classify_chat_with_llm", _llm_ok)
    with LiveStore(path=live_config.LIVE_DB) as s:
        s.chat_set_slowmode(0.0)

    r1 = client.post(
        "/api/chat",
        json={"text": "message from author A", "turnstile_token": "x"},
        headers={"CF-Connecting-IP": "172.22.0.1"},
    )
    assert r1.status_code == 200

    r1b = client.post(
        "/api/chat",
        json={"text": "second message from author A", "turnstile_token": "x"},
        headers={"CF-Connecting-IP": "172.22.0.1"},
    )
    assert r1b.status_code == 422
    assert r1b.json()["reason"] == "chat_busy"

    r2 = client.post(
        "/api/chat",
        json={"text": "message from author B", "turnstile_token": "x"},
        headers={"CF-Connecting-IP": "172.22.0.2"},
    )
    assert r2.status_code == 200


# -- R1: per-reporter and site-wide auto-hide caps (round-2 security review) ---------------------


def _make_eligible_reporters(client, monkeypatch, n: int, ip_prefix: str) -> list[tuple[str, str]]:
    """Returns `[(ip, chat_token), ...]` for `n` distinct, eligible (>=1 accepted message,
    old-enough session) reporter identities."""
    reporters = []
    for i in range(n):
        ip = f"{ip_prefix}.{i}.1"
        _accepted_message_from(client, monkeypatch, ip, f"hi, I'm reporter {i}")
        reporters.append((ip, _eligible_reporter_token(ip)))
    return reporters


def test_chat_report_five_identities_cannot_hide_unlimited_messages_r1(client, monkeypatch):
    """Reproduces the round-2 security review's R1 exploit shape directly: 5 eligible identities
    (their own accepted message + a 10-min-old session) report the same set of victim messages.
    Before the fix this hid 20/20 messages, 5 requests apiece, unbounded. After the fix (a
    per-reporter cap + a site-wide auto-hide cap, both applied in the SAME 10-minute window as the
    report threshold), the number of messages actually hidden is bounded and strictly less than
    the number of victims."""
    from bioreservoir.live import config as live_config

    reporters = _make_eligible_reporters(client, monkeypatch, live_config.CHAT_REPORT_THRESHOLD, "172.16")

    victim_ids = []
    for i in range(20):
        posted = _accepted_message_from(client, monkeypatch, f"172.20.{i}.1", f"victim message {i}")
        victim_ids.append(posted["id"])

    hidden_count = 0
    last_resp = None
    for vid in victim_ids:
        for ip, token in reporters:
            last_resp = client.post(
                f"/api/chat/{vid}/report", json={"chat_token": token}, headers={"CF-Connecting-IP": ip}
            )
            assert last_resp.status_code == 200
        # Every report is still recorded regardless of whether the hide was withheld.
        assert last_resp.json()["reports"] == live_config.CHAT_REPORT_THRESHOLD
        if last_resp.json()["auto_hidden"]:
            hidden_count += 1

    assert hidden_count <= live_config.CHAT_AUTO_HIDE_PER_10MIN
    assert hidden_count < 20, "the whole point of R1: 5 identities must not be able to hide every victim"


def test_chat_report_per_reporter_cap_withholds_hide_but_still_records(client, monkeypatch):
    from bioreservoir.live import config as live_config

    monkeypatch.setattr(live_config, "CHAT_REPORTS_PER_10MIN", 1)
    # Isolate this test from the site-wide cap so only the per-reporter cap is exercised.
    monkeypatch.setattr(live_config, "CHAT_AUTO_HIDE_PER_10MIN", 1000)

    reporters = _make_eligible_reporters(client, monkeypatch, live_config.CHAT_REPORT_THRESHOLD, "172.17")
    msg1 = _accepted_message_from(client, monkeypatch, "172.18.0.1", "victim 1")
    msg2 = _accepted_message_from(client, monkeypatch, "172.18.0.2", "victim 2")

    r = None
    for ip, token in reporters:
        r = client.post(f"/api/chat/{msg1['id']}/report", json={"chat_token": token}, headers={"CF-Connecting-IP": ip})
    assert r.json()["auto_hidden"] is True  # first message: every reporter still has quota

    for ip, token in reporters:
        r = client.post(f"/api/chat/{msg2['id']}/report", json={"chat_token": token}, headers={"CF-Connecting-IP": ip})
    assert r.json()["reports"] == live_config.CHAT_REPORT_THRESHOLD  # still recorded
    assert r.json()["auto_hidden"] is False  # but the crossing reporter is out of quota


def test_chat_report_site_wide_cap_withholds_hide_after_it_is_reached(client, monkeypatch):
    from bioreservoir.live import config as live_config

    monkeypatch.setattr(live_config, "CHAT_AUTO_HIDE_PER_10MIN", 1)
    monkeypatch.setattr(live_config, "CHAT_REPORTS_PER_10MIN", 1000)  # isolate from the per-reporter cap

    reporters = _make_eligible_reporters(client, monkeypatch, live_config.CHAT_REPORT_THRESHOLD, "172.19")
    msg1 = _accepted_message_from(client, monkeypatch, "172.23.0.1", "victim 1")
    msg2 = _accepted_message_from(client, monkeypatch, "172.23.0.2", "victim 2")

    r = None
    for ip, token in reporters:
        r = client.post(f"/api/chat/{msg1['id']}/report", json={"chat_token": token}, headers={"CF-Connecting-IP": ip})
    assert r.json()["auto_hidden"] is True

    for ip, token in reporters:
        r = client.post(f"/api/chat/{msg2['id']}/report", json={"chat_token": token}, headers={"CF-Connecting-IP": ip})
    assert r.json()["reports"] == live_config.CHAT_REPORT_THRESHOLD
    assert r.json()["auto_hidden"] is False  # site-wide cap of 1 already spent on msg1


def test_repeating_the_same_question_still_consumes_the_rate_limit(client):
    """The dedupe serves a repeat without inserting a row, so the row-based per-IP counters alone
    would not see it — attempts.AttemptCounter does, otherwise the same wording could be re-sent
    without limit (and each send still costs a Turnstile verification)."""
    for _ in range(3):
        r = client.post("/api/ask", json={"question": "Will the same words repeat?", "turnstile_token": "x"})
        assert r.status_code == 202
    r = client.post("/api/ask", json={"question": "Will the same words repeat?", "turnstile_token": "x"})
    assert r.status_code == 422
    assert r.json()["reason"] == "rate_limited"


def test_a_busy_classifier_queues_the_question_instead_of_calling_it_spam(client, monkeypatch):
    """The LLM gateway rate-limiting us says nothing about the question: it joins the line with
    `needs_llm` and the worker classifies it before simulating (moderation.LLM_BUSY)."""
    from bioreservoir.live import config, moderation
    from bioreservoir.live.store import LiveStore

    async def busy(question, http_client=None):
        return moderation.LLM_BUSY

    monkeypatch.setattr(moderation, "classify_with_llm", busy)
    monkeypatch.setattr(config, "LLM_BASE_URL", "https://llm.example/v1")
    r = client.post("/api/ask", json={"question": "Will the classifier be busy?", "turnstile_token": "x"})
    assert r.status_code == 202, r.text
    with LiveStore(path=config.LIVE_DB) as s:
        row = s.get(r.json()["id"])
        assert row["status"] == "queued"
        assert row["needs_llm"] == 1
