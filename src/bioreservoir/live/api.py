"""FastAPI app for the live fly page (routes match the frontend contract in
`site/src/api/types.ts` exactly). Run with `uvicorn bioreservoir.live.api:app`; the worker
(`live.worker`) is a separate process reading/writing the same `LIVE_DB` file.

The live chat (`POST`/`GET /api/chat`, `POST /api/chat/{id}/report`) is a second feature bolted
onto this same app and the same `/api/events` SSE stream (`events.ChatEventTracker`) -- see
docs/LIVE.md's "Live chat" section and `chat.py`/`moderation.py`'s chat policy for the design.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from bioreservoir.live import (
    attempts,
    card,
    chat,
    config,
    events,
    moderation,
    pipeline,
    stats,
    store,
    summary,
    validation,
    verdict,
)
from bioreservoir.live import game as game_mod

logger = logging.getLogger(__name__)

# Ask-dedupe: a rejection whose reason is about the ASKER's own request
# (rate limit / captcha), not about the QUESTION's content, must never be replayed for a brand-new
# asker who happens to reuse the same wording -- `moderation.CONTENT_REASON_CODES` is the allowed
# set for that replay.


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.store = store.LiveStore()
    # `validation.build_validation()` reads both brains' harmonized graphs (~1-2s) -- computed
    # lazily on first `GET /api/validation` (below), not here, so every OTHER route (and every
    # test that spins up this app) does not pay that cost up front.
    app.state.validation = None
    # F8 (2026-09-18 security review): chat needs THREE secrets with no default value
    # (CHAT_SESSION_SECRET, CHAT_ID_PEPPER, CHAT_LLM_API_KEY) -- checked once here, not per
    # request, and logged once at startup rather than spamming a warning on every chat call.
    chat_secrets_ok, missing_chat_secrets = config.chat_secrets_configured()
    app.state.chat_secrets_ok = chat_secrets_ok
    if config.LIVE_CHAT_ENABLED and not chat_secrets_ok:
        logger.warning(
            "chat disabled at startup: missing required secret(s) %s -- see docs/LIVE.md's "
            "'Live chat' section for how to provision them; all chat routes will 503 until set",
            ", ".join(missing_chat_secrets),
        )

    # ONE `events.Broadcaster` per process, started here (in the lifespan --
    # simpler than a lazy first-subscriber start, no race to guard around task creation) rather
    # than one `LiveEventTracker`/poll loop per SSE connection. `chat_tracker` is built the exact
    # same way the old per-connection code built its own (seeded to "now"/`chat_latest_id()` so a
    # fresh process never replays history, F6) -- but only ONE ever exists for the whole process,
    # and only when chat is actually available (same gate `_chat_available` checks per-request).
    chat_tracker = None
    if config.LIVE_CHAT_ENABLED and chat_secrets_ok:
        now_iso = datetime.now(UTC).isoformat()
        chat_tracker = events.ChatEventTracker(
            last_id=app.state.store.chat_latest_id(),
            last_deleted_at=now_iso,
            last_restored_at=now_iso,
        )
    app.state.broadcaster = events.Broadcaster(
        app.state.store, config.LIVE_POLL_INTERVAL_S, chat_tracker=chat_tracker
    )
    app.state.broadcaster.start()
    try:
        yield
    finally:
        await app.state.broadcaster.stop()
        app.state.store.close()


app = FastAPI(title="BioReservoir Live Fly", lifespan=lifespan)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Logs every unhandled exception as `UNHANDLED exc_class=<Type> path=<route>` — a fixed,
    greppable format for external log monitoring — before returning a generic 500."""
    if isinstance(exc, (HTTPException, RequestValidationError)):
        raise exc
    request_id = str(uuid.uuid4())
    logger.error(
        "UNHANDLED exc_class=%s path=%s request_id=%s\n%s",
        type(exc).__name__, request.url.path, request_id, traceback.format_exc(),
    )
    return JSONResponse(status_code=500, content={"error": "internal", "request_id": request_id})


def _client_ip(request: Request) -> str:
    # Deployed behind Cloudflare/Traefik — CF-Connecting-IP is the real client IP when
    # present; falls back to the direct peer address for local/dev runs with no proxy in front.
    return request.headers.get("cf-connecting-ip") or (request.client.host if request.client else "unknown")


def _chat_available(request: Request) -> bool:
    """The HARD gate every chat route checks first (F8): the deploy-time LIVE_CHAT_ENABLED
    switch AND all three required chat secrets being present. Distinct from the SOFT runtime
    pause (`live_chat_state.enabled`, `chat_admin off`/`on`), which only POST /api/chat honors —
    see that route's own comment."""
    return config.LIVE_CHAT_ENABLED and request.app.state.chat_secrets_ok


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    turnstile_token: str = ""


def _rejected(reason: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=422, content={"status": "rejected", "reason": reason, "message": message})


def _joined_response(live_store: store.LiveStore, twin_id: int) -> dict:
    return {
        "id": twin_id,
        "status": "queued",
        "position": live_store.position(twin_id),
        "claimed_total": live_store.claimed_total(),
        "avg_cycle_s": live_store.avg_cycle_s(),
        "joined": True,
    }


_ASK_ATTEMPTS = attempts.AttemptCounter()


@app.post("/api/ask", status_code=202)
async def ask(payload: AskRequest, request: Request):
    """Ask-dedupe: `moderate_gate` (length/yes-no/rate-limit/Turnstile) runs
    UNCHANGED for every single ask ("keep Turnstile and the per-IP rate limits exactly as they are
    for every ask") -- only what happens AFTER that gate depends on whether this exact normalized
    wording (`pipeline.question_key`) has been seen before:

      1. already ANSWERED -> return that id, `repeat: true`, no new row, no LLM call (the same
         wording produces the identical answer -- same seeds, same yes-side coin).
      2. already QUEUED/THINKING -> join it: same id, `joined: true`, no new row, no LLM call,
         and this does NOT count toward the asker's own per-IP queue cap since no row
         is created for them.
      3. previously rejected for CONTENT (not rate-limit/captcha/queue-cap) -> replay that same
         rejection, no new row, no LLM call -- a repeated bad question does not get a second LLM
         classification just because a different IP asked it.
      4. otherwise -> `moderate_content` (rules + LLM) runs as before, then `enqueue`.

    Race guard ("two identical asks at the same moment must not create two runs"):
    `store.enqueue`'s unique partial index can still reject a concurrent duplicate that slipped
    past step 2's `active_twin` check (both requests read "no twin yet" before either commits) --
    caught here and turned into the SAME join response step 2 would have given, against whichever
    of the two requests actually won.
    """
    live_store: store.LiveStore = request.app.state.store
    question = moderation.normalize_question(payload.question)
    ip_hash = live_store.hash_ip(_client_ip(request))

    # Counted BEFORE the gate and independently of the dedupe below: a repeat or a joined ask
    # creates no row, so the row-based counters alone would let the same wording be re-sent without
    # limit (and each one still pays for a Turnstile verification). See attempts.py.
    # Counted like the row-based counters are: EARLIER attempts only, this one not included yet.
    n_recent_minute = max(live_store.count_recent(ip_hash, window_seconds=60.0), _ASK_ATTEMPTS.count(ip_hash, 60.0))
    n_today = max(live_store.count_today(ip_hash), _ASK_ATTEMPTS.count(ip_hash, 86_400.0))
    _ASK_ATTEMPTS.record(ip_hash)
    gate_result = await moderation.moderate_gate(
        question=question,
        turnstile_token=payload.turnstile_token,
        n_recent_minute=n_recent_minute,
        n_today=n_today,
        n_queued=live_store.count_queued(ip_hash),
    )
    if not gate_result.ok:
        live_store.reject(question, ip_hash, gate_result.reason, gate_result.message)
        return _rejected(gate_result.reason, gate_result.message)

    key = pipeline.question_key(question)

    answered_twin = live_store.answered_twin(key)
    if answered_twin is not None:
        return {"id": answered_twin["id"], "status": "answered", "repeat": True}

    active_twin = live_store.active_twin(key)
    if active_twin is not None:
        return _joined_response(live_store, active_twin["id"])

    rejected_twin = live_store.rejected_twin(key)
    if rejected_twin is not None and rejected_twin["reason"] in moderation.CONTENT_REASON_CODES:
        return _rejected(rejected_twin["reason"], rejected_twin["message"])

    content_result = await moderation.moderate_content(question)
    # The classifier was busy (a crowd hitting its per-minute cap, moderation.LLM_BUSY): the
    # question joins the line with `needs_llm`, and the worker classifies it right before
    # simulating — one call a minute instead of one per ask, and no visitor is told their real
    # question looks like spam because we were rate-limited.
    needs_llm = not content_result.ok and content_result.reason == moderation.LLM_BUSY
    if not content_result.ok and not needs_llm:
        live_store.reject(question, ip_hash, content_result.reason, content_result.message, question_key=key)
        return _rejected(content_result.reason, content_result.message)

    try:
        id_ = live_store.enqueue(question, ip_hash, question_key=key, needs_llm=needs_llm)
    except store.DuplicateActiveQuestion:
        # Lost a race against another request for the same key that committed its own `enqueue`
        # between this request's `active_twin` check above and its own insert just now.
        winner = live_store.active_twin(key)
        assert winner is not None  # the row that won the race must exist by the time we get here
        return _joined_response(live_store, winner["id"])

    return {
        "id": id_,
        "status": "queued",
        "position": live_store.position(id_),
        "claimed_total": live_store.claimed_total(),
        "avg_cycle_s": live_store.avg_cycle_s(),
        "joined": False,
    }


@app.get("/api/answers/{id}")
async def get_answer(id: int, request: Request, response: Response):
    """Cache-Control varies by status: `queued`/`thinking`/`rejected` are `no-store`
    (they change or stop existing), `answered` is `public, max-age=31536000, immutable` (an
    answered row's `answer_json` never changes again -- the id is a permanent, cacheable URL, for
    CDN-friendliness). The asker gets the FULL `Answer` (turn_strength backfilled via
    `verdict.with_turn_strength` for rows answered before that field existed).

    An EMBARGOED election answer is the one exception: it carries no `answer` at all unless the
    caller is the asker (same `ip_hash`), and it is never cached. Holding the id used to be treated
    as proof of being the asker -- but ids are public (the feed, and now the whole questions
    archive), so anyone could read a sealed verdict straight off the API while the page carefully
    hid it (2026-09-19 external review). The asker's own tab still gets it: same IP, same day."""
    live_store: store.LiveStore = request.app.state.store
    row = live_store.get(id)
    if row is None:
        raise HTTPException(status_code=404, detail="not found")
    status_ = row["status"]
    resp: dict = {"id": row["id"], "question": row["question"], "status": status_}
    if status_ in ("queued", "thinking"):
        # The side a YES would face, before the answer exists — see `pipeline.yes_side_for`.
        resp["yes_side"] = pipeline.yes_side_for(row["question"])
    if status_ == "queued":
        resp["position"] = live_store.position(id)
        resp["claimed_total"] = live_store.claimed_total()
        resp["avg_cycle_s"] = live_store.avg_cycle_s()
        response.headers["Cache-Control"] = "no-store"
    elif status_ == "thinking":
        resp["started_at"] = row["started_at"]
        response.headers["Cache-Control"] = "no-store"
    elif status_ == "answered":
        full_answer = verdict.with_turn_strength(json.loads(row["answer_json"]))
        embargoed = card.election_embargo(full_answer)
        resp["embargoed"] = embargoed
        is_asker = embargoed and row["ip_hash"] == live_store.hash_ip(_client_ip(request))
        if embargoed and not is_asker:
            # No verdict, and nothing cacheable: this response depends on who is asking.
            response.headers["Cache-Control"] = "no-store"
        else:
            resp["answer"] = full_answer
            response.headers["Cache-Control"] = (
                "no-store" if embargoed else "public, max-age=31536000, immutable"
            )
    elif status_ == "rejected":
        resp["reason"] = row["reason"]
        resp["message"] = row["message"]
        response.headers["Cache-Control"] = "no-store"
    return resp


_GUESS_SLOTS = ("A", "B", "C")
_GUESS_SKIP = "skip"


class GuessRequest(BaseModel):
    id: int
    pick: str


def _within_guess_count_window(answered_at_iso: str | None, now: datetime | None = None) -> bool:
    """F2: a guess only counts toward the site-wide stat if it lands within
    `config.GUESS_COUNT_WINDOW_S` of the answer -- closes the "look the answer up elsewhere,
    come back later and guess" loophole. `answered_at_iso` missing/unparsable never counts
    (fail closed, matching this feature's other "never silently count" rules)."""
    if not answered_at_iso:
        return False
    try:
        answered_at = datetime.fromisoformat(answered_at_iso)
    except ValueError:
        return False
    elapsed_s = ((now or datetime.now(UTC)) - answered_at).total_seconds()
    return elapsed_s <= config.GUESS_COUNT_WINDOW_S


def _is_first_answer_for_question(live_store: store.LiveStore, id_: int, question: str) -> bool:
    """Round-2 logic review R2(a): a guess only counts if THIS answer is the FIRST answered row
    for its normalized question text (`pipeline.question_key`) -- otherwise the example chips (or
    any popular/repeated wording) let a visitor guess a question whose answer was already public
    from an earlier asker's reveal, counted or not. Compares against every EARLIER answered row
    (smaller id, i.e. asked before this one) with the same key; an exact rephrasing that still
    normalizes to the same key is caught too, not just an identical string."""
    from bioreservoir.live.pipeline import question_key

    this_key = question_key(question)
    for row in live_store.answered_questions_before(id_):
        if question_key(row["question"]) == this_key:
            return False
    return True


@app.post("/api/guess")
async def guess(payload: GuessRequest, request: Request):
    """The "which one is the real fly?" game (live/game.py). One guess per (answer id, client) —
    a repeat call for an id this client already guessed returns the SAME `{correct, real, counted,
    pick}` it got the first time (F6), `pick` being the STORED pick, not this call's, so the
    frontend never mismarks which slot the visitor actually picked. Anyone can guess (share links,
    the public feed included), but a guess only counts toward the site-wide "visitors spot the
    real brain X% of the time" stat (`GET /api/stats`'s `guesses`, `store.guess_stats`, which sums
    only `counted` rows) if ALL of: (a) this client is the question's own asker
    (`live_questions.ip_hash`), (b) the guess lands within `config.GUESS_COUNT_WINDOW_S` of the
    answer, and (c) this answer is the FIRST answered occurrence of its normalized question text
    (`_is_first_answer_for_question`, R2(a) -- otherwise the example chips or any repeated wording
    let an asker guess with the answer already public from an earlier reveal).

    `pick == "skip"` (R2(b)) records a NEVER-counted guess for this (answer, client) anyway --
    consumes the slot the same way a real pick does, so a visitor who skips and later opens the
    share link (or the id directly) on the same answer gets the idempotent "repeat" path instead
    of a fresh, now-answer-known, countable guess."""
    if payload.pick not in _GUESS_SLOTS and payload.pick != _GUESS_SKIP:
        raise HTTPException(status_code=422, detail="pick must be one of A, B, C, or skip")

    live_store: store.LiveStore = request.app.state.store
    ip_hash = live_store.hash_ip(_client_ip(request))

    row = live_store.get(payload.id)
    if row is None or row["status"] != "answered":
        raise HTTPException(status_code=404, detail="no answer for this id")
    answer = json.loads(row["answer_json"])
    game = answer.get("game")
    if game is None:
        raise HTTPException(status_code=404, detail="this answer has no game")

    real = game_mod.real_slot(game["order"])
    existing = live_store.get_guess(payload.id, ip_hash)
    if existing is not None:
        # A repeat is a cheap, idempotent read -- answered BEFORE the rate limit below, so a
        # legitimate reload/double-click of an id this client already guessed can never be
        # 429'd by the SAME rate limit that exists to stop hammering the endpoint with NEW
        # (would-be-recorded) guesses. Also the ONLY path a skip's slot is ever seen again through.
        return {
            "correct": bool(existing["correct"]),
            "real": real,
            "counted": bool(existing["counted"]),
            "pick": existing["pick"],
            "repeat": True,
        }

    # F2: rate-limit only a genuinely NEW guess (or skip) about to be recorded, reusing the SAME
    # atomic reserve-attempt mechanism chat's report/LLM-quota namespacing already established (a
    # "guess:{ip_hash}" key on the same live_chat_attempts table) -- blocks a fast id-enumeration
    # loop from recording many new guesses per minute, independent of whether any end up counted.
    if not live_store.chat_reserve_attempt(
        f"guess:{ip_hash}",
        slowmode_s=config.GUESS_SLOWMODE_S,
        window_s=config.GUESS_RATE_WINDOW_S,
        max_per_window=config.GUESS_RATE_PER_WINDOW,
    ):
        raise HTTPException(status_code=429, detail="too many guesses — slow down")

    if payload.pick == _GUESS_SKIP:
        live_store.record_guess(payload.id, ip_hash, _GUESS_SKIP, correct=False, counted=False)
        return {"correct": False, "real": real, "counted": False, "pick": _GUESS_SKIP, "repeat": False}

    # F2: `row["ip_hash"]` is the ASKER's hash, set once by POST /api/ask at ask time; both it and
    # `ip_hash` above use LiveStore.hash_ip's per-UTC-day salt, so this comparison is exact only
    # within the same day -- an accepted limitation right at the UTC midnight boundary.
    counted = (
        row["ip_hash"] == ip_hash
        and _within_guess_count_window(row["answered_at"])
        and _is_first_answer_for_question(live_store, payload.id, row["question"])
    )
    correct = payload.pick == real
    live_store.record_guess(payload.id, ip_hash, payload.pick, correct, counted)
    return {"correct": correct, "real": real, "counted": counted, "pick": payload.pick, "repeat": False}


@app.get("/api/now")
async def now(request: Request, response: Response):
    """The single "what's happening right now" call a client makes on load, alongside
    `GET /api/feed` and `GET /api/chat`, BEFORE ever opening the SSE stream (the stream itself
    sends nothing retroactive) -- `thinking`/`last` mirror the SSE `thinking`/`answered` event
    shapes exactly, so the same rendering code handles the initial snapshot and every later live
    update."""
    live_store: store.LiveStore = request.app.state.store
    thinking_row = live_store.current_thinking()
    thinking = None
    if thinking_row is not None:
        thinking = {
            "id": thinking_row["id"],
            "question": thinking_row["question"],
            "started_at": thinking_row["started_at"],
            "embargoed": card.question_embargoed(thinking_row["question"]),
            # Known from the wording alone (pipeline.yes_side_for): the stage shows YES/NO on their
            # real sides while the fly is still deciding, so nothing swaps when the verdict lands.
            "yes_side": pipeline.yes_side_for(thinking_row["question"]),
        }
    last_row = live_store.latest_answered()
    last = (
        summary.answer_summary(json.loads(last_row["answer_json"]), likes=last_row["likes"])
        if last_row is not None
        else None
    )
    response.headers["Cache-Control"] = "public, max-age=0, s-maxage=2"
    return {
        "thinking": thinking,
        "last": last,
        "queue_length": live_store.queue_length(),
        "claimed_total": live_store.claimed_total(),
        "avg_cycle_s": live_store.avg_cycle_s(),
    }


@app.get("/api/feed")
async def feed(request: Request, response: Response, limit: int = 20):
    live_store: store.LiveStore = request.app.state.store
    limit = max(1, min(limit, 50))  # cap at 50 -- no spike data in here regardless
    recent = [
        summary.answer_summary(json.loads(row["answer_json"]), likes=row["likes"])
        for row in live_store.answered_page("recent", limit)
    ]
    response.headers["Cache-Control"] = "public, max-age=0, s-maxage=3"
    return {"queue_length": live_store.queue_length(), "recent": recent}


@app.get("/api/questions")
async def questions(request: Request, response: Response, sort: str = "recent", limit: int = 20, offset: int = 0):
    """The archive every visitor can browse (2026-09-19: "a big list of previous
    questions, and the most liked ones kept around"): `sort=recent` is newest first, `sort=top` is
    most liked first. Deliberately free of anything per-visitor -- which questions THIS client
    liked lives in its own localStorage -- so the whole page is one CDN-cacheable response for
    every tab asking for it, the same reasoning as `/api/feed`.

    Embargoed questions appear with their verdict withheld (`summary.answer_summary`), exactly as
    they do in the feed: the question is public, the answer is not yet."""
    live_store: store.LiveStore = request.app.state.store
    if sort not in ("recent", "top"):
        raise HTTPException(status_code=422, detail="sort must be 'recent' or 'top'")
    limit = max(1, min(limit, config.QUESTIONS_PAGE_MAX))
    offset = max(0, offset)
    rows = live_store.answered_page(sort, limit, offset)
    total = live_store.answered_count()
    response.headers["Cache-Control"] = "public, max-age=0, s-maxage=10"
    return {
        "sort": sort,
        "total": total,
        "has_more": offset + len(rows) < total,
        "items": [summary.answer_summary(json.loads(row["answer_json"]), likes=row["likes"]) for row in rows],
    }


@app.post("/api/answers/{id}/like")
async def like_answer(id: int, request: Request, response: Response):
    """Like an answered question, or take the like back (the same call toggles). One like per
    client per question is the likes table's primary key, not a check here, so a double tap or a
    replayed request can never inflate a count."""
    live_store: store.LiveStore = request.app.state.store
    # A STABLE identity (chat's peppered author_id), not LiveStore.hash_ip's daily-rotating salt:
    # with the daily hash, tomorrow's tap on a filled heart would not find yesterday's like and
    # would add a second one instead of taking it back (2026-09-19 external review).
    ip_hash = chat.author_id(_client_ip(request), config.CHAT_ID_PEPPER)
    if not live_store.chat_reserve_attempt(
        f"like:{ip_hash}",
        slowmode_s=config.LIKE_SLOWMODE_S,
        window_s=config.LIKE_RATE_WINDOW_S,
        max_per_window=config.LIKE_RATE_PER_WINDOW,
    ):
        raise HTTPException(status_code=429, detail="too many likes — slow down")
    result = live_store.toggle_like(id, ip_hash)
    if result is None:
        raise HTTPException(status_code=404, detail="no answer for this id")
    likes, liked = result
    response.headers["Cache-Control"] = "no-store"
    return {"id": id, "likes": likes, "liked": liked}


@app.get("/api/events")
async def live_events(request: Request):
    """Subscribes to the ONE process-wide `events.Broadcaster` (`app.state.broadcaster`, started
    in `lifespan`) instead of running its own poll loop -- this route's per-connection work is now
    just `await queue.get()` (O(1) per event) plus the existing disconnect check, no store access
    at all."""
    broadcaster: events.Broadcaster = request.app.state.broadcaster
    _subscriber_id, subscriber_queue = broadcaster.subscribe()

    async def generator():
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    name, data = await asyncio.wait_for(
                        subscriber_queue.get(), timeout=config.LIVE_SSE_HEARTBEAT_S
                    )
                except TimeoutError:
                    # No event within one heartbeat interval -- loop back to the disconnect check;
                    # `sse-starlette`'s own `ping=` below still keeps the connection alive.
                    continue
                yield {"event": name, "data": data}
        except asyncio.CancelledError:
            pass
        finally:
            broadcaster.unsubscribe(_subscriber_id)

    return EventSourceResponse(generator(), ping=config.LIVE_SSE_HEARTBEAT_S)


# -- live chat (docs/LIVE.md "Live chat") -----------------------------------------------------------
# Same /api/events stream broadcasts it (ChatEventTracker above) -- no second SSE endpoint.


def _chat_reject(reason: str, chat_token: str | None = None) -> JSONResponse:
    content: dict = {
        "status": "rejected",
        "reason": reason,
        "message": moderation.CHAT_FRIENDLY_MESSAGES[reason],
    }
    if chat_token:
        content["chat_token"] = chat_token
    return JSONResponse(status_code=422, content=content)


class ChatRequest(BaseModel):
    # No Field(min_length=...) constraints on purpose: every length/format rule here goes through
    # moderation.py so a violation gets OUR structured {"status": "rejected", ...} 422 body, not
    # FastAPI's own differently-shaped validation-error response (see AskRequest above, which
    # does hit that edge case on an empty `question` -- not repeated here deliberately).
    text: str = ""
    nickname: str = ""
    turnstile_token: str = ""
    chat_token: str = ""


class ChatReportRequest(BaseModel):
    chat_token: str = ""


@app.post("/api/chat")
async def post_chat(payload: ChatRequest, request: Request):
    if not _chat_available(request):
        raise HTTPException(status_code=503, detail="chat disabled")

    live_store: store.LiveStore = request.app.state.store
    client_ip = _client_ip(request)
    # F4: author_id is the STABLE identity (HMAC(CHAT_ID_PEPPER, ip), IPv6 normalized to /64) --
    # NOT LiveStore.hash_ip's daily-rotating salt. Bans, rate limiting, duplicate suppression,
    # session tokens, and anonymous handles all key off this now, so none of them evaporate or
    # rotate at a UTC day boundary.
    aid = chat.author_id(client_ip, config.CHAT_ID_PEPPER)

    if live_store.chat_is_banned(aid):
        return _chat_reject("banned")

    state = live_store.chat_state()
    if not state["enabled"]:
        return _chat_reject("chat_paused")

    # F1/F2 (2026-09-18 security review): reserve the rate-limit slot atomically BEFORE anything
    # else that could `await` (Turnstile verification, the LLM classifier) -- a single SQLite
    # transaction (store.py's chat_reserve_attempt) closes the check-then-act race a concurrent
    # burst exploited to bypass slow-mode/the 10-minute cap entirely, and it records the attempt
    # regardless of what happens next, so a client cannot dodge the counter by sending content it
    # knows will be rejected (F2's "unlimited LLM calls").
    if not live_store.chat_reserve_attempt(aid, state["slowmode_s"]):
        return _chat_reject("rate_limited")

    # -- session auth: a valid chat_token (bound to THIS author_id -- F3) skips Turnstile;
    # otherwise verify Turnstile and issue one for every later message this session
    # ("Turnstile is verified on the first message only"). The new token rides along on EVERY
    # response from here on, success or rejection, so a message that gets bounced for content
    # still leaves the client authenticated for its next try.
    new_token: str | None = None
    if payload.chat_token and chat.verify_chat_token(payload.chat_token, config.CHAT_SESSION_SECRET, aid):
        pass
    else:
        if not await moderation.verify_turnstile(payload.turnstile_token, remote_ip=client_ip):
            return _chat_reject("captcha")
        new_token = chat.issue_chat_token(config.CHAT_SESSION_SECRET, aid, config.CHAT_TOKEN_TTL_S)

    text = payload.text.strip()
    dup_result = moderation.check_chat_duplicate(text, live_store.chat_last_text(aid))
    if not dup_result.ok:
        return _chat_reject(dup_result.reason, new_token)

    nickname = payload.nickname.strip()
    if nickname:
        nick_result = moderation.check_nickname(nickname)
        if not nick_result.ok:
            return _chat_reject(nick_result.reason, new_token)
    else:
        nickname = chat.anonymous_handle(aid)

    content_result = await moderation.moderate_chat_text(
        text,
        # R2 (round-2 security review): a per-author quota, checked before the site-wide breaker,
        # so a handful of identities can't keep everyone else's "chat_busy" tripped. Reuses the
        # same chat_reserve_attempt path as the rate limiter above, namespaced so it can never
        # collide with a real ip_hash/author_id or the "report:" namespace below.
        reserve_author_llm_call=lambda: live_store.chat_reserve_attempt(
            f"llm:{aid}",
            slowmode_s=0.0,
            window_s=config.CHAT_LLM_BREAKER_WINDOW_S,
            max_per_window=config.CHAT_LLM_PER_AUTHOR_PER_MINUTE,
        ),
        reserve_llm_call=live_store.chat_reserve_llm_call,
    )
    if not content_result.ok:
        return _chat_reject(content_result.reason, new_token)

    id_ = live_store.chat_send(nickname, text, aid)
    row = live_store.chat_get(id_)
    resp: dict = {
        "status": "ok",
        "id": id_,
        "nickname": nickname,
        "text": text,
        "created_at": row["created_at"],
    }
    if new_token:
        resp["chat_token"] = new_token
    return resp


@app.get("/api/chat")
async def get_chat(
    request: Request, limit: int = Query(100, ge=1, le=200), before_id: int | None = Query(None, gt=0)
):
    """`before_id` given -> a history page strictly older than that id ("for history
    paging"); omitted -> the latest `limit` messages, unchanged behaviour. Both branches return the
    same shape, `has_more` included either way, so the frontend's "load older messages" affordance
    works identically regardless of which call first populated the page."""
    if not _chat_available(request):
        raise HTTPException(status_code=503, detail="chat disabled")
    live_store: store.LiveStore = request.app.state.store
    rows = live_store.chat_before(before_id, limit) if before_id is not None else live_store.chat_recent(limit)
    has_more = live_store.chat_has_more_before(rows[0]["id"]) if rows else False
    return {
        "messages": [
            {"id": r["id"], "nickname": r["nickname"], "text": r["text"], "created_at": r["created_at"]}
            for r in rows
        ],
        "state": live_store.chat_state(),
        "has_more": has_more,
    }


@app.post("/api/chat/{id}/report")
async def report_chat(id: int, payload: ChatReportRequest, request: Request):
    if not _chat_available(request):
        raise HTTPException(status_code=503, detail="chat disabled")

    live_store: store.LiveStore = request.app.state.store
    client_ip = _client_ip(request)
    aid = chat.author_id(client_ip, config.CHAT_ID_PEPPER)

    # F3: banned clients cannot report (previously unchecked -- a banned troll kept a valid token
    # and could keep reporting other people's messages).
    if live_store.chat_is_banned(aid):
        return _chat_reject("banned")

    claims = chat.verify_chat_token(payload.chat_token, config.CHAT_SESSION_SECRET, aid)
    if claims is None:
        return _chat_reject("captcha")

    row = live_store.chat_get(id)
    if row is None:
        raise HTTPException(status_code=404, detail="not found")

    # F3: a report only counts toward the auto-hide tally if the reporter's session is at least
    # CHAT_REPORT_MIN_SESSION_AGE_S old AND they have >=1 accepted message -- closes "solve
    # Turnstile once, report from a few IPs" instant censorship. The call still succeeds (no error
    # surfaced to the client, no hint about WHY it didn't count -- avoids leaking the eligibility
    # rule to a would-be abuser), it just doesn't move the tally.
    session_age_s = time.time() - claims.issued_at
    eligible = (
        session_age_s >= config.CHAT_REPORT_MIN_SESSION_AGE_S
        and live_store.chat_has_accepted_message(aid)
    )
    if not eligible:
        return {
            "status": "ok",
            "id": id,
            "reports": live_store.chat_report_count(id),
            "auto_hidden": bool(row["deleted"] and row["deleted_reason"] == "reported"),
        }

    # F3: "reporters on the same IPv4 /24 or IPv6 /64 count once" -- a COARSER grouping than
    # author_id (whose IPv4 side is the full address), used only for this dedupe.
    group_id = chat.report_group_id(client_ip, config.CHAT_ID_PEPPER)
    # R1 (round-2 security review): round 1's eligibility rules alone still let 5 colluding,
    # patient identities hide unlimited messages -- reproduced 20/20. Two independent caps, BOTH
    # checked only once the report threshold is actually crossed (so an ordinary report that
    # doesn't tip a message over never spends either budget): a per-reporter quota (reused
    # through the same atomic chat_reserve_attempt path, namespaced "report:{author_id}") and a
    # site-wide cap on how many auto-hides can happen in one window. The report is ALWAYS
    # recorded regardless -- only the hide is withheld -- so a message that hits either cap still
    # shows an elevated report count for operator review (`chat_admin list`).
    count = live_store.chat_report(id, group_id)
    auto_hidden = False
    if count >= config.CHAT_REPORT_THRESHOLD and not row["deleted"]:
        reporter_under_cap = live_store.chat_reserve_attempt(
            f"report:{aid}",
            slowmode_s=0.0,
            window_s=config.CHAT_RATE_WINDOW_S,
            max_per_window=config.CHAT_REPORTS_PER_10MIN,
        )
        site_under_cap = (
            live_store.chat_auto_hide_count_recent(config.CHAT_RATE_WINDOW_S)
            < config.CHAT_AUTO_HIDE_PER_10MIN
        )
        if reporter_under_cap and site_under_cap:
            auto_hidden = live_store.chat_auto_hide(id)
            if auto_hidden:
                # F3: "logged" -- goes into an operator review queue (`chat_admin list
                # --reported`) and is restorable (`chat_admin restore <id>`), but the auto-hide
                # event itself must be visible in the log too, not just discoverable by
                # remembering to check the CLI.
                logger.info("chat auto-hide id=%s reports=%s", id, count)
        else:
            # A cap held the hide back. Without this line a burned site-wide cap would silently
            # swallow honest reports on real abuse (security review round 3, S1).
            logger.warning(
                "chat auto-hide WITHHELD id=%s reports=%s reporter_under_cap=%s site_under_cap=%s",
                id, count, reporter_under_cap, site_under_cap,
            )
    return {"status": "ok", "id": id, "reports": count, "auto_hidden": auto_hidden}


@app.get("/api/card/{id}.png")
async def get_card(id: int, request: Request):
    live_store: store.LiveStore = request.app.state.store
    row = live_store.get(id)
    if row is None or row["status"] != "answered":
        raise HTTPException(status_code=404, detail="no answer for this id")
    answer = json.loads(row["answer_json"])
    png_bytes = card.render_card(answer)
    return Response(content=png_bytes, media_type="image/png")


@app.get("/a/{id}", response_class=HTMLResponse)
async def share_page(id: int, request: Request):
    live_store: store.LiveStore = request.app.state.store
    row = live_store.get(id)
    if row is None or row["status"] != "answered":
        raise HTTPException(status_code=404, detail="no answer for this id")
    answer = json.loads(row["answer_json"])
    base_url = str(request.base_url).rstrip("/")
    return card.render_share_html(answer, base_url)


@app.get("/api/stats")
async def get_stats(request: Request, response: Response):
    live_store: store.LiveStore = request.app.state.store
    result = stats.compute_stats(live_store.all_answered(), live_store.earliest_created_at())
    # Guesses live in their own table (live_guesses), not inside any answer_json row, so they are
    # merged in here rather than threaded through stats.compute_stats's pure aggregation.
    result["guesses"] = live_store.guess_stats()
    response.headers["Cache-Control"] = "public, max-age=0, s-maxage=10"
    return result


@app.get("/api/validation")
async def get_validation(request: Request, response: Response):
    """Brain passport, calibration, lateral-validation verdict, handedness b0 and OpenTimestamps
    stamps — computed once on first request and cached on `app.state` (graph
    metadata/calibration/stamps do not change without a redeploy; the ledger read is cheap
    enough to not need its own cache, but is bundled with everything else here for one
    `/api/validation` call)."""
    if request.app.state.validation is None:
        request.app.state.validation = validation.build_validation()
    response.headers["Cache-Control"] = "public, max-age=0, s-maxage=300"
    return request.app.state.validation


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}
