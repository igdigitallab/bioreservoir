"""Env vars and tunables for the live fly page (names only — see docs/LIVE.md, never commit
values). Read once per process at import time via `os.environ`, same convention as
`oracle.encode.load_encoder`'s `HF_HUB_OFFLINE` defaults: every value here has a safe default so
`import bioreservoir.live.config` never raises just because an operator hasn't set every var yet.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_DIR = REPO_ROOT / "experiments" / "001-fly-oracle"
LIVE_STATES_YAML = EXPERIMENT_DIR / "live-states.yaml"

# Reference (read-only) to the main experiment's ledger, for `oracle.handedness.load_b0` —
# never opened for writing, and only if it exists (worker.py falls back to computing its own b0
# from reference.yaml if this file is absent or has no matching (malecns, real) reference runs
# yet).
MAIN_LEDGER_PATH = REPO_ROOT / "experiments" / "001-fly-oracle" / "ledger.sqlite"

# -- storage -----------------------------------------------------------------------------------
LIVE_DB = Path(os.environ.get("LIVE_DB", str(REPO_ROOT / "data" / "live" / "live.sqlite")))
B0_CACHE_PATH = Path(
    os.environ.get("LIVE_B0_CACHE", str(REPO_ROOT / "data" / "live" / "b0_cache.json"))
)

# -- environment ---------------------------------------------------------------------------------
LIVE_ENV = os.environ.get("LIVE_ENV", "prod")  # "dev" | "prod"

# -- moderation ------------------------------------------------------------------------------------
MAX_QUESTION_LENGTH = 140
RATE_LIMIT_PER_MINUTE = int(os.environ.get("LIVE_RATE_PER_MINUTE", "3"))
RATE_LIMIT_PER_DAY = int(os.environ.get("LIVE_RATE_PER_DAY", "30"))
QUEUE_CAP_PER_IP = int(os.environ.get("LIVE_QUEUE_CAP_PER_IP", "3"))
REJECTED_RETENTION_DAYS = 7

TURNSTILE_SECRET = os.environ.get("TURNSTILE_SECRET")
TURNSTILE_VERIFY_URL = "https://challenges.cloudflare.com/turnstile/v0/siteverify"

LLM_BASE_URL = os.environ.get("LLM_BASE_URL")
LLM_API_KEY = os.environ.get("LLM_API_KEY")
LLM_MODEL = os.environ.get("LLM_MODEL", "auto-json")
LLM_TIMEOUT_S = float(os.environ.get("LLM_TIMEOUT_S", "8.0"))

# -- provenance ------------------------------------------------------------------------------------
# `oracle.ledger.git_sha()` needs a `.git` checkout, but the deployed container never has one
# (the deploy image only copies `src/`/`experiments/`/`pyproject.toml` out of a verified
# checkout — no `.git`). The deploy script already knows and verifies the exact commit it is
# deploying and passes it in as a build ARG -> env var, so
# `worker.build_resources()` has a real commit to fall back to instead of `None`
# (the alternative would be shipping `provenance.code_sha` as null in production).
CODE_SHA_FALLBACK = os.environ.get("BIORESERVOIR_CODE_SHA")

# -- worker / simulation -------------------------------------------------------------------------
LIVE_BRAIN = "malecns"  # always the real male brain (MaleCNS) — live mode never runs BANC.
LIVE_N_TRIALS = 3
LIVE_SIDE_BASE = int(os.environ.get("LIVE_SIDE_BASE", "20260918777"))  # distinct from exp 001's
# config.yaml seed.side_base (20260918107) on purpose: live questions are not the 33
# pre-registered scored questions, and must never share their per-question coin sequence.
# ⚠️ api and worker must read the SAME value: the api publishes the YES side with the question
# (pipeline.yes_side_for -> /api/now, /api/answers/{id}, SSE thinking) and the worker records it
# in the answer. Divergent env between the two containers = labels that swap at the verdict.
# The deployed stack gives both services one shared env_file, so keep it out of per-service env.
LIVE_POLL_INTERVAL_S = float(os.environ.get("LIVE_POLL_INTERVAL_S", "1.0"))
LIVE_SSE_HEARTBEAT_S = float(os.environ.get("LIVE_SSE_HEARTBEAT_S", "15.0"))

# -- worker liveness (2026-09-19 logic review F3/F4) ---------------------------------------------
# `run_loop` touches this file on every iteration (idle poll or a finished question) -- a plain
# `pgrep -f bioreservoir.live.worker` stays green even when the process is wedged (e.g. hung on
# an ER subprocess with no timeout, the exact bug this review found), because the parent process
# itself is still alive. The compose healthcheck instead checks this file's mtime for freshness;
# see docs/LIVE.md's "The game" section for the exact command and the threshold's reasoning.
WORKER_HEARTBEAT_PATH = Path(
    os.environ.get("LIVE_WORKER_HEARTBEAT", str(REPO_ROOT / "data" / "live" / "worker.heartbeat"))
)

# -- the game: random-graph control timeouts (F3/F4/F5) -------------------------------------------
# One-time startup warm-up (game.ErContenderPool.warm_up): generous, since it includes the ER
# subprocess's own one-time Brian2/cython compile (~15s measured on a real MaleCNS-scale network,
# docs/LIVE.md) plus process spawn -- failing this should mean "actually broken", not "a bit slow".
# The "which one is the real fly?" game (game.py) is OFF by default: it was removed from
# the page on 2026-09-19 (three anonymous answers after every question read as clutter). Off means
# the worker never builds the random-graph contender (no extra ~1.9 GB, no extra CPU per question)
# and answers carry no `game` block, so the page shows the plain answer.
LIVE_GAME_ENABLED = os.environ.get("LIVE_GAME", "0") == "1"

ER_WARMUP_TIMEOUT_S = float(os.environ.get("LIVE_ER_WARMUP_TIMEOUT_S", "180.0"))

# -- the game: guess counting + rate limiting (2026-09-19 logic review F2) ------------------------
# Only the ASKER's own first guess on their OWN question counts toward the site-wide "visitors
# spot the real brain X% of the time" stat (api.py's POST /api/guess) -- anyone can still guess
# (share links, the public feed), but api.py marks those `counted: false`. A guess must be within
# this many seconds of the answer to count, closing the "look up the answer elsewhere, then guess"
# loophole; `LiveStore.hash_ip`'s per-UTC-day salt makes the asker/guesser ip_hash comparison exact
# only within the same day, an accepted limitation right at the UTC midnight boundary.
GUESS_COUNT_WINDOW_S = float(os.environ.get("LIVE_GUESS_COUNT_WINDOW_S", str(15 * 60)))
# Rate limit for POST /api/guess itself, reusing store.py's chat_reserve_attempt pattern
# (namespaced "guess:{ip_hash}" key on the SAME live_chat_attempts table, same convention as the
# chat report/LLM-quota namespacing) -- generous enough for a real visitor guessing on several
# questions, tight enough to block a fast id-enumeration loop.
GUESS_SLOWMODE_S = float(os.environ.get("LIVE_GUESS_SLOWMODE_S", "1.0"))
GUESS_RATE_WINDOW_S = 60.0
GUESS_RATE_PER_WINDOW = int(os.environ.get("LIVE_GUESS_RATE_PER_MINUTE", "10"))

# Likes on answered questions (POST /api/answers/{id}/like) and the questions list
# (GET /api/questions). The rate limit reuses store.chat_reserve_attempt with a "like:{ip_hash}"
# namespace, same as guesses -- one like per client per question is already enforced by the
# table's primary key, so this only exists to stop a loop from spraying likes across every id.
# No slowmode: liking and immediately unliking is a normal double tap, and the per-question
# primary key already makes a fast loop pointless. The per-minute cap below is the guard.
LIKE_SLOWMODE_S = float(os.environ.get("LIVE_LIKE_SLOWMODE_S", "0"))
LIKE_RATE_WINDOW_S = 60.0
LIKE_RATE_PER_WINDOW = int(os.environ.get("LIVE_LIKE_RATE_PER_MINUTE", "30"))
QUESTIONS_PAGE_MAX = 50

# -- frames (atlas visualisation) ------------------------------------------------------------------
FRAMES_BIN_MS = 25.0
FRAMES_MAX_ACTIVE_PER_BIN = 6000
ATLAS_DIR = REPO_ROOT / "site" / "public" / "data" / "atlas"

# -- share card ------------------------------------------------------------------------------------
# Election embargo, ON 2026-09-19 -> OFF 2026-09-21: the site is a fly answering
# visitors' questions; nothing on it is a forecast from the lab, so a visitor's election question is just
# another visitor question). While it was on, the gate read as the exact opposite of that -- the
# public archive showed "HELD BACK" on three of its seven rows -- and it was visibly arbitrary:
# `Will Obama win 2028?` was held while `Will Kamala win 2028?` published its YES, because the
# name list below has one and not the other.
#
# What replaces it: the disclaimer rides on the viral surface itself (a permanent caveat line on
# the share-card PNG and in its OG description) instead of hiding the verdict -- label, do not
# conceal. Flipping this back to True re-arms the whole mechanism, which is still tested end to
# end (tests/test_live_card.py, test_live_api.py, test_live_events.py).
ELECTION_CARD_EMBARGO_ENABLED = False
ELECTION_CARD_EMBARGO_UNTIL = "2026-11-04T17:00:00+00:00"

CARD_WIDTH = 1200
CARD_HEIGHT = 630

# -- live chat ---------------------------------------------------------------------------------
# Hard, deploy-time kill switch: off -> POST/GET /api/chat return 503 and the SSE
# stream never wires up a ChatEventTracker at all. Distinct from the SOFT, runtime-adjustable
# pause an operator flips with `chat_admin off`/`on` (store.py's `live_chat_state.enabled`,
# reacted to live over the `chat_state` SSE event) -- this env var is the belt-and-suspenders
# override that is not expected to change without a redeploy.
LIVE_CHAT_ENABLED = os.environ.get("LIVE_CHAT_ENABLED", "1") != "0"

CHAT_MAX_LENGTH = 200
NICKNAME_MIN_LENGTH = 2
NICKNAME_MAX_LENGTH = 20

# Seed value only for `live_chat_state.slowmode_s` on first `LiveStore` init at this path -- after
# that, `chat_admin slowmode <seconds>` owns the live value (store.py's `chat_state()`); this env
# var never overrides a value already persisted in LIVE_DB.
CHAT_DEFAULT_SLOWMODE_S = float(os.environ.get("CHAT_DEFAULT_SLOWMODE_S", "4.0"))
CHAT_RATE_WINDOW_S = 600.0
CHAT_RATE_PER_WINDOW = int(os.environ.get("CHAT_RATE_PER_10MIN", "15"))
# Raised 3 -> 5 (2026-09-18 security review F3): 3 identities could otherwise censor any message.
CHAT_REPORT_THRESHOLD = int(os.environ.get("CHAT_REPORT_THRESHOLD", "5"))
# F3: a report only counts toward CHAT_REPORT_THRESHOLD if the reporter's session token is at
# least this old AND the reporter has >=1 accepted message -- closes the "solve Turnstile once,
# report from a few IPs" flood-censorship exploit the security review reproduced.
CHAT_REPORT_MIN_SESSION_AGE_S = float(os.environ.get("CHAT_REPORT_MIN_SESSION_AGE_S", "600"))

# R1 (round-2 security review): round 1's per-message report threshold + eligibility rules still
# let 5 colluding identities hide unlimited messages, 5 requests apiece, forever -- a per-author
# report-ACTION rate limit (reused through chat_reserve_attempt with a "report:{author_id}"
# namespaced key) plus a site-wide cap on how many auto-hides can happen in one window. Beyond
# either cap the report still gets recorded (api.py's report_chat) but does not trigger a hide --
# it just raises that message's report count for `chat_admin list` review.
CHAT_REPORTS_PER_10MIN = int(os.environ.get("CHAT_REPORTS_PER_10MIN", "5"))
CHAT_AUTO_HIDE_PER_10MIN = int(os.environ.get("CHAT_AUTO_HIDE_PER_10MIN", "10"))

# F2/R2: site-wide circuit breaker on chat LLM classification calls, independent of any single
# client's identity -- caps the shared LiteLLM spend/rate-limit exposure from ANY combination of
# clients hammering POST /api/chat with content that always reaches the LLM step. Round-2 review:
# 30/min tripped under 35 ordinary concurrent chatters (12 false "chat_busy"); raised to 200 --
# the LiteLLM virtual key's own budget is the real spend backstop. A SEPARATE, smaller per-author
# quota (checked BEFORE this global one) stops a handful of identities from keeping the global
# breaker tripped for everyone else.
CHAT_LLM_MAX_PER_MINUTE = int(os.environ.get("CHAT_LLM_MAX_PER_MINUTE", "200"))
CHAT_LLM_BREAKER_WINDOW_S = 60.0
CHAT_LLM_PER_AUTHOR_PER_MINUTE = int(os.environ.get("CHAT_LLM_PER_AUTHOR_PER_MINUTE", "6"))

# Chat session token TTL ("~1h"). The token is an HMAC-signed
# "<author_id>.<issued_at>.<expires_at>.<sig>" (chat.py's issue_chat_token/verify_chat_token) --
# bound to author_id (F3: "put the client hash in the HMAC payload; reject on mismatch"), which is
# itself derived from CHAT_ID_PEPPER (below), NOT LiveStore.hash_ip's daily-rotating salt, so a
# token/ban issued near a UTC day boundary does not spuriously expire/reset.
CHAT_TOKEN_TTL_S = float(os.environ.get("CHAT_TOKEN_TTL_S", "3600"))

# -- chat secrets (2026-09-18 security review F8: NO default value for any of these three --
# missing any one of them disables chat entirely (503), logged once at process startup by
# api.py's lifespan, rather than silently degrading to an ephemeral or shared fallback). ---------

# HMAC secret for chat session tokens. Generate with `openssl rand -hex 32` (see docs/LIVE.md).
CHAT_SESSION_SECRET = os.environ.get("CHAT_SESSION_SECRET")

# Server pepper for the stable author identity (chat.py's author_id) used for bans, reports, and
# anonymous handles -- HMAC-SHA256(CHAT_ID_PEPPER, normalized_ip), IPv6 normalized to /64. Never
# derived from LiveStore.hash_ip's per-UTC-day salt (F4: bans/handles must not evaporate/rotate at
# midnight UTC -- 4pm PST on US election day). Generate with `openssl rand -hex 32`.
CHAT_ID_PEPPER = os.environ.get("CHAT_ID_PEPPER")

# Chat's OWN LiteLLM virtual key (F2) -- NEVER falls back to LLM_API_KEY (the question pipeline's
# key), so chat abuse exhausting its own gateway budget/rate-limit cannot take POST /api/ask down
# too. Same LLM_BASE_URL/LLM_MODEL gateway, different key -> the gateway enforces the budget split.
CHAT_LLM_API_KEY = os.environ.get("CHAT_LLM_API_KEY")


def chat_secrets_configured() -> tuple[bool, list[str]]:
    """`(ok, missing_names)` -- api.py's lifespan calls this once at startup (F8: "log it once at
    startup") and caches `ok` on `app.state`; every chat route 503s when `not ok`, same as the
    hard LIVE_CHAT_ENABLED kill switch."""
    missing = [
        name
        for name, value in (
            ("CHAT_SESSION_SECRET", CHAT_SESSION_SECRET),
            ("CHAT_ID_PEPPER", CHAT_ID_PEPPER),
            ("CHAT_LLM_API_KEY", CHAT_LLM_API_KEY),
        )
        if not value
    ]
    return (len(missing) == 0, missing)
