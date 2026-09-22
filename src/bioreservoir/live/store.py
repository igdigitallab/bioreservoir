"""SQLite queue + answer store for the live fly page.

One table (`live_questions`) covers the whole lifecycle (queued -> thinking -> answered, or
rejected at submission time — see `api.py`: rejected attempts never reach the client with an id,
this table keeps them only for the retention-window audit trail).
`live_ip_salts` holds one random, server-generated salt per UTC day, used to hash every IP this
process ever sees (`hash_ip`) — no raw IP is ever written to either table, which is what the
site's Privacy Policy states and what `moderation.py` implements on the other side.

WAL + a busy timeout, same convention as `oracle.ledger.Ledger` (this module's sibling): the API
process and the worker process each open their own connection against the same on-disk file.

`live_chat_*` tables (messages, bans, reports, and a single-row runtime-state table) are the live
chat's store, opened against this SAME `LIVE_DB` file and `LiveStore` instance -- no second
database, no second connection. The operator CLI (`chat_admin.py`) and the API process each open
their own short-lived `LiveStore`, same multi-process convention as the queue above.
"""

from __future__ import annotations

import json
import secrets
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Self

from bioreservoir.live import config
from bioreservoir.live.moderation import check_chat_rate_limit
from bioreservoir.live.moderation import hash_ip as _hash_ip

STATUSES = ("queued", "thinking", "answered", "rejected")


class DuplicateActiveQuestion(Exception):
    """Raised by `enqueue` when the `idx_live_questions_active_key` unique partial index rejects a
    concurrent duplicate insert for the same `question_key` while an earlier row with that key is
    still `queued`/`thinking` (task brief: "two identical asks at the same moment must not create
    two runs"). The caller lost a race against another request that committed its own `enqueue`
    between its own `active_twin` check and this call -- it should re-read `active_twin(key)` now
    and join whichever row actually won, exactly the same response shape a non-racing join takes."""

SCHEMA = """
CREATE TABLE IF NOT EXISTS live_questions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question TEXT NOT NULL,
    status TEXT NOT NULL,
    ip_hash TEXT NOT NULL,
    reason TEXT,
    message TEXT,
    created_at TEXT NOT NULL,
    started_at TEXT,
    answered_at TEXT,
    answer_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_live_questions_status ON live_questions(status);
CREATE INDEX IF NOT EXISTS idx_live_questions_ip_created ON live_questions(ip_hash, created_at);
-- `started_at IS NOT NULL` is `claimed_total`'s own WHERE clause (below) and `thinking_since`'s
-- lower bound -- both scale with total questions ever claimed, not just the current queue, so this
-- index keeps them cheap once the store has millions of rows (scaling task brief).
CREATE INDEX IF NOT EXISTS idx_live_questions_started ON live_questions(started_at);

-- One like per (answered question, client). `ip_hash` is the same daily-rotating salted hash the
-- queue and the game use -- a like is a popularity signal, not a moderation target, so it needs no
-- stable identity, and the daily rotation means yesterday's visitor can like again today (they are
-- a new client as far as this table is concerned). The count itself is denormalized onto
-- `live_questions.likes` (added by `_migrate_columns`) so the "most liked" list is one indexed
-- ORDER BY instead of a GROUP BY over every like ever given.
CREATE TABLE IF NOT EXISTS live_question_likes (
    answer_id INTEGER NOT NULL,
    ip_hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (answer_id, ip_hash)
);
CREATE INDEX IF NOT EXISTS idx_likes_ip_created ON live_question_likes(ip_hash, created_at);

CREATE TABLE IF NOT EXISTS live_ip_salts (
    day TEXT PRIMARY KEY,
    salt TEXT NOT NULL
);

-- The "which one is the real fly?" game (live/game.py): one guess per (answer, client) -- task
-- brief. `ip_hash` is `LiveStore.hash_ip`'s daily-rotating salted hash (same convention as
-- `live_questions`, NOT chat's stable author_id -- a guess is a one-off honesty stat, not a
-- moderation/ban target, so the simpler queue-style hash is enough).
-- `counted` (2026-09-19 logic review F2): whether this guess counts toward the SITE-WIDE
-- "visitors spot the real brain X% of the time" stat -- api.py sets it True only for the
-- QUESTION'S OWN ASKER (ip_hash matches `live_questions.ip_hash` for this answer_id) guessing
-- within `config.GUESS_COUNT_WINDOW_S` of the answer. Anyone else can still guess (share links,
-- the public feed), the row is still stored (so a repeat call stays idempotent either way), it
-- just never counts toward the published percentage -- otherwise `for id in 1..N: GET answer;
-- POST guess real_slot` drives the number to ~100% from one IP, since the payload (order,
-- per-contender answers) is visible to anyone who can read the Answer JSON.
CREATE TABLE IF NOT EXISTS live_guesses (
    answer_id INTEGER NOT NULL,
    ip_hash TEXT NOT NULL,
    pick TEXT NOT NULL,
    correct INTEGER NOT NULL,
    counted INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    PRIMARY KEY (answer_id, ip_hash)
);

CREATE TABLE IF NOT EXISTS live_chat_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    nickname TEXT NOT NULL,
    text TEXT NOT NULL,
    ip_hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    deleted INTEGER NOT NULL DEFAULT 0,
    deleted_at TEXT,
    deleted_reason TEXT,
    restored_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_chat_created ON live_chat_messages(created_at);
CREATE INDEX IF NOT EXISTS idx_chat_ip_created ON live_chat_messages(ip_hash, created_at);
CREATE INDEX IF NOT EXISTS idx_chat_deleted_at ON live_chat_messages(deleted_at);
CREATE INDEX IF NOT EXISTS idx_chat_restored_at ON live_chat_messages(restored_at);

CREATE TABLE IF NOT EXISTS live_chat_bans (
    ip_hash TEXT PRIMARY KEY,
    banned_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS live_chat_reports (
    message_id INTEGER NOT NULL,
    ip_hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (message_id, ip_hash)
);

-- F1/F2 (2026-09-18 security review): one row per POST /api/chat ATTEMPT (accepted, rejected, or
-- failed downstream -- inserted BEFORE moderation ever runs), keyed by author_id. `chat_reserve_
-- attempt` checks-and-inserts inside a single BEGIN IMMEDIATE transaction so a concurrent burst
-- from the same client cannot all observe an under-threshold count before any of them commits
-- (F1's race), and every attempt -- not just accepted messages -- counts against the slow-mode
-- interval and 10-minute cap (F2's "unlimited LLM calls via guaranteed-rejected content").
CREATE TABLE IF NOT EXISTS live_chat_attempts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ip_hash TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chat_attempts_ip_created ON live_chat_attempts(ip_hash, created_at);

-- F2: site-wide circuit breaker on chat LLM classification calls, independent of any one client's
-- identity -- `chat_reserve_llm_call` checks-and-inserts atomically the same way.
CREATE TABLE IF NOT EXISTS live_chat_llm_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chat_llm_calls_created ON live_chat_llm_calls(created_at);

-- Single-row table (id is always 1): the operator-adjustable slow-mode interval and the runtime
-- (soft) kill switch -- `chat_admin slowmode <seconds>` / `chat_admin off|on`. Distinct from
-- config.LIVE_CHAT_ENABLED, the hard env-level switch that api.py checks separately and that
-- requires a restart to flip (see config.py's chat section).
CREATE TABLE IF NOT EXISTS live_chat_state (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    enabled INTEGER NOT NULL DEFAULT 1,
    slowmode_s REAL NOT NULL DEFAULT 4.0,
    updated_at TEXT
);
"""


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _iso(dt: datetime) -> str:
    return dt.isoformat()


@dataclass
class LiveStore:
    path: Path = field(default_factory=lambda: config.LIVE_DB)
    _conn: sqlite3.Connection | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, timeout=30.0)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA busy_timeout = 30000")
        self._conn.executescript(SCHEMA)
        self._migrate_columns()
        self._conn.execute(
            "INSERT OR IGNORE INTO live_chat_state (id, enabled, slowmode_s) VALUES (1, 1, ?)",
            (config.CHAT_DEFAULT_SLOWMODE_S,),
        )
        self._conn.commit()

    def _migrate_columns(self) -> None:
        """Adds `live_questions.question_key` (task brief's `POST /api/ask` dedupe) in place on an
        existing DB -- `sqlite3` has no `ALTER TABLE ... ADD COLUMN IF NOT EXISTS`, so this checks
        `PRAGMA table_info` first, same idempotent-on-every-open shape `SCHEMA`'s own
        `CREATE TABLE/INDEX IF NOT EXISTS` statements already use for a brand-new table. Existing
        rows are backfilled with `pipeline.question_key(question)` (imported lazily -- `pipeline`
        pulls in `oracle.handedness`/`oracle.side_mapping`/`oracle.readout`, modules `store.py`
        itself has no other reason to import at module load, matching `api.py`'s own lazy import of
        the same function in `_is_first_answer_for_question`) so a question asked before this
        column existed can still be recognized as a repeat/join target afterwards.

        The two indexes created here can always run unconditionally (`CREATE INDEX IF NOT EXISTS`
        is a no-op once they exist): a plain index for `answered_twin`/`rejected_twin`'s lookups,
        and a UNIQUE index scoped to `status IN ('queued', 'thinking')` (a partial index, SQLite
        feature) that is the actual race guard task brief asks for -- two concurrent `POST
        /api/ask` calls for the same wording can both pass every earlier check and both attempt an
        `enqueue`, but only one INSERT can ever satisfy this constraint while its twin is still
        active; the loser's `enqueue` raises `DuplicateActiveQuestion` (see below) and the caller
        re-reads `active_twin` to join whichever request actually won. A row leaving `queued`/
        `thinking` (answered or rejected) frees its key for a brand-new question to reuse without
        colliding with its own now-terminal history."""
        # `BEGIN IMMEDIATE` (same convention as `claim_next`/`chat_reserve_attempt`): every
        # process opening this same DB file runs this same check-then-ALTER sequence at startup,
        # so without an explicit write lock two connections racing to open a brand-new file could
        # both see "column missing" before either commits its own `ALTER TABLE`, and the loser's
        # ALTER raises `OperationalError: duplicate column name` (reproduced by this project's own
        # `test_chat_reserve_attempt_is_atomic_under_real_concurrency`, which opens 40 real
        # connections against one fresh file at once). SQLite's DDL is fully transactional, so
        # wrapping the ALTER + backfill in the same transaction as the lock is enough: the loser
        # blocks on `busy_timeout` until the winner commits, then re-checks and finds the column
        # already there.
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            columns = {row["name"] for row in self._conn.execute("PRAGMA table_info(live_questions)")}
            if "needs_llm" not in columns:
                # Set when the classifier was busy at ask time (moderation.LLM_BUSY): the worker
                # classifies such a question right before simulating it, so a crowd hitting the
                # gateway's per-minute cap does not turn normal questions into "spam" rejections.
                self._conn.execute("ALTER TABLE live_questions ADD COLUMN needs_llm INTEGER NOT NULL DEFAULT 0")
            if "likes" not in columns:
                # Denormalized like count (the "most liked questions" tab). Existing rows start at
                # 0, which is the truth: the likes table is created empty alongside this column.
                self._conn.execute("ALTER TABLE live_questions ADD COLUMN likes INTEGER NOT NULL DEFAULT 0")
            if "question_key" not in columns:
                self._conn.execute("ALTER TABLE live_questions ADD COLUMN question_key TEXT")
                from bioreservoir.live.pipeline import question_key as _question_key

                rows = self._conn.execute("SELECT id, question FROM live_questions").fetchall()
                for row in rows:
                    self._conn.execute(
                        "UPDATE live_questions SET question_key = ? WHERE id = ?",
                        (_question_key(row["question"]), row["id"]),
                    )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_live_questions_key ON live_questions(question_key)"
        )
        # The "most liked" page: ties broken by id so the order is total and paging cannot repeat
        # or skip a row between two requests with the same like counts.
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_live_questions_likes "
            "ON live_questions(likes DESC, id DESC) WHERE status = 'answered'"
        )
        self._conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_live_questions_active_key "
            "ON live_questions(question_key) WHERE status IN ('queued', 'thinking')"
        )
        self._conn.commit()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- IP hashing (daily, random, server-side salt) -------------------------------------------

    def hash_ip(self, raw_ip: str, now: datetime | None = None) -> str:
        """Daily-salted IP hash (moderation.py's `hash_ip`) — the salt is a real random secret
        generated on first use each UTC day and persisted in `live_ip_salts`, not derived from
        the date string itself, so a hash cannot be reversed just by knowing what day it is."""
        day = (now or _utcnow()).strftime("%Y-%m-%d")
        row = self._conn.execute(
            "SELECT salt FROM live_ip_salts WHERE day = ?", (day,)
        ).fetchone()
        if row is None:
            salt = secrets.token_hex(32)
            self._conn.execute(
                "INSERT INTO live_ip_salts (day, salt) VALUES (?, ?) ON CONFLICT(day) DO NOTHING",
                (day, salt),
            )
            self._conn.commit()
            row = self._conn.execute(
                "SELECT salt FROM live_ip_salts WHERE day = ?", (day,)
            ).fetchone()
        return _hash_ip(raw_ip, row["salt"])

    # -- rate limiting (moderation.py's step 3 needs these counts) -------------------------------

    def count_recent(self, ip_hash: str, window_seconds: float, now: datetime | None = None) -> int:
        since = _iso((now or _utcnow()) - timedelta(seconds=window_seconds))
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM live_questions WHERE ip_hash = ? AND created_at >= ?",
            (ip_hash, since),
        ).fetchone()
        return row["n"]

    def count_today(self, ip_hash: str, now: datetime | None = None) -> int:
        start = _iso((now or _utcnow()).replace(hour=0, minute=0, second=0, microsecond=0))
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM live_questions WHERE ip_hash = ? AND created_at >= ?",
            (ip_hash, start),
        ).fetchone()
        return row["n"]

    def count_queued(self, ip_hash: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM live_questions WHERE ip_hash = ? AND status IN "
            "('queued', 'thinking')",
            (ip_hash,),
        ).fetchone()
        return row["n"]

    # -- writes ------------------------------------------------------------------------------------

    def enqueue(
        self,
        question: str,
        ip_hash: str,
        question_key: str | None = None,
        now: datetime | None = None,
        needs_llm: bool = False,
    ) -> int:
        """`question_key` defaults to `pipeline.question_key(question)` computed here if the
        caller doesn't already have one on hand (every pre-existing test/caller passes only
        `question`/`ip_hash`, so this keeps them working unchanged) -- `api.py`'s `POST /api/ask`
        passes the already-normalized key it just used for the dedupe lookup, to avoid computing it
        twice. Raises `DuplicateActiveQuestion` if `idx_live_questions_active_key` (the race guard,
        see `_migrate_columns`'s docstring) rejects this insert because another `queued`/
        `thinking` row with the same key already exists -- expected only when two identical asks
        race past the caller's own `active_twin` check at nearly the same instant."""
        if question_key is None:
            from bioreservoir.live.pipeline import question_key as _question_key

            question_key = _question_key(question)
        try:
            cur = self._conn.execute(
                "INSERT INTO live_questions (question, status, ip_hash, created_at, question_key, needs_llm) "
                "VALUES (?, 'queued', ?, ?, ?, ?)",
                (question, ip_hash, _iso(now or _utcnow()), question_key, 1 if needs_llm else 0),
            )
            self._conn.commit()
        except sqlite3.IntegrityError as exc:
            self._conn.rollback()
            raise DuplicateActiveQuestion(question_key) from exc
        return cur.lastrowid

    def reject(
        self, question: str, ip_hash: str, reason: str, message: str,
        question_key: str | None = None, now: datetime | None = None,
    ) -> int:
        """Records a rejected attempt (audit trail only — `api.py` never returns this row's id to
        the client, matching the task brief's `POST /api/ask` 422 contract, which has no `id`
        field). Question text is purged after `REJECTED_RETENTION_DAYS` by `purge_expired` (which
        also blanks `question_key` at the same time -- see its own docstring). `question_key`
        defaults the same way `enqueue` does, and is what `rejected_twin` looks up later to skip a
        repeat content-moderation/LLM call for a question already rejected once (task brief)."""
        if question_key is None:
            from bioreservoir.live.pipeline import question_key as _question_key

            question_key = _question_key(question)
        cur = self._conn.execute(
            "INSERT INTO live_questions (question, status, ip_hash, reason, message, created_at, "
            "question_key) VALUES (?, 'rejected', ?, ?, ?, ?, ?)",
            (question, ip_hash, reason, message, _iso(now or _utcnow()), question_key),
        )
        self._conn.commit()
        return cur.lastrowid

    def claim_next(self, now: datetime | None = None) -> sqlite3.Row | None:
        """Atomically claim the oldest `queued` row (FIFO) for the worker: moves it to `thinking`
        and returns it, or `None` if the queue is empty. `BEGIN IMMEDIATE` takes the write lock
        up front so two worker processes (there is only ever one in practice, but this makes that
        an operational choice, not a correctness requirement) cannot both claim the same row."""
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            row = self._conn.execute(
                "SELECT * FROM live_questions WHERE status = 'queued' ORDER BY id LIMIT 1"
            ).fetchone()
            if row is None:
                self._conn.commit()
                return None
            self._conn.execute(
                "UPDATE live_questions SET status = 'thinking', started_at = ? WHERE id = ?",
                (_iso(now or _utcnow()), row["id"]),
            )
            self._conn.commit()
            return self.get(row["id"])
        except Exception:
            self._conn.rollback()
            raise

    def record_answer(self, id_: int, answer: dict, now: datetime | None = None) -> None:
        self._conn.execute(
            "UPDATE live_questions SET status = 'answered', answer_json = ?, answered_at = ? "
            "WHERE id = ?",
            (json.dumps(answer), _iso(now or _utcnow()), id_),
        )
        self._conn.commit()

    def reject_claimed(self, id_: int, reason: str, message: str, now: datetime | None = None) -> None:
        """A question the worker took but must not simulate after all -- today only: it was queued
        with `needs_llm` (the classifier was busy when it was asked) and the classification, run
        just before simulating, came back as a rejection. The asker sees it on their own
        `GET /api/answers/{id}`; nothing about it is ever broadcast."""
        self._conn.execute(
            "UPDATE live_questions SET status = 'rejected', reason = ?, message = ?, answered_at = ? WHERE id = ?",
            (reason, message, _iso(now or _utcnow()), id_),
        )
        self._conn.commit()

    def mark_failed(self, id_: int, message: str, now: datetime | None = None) -> None:
        """A trial that raised (e.g. a Brian2 error) does not silently vanish from the queue —
        recorded as `rejected`/`spam` so the client's poll/SSE stream gets a terminal status
        instead of hanging on `thinking` forever."""
        self._conn.execute(
            "UPDATE live_questions SET status = 'rejected', reason = 'spam', message = ?, "
            "answered_at = ? WHERE id = ?",
            (message, _iso(now or _utcnow()), id_),
        )
        self._conn.commit()

    # -- reads -------------------------------------------------------------------------------------

    def get(self, id_: int) -> sqlite3.Row | None:
        return self._conn.execute(
            "SELECT * FROM live_questions WHERE id = ?", (id_,)
        ).fetchone()

    def position(self, id_: int) -> int | None:
        """1-based FIFO rank among currently `queued` rows, or `None` if `id_` is not queued
        (already claimed, answered, rejected, or never existed)."""
        row = self.get(id_)
        if row is None or row["status"] != "queued":
            return None
        rank = self._conn.execute(
            "SELECT COUNT(*) AS n FROM live_questions WHERE status = 'queued' AND id <= ?",
            (id_,),
        ).fetchone()
        return rank["n"]

    def queue_length(self) -> int:
        """Questions WAITING for the worker. The one being simulated right now is not among them:
        the live stage shows it as "deciding" and counting it again read as one extra person in
        line (2026-09-19)."""
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM live_questions WHERE status = 'queued'"
        ).fetchone()
        return row["n"]

    def answered_page(self, sort: str = "recent", limit: int = 20, offset: int = 0) -> list[sqlite3.Row]:
        """One page of answered questions for `GET /api/questions` — `(answer_json, likes)` rows.
        `sort="top"` is most-liked first, `"recent"` is newest first; both break ties by id, so a
        visitor paging through does not see a row twice or miss one between two requests."""
        order = "likes DESC, id DESC" if sort == "top" else "id DESC"
        return self._conn.execute(
            f"SELECT id, answer_json, likes FROM live_questions WHERE status = 'answered' "
            f"ORDER BY {order} LIMIT ? OFFSET ?",
            (max(1, limit), max(0, offset)),
        ).fetchall()

    def answered_count(self) -> int:
        """How many questions have been answered, ever — the questions list shows it as the size
        of the archive and uses it to know whether another page exists."""
        row = self._conn.execute("SELECT COUNT(*) AS n FROM live_questions WHERE status = 'answered'").fetchone()
        return int(row["n"]) if row else 0

    def toggle_like(self, answer_id: int, ip_hash: str, now: datetime | None = None) -> tuple[int, bool] | None:
        """Like an answered question, or take the like back — `(likes, liked_now)`, or `None` if
        there is no answered row with this id. One row per (question, client), so a double tap can
        never inflate the count; the denormalized counter moves inside the same `BEGIN IMMEDIATE`
        transaction as the like row, so it can never drift from the table it counts."""
        stamp = (now or datetime.now(UTC)).isoformat()
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            row = self._conn.execute(
                "SELECT likes FROM live_questions WHERE id = ? AND status = 'answered'", (answer_id,)
            ).fetchone()
            if row is None:
                self._conn.rollback()
                return None
            existing = self._conn.execute(
                "SELECT 1 FROM live_question_likes WHERE answer_id = ? AND ip_hash = ?", (answer_id, ip_hash)
            ).fetchone()
            if existing is None:
                self._conn.execute(
                    "INSERT INTO live_question_likes (answer_id, ip_hash, created_at) VALUES (?, ?, ?)",
                    (answer_id, ip_hash, stamp),
                )
                delta, liked = 1, True
            else:
                self._conn.execute(
                    "DELETE FROM live_question_likes WHERE answer_id = ? AND ip_hash = ?", (answer_id, ip_hash)
                )
                delta, liked = -1, False
            # max(0, ...) keeps a counter that somehow drifted (a hand-edited DB, a restored
            # backup) from going negative and rendering as "-1 likes".
            self._conn.execute(
                "UPDATE live_questions SET likes = MAX(0, likes + ?) WHERE id = ?", (delta, answer_id)
            )
            likes = self._conn.execute("SELECT likes FROM live_questions WHERE id = ?", (answer_id,)).fetchone()["likes"]
            self._conn.commit()
            return int(likes), liked
        except Exception:
            self._conn.rollback()
            raise

    def answered_questions_before(self, id_: int) -> list[sqlite3.Row]:
        """`(id, question)` for every `answered` row with `id < id_` (round-2 logic review R2(a)):
        `api.py`'s `POST /api/guess` uses this to check whether THIS answer is the FIRST answered
        occurrence of its normalized question text -- a repeat or rephrasing-to-the-same-key of a
        question already answered lets a visitor guess with the answer effectively already known
        (the example chips being the common, honest-not-cheating case). Text comparison itself
        (via `pipeline.question_key`) happens in `api.py`, not here -- this method only owns the
        SQL, matching this class's existing split between store-owns-the-query and
        caller-owns-the-business-rule (see `_within_guess_count_window`'s own docstring)."""
        return self._conn.execute(
            "SELECT id, question FROM live_questions WHERE status = 'answered' AND id < ? ORDER BY id",
            (id_,),
        ).fetchall()

    def all_answered(self) -> list[dict]:
        """Every `answered` row's `question`/`answer` (parsed) + `created_at`, oldest first —
        `stats.compute_stats`'s only input (task brief `GET /api/stats`, "no fake values")."""
        rows = self._conn.execute(
            "SELECT answer_json, created_at FROM live_questions WHERE status = 'answered' "
            "ORDER BY id"
        ).fetchall()
        return [{"answer": json.loads(row["answer_json"]), "created_at": row["created_at"]} for row in rows]

    def latest_id(self) -> int:
        """Highest `id` across every row ever seen, or 0 if the store is empty — used by
        `api.py`'s SSE route to seed a fresh `events.LiveEventTracker` at connect time so it only
        streams transitions that happen AFTER the client connects, not a replay of every
        `answered` row in history (which duplicated whatever `GET /api/feed`'s initial load
        already rendered — task brief item 1)."""
        row = self._conn.execute("SELECT MAX(id) AS m FROM live_questions").fetchone()
        return row["m"] or 0

    def earliest_created_at(self) -> str | None:
        """`MIN(created_at)` over every row ever seen (queued, thinking, answered, rejected) —
        `stats.compute_stats`'s `since` field: when this store started seeing traffic at all, not
        just when the first question was answered."""
        row = self._conn.execute("SELECT MIN(created_at) AS m FROM live_questions").fetchone()
        return row["m"]

    def current_thinking(self) -> sqlite3.Row | None:
        """The row currently being answered, or `None` if the worker is idle — `GET /api/now`
        (task brief). A single worker process claims one row at a time, so there is normally at
        most one `thinking` row; `ORDER BY id DESC` is defensive (picks the most recently claimed
        one) rather than a correctness requirement for the documented single-worker deployment."""
        return self._conn.execute(
            "SELECT * FROM live_questions WHERE status = 'thinking' ORDER BY id DESC LIMIT 1"
        ).fetchone()

    def latest_answered(self) -> sqlite3.Row | None:
        """The most recently answered row, or `None` — `GET /api/now`'s `last` field."""
        return self._conn.execute(
            "SELECT * FROM live_questions WHERE status = 'answered' ORDER BY id DESC LIMIT 1"
        ).fetchone()

    def thinking_since(self, last_id: int) -> list[sqlite3.Row]:
        """Every row the worker has ever claimed (`started_at IS NOT NULL`) with `id > last_id`,
        oldest first — regardless of its CURRENT status, so a row that raced from `queued` all the
        way to `answered`/`rejected` between two polls still gets exactly one `thinking` broadcast
        first (`events`'s broadcaster emits `thinking` from this, then separately checks
        `answered_since` for the same id). Task brief's single global SSE broadcaster: one query
        per poll tick, not one per subscriber."""
        return self._conn.execute(
            "SELECT * FROM live_questions WHERE started_at IS NOT NULL AND id > ? ORDER BY id",
            (last_id,),
        ).fetchall()

    def answered_since(self, last_id: int) -> list[sqlite3.Row]:
        """Every row with `status = 'answered'` and `id > last_id`, oldest first — deliberately
        NOT `answered_at IS NOT NULL` alone, since `mark_failed` also stamps `answered_at` on a
        `rejected` row and must never be broadcast as an `answered` SSE event."""
        return self._conn.execute(
            "SELECT * FROM live_questions WHERE status = 'answered' AND id > ? ORDER BY id",
            (last_id,),
        ).fetchall()

    def claimed_total(self) -> int:
        """Monotonic count of every row the worker has EVER claimed (`started_at IS NOT NULL`) —
        task brief: the client computes its live queue position as
        `max(1, position_at_ask - (claimed_total_now - claimed_total_at_ask))`, so this number must
        never decrease and must count a claim exactly once, forever (a row that is later
        `mark_failed`'d into `rejected` keeps its `started_at`, so it still counts here — it WAS
        claimed and processed, just did not produce an answer)."""
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM live_questions WHERE started_at IS NOT NULL"
        ).fetchone()
        return row["n"]

    def avg_cycle_s(self, n: int = 20, fallback: float = 60.0) -> float:
        """Median of `(answered_at - started_at)` over the last `n` ANSWERED rows (task brief),
        rounded to 1 decimal. Median, not mean, so one unusually slow trial (a contended host, a
        retry) does not swing the client-facing ETA for everyone else waiting in line. `fallback`
        (measured production baseline, `config`'s own docstring references it) covers the empty-
        store/no-answers-yet case, matching this codebase's "zeros/defaults are fine, never a fake
        number" convention (`stats.py`)."""
        import statistics

        rows = self._conn.execute(
            "SELECT started_at, answered_at FROM live_questions WHERE status = 'answered' "
            "AND started_at IS NOT NULL AND answered_at IS NOT NULL ORDER BY id DESC LIMIT ?",
            (n,),
        ).fetchall()
        if not rows:
            return fallback
        durations = [
            (datetime.fromisoformat(r["answered_at"]) - datetime.fromisoformat(r["started_at"])).total_seconds()
            for r in rows
        ]
        # Floor: a real run is never under a few seconds (encoding alone takes about one), so a
        # freakishly small median means seeded/imported rows, not a fast worker — and an ETA built
        # on it would promise a wait nobody will get.
        return max(5.0, round(statistics.median(durations), 1))

    # -- ask dedupe (task brief: same normalized wording -> the same answer, or join the same run
    # already in flight, or skip a repeat content-rejection's LLM call) -------------------------

    def answered_twin(self, question_key: str) -> sqlite3.Row | None:
        """The most recent ANSWERED row for this normalized question key, or `None` — `api.py`
        returns its id/`repeat: true` instead of creating a new row or running the pipeline again
        (same wording -> the same seeds/coin -> the identical answer, task brief)."""
        return self._conn.execute(
            "SELECT * FROM live_questions WHERE status = 'answered' AND question_key = ? "
            "ORDER BY id DESC LIMIT 1",
            (question_key,),
        ).fetchone()

    def active_twin(self, question_key: str) -> sqlite3.Row | None:
        """The `queued`/`thinking` row for this key already in flight, or `None` —
        `idx_live_questions_active_key`'s uniqueness guarantees at most one such row ever exists at
        a time, so `LIMIT 1` is a formality, not a real ambiguity."""
        return self._conn.execute(
            "SELECT * FROM live_questions WHERE status IN ('queued', 'thinking') "
            "AND question_key = ? LIMIT 1",
            (question_key,),
        ).fetchone()

    def rejected_twin(self, question_key: str) -> sqlite3.Row | None:
        """The most recent REJECTED row for this key, or `None` — the CALLER (`api.py`) decides
        whether that rejection's `reason` counts as "content" (not rate-limit/captcha/queue-cap)
        before treating it as authoritative for a brand-new asker, matching this module's existing
        store-owns-the-query/caller-owns-the-business-rule split (`answered_questions_before`'s own
        docstring). `None` once `purge_expired` has blanked this row's `question_key` (7-day
        retention, same as its `question` text) — a question rejected long enough ago goes through
        full moderation again rather than staying "remembered" forever."""
        return self._conn.execute(
            "SELECT * FROM live_questions WHERE status = 'rejected' AND question_key = ? "
            "AND question_key IS NOT NULL ORDER BY id DESC LIMIT 1",
            (question_key,),
        ).fetchone()

    # -- game: guesses (live/game.py's "which one is the real fly?") -----------------------------

    def record_guess(
        self, answer_id: int, ip_hash: str, pick: str, correct: bool, counted: bool, now: datetime | None = None
    ) -> bool:
        """One guess per (answer, client) -- `INSERT OR IGNORE` is a silent no-op on a repeat
        (answer_id, ip_hash) pair; `api.py`'s route checks `get_guess` first regardless and
        returns that stored result, so this return value (True iff this call actually inserted a
        new row) is for tests/clarity, not load-bearing. `counted` (F2, live_guesses' own
        docstring above) is decided ONCE, by the caller, at insert time -- never recomputed."""
        cur = self._conn.execute(
            "INSERT OR IGNORE INTO live_guesses (answer_id, ip_hash, pick, correct, counted, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (answer_id, ip_hash, pick, 1 if correct else 0, 1 if counted else 0, _iso(now or _utcnow())),
        )
        self._conn.commit()
        return cur.rowcount > 0

    def get_guess(self, answer_id: int, ip_hash: str) -> sqlite3.Row | None:
        return self._conn.execute(
            "SELECT * FROM live_guesses WHERE answer_id = ? AND ip_hash = ?", (answer_id, ip_hash)
        ).fetchone()

    def guess_stats(self) -> dict:
        """`GET /api/stats`'s `guesses` field (task brief: the site-wide "visitors spot the real
        brain X% of the time" number) — `total`/`correct` over only the guesses marked `counted`
        (F2: the question's own asker, guessing within the count window; every other guess is
        stored for idempotency but excluded here so the published percentage cannot be farmed by
        replaying `GET /api/answers/{id}` -> `POST /api/guess` against ids the client never
        asked)."""
        row = self._conn.execute(
            "SELECT COUNT(*) AS total, COALESCE(SUM(correct), 0) AS correct "
            "FROM live_guesses WHERE counted = 1"
        ).fetchone()
        return {"total": row["total"], "correct": row["correct"]}

    # -- chat: writes --------------------------------------------------------------------------------
    # NOTE (F4, 2026-09-18 security review): every `ip_hash` parameter below the chat section is
    # actually `chat.author_id(ip, CHAT_ID_PEPPER)` -- a STABLE identity, not LiveStore.hash_ip's
    # daily-rotating one -- kept under the historical column/param name to avoid an in-place SQLite
    # column rename on tables with no production data yet. See chat.py's module docstring.

    def chat_send(self, nickname: str, text: str, ip_hash: str, now: datetime | None = None) -> int:
        cur = self._conn.execute(
            "INSERT INTO live_chat_messages (nickname, text, ip_hash, created_at, deleted) "
            "VALUES (?, ?, ?, ?, 0)",
            (nickname, text, ip_hash, _iso(now or _utcnow())),
        )
        self._conn.commit()
        return cur.lastrowid

    def _chat_soft_delete(self, id_: int, reason: str, now: datetime | None = None) -> bool:
        row = self.chat_get(id_)
        if row is None or row["deleted"]:
            return False
        self._conn.execute(
            "UPDATE live_chat_messages SET deleted = 1, deleted_at = ?, deleted_reason = ? "
            "WHERE id = ?",
            (_iso(now or _utcnow()), reason, id_),
        )
        self._conn.commit()
        return True

    def chat_delete(self, id_: int, now: datetime | None = None) -> bool:
        """Operator delete (`chat_admin delete <id>`) -- broadcast as `chat_delete` by
        `events.ChatEventTracker`'s next poll. Returns False if the id doesn't exist or was
        already deleted (idempotent no-op, not an error)."""
        return self._chat_soft_delete(id_, "admin", now)

    def chat_auto_hide(self, id_: int, now: datetime | None = None) -> bool:
        """Report-threshold auto-hide (`api.py`'s report route) -- same soft-delete, tagged
        `deleted_reason='reported'` so `chat_recent`'s exclusion and `chat_admin list --reported`
        can tell it apart from an operator's own `chat_delete`."""
        return self._chat_soft_delete(id_, "reported", now)

    def chat_restore(self, id_: int, now: datetime | None = None) -> bool:
        """`chat_admin restore <id>` -- un-hides the message and clears its report tally (a fresh
        set of reporters is required to re-trigger auto-hide; otherwise the SAME 3 reporters who
        triggered it would instantly re-hide it the moment the operator restores it)."""
        row = self.chat_get(id_)
        if row is None or not row["deleted"]:
            return False
        ts = _iso(now or _utcnow())
        self._conn.execute(
            "UPDATE live_chat_messages SET deleted = 0, deleted_at = NULL, deleted_reason = NULL, "
            "restored_at = ? WHERE id = ?",
            (ts, id_),
        )
        self._conn.execute("DELETE FROM live_chat_reports WHERE message_id = ?", (id_,))
        self._conn.commit()
        return True

    def chat_clear(self, now: datetime | None = None) -> int:
        ts = _iso(now or _utcnow())
        cur = self._conn.execute(
            "UPDATE live_chat_messages SET deleted = 1, deleted_at = ?, deleted_reason = 'admin' "
            "WHERE deleted = 0",
            (ts,),
        )
        self._conn.commit()
        return cur.rowcount

    def chat_ban(self, ip_hash: str, now: datetime | None = None) -> None:
        self._conn.execute(
            "INSERT INTO live_chat_bans (ip_hash, banned_at) VALUES (?, ?) "
            "ON CONFLICT(ip_hash) DO UPDATE SET banned_at = excluded.banned_at",
            (ip_hash, _iso(now or _utcnow())),
        )
        self._conn.commit()

    def chat_is_banned(self, ip_hash: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM live_chat_bans WHERE ip_hash = ?", (ip_hash,)
        ).fetchone()
        return row is not None

    def chat_report(self, message_id: int, report_group_id: str, now: datetime | None = None) -> int:
        """Idempotent: a second report from the same `report_group_id` (F3: an IPv4 /24 or IPv6
        /64 group, coarser than a single author_id -- see chat.py's `report_group_id`) for the
        same message does not count twice. Returns the current distinct-reporter-group count
        (`api.py`'s route compares this against `config.CHAT_REPORT_THRESHOLD` to decide whether
        to auto-hide). The column is still named `ip_hash` for the same reason noted above."""
        self._conn.execute(
            "INSERT INTO live_chat_reports (message_id, ip_hash, created_at) VALUES (?, ?, ?) "
            "ON CONFLICT(message_id, ip_hash) DO NOTHING",
            (message_id, report_group_id, _iso(now or _utcnow())),
        )
        self._conn.commit()
        return self.chat_report_count(message_id)

    def chat_report_count(self, message_id: int) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM live_chat_reports WHERE message_id = ?", (message_id,)
        ).fetchone()
        return row["n"]

    def chat_has_accepted_message(self, ip_hash: str) -> bool:
        """F3: a report only counts toward auto-hide if the reporter has posted >=1 accepted
        message. Since `live_chat_messages` only ever receives rows AFTER moderation passes (any
        row's existence proves the author posted at least once), any row for this author_id --
        deleted or not -- satisfies this, so no extra bookkeeping table is needed."""
        row = self._conn.execute(
            "SELECT 1 FROM live_chat_messages WHERE ip_hash = ? LIMIT 1", (ip_hash,)
        ).fetchone()
        return row is not None

    # -- chat: atomic rate limiting (F1/F2, 2026-09-18 security review) --------------------------

    def chat_reserve_attempt(
        self,
        key: str,
        slowmode_s: float,
        window_s: float | None = None,
        max_per_window: int | None = None,
        now: datetime | None = None,
    ) -> bool:
        """Atomically check-and-record one attempt for `key` (F1: closes the check-then-act race
        a concurrent burst could exploit -- N parallel requests all reading count=0 before any of
        their inserts commits; F2: records EVERY attempt, accepted, rejected, or failed
        downstream -- not just accepted messages -- so a client cannot burn unlimited LLM calls by
        repeatedly sending content it knows will be rejected). `BEGIN IMMEDIATE` takes the write
        lock up front, the same pattern `claim_next` above uses, so two concurrent requests for
        the same key cannot both observe an under-threshold count before either commits its own
        insert. Call this BEFORE running any moderation, including the deterministic rule layer,
        not just before the LLM step.

        `key` need not be a literal author_id/ip_hash -- R1/R2 (round-2 security review) reuse
        this SAME table/transaction with namespaced keys (`f"report:{author_id}"`,
        `f"llm:{author_id}"`) to rate-limit report actions and per-author LLM classification calls
        independently of the per-message posting limit, via `window_s`/`max_per_window` overrides
        (default to the shared POST /api/chat thresholds when omitted)."""
        now_dt = now or _utcnow()
        window_s = config.CHAT_RATE_WINDOW_S if window_s is None else window_s
        max_per_window = config.CHAT_RATE_PER_WINDOW if max_per_window is None else max_per_window
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            n_interval = self._conn.execute(
                "SELECT COUNT(*) AS n FROM live_chat_attempts WHERE ip_hash = ? AND created_at >= ?",
                (key, _iso(now_dt - timedelta(seconds=slowmode_s))),
            ).fetchone()["n"]
            n_window = self._conn.execute(
                "SELECT COUNT(*) AS n FROM live_chat_attempts WHERE ip_hash = ? AND created_at >= ?",
                (key, _iso(now_dt - timedelta(seconds=window_s))),
            ).fetchone()["n"]
            if not check_chat_rate_limit(n_interval, n_window, max_per_window=max_per_window).ok:
                self._conn.commit()  # release the write lock even on a rejection, no insert
                return False
            self._conn.execute(
                "INSERT INTO live_chat_attempts (ip_hash, created_at) VALUES (?, ?)",
                (key, _iso(now_dt)),
            )
            self._conn.commit()
            return True
        except Exception:
            self._conn.rollback()
            raise

    def chat_reserve_llm_call(self, now: datetime | None = None) -> bool:
        """F2's site-wide circuit breaker: atomically check-and-record one chat-LLM classification
        call within `config.CHAT_LLM_BREAKER_WINDOW_S`, independent of any single client's
        identity -- protects the shared chat LiteLLM key's budget/rate limit from ANY combination
        of clients, not just one abusive IP. Same `BEGIN IMMEDIATE` pattern as
        `chat_reserve_attempt` above."""
        now_dt = now or _utcnow()
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            since = _iso(now_dt - timedelta(seconds=config.CHAT_LLM_BREAKER_WINDOW_S))
            n = self._conn.execute(
                "SELECT COUNT(*) AS n FROM live_chat_llm_calls WHERE created_at >= ?", (since,)
            ).fetchone()["n"]
            if n >= config.CHAT_LLM_MAX_PER_MINUTE:
                self._conn.commit()
                return False
            self._conn.execute(
                "INSERT INTO live_chat_llm_calls (created_at) VALUES (?)", (_iso(now_dt),)
            )
            self._conn.commit()
            return True
        except Exception:
            self._conn.rollback()
            raise

    def chat_auto_hide_count_recent(self, window_s: float, now: datetime | None = None) -> int:
        """R1 (round-2 security review): how many messages were auto-hidden by the report
        threshold within the last `window_s` seconds, site-wide -- `api.py`'s report route
        refuses to auto-hide once this reaches `config.CHAT_AUTO_HIDE_PER_10MIN`, capping how
        fast even a large ring of colluding identities can remove content. A plain read (not
        wrapped in `chat_reserve_attempt`'s transaction): the message-level `_chat_soft_delete`
        guard against double-hiding the SAME message already makes the worst case of a race here
        "a couple of hides over the cap", not a bypass."""
        since = _iso((now or _utcnow()) - timedelta(seconds=window_s))
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM live_chat_messages WHERE deleted_reason = 'reported' "
            "AND deleted_at >= ?",
            (since,),
        ).fetchone()
        return row["n"]

    # -- chat: reads -----------------------------------------------------------------------------

    def chat_get(self, id_: int) -> sqlite3.Row | None:
        return self._conn.execute(
            "SELECT * FROM live_chat_messages WHERE id = ?", (id_,)
        ).fetchone()

    def chat_recent(self, limit: int = 100, include_deleted: bool = False) -> list[sqlite3.Row]:
        """Oldest-first among the most recent `limit` rows (task brief `GET /api/chat?limit=100`,
        "returns recent non-deleted messages") -- `include_deleted=True` is for the operator CLI's
        `list --all` only, never used by the public API route."""
        where = "" if include_deleted else "WHERE deleted = 0"
        rows = self._conn.execute(
            f"SELECT * FROM live_chat_messages {where} ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return list(reversed(rows))

    def chat_before(self, before_id: int, limit: int = 50) -> list[sqlite3.Row]:
        """Same "most recent N, oldest-first" shape as `chat_recent`, scoped to `id < before_id` --
        `GET /api/chat?before_id=...`'s history-paging branch (task brief). Never includes deleted
        messages, same as the public `chat_recent` (there is no operator/`--all` use of this one)."""
        rows = self._conn.execute(
            "SELECT * FROM live_chat_messages WHERE id < ? AND deleted = 0 ORDER BY id DESC LIMIT ?",
            (before_id, limit),
        ).fetchall()
        return list(reversed(rows))

    def chat_has_more_before(self, id_: int) -> bool:
        """Whether at least one non-deleted message older than `id_` exists — `GET /api/chat`'s
        `has_more` (task brief), computed the same way whether the page came from `chat_recent` or
        `chat_before` (the caller passes the returned page's own oldest id, or an id one past the
        newest row ever seen if the page was empty)."""
        row = self._conn.execute(
            "SELECT 1 FROM live_chat_messages WHERE id < ? AND deleted = 0 LIMIT 1", (id_,)
        ).fetchone()
        return row is not None

    def chat_reported(self, limit: int = 100) -> list[sqlite3.Row]:
        """`chat_admin list --reported`: messages auto-hidden by the report threshold, newest
        first, for operator review."""
        return self._conn.execute(
            "SELECT * FROM live_chat_messages WHERE deleted = 1 AND deleted_reason = 'reported' "
            "ORDER BY deleted_at DESC LIMIT ?",
            (limit,),
        ).fetchall()

    def chat_count_recent(self, ip_hash: str, window_seconds: float, now: datetime | None = None) -> int:
        since = _iso((now or _utcnow()) - timedelta(seconds=window_seconds))
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM live_chat_messages WHERE ip_hash = ? AND created_at >= ?",
            (ip_hash, since),
        ).fetchone()
        return row["n"]

    def chat_last_text(self, ip_hash: str) -> str | None:
        row = self._conn.execute(
            "SELECT text FROM live_chat_messages WHERE ip_hash = ? ORDER BY id DESC LIMIT 1",
            (ip_hash,),
        ).fetchone()
        return row["text"] if row else None

    def chat_latest_id(self) -> int:
        row = self._conn.execute("SELECT MAX(id) AS m FROM live_chat_messages").fetchone()
        return row["m"] or 0

    def chat_messages_after(self, last_id: int) -> list[sqlite3.Row]:
        """Every row (deleted or not) with `id > last_id`, oldest first -- `events.ChatEventTracker`
        uses this to emit a `chat` event exactly once per newly-inserted row that is not already
        deleted by the time it's seen."""
        return self._conn.execute(
            "SELECT * FROM live_chat_messages WHERE id > ? ORDER BY id", (last_id,)
        ).fetchall()

    def chat_deleted_after(self, since: str) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM live_chat_messages WHERE deleted = 1 AND deleted_at IS NOT NULL "
            "AND deleted_at > ? ORDER BY deleted_at",
            (since,),
        ).fetchall()

    def chat_restored_after(self, since: str) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM live_chat_messages WHERE deleted = 0 AND restored_at IS NOT NULL "
            "AND restored_at > ? ORDER BY restored_at",
            (since,),
        ).fetchall()

    # -- chat: runtime state (slow-mode interval + soft kill switch) -----------------------------

    def chat_state(self) -> dict:
        row = self._conn.execute(
            "SELECT enabled, slowmode_s FROM live_chat_state WHERE id = 1"
        ).fetchone()
        return {"enabled": bool(row["enabled"]), "slowmode_s": row["slowmode_s"]}

    def chat_set_enabled(self, enabled: bool, now: datetime | None = None) -> None:
        self._conn.execute(
            "UPDATE live_chat_state SET enabled = ?, updated_at = ? WHERE id = 1",
            (1 if enabled else 0, _iso(now or _utcnow())),
        )
        self._conn.commit()

    def chat_set_slowmode(self, seconds: float, now: datetime | None = None) -> None:
        self._conn.execute(
            "UPDATE live_chat_state SET slowmode_s = ?, updated_at = ? WHERE id = 1",
            (seconds, _iso(now or _utcnow())),
        )
        self._conn.commit()

    # -- retention -----------------------------------------------------------------------------------

    def purge_expired(self, retention_days: int = config.REJECTED_RETENTION_DAYS, now: datetime | None = None) -> int:
        """Blanks `question`/`message`/`question_key` (not the whole row — `reason`/timestamps
        stay for aggregate stats) on `rejected` rows older than `retention_days` (task brief:
        "delete question text of rejected items after 7 days; keep answered ones"). `question_key`
        is blanked alongside `question` — it is a normalized near-copy of the same text (NFKC/
        casefold/whitespace-collapse, not a hash), so keeping it around after the text itself is
        purged would defeat the retention rule in substance while satisfying it in name; a rejected
        question aged out of `rejected_twin`'s lookup this way simply goes through full moderation
        again on its next ask, same as a question never seen before. Returns the number of rows
        touched."""
        cutoff = _iso((now or _utcnow()) - timedelta(days=retention_days))
        cur = self._conn.execute(
            "UPDATE live_questions SET question = '', message = NULL, question_key = NULL "
            "WHERE status = 'rejected' AND created_at < ? AND question != ''",
            (cutoff,),
        )
        self._conn.commit()
        return cur.rowcount

    def chat_purge_expired(
        self,
        deleted_text_retention_days: int = config.REJECTED_RETENTION_DAYS,
        identity_retention_days: int = 30,
        attempts_retention_days: int = 1,
        now: datetime | None = None,
    ) -> dict:
        """Three independent retention rules (security review F4 + the accompanying "purge
        soft-deleted chat after 7 days" instruction + round-2's "attempts/llm_calls tables are
        never purged"), all matching `purge_expired`'s "blank/drop the sensitive or unbounded
        data, keep what's still useful" shape:

        1. Soft-deleted (admin- or report-hidden) messages older than `deleted_text_retention_days`
           have their `text` blanked -- same 7-day default as rejected questions above.
        2. ANY message (deleted or not) older than `identity_retention_days` has its author
           linkage (`ip_hash`, i.e. author_id) blanked, UNLESS that author is currently banned --
           F4: "purge after 30 days unless banned" (keeps a ban enforceable/auditable without
           holding every past visitor's identity linkage forever).
        3. `live_chat_attempts`/`live_chat_llm_calls` rows older than `attempts_retention_days`
           are hard-deleted -- both only ever need to be queried within a 10-minute window (the
           widest rate-limit window in use), so a 1-day retention is already generous; nothing
           reads them past that.

        Returns `{"text_blanked": n, "identity_blanked": n, "attempts_purged": n, "llm_calls_purged": n}`.
        """
        now_dt = now or _utcnow()
        text_cutoff = _iso(now_dt - timedelta(days=deleted_text_retention_days))
        identity_cutoff = _iso(now_dt - timedelta(days=identity_retention_days))
        attempts_cutoff = _iso(now_dt - timedelta(days=attempts_retention_days))

        cur_text = self._conn.execute(
            "UPDATE live_chat_messages SET text = '' WHERE deleted = 1 AND deleted_at IS NOT NULL "
            "AND deleted_at < ? AND text != ''",
            (text_cutoff,),
        )
        cur_identity = self._conn.execute(
            "UPDATE live_chat_messages SET ip_hash = '' WHERE created_at < ? AND ip_hash != '' "
            "AND ip_hash NOT IN (SELECT ip_hash FROM live_chat_bans)",
            (identity_cutoff,),
        )
        cur_attempts = self._conn.execute(
            "DELETE FROM live_chat_attempts WHERE created_at < ?", (attempts_cutoff,)
        )
        cur_llm_calls = self._conn.execute(
            "DELETE FROM live_chat_llm_calls WHERE created_at < ?", (attempts_cutoff,)
        )
        self._conn.commit()
        return {
            "text_blanked": cur_text.rowcount,
            "identity_blanked": cur_identity.rowcount,
            "attempts_purged": cur_attempts.rowcount,
            "llm_calls_purged": cur_llm_calls.rowcount,
        }
