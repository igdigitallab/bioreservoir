"""SQLite ledger + committed JSONL export (README.md Pipeline step 5).

The SQLite file (`experiments/001-fly-oracle/ledger.sqlite`, gitignored — `*.sqlite` in
`.gitignore`) is the pipeline's working state: every run's status, seeds and per-trial readout
stats, used by `oracle.plan` to decide what is already done (resumability) and by `oracle.score`
to compute the scoreboard. It is not itself the pre-registration artifact.

`predictions/<YYYY-MM-DD>.jsonl` is: one append-only file per UTC day a prediction was newly
exported, never rewritten once a day's file exists (new predictions on a later day go to that
day's own file). `predictions/SHA256SUMS` covers every `predictions/*.jsonl` file and is
recomputed on every export. Both are meant to be `git add`ed and committed by whoever runs the
pipeline. This module deliberately does not commit anything by itself: a research ledger that
auto-commits predictions on a schedule, with nobody in the loop, is one bad cron away from
publishing a run nobody meant to publish. `oracle.ledger.stamp_sha256sums` then timestamps that
commit's `SHA256SUMS` via OpenTimestamps, behind an explicit `--stamp` flag the CLI never turns on
by itself.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Self

from bioreservoir.oracle.config import EXPERIMENT_DIR

LEDGER_PATH = EXPERIMENT_DIR / "ledger.sqlite"
PREDICTIONS_DIR = EXPERIMENT_DIR / "predictions"
SHA256SUMS_PATH = PREDICTIONS_DIR / "SHA256SUMS"

SCHEMA = """
CREATE TABLE IF NOT EXISTS questions (
    question_id TEXT PRIMARY KEY,
    category TEXT NOT NULL,
    resolves_by TEXT NOT NULL,
    added_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    question_id TEXT NOT NULL REFERENCES questions(question_id),
    brain TEXT NOT NULL,
    condition TEXT NOT NULL,
    variant TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    code_git_sha TEXT,
    trial_seeds TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'running',
    error TEXT,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    UNIQUE(question_id, brain, condition, variant, config_hash)
);

CREATE TABLE IF NOT EXISTS predictions (
    run_id INTEGER PRIMARY KEY REFERENCES runs(run_id),
    p_yes REAL NOT NULL,
    mean_bias REAL,
    n_trials INTEGER NOT NULL,
    n_zero_spike_trials INTEGER NOT NULL,
    per_trial_json TEXT NOT NULL,
    b0 REAL,
    corrected_bias REAL,
    left_is_yes INTEGER,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    exported_at TEXT
);

CREATE TABLE IF NOT EXISTS resolutions (
    question_id TEXT PRIMARY KEY REFERENCES questions(question_id),
    outcome TEXT NOT NULL CHECK (outcome IN ('yes', 'no', 'void')),
    resolved_at TEXT NOT NULL,
    source TEXT NOT NULL,
    notes TEXT
);

-- Handedness reference runs (oracle.handedness, oracle.reference; README.md "Handedness and side
-- mapping") — a distinct pair of tables, not a `condition`/`kind` flag on `runs`/`predictions`, so
-- a reference-sentence row can never be joined or filtered together with a question row by
-- accident (e.g. `all_predictions()`/`score.py` simply cannot see these tables).
CREATE TABLE IF NOT EXISTS reference_sentences (
    reference_id TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    added_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS reference_runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    reference_id TEXT NOT NULL REFERENCES reference_sentences(reference_id),
    brain TEXT NOT NULL,
    condition TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    code_git_sha TEXT,
    trial_seeds TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'running',
    error TEXT,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    UNIQUE(reference_id, brain, condition, config_hash)
);

CREATE TABLE IF NOT EXISTS reference_predictions (
    run_id INTEGER PRIMARY KEY REFERENCES reference_runs(run_id),
    mean_bias REAL NOT NULL,
    n_trials INTEGER NOT NULL,
    n_zero_spike_trials INTEGER NOT NULL,
    per_trial_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


def _utcnow() -> str:
    return datetime.now(UTC).isoformat()


def git_sha(dirty_suffix: bool = True) -> str | None:
    """`git rev-parse HEAD` for the repo containing this file, or `None` outside a git checkout
    (e.g. an installed wheel) — `code_git_sha` is best-effort provenance, not load-bearing for
    resumability (`config_hash` is)."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=EXPERIMENT_DIR,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    if dirty_suffix:
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=EXPERIMENT_DIR, capture_output=True, text=True, check=False
        ).stdout.strip()
        if dirty:
            sha += "-dirty"
    return sha


@dataclass
class Ledger:
    path: Path = LEDGER_PATH
    _conn: sqlite3.Connection | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, timeout=30.0)
        self._conn.execute("PRAGMA foreign_keys = ON")
        # WAL + a busy timeout: `oracle.runner.run_all` opens one `Ledger` per worker process, all
        # against the same on-disk file (a sqlite3 connection itself cannot cross a process
        # boundary) — WAL lets concurrent readers/writers coexist instead of serializing on a
        # single rollback-journal lock, and the busy timeout retries instead of raising
        # "database is locked" on the rare write-write collision.
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA busy_timeout = 30000")
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- questions -----------------------------------------------------------------------------

    def upsert_question(self, question_id: str, category: str, resolves_by: str) -> None:
        self._conn.execute(
            "INSERT INTO questions (question_id, category, resolves_by) VALUES (?, ?, ?) "
            "ON CONFLICT(question_id) DO UPDATE SET category = excluded.category, "
            "resolves_by = excluded.resolves_by",
            (question_id, category, resolves_by),
        )
        self._conn.commit()

    # -- runs / resumability ---------------------------------------------------------------------

    def is_done(self, question_id: str, brain: str, condition: str, variant: str, config_hash: str) -> bool:
        row = self._conn.execute(
            "SELECT status FROM runs WHERE question_id = ? AND brain = ? AND condition = ? "
            "AND variant = ? AND config_hash = ?",
            (question_id, brain, condition, variant, config_hash),
        ).fetchone()
        return row is not None and row[0] == "done"

    def start_run(
        self,
        question_id: str,
        brain: str,
        condition: str,
        variant: str,
        config_hash: str,
        trial_seeds: list[int],
    ) -> int:
        """Insert (or reset, if a previous attempt errored) a `running` row and return `run_id`.

        The `UNIQUE(question_id, brain, condition, variant, config_hash)` constraint means a
        second `start_run` for an identical combination overwrites the prior (necessarily
        non-`done`, since `oracle.plan` already skips `done` combinations before calling this)
        attempt rather than accumulating dead rows from crashed workers.
        """
        cur = self._conn.execute(
            "INSERT INTO runs (question_id, brain, condition, variant, config_hash, code_git_sha, "
            "trial_seeds, status, started_at) VALUES (?, ?, ?, ?, ?, ?, ?, 'running', ?) "
            "ON CONFLICT(question_id, brain, condition, variant, config_hash) DO UPDATE SET "
            "code_git_sha = excluded.code_git_sha, trial_seeds = excluded.trial_seeds, "
            "status = 'running', error = NULL, started_at = excluded.started_at, finished_at = NULL "
            "RETURNING run_id",
            (question_id, brain, condition, variant, config_hash, git_sha(), json.dumps(trial_seeds), _utcnow()),
        )
        run_id = cur.fetchone()[0]
        self._conn.commit()
        return run_id

    def finish_run(self, run_id: int, status: str, error: str | None = None) -> None:
        if status not in ("done", "failed"):
            raise ValueError(f"status must be 'done' or 'failed', got {status!r}")
        self._conn.execute(
            "UPDATE runs SET status = ?, error = ?, finished_at = ? WHERE run_id = ?",
            (status, error, _utcnow(), run_id),
        )
        self._conn.commit()

    # -- predictions -----------------------------------------------------------------------------

    def record_prediction(
        self,
        run_id: int,
        p_yes: float,
        mean_bias: float | None,
        n_trials: int,
        n_zero_spike_trials: int,
        per_trial_stats: list[dict],
        b0: float | None = None,
        corrected_bias: float | None = None,
        left_is_yes: bool | None = None,
    ) -> None:
        """`p_yes` is the final, handedness-corrected, side-mapped probability
        (`oracle.handedness.apply_correction`) — `score.py` reads it as-is, it never recomputes
        anything from `mean_bias`. `mean_bias` stays the *raw*, uncorrected trial-averaged bias
        (unchanged meaning from before handedness correction existed); `b0`/`corrected_bias`/
        `left_is_yes` are optional so callers that predate handedness correction (tests, mostly)
        keep working with `NULL`s in those columns."""
        self._conn.execute(
            "INSERT INTO predictions (run_id, p_yes, mean_bias, n_trials, n_zero_spike_trials, "
            "per_trial_json, b0, corrected_bias, left_is_yes) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(run_id) DO UPDATE SET p_yes = excluded.p_yes, mean_bias = excluded.mean_bias, "
            "n_trials = excluded.n_trials, n_zero_spike_trials = excluded.n_zero_spike_trials, "
            "per_trial_json = excluded.per_trial_json, b0 = excluded.b0, "
            "corrected_bias = excluded.corrected_bias, left_is_yes = excluded.left_is_yes, "
            "exported_at = NULL",
            (
                run_id,
                p_yes,
                mean_bias,
                n_trials,
                n_zero_spike_trials,
                json.dumps(per_trial_stats),
                b0,
                corrected_bias,
                None if left_is_yes is None else int(left_is_yes),
            ),
        )
        self._conn.commit()

    # -- resolutions -----------------------------------------------------------------------------

    def record_resolution(self, question_id: str, outcome: str, source: str, notes: str | None = None) -> None:
        if outcome not in ("yes", "no", "void"):
            raise ValueError(f"outcome must be 'yes', 'no' or 'void', got {outcome!r}")
        self._conn.execute(
            "INSERT INTO resolutions (question_id, outcome, resolved_at, source, notes) "
            "VALUES (?, ?, ?, ?, ?) ON CONFLICT(question_id) DO UPDATE SET "
            "outcome = excluded.outcome, resolved_at = excluded.resolved_at, "
            "source = excluded.source, notes = excluded.notes",
            (question_id, outcome, _utcnow(), source, notes),
        )
        self._conn.commit()

    def all_predictions(self) -> list[sqlite3.Row]:
        conn = self._conn
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT r.run_id, r.question_id, r.brain, r.condition, r.variant, r.config_hash, "
            "r.code_git_sha, r.trial_seeds, r.started_at, r.finished_at, "
            "p.p_yes, p.mean_bias, p.n_trials, p.n_zero_spike_trials, p.per_trial_json, "
            "p.b0, p.corrected_bias, p.left_is_yes, "
            "p.created_at, p.exported_at, res.outcome, res.resolved_at "
            "FROM runs r JOIN predictions p ON p.run_id = r.run_id "
            "LEFT JOIN resolutions res ON res.question_id = r.question_id "
            "WHERE r.status = 'done' ORDER BY r.run_id"
        ).fetchall()
        conn.row_factory = None
        return rows

    # -- handedness reference runs -----------------------------------------------------------------

    def upsert_reference_sentence(self, reference_id: str, text: str) -> None:
        self._conn.execute(
            "INSERT INTO reference_sentences (reference_id, text) VALUES (?, ?) "
            "ON CONFLICT(reference_id) DO UPDATE SET text = excluded.text",
            (reference_id, text),
        )
        self._conn.commit()

    def is_reference_done(self, reference_id: str, brain: str, condition: str, config_hash: str) -> bool:
        row = self._conn.execute(
            "SELECT status FROM reference_runs WHERE reference_id = ? AND brain = ? AND "
            "condition = ? AND config_hash = ?",
            (reference_id, brain, condition, config_hash),
        ).fetchone()
        return row is not None and row[0] == "done"

    def start_reference_run(
        self,
        reference_id: str,
        brain: str,
        condition: str,
        config_hash: str,
        trial_seeds: list[int],
    ) -> int:
        """Same reset-on-retry contract as `start_run` (see its docstring): a second call for an
        identical `(reference_id, brain, condition, config_hash)` overwrites the prior attempt
        rather than accumulating dead rows from crashed workers."""
        cur = self._conn.execute(
            "INSERT INTO reference_runs (reference_id, brain, condition, config_hash, "
            "code_git_sha, trial_seeds, status, started_at) VALUES (?, ?, ?, ?, ?, ?, 'running', ?) "
            "ON CONFLICT(reference_id, brain, condition, config_hash) DO UPDATE SET "
            "code_git_sha = excluded.code_git_sha, trial_seeds = excluded.trial_seeds, "
            "status = 'running', error = NULL, started_at = excluded.started_at, finished_at = NULL "
            "RETURNING run_id",
            (reference_id, brain, condition, config_hash, git_sha(), json.dumps(trial_seeds), _utcnow()),
        )
        run_id = cur.fetchone()[0]
        self._conn.commit()
        return run_id

    def finish_reference_run(self, run_id: int, status: str, error: str | None = None) -> None:
        if status not in ("done", "failed"):
            raise ValueError(f"status must be 'done' or 'failed', got {status!r}")
        self._conn.execute(
            "UPDATE reference_runs SET status = ?, error = ?, finished_at = ? WHERE run_id = ?",
            (status, error, _utcnow(), run_id),
        )
        self._conn.commit()

    def record_reference_prediction(
        self,
        run_id: int,
        mean_bias: float,
        n_trials: int,
        n_zero_spike_trials: int,
        per_trial_stats: list[dict],
    ) -> None:
        self._conn.execute(
            "INSERT INTO reference_predictions (run_id, mean_bias, n_trials, n_zero_spike_trials, "
            "per_trial_json) VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(run_id) DO UPDATE SET mean_bias = excluded.mean_bias, "
            "n_trials = excluded.n_trials, n_zero_spike_trials = excluded.n_zero_spike_trials, "
            "per_trial_json = excluded.per_trial_json",
            (run_id, mean_bias, n_trials, n_zero_spike_trials, json.dumps(per_trial_stats)),
        )
        self._conn.commit()

    def reference_mean_biases(self, brain: str, condition: str, config_hash: str) -> list[float]:
        """Every `done` reference sentence's own mean bias for `(brain, condition, config_hash)` —
        the input to `oracle.handedness.compute_b0`. One row per reference sentence (each sentence's
        own trials, if any, are already averaged by whoever called `record_reference_prediction`)."""
        rows = self._conn.execute(
            "SELECT p.mean_bias FROM reference_runs r JOIN reference_predictions p "
            "ON p.run_id = r.run_id "
            "WHERE r.status = 'done' AND r.brain = ? AND r.condition = ? AND r.config_hash = ?",
            (brain, condition, config_hash),
        ).fetchall()
        return [row[0] for row in rows]

    # -- export ----------------------------------------------------------------------------------

    def export_predictions(self, date: str | None = None, export_dir: Path = PREDICTIONS_DIR) -> Path | None:
        """Append every not-yet-exported `done` prediction to `<export_dir>/<date>.jsonl` (UTC
        today unless `date` is given) and refresh `<export_dir>/SHA256SUMS` over all `*.jsonl`
        files there. Returns the day's file path, or `None` if there was nothing new to export.
        """
        date = date or datetime.now(UTC).strftime("%Y-%m-%d")
        export_dir.mkdir(parents=True, exist_ok=True)
        conn = self._conn
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT r.run_id, r.question_id, r.brain, r.condition, r.variant, r.config_hash, "
            "r.code_git_sha, r.trial_seeds, r.started_at, r.finished_at, "
            "p.p_yes, p.mean_bias, p.n_trials, p.n_zero_spike_trials, "
            "p.b0, p.corrected_bias, p.left_is_yes "
            "FROM runs r JOIN predictions p ON p.run_id = r.run_id "
            "WHERE r.status = 'done' AND p.exported_at IS NULL ORDER BY r.run_id"
        ).fetchall()
        conn.row_factory = None
        if not rows:
            return None

        out_path = export_dir / f"{date}.jsonl"
        exported_at = _utcnow()
        with open(out_path, "a") as f:
            for row in rows:
                record = dict(row)
                record["exported_at"] = exported_at
                f.write(json.dumps(record, sort_keys=True) + "\n")
        conn.executemany(
            "UPDATE predictions SET exported_at = ? WHERE run_id = ?",
            [(exported_at, row["run_id"]) for row in rows],
        )
        conn.commit()
        update_sha256sums(export_dir)
        return out_path


def update_sha256sums(export_dir: Path = PREDICTIONS_DIR) -> Path:
    """Recompute `SHA256SUMS` over every `*.jsonl` file in `export_dir`, sorted by name — the
    "predictions can be hash-committed to git before resolution" artifact (task brief)."""
    jsonl_files = sorted(export_dir.glob("*.jsonl"))
    lines = []
    for path in jsonl_files:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.name}")
    sums_path = export_dir / "SHA256SUMS"
    sums_path.write_text("\n".join(lines) + ("\n" if lines else ""))
    return sums_path


def stamp_sha256sums(sums_path: Path = SHA256SUMS_PATH) -> subprocess.CompletedProcess:
    """`ots stamp <sums_path>` (opentimestamps-client, optional dependency — see pyproject.toml's
    `stamp` extra). Task brief: "implement but do not run it" — nothing in this module or the CLI
    calls this function except the `run`/`score` subcommands' own explicit `--stamp` flag, which
    this codebase's own tests and CLI invocations never pass.
    """
    return subprocess.run(["ots", "stamp", str(sums_path)], check=True, capture_output=True, text=True)
