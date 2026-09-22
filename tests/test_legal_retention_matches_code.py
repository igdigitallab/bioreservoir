"""The Privacy Policy's retention promises must equal what the code actually does.

This is the one document on the site that goes stale silently: change
`REJECTED_RETENTION_DAYS` from 7 to 30, or bump the chat identity window, and nothing fails —
the page keeps promising the old number to visitors. Nobody notices until someone checks, and by
then the policy has been a lie for months.

Bound here: every "N days" line in Privacy §5 ("How long things are kept",
`site/content/legal.json`, found by its stable `id`, not by heading number) against the constant
that really governs it.

Where the real numbers live: `worker.py`'s maintenance pass calls `live_store.purge_expired()` and
`live_store.chat_purge_expired()` with NO arguments, so each function's signature DEFAULT is the
retention actually in force. That is what this file reads (via `inspect.signature`) rather than a
second copy of the numbers — a copy would drift the same way the policy does. Moving a default
into `config.py` later keeps this test correct, because the signature still resolves to it.
"""

from __future__ import annotations

import inspect
import json
import re
from pathlib import Path

import pytest

from bioreservoir.live.store import LiveStore

LEGAL_JSON = Path(__file__).resolve().parents[1] / "site" / "content" / "legal.json"


def _retention_default(name: str) -> int:
    """The in-force retention window for one purge argument: the signature default, because both
    callers in worker.py invoke these functions with no arguments at all."""
    for func in (LiveStore.chat_purge_expired, LiveStore.purge_expired):
        param = inspect.signature(func).parameters.get(name)
        if param is not None and param.default is not inspect.Parameter.empty:
            return int(param.default)
    raise AssertionError(f"no purge function takes a {name!r} argument with a default any more")


# (a distinctive phrase from the policy line, the purge argument that governs it). The phrase only
# has to be unique within the section -- the day count itself is read out of the line, never
# written here, so this table cannot disagree with the policy about the number.
RETENTION_LINES = [
    ("rejected question", "retention_days"),
    ("Chat identity", "identity_retention_days"),
    ("Hidden or removed chat message text", "deleted_text_retention_days"),
    ("Rate-limit and moderation-call bookkeeping", "attempts_retention_days"),
]

# Lines in the section that deliberately promise no fixed window. They carry no "N days" and so
# have nothing to check against the code; listed explicitly so that deleting or rewording one
# still trips the completeness assertion below.
UNBOUNDED_LINES = ["kept indefinitely", "Server and Cloudflare request logs"]

_DAYS_RE = re.compile(r"\b(\d+)\s+days?\b")


@pytest.fixture(scope="module")
def retention_bullets() -> list[str]:
    doc = json.loads(LEGAL_JSON.read_text(encoding="utf-8"))["privacy"]
    sections = [s for s in doc["sections"] if s.get("id") == "retention"]
    assert len(sections) == 1, "Privacy needs exactly one section with id 'retention' for this test to bind to"
    return sections[0]["list"]


@pytest.mark.parametrize(("phrase", "argument"), RETENTION_LINES, ids=[a for _, a in RETENTION_LINES])
def test_a_retention_promise_matches_the_window_the_code_enforces(phrase, argument, retention_bullets):
    matching = [line for line in retention_bullets if phrase in line]
    assert len(matching) == 1, f"expected exactly one Privacy §5 line containing {phrase!r}, got {len(matching)}"

    days_in_policy = _DAYS_RE.findall(matching[0])
    assert len(days_in_policy) == 1, f"expected exactly one day count in {matching[0]!r}"

    assert int(days_in_policy[0]) == _retention_default(argument), (
        f"the Privacy Policy tells visitors {days_in_policy[0]} days, but "
        f"{argument} is {_retention_default(argument)} — fix whichever one is wrong, in this commit"
    )


def test_every_retention_line_is_either_checked_or_declared_unbounded(retention_bullets):
    """The guard that keeps this file honest as the policy grows: a new retention promise added to
    §5 with no entry in either table above fails here instead of going unverified forever."""
    checked = {phrase for phrase, _ in RETENTION_LINES} | set(UNBOUNDED_LINES)
    unaccounted = [line for line in retention_bullets if not any(p in line for p in checked)]
    assert not unaccounted, (
        "Privacy §5 lines nobody verifies against the code: "
        f"{unaccounted}. Add each to RETENTION_LINES (if it names a window) or UNBOUNDED_LINES."
    )


def test_no_unbounded_line_quietly_grew_a_deadline(retention_bullets):
    """The inverse drift: a line listed as making no fixed promise must not start naming days
    without being moved into the checked table."""
    for phrase in UNBOUNDED_LINES:
        matching = [line for line in retention_bullets if phrase in line]
        assert len(matching) == 1, f"expected exactly one Privacy §5 line containing {phrase!r}"
        assert not _DAYS_RE.search(matching[0]), (
            f"{matching[0]!r} now names a retention window — move it to RETENTION_LINES so it is "
            "checked against the code"
        )
