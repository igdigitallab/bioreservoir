"""`AnswerSummary` -- the lean, spike-data-free projection of a full Answer dict used by every
PUBLIC surface (`GET /api/feed`, `GET /api/now`, the SSE `answered` event). Scaling task brief:
a full Answer's `lab` object (raster spikes, per-trial rows) is ~150 KB; a summary is a few hundred
bytes, so fanning it out to thousands of open tabs does not mean fanning out the raster to
thousands of open tabs. `GET /api/answers/{id}` still returns the full Answer -- the asker needs
the raster/lab detail to render the reveal animation and the "recompute this" panel.

Kept as its own module (not folded into `card.py` or `api.py`) so it has exactly the two
dependencies it needs (`card.election_embargo` for the embargo gate, `verdict.answer_turn_strength`
for the backfill) without either of THOSE modules needing to import this one back -- `card.py`
renders PNGs from a full Answer, it has no reason to know about the summary shape.
"""

from __future__ import annotations

from bioreservoir.live import card
from bioreservoir.live.verdict import answer_turn_strength

# Every field task brief's `AnswerSummary` names, in that order -- kept as a tuple so tests can
# assert the exact key set without hardcoding it twice.
ANSWER_SUMMARY_FIELDS = (
    "id",
    "question",
    "answer",
    "embargoed",
    "yes_side",
    "lateral_bias",
    "turn_strength",
    "answered_at",
    # How many visitors liked this question (`live_questions.likes`) -- the questions list sorts
    # its "most liked" tab by it, and the live feed shows the same number, so the two never
    # disagree. Not part of the Answer JSON: it changes long after the answer is final.
    "likes",
)


def answer_summary(answer: dict, likes: int = 0) -> dict:
    """Full Answer dict -> `AnswerSummary` dict (task brief). `answer["lab"]["yes_side"]`
    (`lab.build_lab`) is the same handedness coin `answer["lateral_bias"]` was corrected against --
    `.get()`'d defensively since a handful of tests/legacy rows build a minimal Answer fixture with
    no `lab` object at all (`stats.py`'s own docstring notes the same "legacy rows have no `lab`"
    defensiveness rule)."""
    embargoed = card.election_embargo(answer)
    if embargoed:
        return {
            "id": answer["id"],
            "question": answer.get("question", ""),
            "answer": None,
            "embargoed": True,
            "yes_side": None,
            "lateral_bias": None,
            "turn_strength": None,
            "answered_at": answer.get("answered_at"),
            "likes": likes,
        }
    lab = answer.get("lab") or {}
    return {
        "id": answer["id"],
        "question": answer.get("question", ""),
        "answer": answer.get("answer"),
        "embargoed": False,
        "yes_side": lab.get("yes_side"),
        "lateral_bias": answer.get("lateral_bias"),
        "turn_strength": answer_turn_strength(answer),
        "answered_at": answer.get("answered_at"),
        "likes": likes,
    }
