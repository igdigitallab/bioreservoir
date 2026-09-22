"""Per-question YES-side mapping (README.md "Handedness and side mapping").

Replaces the old global `config.yaml` `readout.left_is_yes` flag. A single global flag would have
meant every one of the 33 "will <incumbent party> hold <seat>?" questions shared one answer key —
so the real brain's own intrinsic lateral bias ("handedness", see `oracle.handedness`) would have
read as a uniform partisan artefact rather than as noise (2026-09-18 smoke run, memory
`fly-handedness.md`). Instead, each question id gets its own deterministic coin, seeded from
`config.yaml`'s `seed.side_base` via `oracle.seeding.stable_seed`, deciding whether a leftward
(positive, handedness-corrected) bias means YES or NO *for that question*. The mapping never reads
the question's content or its `incumbent_party` — only its `id` — so it cannot be steered by
anything about the question itself, and it is recorded per prediction in the ledger
(`predictions.left_is_yes`) so every run is auditable after the fact.

This also makes the YES/NO-swap invariance check (README.md "Invariance tests") structural rather
than a manually flipped config flag for one extra run: roughly half of the 33 questions already
use each mapping, so a brain that were quietly using the raw embedding polarity as its answer
(rather than anything question-specific) would show up as no better than chance across the set,
not as a hidden uniform skew. Negation and paraphrase text variants are unrelated to this and are
unchanged (`oracle.questions.TEXT_VARIANTS`).
"""

from __future__ import annotations

from bioreservoir.oracle.seeding import stable_seed


def left_is_yes_for(question_id: str, side_base: int) -> bool:
    """Deterministic per-question coin: `True` if a positive (leftward), handedness-corrected
    bias means YES for `question_id`; `False` if it means NO. Depends only on `question_id` and
    `side_base` — same question id, same config -> same mapping, forever (until `side_base`
    itself changes, which is a config-hash-busting, pre-registration-breaking change like any
    other seed in `config.yaml`'s `seed:` block)."""
    return stable_seed(question_id, base=side_base) % 2 == 0
