"""Loader for `experiments/001-fly-oracle/reference.yaml` (README.md "Handedness and side
mapping"): 24 short, neutral, non-predictive, non-political sentences, run through every (brain,
condition) exactly like a question's `original` variant (`oracle.encode.encode_question_to_input`,
with an empty `context`), used only to measure each group's own intrinsic lateral bias b0
(`oracle.handedness`). Never scored, never assigned a `P(yes)`.

Like `questions.yaml` (`oracle.questions` docstring), this file is committed and not edited in
place: a wording correction lands as a new `id`, so a previously-computed b0 is never silently
redefined by mutating the sentence text underneath an id the ledger already has results for.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from bioreservoir.oracle.config import EXPERIMENT_DIR

REFERENCE_YAML = EXPERIMENT_DIR / "reference.yaml"


@dataclass(frozen=True)
class ReferenceSentence:
    id: str
    text: str


def load_reference(path: Path = REFERENCE_YAML) -> list[ReferenceSentence]:
    raw = yaml.safe_load(path.read_text())
    return [ReferenceSentence(id=row["id"], text=row["text"]) for row in raw]


def reference_by_id(path: Path = REFERENCE_YAML) -> dict[str, ReferenceSentence]:
    return {r.id: r for r in load_reference(path)}
