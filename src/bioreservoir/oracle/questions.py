"""Loader for `experiments/001-fly-oracle/questions.yaml` (33 pre-registered questions, each with
an `original`/`negation`/`paraphrase` text variant — see `scripts/add_question_variants.py` for how
`negation`/`paraphrase` were derived, and README.md's "Invariance tests" for why they exist).

Questions are committed and never edited in place (questions.yaml's own header comment; corrections
land as a new `id` with a `supersedes:` field) — this module only reads, never writes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from bioreservoir.oracle.config import QUESTIONS_YAML, TEXT_VARIANTS


@dataclass(frozen=True)
class Question:
    id: str
    format: str
    category: str
    question: str
    negation: str
    paraphrase: str
    context: str
    resolves_by: str
    resolution_criterion: str
    resolution_source: str
    incumbent_party: str | None = None  # "Republican" | "Democratic" | None (questions.yaml's own
    # field, used only for the side-mapping balance report — oracle.side_mapping never reads it,
    # the mapping is seeded from `id` alone so it cannot be influenced by which party holds a seat)

    def text(self, variant: str) -> str:
        """Question text for `variant` ("original" | "negation" | "paraphrase")."""
        if variant not in TEXT_VARIANTS:
            raise ValueError(f"unknown text variant: {variant!r} (expected one of {TEXT_VARIANTS})")
        if variant == "original":
            return self.question
        return getattr(self, variant)


def load_questions(path: Path = QUESTIONS_YAML) -> list[Question]:
    raw = yaml.safe_load(path.read_text())
    questions = []
    for row in raw:
        if row.get("format") != "scored":
            continue  # format "take" ("Fly's take") is out of scope for the scored pipeline
        questions.append(
            Question(
                id=row["id"],
                format=row["format"],
                category=row["category"],
                question=row["question"],
                negation=row["negation"],
                paraphrase=row["paraphrase"],
                context=row["context"].strip(),
                resolves_by=row["resolves_by"],
                resolution_criterion=row["resolution_criterion"].strip(),
                resolution_source=row["resolution_source"],
                incumbent_party=row.get("incumbent_party"),
            )
        )
    return questions


def questions_by_id(path: Path = QUESTIONS_YAML) -> dict[str, Question]:
    return {q.id: q for q in load_questions(path)}
