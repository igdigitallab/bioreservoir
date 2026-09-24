"""`python -m bioreservoir.oracle score` — accuracy, Brier, log loss, calibration, invariance
(README.md Pipeline step 6 + "What counts as a result" + "Invariance tests").

Reads resolved predictions from the ledger (`oracle.ledger.Ledger.all_predictions`, joined against
`resolutions`); questions with no resolution yet are excluded from every metric (nothing to score
against) but still counted in the report so an unresolved batch is visibly incomplete, not silently
0-scored. Also writes `site/data/scoreboard.json` for the scoreboard page (docs/ROADMAP.md).
"""

from __future__ import annotations

import itertools
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from bioreservoir.oracle.config import REPO_ROOT
from bioreservoir.oracle.ledger import Ledger
from bioreservoir.oracle.questions import Question, load_questions

SCOREBOARD_JSON = REPO_ROOT / "site" / "data" / "scoreboard.json"


@dataclass(frozen=True)
class ScoredPrediction:
    question_id: str
    brain: str
    condition: str
    variant: str
    p_yes: float
    outcome: int  # 1 = yes, 0 = no
    cluster: str


def brier_score(p_yes: float, outcome: int) -> float:
    return (p_yes - outcome) ** 2


def log_loss(p_yes: float, outcome: int, eps: float = 1e-6) -> float:
    p = min(max(p_yes, eps), 1.0 - eps)
    return -(outcome * math.log(p) + (1 - outcome) * math.log(1.0 - p))


def accuracy(predictions: list[ScoredPrediction]) -> float | None:
    """Fraction where `p_yes > 0.5` matches `outcome` (`p_yes == 0.5` counts as a miss either way
    — an honestly-uncertain call should not score as "correct", see README.md's zero-spike-trial
    handling for the same "0.5 is not a free pass" stance)."""
    if not predictions:
        return None
    correct = sum(1 for p in predictions if (p.p_yes > 0.5) == bool(p.outcome))
    return correct / len(predictions)


def calibration_bins(predictions: list[ScoredPrediction], n_bins: int = 5) -> list[dict]:
    """Reliability diagram data: for each `[i/n_bins, (i+1)/n_bins)` bucket of `p_yes`, the mean
    predicted probability vs. the observed frequency of `outcome == 1`, and the bucket's size."""
    edges = [i / n_bins for i in range(n_bins + 1)]
    bins = []
    for lo, hi in itertools.pairwise(edges):
        bucket = [p for p in predictions if (lo <= p.p_yes < hi) or (hi == 1.0 and p.p_yes == 1.0)]
        if not bucket:
            bins.append({"range": [lo, hi], "n": 0, "mean_p_yes": None, "observed_freq_yes": None})
            continue
        bins.append(
            {
                "range": [lo, hi],
                "n": len(bucket),
                "mean_p_yes": sum(p.p_yes for p in bucket) / len(bucket),
                "observed_freq_yes": sum(p.outcome for p in bucket) / len(bucket),
            }
        )
    return bins


def summarize(predictions: list[ScoredPrediction]) -> dict:
    if not predictions:
        return {"n": 0, "accuracy": None, "brier": None, "log_loss": None, "calibration": calibration_bins([])}
    n = len(predictions)
    return {
        "n": n,
        "accuracy": accuracy(predictions),
        "brier": sum(brier_score(p.p_yes, p.outcome) for p in predictions) / n,
        "log_loss": sum(log_loss(p.p_yes, p.outcome) for p in predictions) / n,
        "calibration": calibration_bins(predictions),
    }


def cluster_summarize(predictions: list[ScoredPrediction]) -> dict:
    """One row per `outcome`-cluster (config.yaml's `scoring.cluster_by`, README.md "correlated
    events... count as a cluster rather than as independent draws"): within a cluster, average the
    per-question `p_yes` before scoring, so a 12-race cluster contributes one data point, not 12
    correlated ones, to the clustered view. Callers get both `summarize` (per-race) and this
    (clustered) side by side, never only one."""
    by_cluster: dict[str, list[ScoredPrediction]] = defaultdict(list)
    for p in predictions:
        by_cluster[p.cluster].append(p)
    cluster_points = []
    for cluster, preds in by_cluster.items():
        mean_p = sum(p.p_yes for p in preds) / len(preds)
        # A cluster's own "outcome" only makes sense if every member resolved the same way, which
        # is not guaranteed (a 12-race election cluster has 12 different winners) — so the
        # clustered view reports each cluster as one row of (mean_p_yes, mean_outcome), which is
        # itself a Brier-style score against the cluster's own average resolution rate, not a
        # single yes/no accuracy per cluster.
        mean_outcome = sum(p.outcome for p in preds) / len(preds)
        cluster_points.append(
            {
                "cluster": cluster,
                "n_questions": len(preds),
                "mean_p_yes": mean_p,
                "mean_outcome_rate": mean_outcome,
                "brier_of_cluster_mean": (mean_p - mean_outcome) ** 2,
            }
        )
    return {"n_clusters": len(cluster_points), "clusters": cluster_points}


def invariance_report(all_scored: list[ScoredPrediction]) -> dict:
    """For each (question, brain, condition): does `P(yes|negation) ~ 1 - P(yes|original)`, and
    does `P(yes|paraphrase) ~ P(yes|original)` (README.md "Invariance tests")? Reported as-is, not
    filtered — "inconsistent answers are published as they are, because they are part of the
    result" (README.md)."""
    by_key: dict[tuple[str, str, str], dict[str, float]] = defaultdict(dict)
    for p in all_scored:
        by_key[(p.question_id, p.brain, p.condition)][p.variant] = p.p_yes

    rows = []
    for (question_id, brain, condition), by_variant in by_key.items():
        original = by_variant.get("original")
        negation = by_variant.get("negation")
        paraphrase = by_variant.get("paraphrase")
        row = {
            "question_id": question_id,
            "brain": brain,
            "condition": condition,
            "p_yes_original": original,
            "p_yes_negation": negation,
            "p_yes_paraphrase": paraphrase,
            "negation_complement_gap": None if original is None or negation is None else abs((1.0 - original) - negation),
            "paraphrase_gap": None if original is None or paraphrase is None else abs(original - paraphrase),
        }
        rows.append(row)
    gaps_neg = [r["negation_complement_gap"] for r in rows if r["negation_complement_gap"] is not None]
    gaps_para = [r["paraphrase_gap"] for r in rows if r["paraphrase_gap"] is not None]
    return {
        "rows": rows,
        "mean_negation_complement_gap": sum(gaps_neg) / len(gaps_neg) if gaps_neg else None,
        "mean_paraphrase_gap": sum(gaps_para) / len(gaps_para) if gaps_para else None,
    }


def load_scored_predictions(ledger: Ledger, questions: dict[str, Question], cluster_by: str) -> list[ScoredPrediction]:
    scored = []
    for row in ledger.all_predictions():
        if row["outcome"] is None or row["outcome"] == "void":
            continue  # unresolved, or resolved void (e.g. a runoff cancelled the race) — excluded
        question = questions.get(row["question_id"])
        if question is None:
            continue
        cluster = getattr(question, cluster_by)
        scored.append(
            ScoredPrediction(
                question_id=row["question_id"],
                brain=row["brain"],
                condition=row["condition"],
                variant=row["variant"],
                p_yes=row["p_yes"],
                outcome=1 if row["outcome"] == "yes" else 0,
                cluster=cluster,
            )
        )
    return scored


def build_report(ledger: Ledger, cluster_by: str = "category") -> dict:
    questions = {q.id: q for q in load_questions()}
    scored = load_scored_predictions(ledger, questions, cluster_by)

    # Only score the pre-registered primary text variant ("original") per README.md's "Primary
    # endpoint" — negation/paraphrase/yes-no-swap are the separate invariance report above, not
    # additional (correlated, double-counted) data points in the main scoreboard.
    original_only = [p for p in scored if p.variant == "original"]
    brain_conditions = sorted({(p.brain, p.condition) for p in original_only})

    by_brain_condition: dict[str, dict] = {}
    for brain, condition in brain_conditions:
        key = f"{brain}/{condition}"
        subset = [p for p in original_only if p.brain == brain and p.condition == condition]
        by_brain_condition[key] = {
            "per_race": summarize(subset),
            "clustered": cluster_summarize(subset),
        }

    return {
        "n_scored_predictions_total": len(scored),
        "n_original_variant_predictions": len(original_only),
        "by_brain_condition": by_brain_condition,
        "invariance": invariance_report(scored),
    }


def write_scoreboard_json(report: dict, path: Path = SCOREBOARD_JSON) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=str))
    return path
