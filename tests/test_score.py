"""Scoring math, exercised against a temp ledger seeded with real question ids from
questions.yaml (score.build_report reads the committed question set to know each question's
`cluster_by` field) but fabricated predictions/resolutions — no simulation involved.
"""

from bioreservoir.oracle.ledger import Ledger
from bioreservoir.oracle.questions import load_questions
from bioreservoir.oracle.score import (
    ScoredPrediction,
    accuracy,
    brier_score,
    build_report,
    cluster_summarize,
    invariance_report,
    log_loss,
)

QUESTIONS = load_questions()
Q1, Q2 = QUESTIONS[0].id, QUESTIONS[1].id


def test_brier_and_log_loss_perfect_prediction_is_zero_and_small():
    assert brier_score(1.0, 1) == 0.0
    assert brier_score(0.0, 0) == 0.0
    assert log_loss(1.0 - 1e-9, 1) < 1e-5


def test_brier_and_log_loss_worst_prediction_is_maximal():
    assert brier_score(0.0, 1) == 1.0
    assert log_loss(0.0, 1) > 10  # clipped by eps, not literally infinite


def test_accuracy_counts_ties_at_half_as_incorrect():
    preds = [
        ScoredPrediction(Q1, "malecns", "real", "original", 0.5, 1, "politics"),
        ScoredPrediction(Q1, "malecns", "real", "original", 0.9, 1, "politics"),
    ]
    assert accuracy(preds) == 0.5


def test_cluster_summarize_averages_within_cluster():
    preds = [
        ScoredPrediction(Q1, "malecns", "real", "original", 0.8, 1, "politics"),
        ScoredPrediction(Q2, "malecns", "real", "original", 0.4, 0, "politics"),
    ]
    report = cluster_summarize(preds)
    assert report["n_clusters"] == 1
    row = report["clusters"][0]
    assert row["cluster"] == "politics"
    assert row["n_questions"] == 2
    assert abs(row["mean_p_yes"] - 0.6) < 1e-9
    assert abs(row["mean_outcome_rate"] - 0.5) < 1e-9


def test_invariance_report_flags_perfect_and_broken_negation_complement():
    good = [
        ScoredPrediction(Q1, "malecns", "real", "original", 0.7, 1, "politics"),
        ScoredPrediction(Q1, "malecns", "real", "negation", 0.3, 0, "politics"),
    ]
    report_good = invariance_report(good)
    assert abs(report_good["rows"][0]["negation_complement_gap"]) < 1e-9

    broken = [
        ScoredPrediction(Q1, "malecns", "real", "original", 0.7, 1, "politics"),
        ScoredPrediction(Q1, "malecns", "real", "negation", 0.9, 0, "politics"),  # should be ~0.3
    ]
    report_broken = invariance_report(broken)
    assert report_broken["rows"][0]["negation_complement_gap"] > 0.5


def test_build_report_uses_the_stored_corrected_p_yes_not_the_raw_bias(tmp_path):
    """score.py must trust the ledger's already-corrected `p_yes` (`oracle.handedness.
    apply_correction`, applied at run time) and never recompute anything from `mean_bias`/`b0` --
    here the raw bias is positive (would look YES-favoring under the old, uncorrected convention)
    but the stored, corrected p_yes is deliberately on the NO side; the report's accuracy must
    follow the stored p_yes."""
    ledger = Ledger(path=tmp_path / "ledger.sqlite")
    ledger.upsert_question(Q1, "politics", "2026-12-15")

    raw_bias = 0.3  # would give p_yes = 0.65 uncorrected
    b0 = 0.5  # this group's own handedness is even more leftward than the question's raw bias
    corrected = raw_bias - b0  # -0.2
    p_yes = (1.0 + corrected) / 2.0  # 0.4 -- on the NO side, opposite of the raw/uncorrected call
    assert p_yes < 0.5

    run_id = ledger.start_run(Q1, "malecns", "real", "original", "cfg1", trial_seeds=[])
    ledger.record_prediction(
        run_id, p_yes=p_yes, mean_bias=raw_bias, n_trials=10, n_zero_spike_trials=0, per_trial_stats=[],
        b0=b0, corrected_bias=corrected, left_is_yes=True,
    )
    ledger.finish_run(run_id, "done")
    ledger.record_resolution(Q1, "no", source="https://apnews.com/")  # matches the corrected call

    report = build_report(ledger, cluster_by="category")
    per_race = report["by_brain_condition"]["malecns/real"]["per_race"]
    assert per_race["accuracy"] == 1.0  # correct because p_yes (corrected) < 0.5 and outcome is "no"
    ledger.close()


def test_build_report_end_to_end_on_a_temp_ledger(tmp_path):
    ledger = Ledger(path=tmp_path / "ledger.sqlite")
    ledger.upsert_question(Q1, "politics", "2026-12-15")

    def add(brain, condition, variant, p_yes, outcome=None):
        run_id = ledger.start_run(Q1, brain, condition, variant, "cfg1", trial_seeds=[])
        ledger.record_prediction(run_id, p_yes, mean_bias=0.0, n_trials=1, n_zero_spike_trials=0, per_trial_stats=[])
        ledger.finish_run(run_id, "done")

    add("malecns", "real", "original", 0.8)
    add("malecns", "real", "negation", 0.2)
    add("malecns", "rewired", "original", 0.55)
    ledger.record_resolution(Q1, "yes", source="https://apnews.com/")

    report = build_report(ledger, cluster_by="category")
    assert report["n_scored_predictions_total"] == 3
    assert report["n_original_variant_predictions"] == 2  # real + rewired, negation excluded
    assert "malecns/real" in report["by_brain_condition"]
    assert report["by_brain_condition"]["malecns/real"]["per_race"]["n"] == 1
    assert report["by_brain_condition"]["malecns/real"]["per_race"]["accuracy"] == 1.0
    assert report["invariance"]["mean_negation_complement_gap"] is not None
    ledger.close()
