"""stats.py: `compute_stats` — pure aggregation over plain answer dicts (task brief `GET
/api/stats`, "no fake values; zeros are fine"). No SQLite here (`store.LiveStore.all_answered`
owns the query).
"""

from __future__ import annotations

import pytest

from bioreservoir.live import stats
from bioreservoir.live.states import STATE_NAMES


def _row(answer: str, **overrides):
    base = {
        "question": overrides.pop("question", "Will it rain?"),
        "answer": answer,
        "sim_ms": 250.0,
        "n_trials": 3,
        "lateral_bias": 0.1,
        "states": {name: None for name in STATE_NAMES},
        "lab": {
            "total_spikes": 1000,
            "active_fraction": 0.05,
            "provenance": {"n_neurons": 165122, "dt_ms": 0.1},
        },
    }
    base.update(overrides)
    return {"answer": base, "created_at": overrides.get("created_at", "2026-09-18T00:00:00+00:00")}


def test_compute_stats_on_an_empty_store_is_all_zeros():
    result = stats.compute_stats([], earliest_created_at=None)
    assert result["answered"] == 0
    assert result["yes"] == 0
    assert result["no"] == 0
    assert result["total_spikes"] == 0
    assert result["simulated_ms"] == 0.0
    assert result["neuron_updates"] == 0
    assert result["mean_abs_corrected_bias"] == 0.0
    assert result["mean_active_fraction"] == 0.0
    assert result["state_counts"] == dict.fromkeys(STATE_NAMES, 0)
    assert result["repeat_consistency"] == {"repeated_questions": 0, "identical_answers": 0}
    assert result["since"] is None


def test_compute_stats_counts_yes_and_no():
    rows = [_row("yes"), _row("yes"), _row("no")]
    result = stats.compute_stats(rows, "2026-09-18T00:00:00+00:00")
    assert result["answered"] == 3
    assert result["yes"] == 2
    assert result["no"] == 1


def test_compute_stats_sums_total_spikes_and_simulated_ms():
    rows = [_row("yes", lab={"total_spikes": 100, "active_fraction": 0.1, "provenance": {"n_neurons": 1000, "dt_ms": 0.1}}), _row("no", lab={"total_spikes": 200, "active_fraction": 0.2, "provenance": {"n_neurons": 1000, "dt_ms": 0.1}})]
    result = stats.compute_stats(rows, "2026-09-18T00:00:00+00:00")
    assert result["total_spikes"] == 300
    assert result["simulated_ms"] == pytest.approx(250.0 * 3 * 2)  # sim_ms x n_trials, per row


def test_compute_stats_neuron_updates_is_neurons_times_steps_times_trials():
    rows = [_row("yes", sim_ms=250.0, n_trials=3, lab={"total_spikes": 0, "active_fraction": 0.0, "provenance": {"n_neurons": 1000, "dt_ms": 0.1}})]
    result = stats.compute_stats(rows, "2026-09-18T00:00:00+00:00")
    # steps = 250 / 0.1 = 2500; neuron_updates = 1000 neurons x 3 trials x 2500 steps
    assert result["neuron_updates"] == 1000 * 3 * 2500


def test_compute_stats_mean_abs_corrected_bias_and_active_fraction():
    rows = [
        _row("yes", lateral_bias=0.2, lab={"total_spikes": 0, "active_fraction": 0.10, "provenance": {"n_neurons": 100, "dt_ms": 0.1}}),
        _row("no", lateral_bias=-0.4, lab={"total_spikes": 0, "active_fraction": 0.30, "provenance": {"n_neurons": 100, "dt_ms": 0.1}}),
    ]
    result = stats.compute_stats(rows, "2026-09-18T00:00:00+00:00")
    assert result["mean_abs_corrected_bias"] == pytest.approx((0.2 + 0.4) / 2)
    assert result["mean_active_fraction"] == pytest.approx((0.10 + 0.30) / 2)


def test_compute_stats_state_counts_only_counts_non_null_values():
    rows = [
        _row("yes", states={"appetite": None, "fear": 0.5, "backoff": None, "courtship": None, "arousal": 0.3}),
        _row("no", states={"appetite": 0.1, "fear": None, "backoff": None, "courtship": None, "arousal": 0.9}),
    ]
    result = stats.compute_stats(rows, "2026-09-18T00:00:00+00:00")
    assert result["state_counts"] == {"appetite": 1, "fear": 1, "backoff": 0, "courtship": 0, "arousal": 2}


def test_compute_stats_repeat_consistency_counts_distinct_repeated_questions():
    rows = [
        _row("yes", question="Will it rain?"),
        _row("yes", question="Will it rain?"),  # same question, same answer -> consistent
        _row("yes", question="Is the sky blue?"),
        _row("no", question="Is the sky blue?"),  # same question, DIFFERENT answer -> inconsistent
        _row("no", question="Only asked once"),  # not repeated at all
    ]
    result = stats.compute_stats(rows, "2026-09-18T00:00:00+00:00")
    assert result["repeat_consistency"]["repeated_questions"] == 2  # "Will it rain?" and "Is the sky blue?"
    assert result["repeat_consistency"]["identical_answers"] == 1  # only "Will it rain?" was consistent


def test_compute_stats_handles_missing_lab_gracefully_zeros_not_crash():
    row = _row("yes")
    del row["answer"]["lab"]
    result = stats.compute_stats([row], "2026-09-18T00:00:00+00:00")
    assert result["answered"] == 1
    assert result["total_spikes"] == 0
    assert result["neuron_updates"] == 0
    assert result["mean_active_fraction"] == 0.0


def test_compute_stats_since_passes_through_earliest_created_at():
    result = stats.compute_stats([], earliest_created_at="2026-01-01T00:00:00+00:00")
    assert result["since"] == "2026-01-01T00:00:00+00:00"
