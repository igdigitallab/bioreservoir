"""validation.py: `GET /api/validation` payload — assembled from committed files only (task
brief: "every number shown must come from the simulation or committed docs"). Reads the REAL,
already-committed `experiments/001-fly-oracle/calibration/*.json` and `stamps/` files, and the
real (already-cached) harmonized graphs for the brain passport — no Brian2, no simulation, same
convention as every other data-loading test in this suite (`test_encode.py`, `test_live_worker.py`).
"""

from __future__ import annotations

import json

import pytest

from bioreservoir.live import config, validation


def test_brain_passport_matches_docs_data_md_for_malecns():
    passport = validation.brain_passport("malecns", min_syn=5)
    # The numbers already published in docs/DATA.md.
    assert passport["n_neurons"] == 165_122
    assert passport["n_connections"] == 6_235_682
    assert passport["n_synapses"] > passport["n_connections"]  # multiple synapses per connection
    assert passport["min_syn"] == 5


def test_brain_passport_works_for_banc_too():
    passport = validation.brain_passport("banc", min_syn=5)
    assert passport["n_neurons"] == 114_456
    assert passport["dataset"] == "banc"


def test_calibration_summary_reads_the_committed_json_files_verbatim():
    calibration = validation.calibration_summary()
    assert calibration["malecns"] is not None
    assert calibration["banc"] is not None
    malecns = calibration["malecns"]
    assert malecns["source_file"].endswith("malecns.json")
    # docs/MODEL.md quotes exactly this number for 150Hz dose-response.
    assert malecns["dose_response"]["150hz"]["mn9_rate_hz"]["mean"] == pytest.approx(128.5, abs=0.1)
    assert malecns["bitter_suppression"]["suppressed"] is True

    # Cross-check directly against the raw file -- calibration_summary must not retype anything.
    raw = json.loads((config.EXPERIMENT_DIR / "calibration" / malecns["source_file"]).read_text())
    assert malecns["dose_response"] == raw["dose_response"]
    assert malecns["lateral_bias"] == raw["lateral_bias"]


def test_calibration_summary_missing_dataset_is_none(tmp_path):
    result = validation.calibration_summary(calibration_dir=tmp_path)
    assert result == {"malecns": None, "banc": None}


def test_lateral_validation_verdict_malecns_passes_banc_fails():
    calibration = validation.calibration_summary()
    verdict = validation.lateral_validation_verdict(calibration)
    assert verdict["malecns"]["passed"] is True
    assert verdict["banc"]["passed"] is False
    # docs/MODEL.md: male left +0.118 / right -0.157; female collapses to -1.0/-1.0.
    assert verdict["malecns"]["jo_left_driven_descending_bias"]["mean"] > 0
    assert verdict["malecns"]["jo_right_driven_descending_bias"]["mean"] < 0
    assert verdict["banc"]["jo_left_driven_descending_bias"]["mean"] == verdict["banc"]["jo_right_driven_descending_bias"]["mean"]


def test_lateral_validation_verdict_handles_missing_calibration():
    assert validation.lateral_validation_verdict({"malecns": None})["malecns"] is None


def test_handedness_b0_reads_a_populated_ledger_under_the_current_config_hash(monkeypatch, tmp_path):
    """Deterministic version of "read the main ledger": builds a throwaway ledger (never the
    real production-batch file — `config.MAIN_LEDGER_PATH` is `REPO_ROOT`-relative, which inside
    THIS worktree does not even resolve to the main checkout's ledger, so a test against the real
    path would be a no-op here) with 24 `done` (malecns, real) reference runs under the CURRENT
    `config.yaml`'s own config_hash, and confirms `handedness_b0` reads it correctly."""
    from bioreservoir.oracle import config as oracle_config
    from bioreservoir.oracle.ledger import Ledger

    cfg = oracle_config.load_config()
    cfg_hash = oracle_config.config_hash(cfg)

    ledger_path = tmp_path / "ledger.sqlite"
    lg = Ledger(path=ledger_path)
    biases = [-0.05, -0.08, -0.11] * 8  # 24 entries
    for i, bias in enumerate(biases):
        ref_id = f"ref-{i:02d}"
        lg.upsert_reference_sentence(ref_id, f"sentence {i}")
        run_id = lg.start_reference_run(ref_id, "malecns", "real", cfg_hash, trial_seeds=[])
        lg.record_reference_prediction(run_id, mean_bias=bias, n_trials=1, n_zero_spike_trials=0, per_trial_stats=[])
        lg.finish_reference_run(run_id, "done")
    lg.close()

    monkeypatch.setattr(config, "MAIN_LEDGER_PATH", ledger_path)
    result = validation.handedness_b0()
    assert result["brain"] == "malecns"
    assert result["condition"] == "real"
    assert result["config_hash"] == cfg_hash
    assert result["n_reference_sentences"] == 24
    assert result["b0"] == pytest.approx(sum(biases) / len(biases))


def test_handedness_b0_returns_none_for_a_nonexistent_ledger(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "MAIN_LEDGER_PATH", tmp_path / "nope.sqlite")
    result = validation.handedness_b0()
    assert result["b0"] is None
    assert result["n_reference_sentences"] == 0


def test_stamps_summary_reads_committed_manifests_verbatim():
    stamps = validation.stamps_summary()
    assert len(stamps) >= 2
    names = {s["manifest_file"] for s in stamps}
    assert "2026-09-18-pre-batch.sha256" in names
    assert "2026-09-18-preregistration.sha256" in names
    pre_batch = next(s for s in stamps if s["manifest_file"] == "2026-09-18-pre-batch.sha256")
    assert pre_batch["ots_file"] == "2026-09-18-pre-batch.sha256.ots"
    assert any(line.startswith("#") for line in pre_batch["lines"])
    assert any("questions.yaml" in line for line in pre_batch["lines"])

    raw_text = (config.EXPERIMENT_DIR / "stamps" / "2026-09-18-pre-batch.sha256").read_text()
    assert pre_batch["lines"] == raw_text.splitlines()  # verbatim, not reformatted


def test_stamps_summary_empty_dir_returns_empty_list(tmp_path):
    assert validation.stamps_summary(stamps_dir=tmp_path / "nowhere") == []


def test_build_validation_has_every_documented_top_level_key():
    result = validation.build_validation()
    for key in ("model", "brain_passport", "calibration", "lateral_validation", "handedness_b0", "stamps"):
        assert key in result
    assert set(result["brain_passport"].keys()) == {"malecns", "banc"}
