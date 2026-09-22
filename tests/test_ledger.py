import json

from bioreservoir.oracle.ledger import Ledger, update_sha256sums


def _ledger(tmp_path):
    return Ledger(path=tmp_path / "ledger.sqlite")


def test_is_done_false_until_finish_run_done(tmp_path):
    lg = _ledger(tmp_path)
    lg.upsert_question("q1", "politics", "2026-12-15")
    assert lg.is_done("q1", "malecns", "real", "original", "cfg1") is False

    run_id = lg.start_run("q1", "malecns", "real", "original", "cfg1", trial_seeds=[1, 2])
    assert lg.is_done("q1", "malecns", "real", "original", "cfg1") is False  # still 'running'

    lg.finish_run(run_id, "done")
    assert lg.is_done("q1", "malecns", "real", "original", "cfg1") is True
    lg.close()


def test_resume_skips_done_but_not_a_different_config_hash(tmp_path):
    lg = _ledger(tmp_path)
    lg.upsert_question("q1", "politics", "2026-12-15")
    run_id = lg.start_run("q1", "malecns", "real", "original", "cfg1", trial_seeds=[1])
    lg.finish_run(run_id, "done")

    assert lg.is_done("q1", "malecns", "real", "original", "cfg1") is True
    # A config change (different seeds/params -> different hash) must NOT be silently skipped.
    assert lg.is_done("q1", "malecns", "real", "original", "cfg2") is False
    lg.close()


def test_start_run_on_same_key_reuses_row_not_duplicates(tmp_path):
    lg = _ledger(tmp_path)
    lg.upsert_question("q1", "politics", "2026-12-15")
    run_id_1 = lg.start_run("q1", "malecns", "real", "original", "cfg1", trial_seeds=[1])
    lg.finish_run(run_id_1, "failed", error="boom")
    assert lg.is_done("q1", "malecns", "real", "original", "cfg1") is False

    run_id_2 = lg.start_run("q1", "malecns", "real", "original", "cfg1", trial_seeds=[1, 2, 3])
    assert run_id_2 == run_id_1
    lg.finish_run(run_id_2, "done")
    assert lg.is_done("q1", "malecns", "real", "original", "cfg1") is True
    lg.close()


def test_record_prediction_and_all_predictions_roundtrip(tmp_path):
    lg = _ledger(tmp_path)
    lg.upsert_question("q1", "politics", "2026-12-15")
    run_id = lg.start_run("q1", "malecns", "real", "original", "cfg1", trial_seeds=[1, 2])
    lg.record_prediction(run_id, p_yes=0.7, mean_bias=0.4, n_trials=2, n_zero_spike_trials=0, per_trial_stats=[{"left": 3, "right": 1}])
    lg.finish_run(run_id, "done")

    rows = lg.all_predictions()
    assert len(rows) == 1
    row = dict(rows[0])
    assert row["p_yes"] == 0.7
    assert json.loads(row["per_trial_json"]) == [{"left": 3, "right": 1}]
    assert row["outcome"] is None  # not resolved yet

    lg.record_resolution("q1", "yes", source="https://apnews.com/")
    rows2 = lg.all_predictions()
    assert dict(rows2[0])["outcome"] == "yes"
    lg.close()


def test_export_predictions_is_append_only_and_idempotent(tmp_path):
    lg = _ledger(tmp_path)
    export_dir = tmp_path / "predictions"
    lg.upsert_question("q1", "politics", "2026-12-15")
    run_id = lg.start_run("q1", "malecns", "real", "original", "cfg1", trial_seeds=[1])
    lg.record_prediction(run_id, p_yes=0.6, mean_bias=0.2, n_trials=1, n_zero_spike_trials=0, per_trial_stats=[])
    lg.finish_run(run_id, "done")

    path1 = lg.export_predictions(date="2026-09-18", export_dir=export_dir)
    assert path1 is not None
    lines_after_first = path1.read_text().strip().splitlines()
    assert len(lines_after_first) == 1

    # nothing new -> None, file untouched
    assert lg.export_predictions(date="2026-09-18", export_dir=export_dir) is None
    assert path1.read_text().strip().splitlines() == lines_after_first

    # a second prediction on a later day appends to a NEW file, the first file is untouched
    lg.upsert_question("q2", "politics", "2026-12-15")
    run_id_2 = lg.start_run("q2", "banc", "real", "original", "cfg1", trial_seeds=[1])
    lg.record_prediction(run_id_2, p_yes=0.4, mean_bias=-0.2, n_trials=1, n_zero_spike_trials=0, per_trial_stats=[])
    lg.finish_run(run_id_2, "done")
    path2 = lg.export_predictions(date="2026-09-19", export_dir=export_dir)
    assert path2 != path1
    assert path1.read_text().strip().splitlines() == lines_after_first  # day-1 file untouched

    sums = (export_dir / "SHA256SUMS").read_text()
    assert path1.name in sums and path2.name in sums
    lg.close()


def test_record_prediction_stores_handedness_fields(tmp_path):
    lg = _ledger(tmp_path)
    lg.upsert_question("q1", "politics", "2026-12-15")
    run_id = lg.start_run("q1", "malecns", "real", "original", "cfg1", trial_seeds=[1])
    lg.record_prediction(
        run_id, p_yes=0.62, mean_bias=-0.1, n_trials=10, n_zero_spike_trials=0, per_trial_stats=[],
        b0=-0.09, corrected_bias=-0.01, left_is_yes=True,
    )
    lg.finish_run(run_id, "done")

    row = dict(lg.all_predictions()[0])
    assert row["b0"] == -0.09
    assert row["corrected_bias"] == -0.01
    assert row["left_is_yes"] == 1  # sqlite stores bool as int
    lg.close()


def test_record_prediction_handedness_fields_default_to_null(tmp_path):
    """Callers that predate handedness correction (e.g. the coin control) omit b0/corrected_bias/
    left_is_yes entirely -- they must round-trip as NULL, not error."""
    lg = _ledger(tmp_path)
    lg.upsert_question("q1", "politics", "2026-12-15")
    run_id = lg.start_run("q1", "n/a", "coin", "original", "cfg1", trial_seeds=[])
    lg.record_prediction(run_id, p_yes=1.0, mean_bias=None, n_trials=1, n_zero_spike_trials=0, per_trial_stats=[])
    lg.finish_run(run_id, "done")

    row = dict(lg.all_predictions()[0])
    assert row["b0"] is None
    assert row["corrected_bias"] is None
    assert row["left_is_yes"] is None
    lg.close()


def test_reference_run_roundtrip(tmp_path):
    lg = _ledger(tmp_path)
    lg.upsert_reference_sentence("ref-01", "The kettle is on the stove.")
    assert lg.is_reference_done("ref-01", "malecns", "real", "cfg1") is False

    run_id = lg.start_reference_run("ref-01", "malecns", "real", "cfg1", trial_seeds=[1, 2])
    assert lg.is_reference_done("ref-01", "malecns", "real", "cfg1") is False  # still 'running'

    lg.record_reference_prediction(
        run_id, mean_bias=-0.08, n_trials=2, n_zero_spike_trials=0, per_trial_stats=[{"left": 3, "right": 4}]
    )
    lg.finish_reference_run(run_id, "done")
    assert lg.is_reference_done("ref-01", "malecns", "real", "cfg1") is True

    biases = lg.reference_mean_biases("malecns", "real", "cfg1")
    assert biases == [-0.08]
    lg.close()


def test_reference_run_resumable_skips_done_but_not_a_different_config_hash(tmp_path):
    lg = _ledger(tmp_path)
    lg.upsert_reference_sentence("ref-01", "The kettle is on the stove.")
    run_id = lg.start_reference_run("ref-01", "malecns", "real", "cfg1", trial_seeds=[1])
    lg.record_reference_prediction(run_id, mean_bias=-0.05, n_trials=1, n_zero_spike_trials=0, per_trial_stats=[])
    lg.finish_reference_run(run_id, "done")

    assert lg.is_reference_done("ref-01", "malecns", "real", "cfg1") is True
    assert lg.is_reference_done("ref-01", "malecns", "real", "cfg2") is False  # different config hash
    lg.close()


def test_reference_run_retry_reuses_row_not_duplicates(tmp_path):
    lg = _ledger(tmp_path)
    lg.upsert_reference_sentence("ref-01", "The kettle is on the stove.")
    run_id_1 = lg.start_reference_run("ref-01", "malecns", "real", "cfg1", trial_seeds=[1])
    lg.finish_reference_run(run_id_1, "failed", error="boom")
    assert lg.is_reference_done("ref-01", "malecns", "real", "cfg1") is False

    run_id_2 = lg.start_reference_run("ref-01", "malecns", "real", "cfg1", trial_seeds=[1, 2])
    assert run_id_2 == run_id_1
    lg.record_reference_prediction(run_id_2, mean_bias=-0.02, n_trials=2, n_zero_spike_trials=0, per_trial_stats=[])
    lg.finish_reference_run(run_id_2, "done")
    assert lg.is_reference_done("ref-01", "malecns", "real", "cfg1") is True
    lg.close()


def test_update_sha256sums_covers_all_jsonl_files(tmp_path):
    export_dir = tmp_path / "predictions"
    export_dir.mkdir()
    (export_dir / "2026-09-18.jsonl").write_text('{"a": 1}\n')
    (export_dir / "2026-09-19.jsonl").write_text('{"b": 2}\n')
    sums_path = update_sha256sums(export_dir)
    text = sums_path.read_text()
    assert "2026-09-18.jsonl" in text
    assert "2026-09-19.jsonl" in text
    assert len(text.strip().splitlines()) == 2
