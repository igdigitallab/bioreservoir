from bioreservoir.oracle import plan
from bioreservoir.oracle.config import load_config
from bioreservoir.oracle.ledger import Ledger


def test_build_plan_covers_every_combination():
    items = plan.build_plan(["q1", "q2"], brains=("malecns", "banc"))
    sim_items = [i for i in items if i.condition in plan.SIM_CONDITIONS]
    # 2 questions x 2 brains x 3 sim conditions x 3 text variants
    assert len(sim_items) == 2 * 2 * 3 * 3
    no_brain = [i for i in items if i.condition == "no_brain"]
    assert len(no_brain) == 2 * 2 * 3  # 2 questions x 2 brains x 3 variants
    coin = [i for i in items if i.condition == "coin"]
    assert len(coin) == 2  # one coin per question, brain/variant-independent


def test_coin_items_are_brain_independent_and_original_variant_only():
    items = plan.build_plan(["q1"])
    coin_items = [i for i in items if i.condition == "coin"]
    assert len(coin_items) == 1
    assert coin_items[0].brain == "n/a"
    assert coin_items[0].variant == "original"


def test_group_sim_items_groups_by_brain_and_condition():
    items = plan.build_plan(["q1"], brains=("malecns", "banc"))
    groups = plan.group_sim_items(items)
    assert set(groups.keys()) == {
        (b, c) for b in ("malecns", "banc") for c in plan.SIM_CONDITIONS
    }
    for key, group_items in groups.items():
        assert len(group_items) == 3  # 3 text variants
        assert all(i.brain == key[0] and i.condition == key[1] for i in group_items)


def test_pending_items_skips_done_combinations(tmp_path):
    ledger = Ledger(path=tmp_path / "ledger.sqlite")
    ledger.upsert_question("q1", "politics", "2026-12-15")
    items = plan.build_plan(["q1"], brains=("malecns",))
    one = items[0]
    run_id = ledger.start_run(one.question_id, one.brain, one.condition, one.variant, "cfgA", trial_seeds=[])
    ledger.record_prediction(run_id, p_yes=0.5, mean_bias=0.0, n_trials=1, n_zero_spike_trials=0, per_trial_stats=[])
    ledger.finish_run(run_id, "done")

    remaining = plan.pending_items(items, ledger, "cfgA")
    assert one not in remaining
    assert len(remaining) == len(items) - 1

    # a different config hash must not be treated as already done
    remaining_other_cfg = plan.pending_items(items, ledger, "cfgB")
    assert len(remaining_other_cfg) == len(items)
    ledger.close()


def test_dry_run_report_shape():
    config = load_config()
    report = plan.dry_run_report(["q1", "q2", "q3"], config, n_parallel=3, reference_ids=["r1", "r2"])
    assert report["n_questions"] == 3
    assert report["cost_estimate"]["n_questions"] == 3
    assert report["cost_estimate"]["parallel_hours"] > 0
    assert report["cost_estimate"]["sequential_hours"] >= report["cost_estimate"]["parallel_hours"]
    # 3 questions x 2 brains x 3 conditions x 3 variants x n_trials
    expected_trials = 3 * 2 * 3 * 3 * config.trial.n_trials
    assert report["cost_estimate"]["total_trials"] == expected_trials

    # reference batch: 2 sentences x 2 brains x 3 conditions x 1 "variant" x n_trials
    expected_reference_trials = 2 * 2 * 3 * config.trial.n_trials
    assert report["n_reference_sentences"] == 2
    assert report["reference_cost_estimate"]["total_trials"] == expected_reference_trials
    assert report["combined_cost_estimate"]["total_trials"] == (
        report["cost_estimate"]["total_trials"] + report["reference_cost_estimate"]["total_trials"]
    )


def test_dry_run_report_defaults_reference_ids_from_config_handedness_file():
    config = load_config()
    report = plan.dry_run_report(["q1"], config, n_parallel=3)
    assert report["n_reference_sentences"] == 24  # experiments/001-fly-oracle/reference.yaml


def test_build_reference_plan_covers_every_combination():
    items = plan.build_reference_plan(["r1", "r2"], brains=("malecns", "banc"))
    # 2 reference sentences x 2 brains x 4 conditions (real, rewired, er, no_brain)
    assert len(items) == 2 * 2 * 4
    assert all(i.condition in plan.REFERENCE_CONDITIONS for i in items)
    assert "coin" not in {i.condition for i in items}


def test_pending_reference_items_skips_done_combinations(tmp_path):
    ledger = Ledger(path=tmp_path / "ledger.sqlite")
    ledger.upsert_reference_sentence("r1", "The kettle is on the stove.")
    items = plan.build_reference_plan(["r1"], brains=("malecns",))
    one = items[0]
    run_id = ledger.start_reference_run(one.reference_id, one.brain, one.condition, "cfgA", trial_seeds=[])
    ledger.record_reference_prediction(run_id, mean_bias=0.0, n_trials=1, n_zero_spike_trials=0, per_trial_stats=[])
    ledger.finish_reference_run(run_id, "done")

    remaining = plan.pending_reference_items(items, ledger, "cfgA")
    assert one not in remaining
    assert len(remaining) == len(items) - 1

    remaining_other_cfg = plan.pending_reference_items(items, ledger, "cfgB")
    assert len(remaining_other_cfg) == len(items)
    ledger.close()


def test_group_reference_sim_items_excludes_no_brain():
    items = plan.build_reference_plan(["r1"], brains=("malecns", "banc"))
    groups = plan.group_reference_sim_items(items)
    assert set(groups.keys()) == {(b, c) for b in ("malecns", "banc") for c in plan.SIM_CONDITIONS}
    for group_items in groups.values():
        assert len(group_items) == 1  # 1 reference sentence, no text variants
    assert all(i.condition != "no_brain" for group in groups.values() for i in group)
