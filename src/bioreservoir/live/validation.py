"""`GET /api/validation` — assembled ONLY from files this repo already ships (operator
requirement: "every number shown must come from the simulation or committed docs"): the
harmonized graph's own neuron/connection/synapse counts, `docs/MODEL.md`'s calibration JSONs
(`experiments/001-fly-oracle/calibration/*.json`, read as-is, never retyped), the OpenTimestamps
stamp manifests (`experiments/001-fly-oracle/stamps/`), and (if the main ledger has them) the
`(malecns, real)` handedness reference runs. No network access, no simulation.
"""

from __future__ import annotations

from pathlib import Path

from bioreservoir.live import config

MODEL_CITATION = "LIF, Shiu et al. 2024"


def brain_passport(dataset: str, min_syn: int) -> dict:
    """Neuron/connection/synapse counts straight from the harmonized graph (same computation as
    `worker.build_resources`'s `lab.LabContext` fields, for the OTHER brain too — `/api/validation`
    is not live-page-specific, it documents both MaleCNS and BANC)."""
    import numpy as np

    from bioreservoir.sim import bench

    n_neurons, pre_idx, _post_idx, weight, _id_to_dense = bench.graph_to_arrays(dataset, min_syn=min_syn)
    return {
        "dataset": dataset,
        "n_neurons": n_neurons,
        "n_connections": len(pre_idx),
        "n_synapses": int(np.abs(weight).sum()),
        "min_syn": min_syn,
    }


def _latest_calibration_file(dataset: str, calibration_dir: Path) -> Path | None:
    candidates = sorted(calibration_dir.glob(f"*-{dataset}.json"))
    return candidates[-1] if candidates else None  # YYYY-MM-DD- prefix sorts chronologically


def calibration_summary(calibration_dir: Path = config.EXPERIMENT_DIR / "calibration") -> dict:
    """The specific calibration numbers `docs/MODEL.md`'s Calibration section quotes, read
    straight out of the committed JSON — dose-response, bitter suppression, lateral bias — for
    both brains. `None` for a dataset with no calibration file committed yet."""
    import json

    result: dict[str, dict | None] = {}
    for dataset in ("malecns", "banc"):
        path = _latest_calibration_file(dataset, calibration_dir)
        if path is None:
            result[dataset] = None
            continue
        raw = json.loads(path.read_text())
        result[dataset] = {
            "source_file": path.name,
            "n_neurons": raw.get("n_neurons"),
            "n_synapses": raw.get("n_synapses"),
            "dose_response": raw.get("dose_response"),
            "bitter_suppression": raw.get("bitter_suppression"),
            "lateral_bias": raw.get("lateral_bias"),
        }
    return result


def lateral_validation_verdict(calibration: dict) -> dict:
    """"male passed / female failed" (task brief), derived from the SAME numbers
    `calibration_summary` already carries — docs/MODEL.md's own qualitative criterion: a stable,
    opposite-signed bias that tracks which side is driven (male), vs. a bias that collapses to a
    stimulus-independent constant regardless of which side is driven (female). `passed` is
    computed, not asserted: `left_vs_right_bias_gap > 0.05` (an order of magnitude below the
    male's own measured gap of ~0.27, so this is not a knife-edge threshold) AND the two driven
    sides read opposite-signed."""
    verdict: dict[str, dict | None] = {}
    for dataset, cal in calibration.items():
        lateral = (cal or {}).get("lateral_bias")
        if not lateral:
            verdict[dataset] = None
            continue
        left = (lateral.get("jo_left_driven") or {}).get("descending_bias") or {}
        right = (lateral.get("jo_right_driven") or {}).get("descending_bias") or {}
        gap = lateral.get("left_vs_right_bias_gap")
        left_mean, right_mean = left.get("mean"), right.get("mean")
        passed = bool(
            left_mean is not None
            and right_mean is not None
            and gap is not None
            and left_mean > 0 > right_mean
            and gap > 0.05
        )
        verdict[dataset] = {
            "passed": passed,
            "jo_left_driven_descending_bias": left,
            "jo_right_driven_descending_bias": right,
            "left_vs_right_bias_gap": gap,
        }
    return verdict


def handedness_b0(brain: str = config.LIVE_BRAIN, condition: str = "real") -> dict:
    """`(malecns, real)` handedness b0, read-only from the main ledger's already-`done`
    reference runs under the CURRENT `config.yaml`'s config hash (`worker.
    read_only_reference_mean_biases` — never a write, never `oracle.ledger.Ledger` against the
    live production-batch file). `b0: None` if that group has no `done` reference runs yet."""
    from bioreservoir.live import worker
    from bioreservoir.oracle import config as oracle_config
    from bioreservoir.oracle import handedness as handedness_mod

    cfg = oracle_config.load_config()
    cfg_hash = oracle_config.config_hash(cfg)
    biases = worker.read_only_reference_mean_biases(config.MAIN_LEDGER_PATH, brain, condition, cfg_hash)
    if not biases:
        return {"brain": brain, "condition": condition, "config_hash": cfg_hash, "b0": None, "n_reference_sentences": 0}
    return {
        "brain": brain,
        "condition": condition,
        "config_hash": cfg_hash,
        "b0": handedness_mod.compute_b0(biases),
        "n_reference_sentences": len(biases),
    }


def stamps_summary(stamps_dir: Path = config.EXPERIMENT_DIR / "stamps") -> list[dict]:
    """One entry per `*.sha256` manifest: its filename, its raw lines (comments + `hash  path`
    rows, verbatim — never re-parsed/reformatted) and its `.ots` OpenTimestamps proof's filename
    if one has been committed alongside it."""
    if not stamps_dir.exists():
        return []
    result = []
    for sha_path in sorted(stamps_dir.glob("*.sha256")):
        ots_path = sha_path.with_name(sha_path.name + ".ots")
        result.append(
            {
                "manifest_file": sha_path.name,
                "lines": sha_path.read_text().splitlines(),
                "ots_file": ots_path.name if ots_path.exists() else None,
            }
        )
    return result


def build_validation() -> dict:
    from bioreservoir.oracle import config as oracle_config

    cfg = oracle_config.load_config()
    passports = {ds: brain_passport(ds, min_syn=cfg.trial.min_syn) for ds in ("malecns", "banc")}
    calibration = calibration_summary()
    return {
        "model": MODEL_CITATION,
        "brain_passport": passports,
        "calibration": calibration,
        "lateral_validation": lateral_validation_verdict(calibration),
        "handedness_b0": handedness_b0(),
        "stamps": stamps_summary(),
    }
