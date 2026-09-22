"""Operator requirement (2026-09-18): "every number shown must come from the simulation or
committed docs — never decorative". Pure "lab" statistics for `Answer.lab` — no Brian2 import,
consumes already-run `pipeline.LiveTrial` results the same way `pipeline.py` does, so every piece
here (raster selection, top cell types, readout latency, stimulated-population tally) is
independently unit-testable with synthetic dense-index arrays, no real connectome required.

`LabContext` bundles everything about the running process/brain that does NOT change between
questions (graph metadata, per-neuron annotation lookups, provenance strings) — built once by
`worker.build_resources()` (or the `reproduce` CLI) and passed into every `pipeline.compute_answer`
call, so per-question logic never re-reads graph metadata or annotation files.
"""

from __future__ import annotations

import base64
import os
import shlex
from dataclasses import dataclass

import numpy as np

RASTER_MAX_NEURONS = 300
TOP_CELL_TYPES_N = 12
MODEL_CITATION = "LIF, Shiu et al. 2024"

# The "Recompute this answer" command has to work for a stranger with an empty terminal, not only
# inside an already-set-up checkout, so it starts from a bare `git clone`. The repository went
# public on 2026-09-21 and this is its URL; override it with BIORESERVOIR_REPO_URL (set it empty
# to drop the clone line entirely, which is what a fork without a public mirror should do — a
# command that 404s is worse than no command). Same gate as the frontend's VITE_REPO_URL
# (site/src/pages/aboutLab.ts).
REPO_URL = os.environ.get("BIORESERVOIR_REPO_URL", "https://github.com/igdigitallab/bioreservoir").strip()


@dataclass(frozen=True)
class LabContext:
    brain_name: str  # e.g. "MaleCNS v1.0" -> Answer.lab.provenance.brain
    n_neurons: int
    n_connections: int
    n_synapses: int
    min_syn: int
    dt_ms: float
    code_sha: str | None
    config_hash: str
    cell_type_lookup: dict[int, str]  # dense_idx -> cell_type ("unknown" if null)
    super_class_lookup: dict[int, str]  # dense_idx -> super_class ("unknown" if null)
    modality_lookup: dict[int, str]  # dense_idx -> raw per-dataset modality ("unknown" if null)
    readout_left_idx: np.ndarray
    readout_right_idx: np.ndarray
    dense_to_atlas: dict[int, int] | None


def stimulated_summary(
    stimulated_dense_idx: np.ndarray,
    stimulated_left_idx: np.ndarray,
    stimulated_right_idx: np.ndarray,
    modality_lookup: dict[int, str],
) -> dict:
    by_modality: dict[str, int] = {}
    for idx in stimulated_dense_idx.tolist():
        modality = modality_lookup.get(int(idx), "unknown")
        by_modality[modality] = by_modality.get(modality, 0) + 1
    return {
        "total": int(stimulated_dense_idx.size),
        "left": int(stimulated_left_idx.size),
        "right": int(stimulated_right_idx.size),
        "by_modality": by_modality,
    }


def readout_latency_ms(trials: list, readout_dense_idx: np.ndarray) -> float | None:
    """Mean, over trials that recorded any readout-population spike at all, of that trial's
    earliest readout spike time (task: "mean over trials of first descending_all spike after
    stimulus onset" — onset is t=0, `LIFNetwork.run_trial` drives the Poisson population from the
    start of the trial). `None` if no trial has recorded spike times, or none of them ever fired
    a readout neuron — never a fabricated number."""
    if readout_dense_idx.size == 0:
        return None
    latencies = []
    for t in trials:
        if t.spike_neuron_idx is None or t.spike_time_ms is None or t.spike_neuron_idx.size == 0:
            continue
        mask = np.isin(t.spike_neuron_idx, readout_dense_idx)
        if mask.any():
            latencies.append(float(t.spike_time_ms[mask].min()))
    if not latencies:
        return None
    return float(np.mean(latencies))


def top_cell_types(
    trials: list,
    stimulated_dense_idx: np.ndarray,
    cell_type_lookup: dict[int, str],
    super_class_lookup: dict[int, str],
    n_trials: int,
    duration_ms: float,
    top_n: int = TOP_CELL_TYPES_N,
) -> list[dict]:
    """Top `top_n` cell types by total spike count, EXCLUDING directly stimulated input neurons
    (task brief) — pooled across every trial's recorded spikes. `rate_hz` is each type's total
    spikes divided by (how many of its neurons actually fired) x (total simulated seconds across
    all trials) — the mean per-neuron firing rate of the neurons of that type that were active in
    this answer, not diluted by the type's silent members."""
    stimulated_set = {int(i) for i in stimulated_dense_idx.tolist()}
    spikes_by_neuron: dict[int, int] = {}
    for t in trials:
        if t.spike_neuron_idx is None:
            continue
        for idx in t.spike_neuron_idx.tolist():
            idx = int(idx)
            if idx in stimulated_set:
                continue
            spikes_by_neuron[idx] = spikes_by_neuron.get(idx, 0) + 1

    per_type: dict[str, dict] = {}
    for idx, n_spikes in spikes_by_neuron.items():
        cell_type = cell_type_lookup.get(idx, "unknown")
        super_class = super_class_lookup.get(idx, "unknown")
        entry = per_type.setdefault(
            cell_type, {"cell_type": cell_type, "super_class": super_class, "n_neurons": 0, "spikes": 0}
        )
        entry["n_neurons"] += 1
        entry["spikes"] += n_spikes

    total_sim_s = (n_trials * duration_ms) / 1000.0
    rows = []
    for entry in per_type.values():
        rate = entry["spikes"] / (entry["n_neurons"] * total_sim_s) if entry["n_neurons"] and total_sim_s > 0 else 0.0
        rows.append({**entry, "rate_hz": rate})
    rows.sort(key=lambda r: r["spikes"], reverse=True)
    return rows[:top_n]


def select_raster_neurons(
    trial0, readout_dense_idx: np.ndarray, seed: int, max_neurons: int = RASTER_MAX_NEURONS
) -> np.ndarray:
    """<= `max_neurons` dense indices: every readout neuron first (seeded-subsampled down to the
    cap if the readout population alone exceeds it), then a seeded sample of trial 0's other
    active neurons filling any remaining budget (task brief)."""
    readout = np.unique(readout_dense_idx.astype(np.int64))
    rng = np.random.default_rng(seed)
    if readout.size >= max_neurons:
        chosen = rng.choice(readout, size=max_neurons, replace=False)
        return np.sort(chosen)

    remaining = max_neurons - readout.size
    if trial0 is not None and trial0.spike_neuron_idx is not None and trial0.spike_neuron_idx.size:
        active = np.unique(trial0.spike_neuron_idx.astype(np.int64))
        pool = np.setdiff1d(active, readout, assume_unique=True)
    else:
        pool = np.array([], dtype=np.int64)
    if pool.size > remaining:
        pool = rng.choice(pool, size=remaining, replace=False)
    return np.sort(np.unique(np.concatenate([readout, pool])))


def build_raster(trial0, readout_dense_idx: np.ndarray, dense_to_atlas: dict[int, int] | None, cell_type_lookup: dict[int, str], seed: int) -> dict:
    """`Answer.lab.raster` — trial 0's spikes only (task brief), quantized to 0.1 ms ticks
    (`sim.lif.DT_MS`, Brian2's own integration step, so no time resolution is invented beyond
    what the simulation actually resolves)."""
    selected = select_raster_neurons(trial0, readout_dense_idx, seed)
    row_of = {int(idx): i for i, idx in enumerate(selected.tolist())}
    atlas_indices = [int(dense_to_atlas.get(int(idx), -1)) if dense_to_atlas is not None else -1 for idx in selected.tolist()]
    cell_types = [cell_type_lookup.get(int(idx), "unknown") for idx in selected.tolist()]

    pairs: list[tuple[int, int]] = []
    if trial0 is not None and trial0.spike_neuron_idx is not None and trial0.spike_time_ms is not None:
        for neuron, t_ms in zip(trial0.spike_neuron_idx.tolist(), trial0.spike_time_ms.tolist(), strict=True):
            row = row_of.get(int(neuron))
            if row is None:
                continue
            ticks = round(t_ms / 0.1)
            ticks = max(0, min(ticks, 65535))  # uint16 range; a 250ms trial never gets close
            pairs.append((row, ticks))

    arr = np.array(pairs, dtype="<u2") if pairs else np.empty((0, 2), dtype="<u2")
    spikes_b64 = base64.b64encode(arr.tobytes()).decode("ascii")
    return {"atlas_indices": atlas_indices, "cell_types": cell_types, "spikes_b64": spikes_b64}


def decode_raster_spikes_b64(encoded: str) -> np.ndarray:
    """Inverse of `build_raster`'s `spikes_b64` -> `(n_spikes, 2)` uint16 array of `(row, ticks)`
    pairs. Used by tests, not by the live pipeline itself."""
    flat = np.frombuffer(base64.b64decode(encoded), dtype="<u2")
    return flat.reshape(-1, 2)


def _reproduce_command(code_sha: str | None, id_: int, question: str) -> str | None:
    """Self-contained shell one-liner a stranger can paste from the "Recompute this answer"
    panel: clone the public repo, check out the exact code that produced this answer, install,
    fetch the (public, CC BY 4.0) connectome data, then run the same CLI `reproduce.py`'s own
    docstring describes. Not just `python -m bioreservoir.live.reproduce ...` on its own — that
    only works if you already happen to have a matching checkout with data downloaded, which is
    never true for someone arriving from the site.

    The checkout target is a TAG, not the raw `code_sha`: the public repository is published as
    squashed snapshots, so our internal commit ids do not exist as objects there. Every published
    snapshot is tagged `snapshot-<short internal sha>` (scripts/publish_public_snapshot.sh), which
    is exactly what `code_sha` shortens to — so the tag resolves in the public clone and still
    names the precise internal commit in our own history.

    `None` when `REPO_URL` is unset (a fork with no public mirror) — a paste-ready command whose
    very first step 404s is worse than no command, and the panel renders the honest note instead.
    """
    if not REPO_URL:
        return None
    # No code_sha (local/dev runs outside a checkout) -> the published tip.
    checkout = shlex.quote(f"snapshot-{code_sha[:12]}" if code_sha else "HEAD")
    return (
        f"git clone {REPO_URL} && cd bioreservoir && git checkout {checkout} && "
        "uv sync --extra sim --extra encode --extra live && python scripts/fetch_data.py && "
        f"uv run python -m bioreservoir.live.reproduce --id {id_} --question {shlex.quote(question)}"
    )


def build_provenance(lab_ctx: LabContext, id_: int, question: str, sim_ms: float) -> dict:
    return {
        "brain": lab_ctx.brain_name,
        "n_neurons": lab_ctx.n_neurons,
        "n_connections": lab_ctx.n_connections,
        "n_synapses": lab_ctx.n_synapses,
        "min_syn": lab_ctx.min_syn,
        "model": MODEL_CITATION,
        "dt_ms": lab_ctx.dt_ms,
        "sim_ms": sim_ms,
        "code_sha": lab_ctx.code_sha,
        "config_hash": lab_ctx.config_hash,
        "reproduce": _reproduce_command(lab_ctx.code_sha, id_, question),
    }


def build_lab(
    id_: int,
    question: str,
    trials: list,
    b0: float,
    corrected_bias: float | None,
    left_is_yes: bool,
    stimulated_dense_idx: np.ndarray,
    stimulated_left_idx: np.ndarray,
    stimulated_right_idx: np.ndarray,
    active_neurons: int,
    active_fraction: float,
    lab_ctx: LabContext,
    duration_ms: float,
) -> dict:
    """`Answer.lab` (task brief). `trials` are `pipeline.LiveTrial`s (each now carrying `.seed`
    and `.total_spikes` — see pipeline.py)."""
    from bioreservoir.oracle.readout import trial_readout

    trial_rows = []
    total_spikes = 0
    for t in trials:
        bias = trial_readout(t.left_spikes, t.right_spikes).bias
        trial_rows.append(
            {
                "seed": t.seed,
                "spikes_left": t.left_spikes,
                "spikes_right": t.right_spikes,
                "bias": bias if bias is not None else 0.0,
            }
        )
        total_spikes += t.total_spikes

    readout_dense_idx = np.union1d(lab_ctx.readout_left_idx, lab_ctx.readout_right_idx)
    trial0 = trials[0] if trials else None

    return {
        "trials": trial_rows,
        "b0": b0,
        "corrected_bias": corrected_bias if corrected_bias is not None else 0.0,
        "yes_side": "left" if left_is_yes else "right",
        "stimulated": stimulated_summary(stimulated_dense_idx, stimulated_left_idx, stimulated_right_idx, lab_ctx.modality_lookup),
        "active_neurons": active_neurons,
        "active_fraction": active_fraction,
        "total_spikes": total_spikes,
        "readout_latency_ms": readout_latency_ms(trials, readout_dense_idx),
        "top_cell_types": top_cell_types(trials, stimulated_dense_idx, lab_ctx.cell_type_lookup, lab_ctx.super_class_lookup, len(trials), duration_ms),
        "raster": build_raster(trial0, readout_dense_idx, lab_ctx.dense_to_atlas, lab_ctx.cell_type_lookup, seed=id_),
        "provenance": build_provenance(lab_ctx, id_, question, duration_ms),
    }
