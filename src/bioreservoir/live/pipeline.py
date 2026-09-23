"""Question -> Answer, the pure part (task brief's Answer schema). No Brian2 import at module
load — `worker.py` is the only place that touches `LIFNetwork` directly; this module takes
already-run trial results (`LiveTrial`) and turns them into the exact dict `api.py` serializes as
`Answer`. Kept separate so it is unit-testable with tiny synthetic spike arrays, the same split
`oracle.runner`/`oracle.readout`/`oracle.handedness` already use for the batch pipeline.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np

from bioreservoir.live import config
from bioreservoir.live import frames as frames_mod
from bioreservoir.live import game as game_mod
from bioreservoir.live import lab as lab_mod
from bioreservoir.live import states as states_mod
from bioreservoir.live.verdict import (  # noqa: F401 (re-exported)
    BIAS_CONFIDENCE_SCALE,
    answer_yes_no,
    confidence_from_bias,
    turn_strength_from_bias,
)
from bioreservoir.oracle import handedness, side_mapping
from bioreservoir.oracle.readout import aggregate_trials, trial_readout


def question_key(question: str) -> str:
    """The identity of a live question for seeding: NFKC, casefolded, whitespace collapsed, trailing
    punctuation dropped. Identical wording gets the same Poisson seeds and the same YES-side coin,
    so asking twice gives the same answer; a rephrasing is a new key and can land differently,
    which is the honest behaviour to show (2026-09-18: the same question had answered NO, then
    YES, when both were keyed by the queue row id)."""
    text = unicodedata.normalize("NFKC", question).casefold()
    text = re.sub(r"\s+", " ", text).strip()
    return text.rstrip(" ?!.")

def yes_side_for(question: str, side_base: int = config.LIVE_SIDE_BASE) -> str:
    """Which physical side of the fly means YES for this wording — `"left"` or `"right"`.

    Deterministic from the text alone (`question_key` + `oracle.side_mapping`), so it is known the
    moment a question is queued, long before any neuron fires. The live stage uses that to put both
    answer labels on their real sides while the fly is still deciding (operator, 2026-09-19: "the
    options must be on screen straight away"), and the side never swaps when the verdict lands —
    `build_answer` records the same coin as `lab.yes_side`. Knowing the side reveals nothing about
    the answer: it only says which way a YES would look.
    """
    return "left" if side_mapping.left_is_yes_for("q:" + question_key(question), side_base) else "right"


# BIAS_CONFIDENCE_SCALE/confidence_from_bias/answer_yes_no moved to `verdict.py` (2026-09-18, the
# "which one is the real fly?" game) so the SAME formula scores the real answer's `confidence` and
# every game contender's `decisiveness` -- re-exported above unchanged, every existing caller here
# keeps working.


@dataclass(frozen=True)
class LiveTrial:
    """One `LIFNetwork.run_trial` call's results, reduced to what the live pipeline needs.

    `seed` is that trial's own Poisson seed (`worker.live_trial_seed`) — carried through so
    `Answer.lab.trials` can show it (operator requirement: every number traces to the
    simulation). `state_spike_counts` is `{population_name: spikes_this_trial}` for the named
    state populations (`states.py`'s `STATE_NAMES` minus `arousal`, which uses
    `n_active_neurons` instead). `total_spikes` is this trial's whole-network spike count (task:
    `Answer.lab.total_spikes` sums this across trials). `spike_neuron_idx`/`spike_time_ms` are
    only set when the network was built with `record_spike_times=True` (`sim/lif.py`) — `None`
    otherwise, meaning "not recorded", not "zero spikes".
    """

    left_spikes: int
    right_spikes: int
    seed: int = 0
    total_spikes: int = 0
    state_spike_counts: dict[str, int] = field(default_factory=dict)
    n_active_neurons: int = 0
    spike_neuron_idx: np.ndarray | None = None
    spike_time_ms: np.ndarray | None = None


def compute_answer(
    id_: int,
    question: str,
    trials: list[LiveTrial],
    b0: float,
    state_specs: dict[str, states_mod.StateSpec],
    duration_ms: float,
    lab_ctx: lab_mod.LabContext,
    stimulated_dense_idx: np.ndarray,
    stimulated_left_idx: np.ndarray,
    stimulated_right_idx: np.ndarray,
    side_base: int = config.LIVE_SIDE_BASE,
    brain: str = config.LIVE_BRAIN,
    now: datetime | None = None,
    game_inputs: game_mod.GameInputs | None = None,
) -> dict:
    """The full Answer dict (task brief schema, including the `lab` object). The
    `oracle.side_mapping` per-question coin is keyed by `question_key(question)`, not the queue
    row id, so the same wording always maps YES to the same side; `config.LIVE_SIDE_BASE` keeps
    it a distinct seed space from the 33 pre-registered questions. `lab_ctx` carries everything about the running brain/process that does
    not change between questions (graph metadata, annotation lookups, provenance strings) — see
    `lab.LabContext`.

    `game_inputs`, if given, adds the `game` field (the "which one is the real fly?" game --
    `game.build_game`): `None` (the default, and every caller before 2026-09-18) omits the field
    entirely -- not `null` -- so old stored answers and every caller that has not been updated to
    supply the two extra contenders keep rendering exactly as before (task brief: "the game is
    simply absent").
    """
    readout_trials = [trial_readout(t.left_spikes, t.right_spikes) for t in trials]
    agg = aggregate_trials(readout_trials, left_is_yes=True, zero_spike_probability=0.5)
    raw_bias = None if agg.all_trials_zero_spikes else agg.mean_bias
    left_is_yes = yes_side_for(question, side_base) == "left"
    p_yes, corrected_bias = handedness.apply_correction(
        raw_bias, b0, left_is_yes, zero_spike_probability=0.5
    )

    summed_state_counts: dict[str, int] = {}
    for t in trials:
        for pop, n in t.state_spike_counts.items():
            summed_state_counts[pop] = summed_state_counts.get(pop, 0) + n
    mean_active = (sum(t.n_active_neurons for t in trials) / len(trials)) if trials else 0.0
    n_total_neurons = lab_ctx.n_neurons
    state_values = states_mod.compute_states(
        state_specs, summed_state_counts, round(mean_active), n_total_neurons
    )

    frames_field = None
    if lab_ctx.dense_to_atlas is not None:
        # Frames come from the first trial with recorded spike times only, not merged across
        # trials -- merging would blur the 25ms bin structure the visual exists to show, and the
        # task brief does not ask for a cross-trial aggregate here.
        first_recorded = next(
            (t for t in trials if t.spike_neuron_idx is not None and t.spike_time_ms is not None),
            None,
        )
        if first_recorded is not None:
            frames_field = frames_mod.frames_payload(
                first_recorded.spike_neuron_idx,
                first_recorded.spike_time_ms,
                duration_ms,
                lab_ctx.dense_to_atlas,
                seed=id_,
            )

    active_fraction = (mean_active / n_total_neurons) if n_total_neurons else 0.0
    lab_field = lab_mod.build_lab(
        id_=id_,
        question=question,
        trials=trials,
        b0=b0,
        corrected_bias=corrected_bias,
        left_is_yes=left_is_yes,
        stimulated_dense_idx=stimulated_dense_idx,
        stimulated_left_idx=stimulated_left_idx,
        stimulated_right_idx=stimulated_right_idx,
        active_neurons=round(mean_active),
        active_fraction=active_fraction,
        lab_ctx=lab_ctx,
        duration_ms=duration_ms,
    )

    result = {
        "id": id_,
        "question": question,
        "answer": answer_yes_no(p_yes),
        "confidence": confidence_from_bias(corrected_bias),
        "lateral_bias": corrected_bias if corrected_bias is not None else 0.0,
        # Task brief (scaling contract): a 0..1 fraction of a real, named biological effect size
        # (the male brain's own single-antenna Johnston's-organ calibration, `verdict.py`) --
        # computed from the SAME `corrected_bias` `lateral_bias` above already carries, not a
        # second, independently-drifting number.
        "turn_strength": turn_strength_from_bias(corrected_bias),
        "states": state_values,
        "frames": frames_field,
        "brain": brain,
        "sim_ms": duration_ms,
        "n_trials": len(trials),
        "answered_at": (now or datetime.now(UTC)).isoformat(),
        "lab": lab_field,
    }
    if game_inputs is not None:
        result["game"] = game_mod.build_game(
            left_is_yes=left_is_yes,
            question_key="q:" + question_key(question),
            real_answer=result["answer"],
            real_corrected_bias=result["lateral_bias"],
            real_confidence=result["confidence"],
            game_inputs=game_inputs,
        )
    return result
