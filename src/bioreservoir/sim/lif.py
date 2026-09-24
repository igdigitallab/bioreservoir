"""Whole-brain leaky integrate-and-fire (LIF) network, ported from Shiu et al., *A leaky
integrate-and-fire computational model based on the connectome of the entire adult Drosophila
brain reveals insights into sensorimotor processing*, Nature 634 (2024).

Source code read (MIT license, code only — never any data file from that repo, because the
FlyWire release it ships is CC BY-NC 4.0 and this project uses none of it, see docs/DATA.md):
https://github.com/philshiu/Drosophila_brain_model,
commit pinned by ``git clone`` on 2026-09-18, files ``model.py`` and ``environment.yml``. The
repo was cloned read-only to ``/tmp/shiu-ref`` and deleted after this port was written; the data
files it ships (``Connectivity_783.parquet``, ``Completeness_783.csv``, CC BY-NC 4.0) were never
opened by any code here.

Parameters (``model.py`` lines 15-53, dict ``default_params``, values and citations verbatim):
    v_rest = v_reset = -52 mV, v_threshold = -45 mV
        Kakaria & de Bivort 2017, https://doi.org/10.3389/fnbeh.2017.00008
    tau_m (membrane time constant) = 20 ms
        same source (capacitance * resistance = .002 uF * 10 Mohm, per that file's comment)
    tau_syn (synaptic decay) = 5 ms
        Juergensen et al. 2022, https://doi.org/10.1088/2634-4386/ac3ba6
    refractory period = 2.2 ms
        Lazar et al. 2021, https://doi.org/10.7554/eLife.62362
    synaptic delay = 1.8 ms
        Paul et al. 2015, https://doi.org/10.3389/fncel.2015.00029
    weight per synapse = 0.275 mV
        free parameter, tuned by the original authors (their comment: "modulated by exponential
        decay"), used as-is here
    Poisson drive: default rate 150 Hz, synapse-weight scale factor 250 (so the weight of one
        Poisson input event is ``weight_per_synapse * 250`` = 68.75 mV, injected directly into
        the membrane potential ``v``, bypassing the synaptic filter ``g``)

Equations (``model.py`` lines 44-48, variable names translated: ``v_0`` -> ``v_rest``, ``t_mbr``
-> ``tau_m``, ``tau`` -> ``tau_syn``)::

    dv/dt = (v_rest - v + g) / tau_m   (unless refractory)
    dg/dt = -g / tau_syn               (unless refractory)
    rfc : second                        # per-neuron refractory duration, itself a state variable

Threshold ``v > v_threshold``. Reset (``model.py`` line 52) is ``'v = v_rst; w = 0; g = 0 * mV'``
— the ``w = 0`` clause is dead code in the source: no neuron-level state variable named ``w``
exists anywhere in ``default_params['eqs']`` (``w`` is only the name Shiu's ``Synapses`` object
gives its own per-*synapse* weight variable, a different namespace). Verified empirically before
writing this port: instantiating Brian2 2.10.1 with that exact reset string and no ``w`` state
variable builds and runs without error or warning — Brian2 treats the assignment as a discarded
local temporary. This port omits the no-op and resets only ``v`` and ``g``.

Two mechanisms use ``v`` directly instead of the synaptic filter ``g``:

1. Recurrent synapses (``model.py`` ``create_model``, lines 174-183): one ``Synapses(neu, neu,
   'w : volt', on_pre='g += w', delay=t_dly)`` for the whole connectome, weight
   ``w = (sign(pre) * synapse_count) * weight_per_synapse`` — this is exactly the
   ``signed_weight`` column already computed by
   :func:`bioreservoir.connectomes.harmonize.load_graph` (sign from the pre-synaptic neuron's
   predicted fast neurotransmitter, ``schema.NT_SIGN``), so no extra sign lookup happens here.
2. Poisson drive (``model.py`` ``poi()``, lines 58-106): one ``PoissonInput(target_var='v',
   N=1, rate=r_poi, weight=w_syn*f_poi)`` per stimulated neuron, and that neuron's refractory
   duration is set to 0 ("no refractory period for Poisson targets", ``model.py`` line 92/103)
   for the duration it is driven.

``PoissonInput``'s own ``rate`` is fixed at construction and is not meant to be changed at
runtime (Brian2 docs, ``brian2.input.poissoninput.PoissonInput``: it precomputes internal state
in ``before_run``/is built for a static rate; it exists specifically so *not* to build a
``PoissonGroup`` for every input, "much more efficient... because synaptic events are generated
... and are not preloaded"). Task 3 requires a network built once and reused across trials with a
*different* stimulated set and rate each time (see module docstring of ``run_trial`` below), so
per-neuron ``PoissonInput`` objects (one create call per trial's stimulated set) are not usable
here without rebuilding the network every trial — exactly the ~20-minute-per-trial cold start
this port exists to avoid. Instead this module builds ONE ``PoissonGroup`` covering all neurons
with a settable per-neuron ``rates`` array, connected 1:1 to the real neurons with the same
weight-into-``v`` semantics as Shiu's ``PoissonInput``. Verified empirically
(``/tmp/poisson_test.py`` during development) that ``PoissonGroup.rates`` is an ordinary,
independently settable state array that survives ``Network.store()``/``restore()`` and produces
deterministic per-trial spike counts under ``brian2.seed()`` without any recompilation between
trials — the efficiency Brian2's docs describe for ``PoissonInput`` is traded for that
flexibility; for the population sizes here (hundreds to a few thousand stimulated neurons out of
up to 165k) this trade is cheap.

Codegen target
--------------
Brian2 offers two *runtime* targets (``numpy``, no compilation; ``cython``, compiled, needs a
C++ compiler — https://brian2.readthedocs.io/en/stable/user/computation.html) and one
*standalone* target (``cpp_standalone``, ahead-of-time compiled C++ program). Task 3 requires the
network to be "built once and reused across trials via Brian2 store()/restore()" — but
``Network.store()``/``restore()`` is documented to work only in runtime mode; issuing it under
``cpp_standalone`` raises ``NotImplementedError: The store/restore mechanism is not supported in
the C++ standalone`` (brian-team/brian2 issue #958; brian.discourse.group thread "NotImplemented
Error: The store/restore mechanism is not supported in the C++ standalone"). That rules out
standalone regardless of its raw speed. Between the two runtime targets, this port picks
``cython``: this machine has a working ``g++`` (checked: ``g++ (Debian 14.2.0-19) 14.2.0``) and
the ``cython`` package (installed via the ``sim`` extra); the one-time compilation cost happens
once per process at the first ``Network.run()`` call and is amortized over every subsequent trial
run on the same persistent :class:`LIFNetwork`, which is the whole point of the store/restore
design. If no C++ toolchain is available, this module falls back to ``numpy`` (uncompiled, correct
but visibly slower per simulated second — see ``docs/MODEL.md`` for the measured difference).
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass

import numpy as np

VALID_CODEGEN_TARGETS = ("cython", "numpy")

# --- Shiu et al. 2024 parameters, see module docstring for exact sources ---------------------
V_REST_MV = -52.0
V_RESET_MV = -52.0
V_THRESHOLD_MV = -45.0
TAU_M_MS = 20.0
TAU_SYN_MS = 5.0
REFRACTORY_MS = 2.2
SYNAPTIC_DELAY_MS = 1.8
WEIGHT_PER_SYNAPSE_MV = 0.275
DEFAULT_POISSON_RATE_HZ = 150.0
POISSON_WEIGHT_SCALE = 250  # PoissonInput weight = weight_per_synapse * this factor

# Brian2's own default integration step (verified: `brian2.defaultclock.dt == 100*us == 0.1*ms`,
# 2026-09-18) — never set explicitly anywhere in this module, so this constant documents the
# actual value in force rather than leaving it implicit. Used by `bioreservoir.live` for
# "neuron updates" (neurons x steps) provenance/statistics, not by the simulation itself.
DT_MS = 0.1

_EQUATIONS = """
dv/dt = (v_rest - v + g) / tau_m : volt (unless refractory)
dg/dt = -g / tau_syn : volt (unless refractory)
rfc : second
"""
_THRESHOLD = "v > v_threshold"
_RESET = "v = v_reset; g = 0 * mV"  # Shiu's dead 'w = 0' clause dropped, see module docstring


def select_codegen_target(explicit: str | None = None) -> str:
    """Pick 'cython' if a C++ compiler is available, else fall back to 'numpy'.

    See the module docstring ("Codegen target") for why standalone mode is not an option and why
    cython is preferred over numpy when both work.
    """
    if explicit is not None:
        if explicit not in VALID_CODEGEN_TARGETS:
            raise ValueError(f"codegen target must be one of {VALID_CODEGEN_TARGETS}")
        return explicit
    return "cython" if shutil.which("g++") is not None else "numpy"


@dataclass(frozen=True)
class TrialResult:
    """Output of one :meth:`LIFNetwork.run_trial` call.

    ``spike_neuron_idx``/``spike_time_ms`` are only populated when the network was built with
    ``record_spike_times=True`` (opt-in, see ``LIFNetwork.__init__``) -- ``None`` otherwise,
    meaning "not recorded", never "zero spikes" (an empty array with ``record_spike_times=True``
    and truly no spikes is a real, distinct value: ``np.array([])``, not ``None``). Added for
    ``bioreservoir.live.frames`` (the live fly page's per-25ms-bin atlas visualisation); every
    existing caller passes ``record_spike_times=False`` (the default) and sees no change at all.
    """

    spike_counts: np.ndarray  # int, aligned to `monitor_idx` (or all neurons if None)
    wall_time_s: float
    spike_neuron_idx: np.ndarray | None = None  # dense neuron index per recorded spike
    spike_time_ms: np.ndarray | None = None  # matching spike time, milliseconds since trial start


class LIFNetwork:
    """A whole-brain LIF network, built once from connectome arrays and reused across trials.

    Parameters
    ----------
    n_neurons : total number of neurons (NeuronGroup size).
    pre_idx, post_idx : dense 0..n_neurons-1 indices of each synapse's pre/post neuron.
        Must already be dense indices into `n_neurons`, not raw dataset neuron IDs — mapping
        dataset IDs to dense indices is the caller's job (see `bench.py`'s `graph_to_arrays`).
    weight : signed synapse count per edge (same order/length as pre_idx/post_idx), i.e.
        `sign(pre neuron's neurotransmitter) * syn_count` — exactly
        `bioreservoir.connectomes.harmonize.load_graph(...)`'s `signed_weight` column. Multiplied
        internally by `WEIGHT_PER_SYNAPSE_MV` to get the actual synaptic weight in mV, matching
        Shiu's `syn.w = df_con['Excitatory x Connectivity'] * w_syn`.
    monitor_idx : neuron indices whose spike counts `run_trial` returns; `None` monitors (and
        returns counts for) all `n_neurons` neurons. Spike *counts* are tracked for every
        neuron regardless (Brian2's `SpikeMonitor(record=False)` still populates `.count` for
        the whole source group — https://brian2.readthedocs.io/en/stable/reference/
        brian2.monitors.spikemonitor.SpikeMonitor.html) — `monitor_idx` only controls what this
        class hands back, not what Brian2 tracks internally, so it costs nothing to change
        between construction calls.
    codegen_target : 'cython', 'numpy', or None to auto-detect (see `select_codegen_target`).
    record_spike_times : opt-in (default `False`, no behaviour change for any existing caller).
        When `True`, an additional `SpikeMonitor(record=True)` is built alongside the existing
        counts-only monitor, and `run_trial` returns each spike's neuron index and time
        (`TrialResult.spike_neuron_idx`/`spike_time_ms`) — needed by
        `bioreservoir.live.frames` to bin activity for the atlas visualisation. Costs extra
        memory proportional to the number of *spikes* in a trial (a few % of `n_neurons` are
        active per the calibration in docs/MODEL.md), not to `n_neurons` itself, so this is cheap
        at the population sizes and trial durations this repo actually runs — left opt-in anyway
        so every non-live caller (`oracle.runner`, `sim.bench`, `sim.calibrate`) keeps building
        the exact same network it always has.
    """

    def __init__(
        self,
        n_neurons: int,
        pre_idx: np.ndarray,
        post_idx: np.ndarray,
        weight: np.ndarray,
        monitor_idx: np.ndarray | None = None,
        codegen_target: str | None = None,
        record_spike_times: bool = False,
    ) -> None:
        import time

        from brian2 import (
            Hz,
            Network,
            NeuronGroup,
            PoissonGroup,
            SpikeMonitor,
            Synapses,
            ms,
            mV,
            prefs,
        )

        self.n_neurons = int(n_neurons)
        self.monitor_idx = None if monitor_idx is None else np.asarray(monitor_idx, dtype=np.int64)
        self.record_spike_times = record_spike_times
        self.codegen_target = select_codegen_target(codegen_target)
        prefs.codegen.target = self.codegen_target

        t0 = time.monotonic()

        neurons = NeuronGroup(
            self.n_neurons,
            model=_EQUATIONS,
            method="linear",  # eqs are linear ODEs -> Brian2's exact analytic update, no step error
            threshold=_THRESHOLD,
            reset=_RESET,
            refractory="rfc",
            name="brain_neurons",
            namespace={
                "v_rest": V_REST_MV * mV,
                "v_reset": V_RESET_MV * mV,
                "v_threshold": V_THRESHOLD_MV * mV,
                "tau_m": TAU_M_MS * ms,
                "tau_syn": TAU_SYN_MS * ms,
            },
        )
        neurons.v = V_REST_MV * mV
        neurons.g = 0 * mV
        neurons.rfc = REFRACTORY_MS * ms

        synapses = Synapses(
            neurons,
            neurons,
            "w : volt",
            on_pre="g += w",
            delay=SYNAPTIC_DELAY_MS * ms,
            name="brain_synapses",
        )
        pre_idx = np.asarray(pre_idx, dtype=np.int64)
        post_idx = np.asarray(post_idx, dtype=np.int64)
        weight = np.asarray(weight, dtype=np.float64)
        synapses.connect(i=pre_idx, j=post_idx)
        synapses.w = weight * WEIGHT_PER_SYNAPSE_MV * mV

        # Poisson drive: one PoissonGroup covering all neurons, rate settable per-trial (see
        # module docstring for why this replaces Shiu's per-neuron PoissonInput).
        poisson = PoissonGroup(self.n_neurons, rates=np.zeros(self.n_neurons) * Hz)
        poisson_synapses = Synapses(
            poisson,
            neurons,
            on_pre="v_post += w_poisson",
            namespace={"w_poisson": WEIGHT_PER_SYNAPSE_MV * POISSON_WEIGHT_SCALE * mV},
            name="poisson_synapses",
        )
        poisson_synapses.connect(j="i")

        spikes = SpikeMonitor(neurons, record=False, name="brain_spikes")
        network_objects = [neurons, synapses, poisson, poisson_synapses, spikes]

        self._spikes_timed = None
        if record_spike_times:
            self._spikes_timed = SpikeMonitor(neurons, record=True, name="brain_spikes_timed")
            network_objects.append(self._spikes_timed)

        self._neurons = neurons
        self._poisson = poisson
        self._spikes = spikes
        self._network = Network(*network_objects)
        self._network.store("baseline")

        self.build_time_s = time.monotonic() - t0
        self.n_synapses = len(pre_idx)

    def run_trial(
        self,
        inputs: dict[int, float] | tuple[np.ndarray, np.ndarray],
        duration_ms: float,
        seed: int,
    ) -> TrialResult:
        """Run one trial from the stored baseline state and return spike counts.

        `inputs` is either `{neuron_index: rate_hz}` or `(idx_array, rate_hz_array)`: the
        neurons driven by the Poisson population for this trial, and at what rate. Every other
        neuron's Poisson rate is 0 for this trial (restored from the baseline snapshot, which was
        captured with all rates at 0). Stimulated neurons also get their refractory period set to
        0 for this trial, matching Shiu's `poi()` (see module docstring).

        Deterministic given `seed`: `brian2.seed(seed)` is called after `restore()` and before
        `run()`, so the Poisson process draws (the only source of randomness in this model) are
        reproducible; `restore()` itself resets the spike monitor's counts to 0 before every run.
        """
        import time

        from brian2 import Hz, ms
        from brian2 import seed as brian2_seed

        if isinstance(inputs, dict):
            idx = np.fromiter(inputs.keys(), dtype=np.int64, count=len(inputs))
            rate = np.fromiter(inputs.values(), dtype=np.float64, count=len(inputs))
        else:
            idx, rate = inputs
            idx = np.asarray(idx, dtype=np.int64)
            rate = np.asarray(rate, dtype=np.float64)

        t0 = time.monotonic()
        self._network.restore("baseline")

        rate_array = np.zeros(self.n_neurons)
        rate_array[idx] = rate
        self._poisson.rates = rate_array * Hz
        if idx.size:
            self._neurons.rfc[idx] = 0 * ms

        brian2_seed(int(seed))
        self._network.run(duration_ms * ms)

        counts = np.asarray(self._spikes.count[:], dtype=np.int64)
        if self.monitor_idx is not None:
            counts = counts[self.monitor_idx]

        spike_neuron_idx = None
        spike_time_ms = None
        if self._spikes_timed is not None:
            spike_neuron_idx = np.asarray(self._spikes_timed.i[:], dtype=np.int64)
            # `.t[:]` is a Brian2 `Quantity` in seconds; `/ ms` (dimensionless division by the
            # same unit) is Brian2's own documented way to get a plain-float array back, matching
            # every other `* ms`/`* mV` use already in this module.
            spike_time_ms = np.asarray(self._spikes_timed.t[:] / ms, dtype=np.float64)

        return TrialResult(
            spike_counts=counts,
            wall_time_s=time.monotonic() - t0,
            spike_neuron_idx=spike_neuron_idx,
            spike_time_ms=spike_time_ms,
        )
