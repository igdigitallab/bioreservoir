# Simulation model

Whole-brain leaky integrate-and-fire (LIF) network, ported from Shiu et al., *Nature* 634 (2024),
code https://github.com/philshiu/Drosophila_brain_model (MIT; code read, never its CC BY-NC 4.0
connectome files). Implementation + full parameter citations: `sim/lif.py` module docstring.

## Equations and parameters

```
dv/dt = (v_rest - v + g) / tau_m   (unless refractory)
dg/dt = -g / tau_syn                (unless refractory)
threshold: v > v_threshold
reset:     v = v_reset; g = 0
```

| parameter | value | source |
|---|---|---|
| v_rest, v_reset | -52 mV | Kakaria & de Bivort 2017, doi:10.3389/fnbeh.2017.00008 |
| v_threshold | -45 mV | same |
| tau_m (membrane) | 20 ms | same |
| tau_syn (synaptic decay) | 5 ms | Juergensen et al. 2022, doi:10.1088/2634-4386/ac3ba6 |
| refractory | 2.2 ms | Lazar et al. 2021, doi:10.7554/eLife.62362 |
| synaptic delay | 1.8 ms | Paul et al. 2015, doi:10.3389/fncel.2015.00029 |
| weight / synapse | 0.275 mV | free parameter (Shiu et al.) |
| Poisson rate (default) | 150 Hz | Shiu et al. default |
| Poisson weight scale | x250 (= 68.75 mV/event, into `v` directly) | Shiu et al. `f_poi` |

Recurrent weight = `signed_weight * 0.275 mV` (`signed_weight` = presynaptic neuron's NT sign x
synapse count, `harmonize.load_graph()`'s own column). Poisson-driven neurons get refractory = 0
for the trial (Shiu's "no refractory period for Poisson targets"). Shiu's reset string also has
a `w = 0` clause for a variable that exists nowhere in the model (`w` is only the *synapse*
weight, different namespace) — verified a silent Brian2 2.10.1 no-op, dropped here.

## Codegen target: runtime `cython`

Compiled, reused across trials via `Network.store()`/`restore()`. `cpp_standalone` ruled out
(store/restore is runtime-only, standalone raises `NotImplementedError`, brian2 issue #958);
`numpy` (uncompiled) loses once the network is built once and reused across trials.

## CPU benchmark

Cage (`MemoryMax=6G`, `CPUQuota=400%`, single process), `min_syn=5`, `dt=0.1 ms` (Brian2
default), 5 trials of 1000 ms after a warm-up run:

| brain | neurons | synapses | build time | peak RSS | s wall / s simulated |
|---|---|---|---|---|---|
| synthetic (165k / 20M, 80/20 E/I) | 165,000 | 20,000,000 | 1.9 s | 1,450 MB | 28.08 |
| MaleCNS (male) | 165,122 | 6,235,682 | 1.2 s | 1,588 MB | 27.48 |
| BANC (female) | 114,456 | 1,395,876 | 0.6 s | 827 MB | 18.98 |

Wall time/simulated-second tracks neuron count, not synapse count (MaleCNS's 6.2M synapses cost
about the same as the 20M-synapse synthetic graph, both 165k neurons) — dominated by N x 10,000
steps/s of state checks, not synaptic events. Single-core (`cython` doesn't parallelize one
`Network.run()`). Compiled once per process, reused after — "build" above is graph loading only.

## Calibration (2026-09-18, `python -m bioreservoir.sim.calibrate`, seed 42, 10 trials/condition
unless noted; only biological stimuli, never question text, to avoid fitting the model to the
questions it will later be asked). No-input baseline:
**0 spikes** in both brains, all populations (no pacemaker current, no gap junctions).

### Sign convention fix (see `docs/DATA.md`, `schema.py`)

Signal was near-dead before 2026-09-18: `sign` mapped ~13% of MaleCNS neurons (dopamine/
serotonin/octopamine/histamine/unclear/null) to 0, zeroing *every outgoing synapse* those
neurons make, not just their classification. Fixed to Shiu et al.'s own binary rule (GABA/
glutamate → -1, everything else → +1, no "unknown"). Same 1000 ms/150 Hz sugar-GRN → MN9 check,
before vs. after (before also used the wrong, 1,428-neuron unsplit `gustatory_all` input; after
uses the correctly cross-matched, Shiu-scale `gustatory_sugar`, n=28 — docs/DATA.md task 3):

| | before (sign + wrong population) | after (both fixes) |
|---|---|---|
| MN9 spikes / 1000 ms | 2 | 128.5 (mean of 10 trials, std 21.3) |

### Canonical result: sugar/bitter GRN → MN9 (MaleCNS only; BANC has no confirmed MN9 homolog)

Dose-response, `gustatory_sugar` (n=28) driven 1000 ms, 10 trials/rate — monotonic, **not**
saturating (theoretical ceiling ≈ 1000/2.2 ms refractory ≈ 454 spikes/s):

| rate | MN9 rate (mean ± std, Hz) | fraction of whole network active |
|---|---|---|
| 100 Hz | 113.3 ± 25.4 | 8.0% ± 1.3 |
| 150 Hz | 128.5 ± 21.3 | 7.5% ± 1.8 |
| 200 Hz | 142.1 ± 5.2 | 6.6% ± 2.3 |

Bitter suppression (Shiu's other headline gustatory result — feed-forward inhibition): sugar
alone → MN9 128.1 ± 11.9 spikes/1000 ms; sugar + `gustatory_bitter` (n=17) together → **4.1 ±
1.8** — a 96.8% reduction, cleanly reproducing Shiu's direction. BANC's equivalent sugar drive
(n=200, much larger cross-matched population) shows no MN9 (absent) but a smaller network-wide
footprint than MaleCNS (2.8–4.5% active vs. 6.6–8.0%) at every rate tested — consistent with
BANC's 3x lower mean degree (docs/DATA.md).

### Chosen readout: `descending_all_left`/`right` lateral bias `(L-R)/(L+R)`

README.md's stated primary readout (MN9 vs. DNp01 giant fiber) **cannot score BANC** — MN9 is
confirmed absent there (docs/DATA.md). The stated alternative, DNa01/DNa02 turning (2+2 neurons
per brain), is too small-N to trust: MaleCNS's own DNa-only bias hit a degenerate ±1.0/0.95 in
every trial (all-or-nothing on 4 neurons). `descending_all_*` (populations.yaml, all
`super_class=="descending"`-equivalent neurons, ~650/side, balanced without downsampling in
both brains) is far more stable. Lateralised Johnston's-organ drive (balanced L/R counts,
`bench.balanced_lateral_population`), 5 trials/side:

| | MaleCNS bias (mean ± std) | BANC bias (mean ± std) |
|---|---|---|
| JO left driven | +0.118 ± 0.002 | -1.0 ± 0.0 |
| JO right driven | -0.157 ± 0.005 | -1.0 ± 0.0 |

MaleCNS: a small but extremely tight (std two orders below the effect size), correctly-signed
(left drive → left-biased, right → right-biased) lateral signal — usable. **BANC: currently
unusable** — collapses to a constant, stimulus-independent all-right response every trial, most
likely because BANC's public snapshot has 45% more proofread neurons on the right side
(docs/DATA.md): the driven/readout population *sizes* were balanced, the *intermediate
circuitry* between them was not. Recommend `descending_all` bias as the primary readout anyway
(present, balanced, large-N in both brains, unlike MN9), but treat BANC's numbers as unreliable
until that density asymmetry is fixed — a follow-up, out of this calibration's scope.

### BANC lateral bias collapse: root cause (2026-09-18, follow-up debug)

**Not a bug** — side labels, index mapping, `_lateral_bias` sign formula, and
`balanced_lateral_population`'s downsampling all verified correct. Structural BFS from
JO-left/right reaches ~97% of *both* `descending_all_left/right` within 4 hops (no reachability
gap), and random `sensory_all` subsets give a graded bias (-0.20 to -0.27), not -1.0 — the
readout machinery works. Only the narrow JO-only stimulus (~322-430 neurons) collapses to
exactly -1.0 in 10/10 trials (`descending_all_left` = 0 spikes every time, either JO side) — a
real narrow-pathway propagation failure in BANC's sparser graph (3x lower mean degree), not code.

**Fix tried, failed (code removed):** a bilaterally balanced BANC subgraph (seeded per
`cell_type`/`cell_class` down-sampling, 114,456 → 71,204 neurons, JO 308/308, DN 576/574) still
gave exactly -1.0 ± 0.0 at 250 and 1000 ms. Balancing *counts* doesn't help; what matters is which
neurons survived proofreading.

**No denser public BANC exists:** Dataverse doi:10.7910/DVN/8TFGGB v8.1 (CC BY 4.0) has only one
connectivity file, materialization 626 (already used); `banc_meta_821.tab` is annotations-only,
no matching edges. Live BANC needs a restricted CAVE key — not public.

**Not saturation:** ~0.3% neurons active under JO drive, 0 at no-input baseline.

**Broad-drive side check (the pipeline's actual input).** `sensory_all`, 250 neurons per side,
250 ms, 5 trials, left/right rates 175/175, 300/50 and 50/300 Hz:

| | symmetric | left-heavy | right-heavy | follows side? |
|---|---|---|---|---|
| MaleCNS | -0.100 ± 0.011 | -0.068 ± 0.013 | -0.133 ± 0.014 | yes, ±0.03 around its own bias |
| BANC | -0.216 ± 0.010 | -0.237 ± 0.004 | -0.230 ± 0.013 | no, within noise |

Both brains carry an intrinsic lateral bias (handedness), which the pipeline subtracts using
neutral reference sentences. Only the male's readout tracks which side is driven. **The female
brain fails this validation.** In experiment 001 her answers are reported for transparency,
flagged, and excluded from every headline claim until a denser public BANC release exists.

Approach (`gustatory_sugar`) vs. escape (`gustatory_bitter`+`MDN`) contrast, MaleCNS: appetitive
→ MN9 135.4 ± 12.6, escape-pool 14.2 ± 7.6; aversive → **both 0** — bitter alone does not drive
the giant-fiber/backward-walking pool (normally visual-loom/strong-mechanosensory driven, not
taste), so this contrast only shows a real effect on the approach side.

### Input regime and trial duration

Random `sensory_all` subsets at n∈{100,1000} × rate∈{50,150,300} Hz (3 trials/point):
`descending_all` responds above baseline even at n=100/50 Hz and stays far from saturation at
the largest (fraction active 6.6–11.5% MaleCNS, 4.3–6.0% BANC) — nothing in this range is
runaway. Duration scan (`gustatory_sugar` 150 Hz, MaleCNS, 5 trials/duration): MN9 rate is
statistically indistinguishable at 250/500/1000 ms (139.2 ± 9.3 / 138.8 ± 14.4 / 123.2 ± 12.9) —
**the effect is fully present by 250 ms**, at 1/4 the wall time. Recommend **250 ms trials,
n≈300–1000 driven sensory neurons, ≥10 trials/condition**; MaleCNS-verified only, extrapolated
(not independently confirmed) to BANC, which has no MN9 to duration-scan against.

### Throughput

Measured mean wall time/trial (`duration_scan`, cross-checked against `bench.py`'s own
benchmark): MaleCNS 27.65 s/1000 ms (6.99 s/250 ms), BANC 20.04 s/1000 ms (5.08 s/250 ms) — graph
variant (real/rewired/Erdős–Rényi) doesn't change this (same neuron count; cost tracks neurons,
not synapses). One question = 2 brains × 3 graph variants × 3 text variants × 10 trials = 180
trials; 33 questions = 5,940 trials; 3 parallel processes/cage (hard rule, see
`experiments/001-fly-oracle/RUNNING.md`; verified: 2 processes together stayed under 2.2 GB RSS
of the 6 GB budget):

| trial duration | sequential | 3 parallel (recommended) |
|---|---|---|
| 1000 ms (conservative) | 39.3 h | **13.1 h** |
| 250 ms (per duration-scan finding) | 10.0 h | **3.3 h** |

## Known limits

No plasticity, no neuromodulation, no gap junctions. NT sign is a per-neuron ML prediction (~94%
accurate), not per-synapse. `min_syn=5`/proofread-only filters drop real but low-confidence
connectivity (`docs/DATA.md`). Poisson drive is a `PoissonGroup`, not Shiu's per-neuron
`PoissonInput` — traded for changing the stimulated set per trial with no rebuild.
