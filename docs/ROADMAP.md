# Roadmap

Status as of 2026-09-18. Nothing is simulated until phase 1.

## Experiment 001: Fly Oracle

Two brains, male (MaleCNS v1.0) and female (BANC), forecast the **US midterm elections of
November 3, 2026** first, then other public events. Results are published as a long-form
write-up and a live scoreboard.

### Phase 0: design (current)

- [x] Background research: connectomes and licenses, simulation, question sourcing and scoring
- [x] Datasets chosen: MaleCNS v1.0 and BANC, both CC BY 4.0. FlyWire's public release is
  CC BY-NC 4.0 and is not used.
- [x] Protocol with a no-brain baseline, rewired controls and invariance tests
- [ ] Data fetched with a checksum manifest; loaders into one common schema for both brains
- [ ] Midterm question set (`questions.yaml`), published before any run

### Phase 1: brains

- [ ] Port the Shiu et al. LIF model to the common schema, so it runs on either brain
- [ ] Snapshot the network with `store()`/`restore()` so a trial skips the full rebuild
- [ ] Benchmark memory and wall-clock time per simulated second on CPU
- [ ] Sanity check on both brains: stimulating sugar receptor neurons drives MN9 (proboscis extension)

### Phase 2: pipeline

- [ ] Encoder: sentence embedding → fixed projection → Poisson rates on sensory neurons
- [ ] Readout: a fixed, untrained contrast between named descending/motor populations → P(yes)
- [ ] Controls on every question: no-brain baseline, rewired graph (degrees, weights, signs,
  cell types preserved), Erdős–Rényi graph, coin flip
- [ ] Invariance runs: negation, paraphrase, YES/NO swap
- [ ] Ledger (SQLite), hash commit in git and an OpenTimestamps proof before resolution

### Phase 3: forecasts

- [ ] Midterm forecasts from both brains committed before November 3, 2026
- [ ] Ongoing questions (sports, economics, culture), for at least 85 resolved questions in total
  before any "beats / loses to a coin flip" claim

### Phase 4: publish

- [ ] Scoreboard with live metrics: accuracy, Brier score and calibration per brain and per control
- [ ] Long-form write-up with methods, controls, the full ledger and the male vs female comparison
- [ ] A visualisation of the neurons that fire for each question

## Next brains

BANC and MaleCNS are the first. Next candidates as data appears: the *Drosophila* larva,
*C. elegans*, MICrONS (mouse visual cortex) and any rodent or primate whole-brain release.
Each gets a license check before any work starts.
