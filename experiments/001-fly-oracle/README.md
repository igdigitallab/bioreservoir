# Experiment 001: Fly Oracle

**Question:** When a real fly connectome is driven by text about a real-world question, does its
response differ from that of a scrambled graph, how does it score against a coin flip, and do a
male and a female brain answer differently?

**First target:** the US midterm elections of November 3, 2026, followed by ongoing public
questions (sports, economics, culture).

Status: **midterm forecasts sealed** (male brain, 2026-09-18). `predictions/SHA256SUMS` and its
OpenTimestamps proof are committed now; the run-level export they hash
(`predictions/2026-09-19.jsonl`, 429 runs over 33 races: the real brain plus every control)
is released at resolution, after November 3, 2026. Commit first, reveal later — that order is
the whole point, and the timestamp makes it checkable.

## Brains

| | Male | Female |
|---|---|---|
| Dataset | Janelia MaleCNS v1.0 | BANC (Brain And Nerve Cord) |
| Coverage | brain + ventral nerve cord | brain + ventral nerve cord |
| License | CC BY 4.0 | CC BY 4.0 |

Both brains run the same model: a whole-brain leaky integrate-and-fire network following Shiu et
al. (*Nature* 2024), built on a common schema (see [docs/DATA.md](../../docs/DATA.md)). Two
individuals of opposite sex help separate what comes from shared fly topology from what comes
from one animal's wiring. Only the male has
the courtship circuits (e.g. P1 neurons).

## Two output formats

| | A. Scored forecast | B. Fly's take |
|---|---|---|
| Input | yes/no question with a resolution date and criterion | a trending news topic |
| Output | P(yes) | one of two graph responses ("approach" / "avoid") |
| Scored | yes: Brier, log loss, calibration | no, labelled as entertainment |
| Pre-registered | yes | no |

Format B depends on format A. The daily "fly's take" is only credible because a scored,
pre-registered forecast record runs alongside it.

## Questions

Questions are **curated by hand** from the public agenda: elections, sports, economics and trending
news. We write them in our own words, each with an unambiguous resolution criterion and source.
We don't scrape or call prediction-market APIs, and no platform data goes into the model.
Market consensus may be quoted as context with a date and a link.

Schema: see [`questions.example.yaml`](questions.example.yaml).

## Pipeline

1. **Encode.** The question text and a short neutral context summary are embedded with
   `all-MiniLM-L6-v2`. The embedding goes through a fixed random projection (the seed is
   committed) and becomes Poisson input rates on sensory populations: gustatory (sugar / bitter),
   olfactory receptor neurons, photoreceptors and mechanosensory neurons.
2. **Simulate.** The whole-brain LIF network runs N independent trials with fixed seeds.
3. **Read out.** An **untrained**, fixed contrast between two named populations produces a lateral
   bias: MN9 (proboscis extension, "feed / approach") against the giant fiber DNp01
   ("escape / avoid"). Alternative readout: DNa01/DNa02 turning. The contrast is a raw bias, not
   yet a probability — see step 4.
4. **Handedness and side mapping.** A 2026-09-18 supervised smoke run found a real-connectome
   lateral bias of about −0.10 for a test question regardless of whether it was phrased as the
   question or its negation — the fly's own intrinsic left/right asymmetry ("handedness"), not a
   question-driven signal. Because every one of the 33 election questions is phrased "will
   \<incumbent party\> hold \<seat\>?", a constant bias would have read as a uniform partisan
   artefact. Two fixes, both decided before any election question runs: (1) 24 short, neutral,
   non-political sentences ([`reference.yaml`](reference.yaml)) are run through every
   `(brain, condition)` the same way, and their mean lateral bias (`b0`) is subtracted from every
   question's raw bias; (2) each question id gets its own deterministic coin
   (`seed.side_base` in [`config.yaml`](config.yaml)) deciding whether a leftward
   (handedness-corrected) bias means YES or NO for that question, replacing the old global
   `left_is_yes` flag. `P(yes)` is then computed from the corrected, mapped bias (clipped as
   before). See `oracle.handedness` and `oracle.side_mapping`.
5. **Controls.** Every question also goes through:
   - **a no-brain baseline**: the same embedding, projection and readout wired directly, with no
     connectome. The meaning comes from the embedding, so the fly has to be compared against
     this, not only against a coin;
   - a rewired connectome that preserves degrees, weights, signs and cell types;
   - an Erdős–Rényi graph of the same size and density (auxiliary);
   - a seeded coin flip.
6. **Commit.** The protocol, seeds and question list are published *before* any run. The pipeline
   is deterministic, so anyone can recompute every prediction and nothing can be quietly
   discarded. Predictions go to the ledger, their hash is committed to git and an
   OpenTimestamps proof is added, all before the resolution date.
7. **Score.** After resolution: Brier score, log loss and calibration for the fly, each control
   and the market consensus snapshot.

## Invariance tests

Each scored question also runs as its negation ("will X win" / "will X lose") and as a paraphrase.
The YES/NO-swap check is no longer a separate manual run: step 4's per-question side mapping
already gives roughly half the 33 questions each polarity, so a brain that were quietly reading
the raw embedding's own polarity as its answer (rather than anything question-specific) shows up
as no better than chance across the set, not as a hidden uniform skew. Inconsistent answers are
published as they are, because they are part of the result.

## What counts as a result

- Primary endpoint: accuracy against the coin flip **and** against the no-brain baseline. Brier
  score and calibration are secondary.
- At least 85 resolved questions before any claim against the coin flip (a 65% vs 50% effect,
  α = .05, power .80). A 60% effect needs about 194. Correlated events, such as the races in one
  election, count as a cluster rather than as independent draws.
- "Real brain ≈ rewired brain" is a valid result, and it gets published.
- The readout is never trained on outcomes, so the brain is not quietly turned into an embedding
  classifier.

## Known limits

Connectome-only models have no plasticity, no neuromodulation and no gap junctions: the wiring
is frozen at the moment the animal was fixed, and everything that biology does with dopamine,
octopamine or electrical coupling is absent here. Neurotransmitter identity is itself a machine
prediction (about 94% accurate per neuron; each dataset ships its own predicted-transmitter
table, see docs/DATA.md), so a fraction of the signs in the graph are wrong. Synapse counts stand in for synaptic weight, which is an assumption, not a
measurement. None of this is fatal to the question being asked — a scrambled graph carries the
same limitations — but no result here should be read as a statement about a living fly.

## Verifying the timestamps

See the "Verifying a pre-registration" section of the repository [README](../../README.md): the
proofs are checked with `ots verify`, and the files are matched against the manifests with
`sha256sum`.
