# BioReservoir

**The digital-brain research lab of [IG Digital Lab](https://igdigi.com).**

[![A question going through all 165,122 neurons of the male fruit-fly connectome, live at fly.igdigi.com](docs/media/fly-preview.gif)](https://igdigi.com/assets/video/fly-24s-musicB.mp4)

A question going through all 165,122 neurons of the male fruit-fly connectome, live at
[fly.igdigi.com](https://fly.igdigi.com). Watch the full video (24s, with sound):
[igdigi.com/assets/video/fly-24s-musicB.mp4](https://igdigi.com/assets/video/fly-24s-musicB.mp4) —
also archived in this repo at
[docs/media/fly-24s.mp4](https://raw.githubusercontent.com/igdigitallab/bioreservoir/main/docs/media/fly-24s.mp4)
(downloads; GitHub does not preview a file this size inline).

Whole-brain connectomes are now public. The complete wiring diagram of a fruit fly's nervous
system is a real, measured biological network, and anyone can download it. As each new brain
is released (a fly today, maybe a rat or a primate later), BioReservoir runs open experiments on
it and asks one question: what does this real wiring do when we drive it with input from the
human world?

We treat each connectome as a fixed *reservoir*. We simulate it as a spiking network, feed it
input and read its response without training the brain itself. Every experiment ships with
controls. The key control is the same graph with its wiring shuffled, because a result only
means something if the real brain behaves differently from a scrambled one.

> **Honest framing.** A simulated fly does not understand elections, sports or the news.
> We are testing whether real biological network topology produces structured, non-random
> responses, and we publish the result either way.

## Experiments

| # | Name | Brains | Status |
|---|---|---|---|
| 001 | [Fly Oracle](experiments/001-fly-oracle/): two fly brains forecast the 2026 US midterms and other public events | male *Drosophila* (MaleCNS v1.0) and female *Drosophila* (BANC) | design |

## Principles

- **Open code, open method.** Everything needed to reproduce a run lives in this repo.
- **Pre-registered.** Questions, seeds and protocol are published before any run. Predictions are
  hash-committed before the outcome is known.
- **Controls, always.** Every brain is compared against a no-brain baseline, a rewired
  connectome and a coin flip.
- **Paper only.** No real-money betting and no trading accounts.
- **No scraping.** Questions are curated by hand from the public agenda and worded by us.

## Repository

```
src/bioreservoir/     shared core: connectome loaders, the LIF simulation, encoding,
                      readout and scoring, plus the live service behind fly.igdigi.com
experiments/NNN-*/    one folder per experiment: protocol, question set, seeds, results
scripts/              data fetching with a checksum manifest, atlas and image export
tests/                unit tests for everything that does not need a real connectome
site/                 the public page (Vite + TypeScript + Three.js)
docs/                 data sources and licenses, model and calibration, the live service
```

Getting the data: see [docs/DATA.md](docs/DATA.md). The model, its parameters and the calibration
runs that reproduce the Shiu et al. reference result: [docs/MODEL.md](docs/MODEL.md).

## Verifying a pre-registration

Every experiment publishes the hash of its inputs before it runs and the hash of its predictions
before the outcome is known, each with an [OpenTimestamps](https://opentimestamps.org) proof
anchored in the Bitcoin blockchain. To check one yourself:

```bash
# 1. the timestamp proof is valid and names a block mined at that date
ots verify experiments/001-fly-oracle/stamps/2026-09-18-preregistration.sha256.ots

# 2. the files in this repo are the ones that were stamped
sha256sum experiments/001-fly-oracle/questions.yaml \
          experiments/001-fly-oracle/README.md \
          experiments/001-fly-oracle/populations.yaml
# compare against experiments/001-fly-oracle/stamps/2026-09-18-preregistration.sha256
```

A timestamp proves that these exact bytes existed on that date. It does not prove that the stated
system produced them — that is what the published code, the fixed seeds and the deterministic
pipeline are for: anyone can rerun a prediction and get the same number.

## Data and attribution

Connectome data comes from the Janelia **MaleCNS** release and the **BANC** (Brain And Nerve Cord)
release, both under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Citations and changes are listed in
[NOTICE](NOTICE) and [docs/DATA.md](docs/DATA.md). The simulation approach follows the
whole-brain leaky integrate-and-fire model of Shiu et al., *Nature* 634 (2024).

BioReservoir is not affiliated with or endorsed by Janelia, HHMI, the BANC or FlyWire teams, the
dataset authors or any prediction-market platform.

## License

Code: [Apache License 2.0](LICENSE). Third-party data and models keep their own licenses, all of
them listed with citations and a record of what we changed in [NOTICE](NOTICE).

## About this repository

This is the public repository of an internal working tree: it is published as squashed snapshots,
so the commit history here is release history, not the minute-by-minute history of the lab. What
ships is the whole system — the loaders, the simulation, the experiment protocol, the scoring, the
site and the tests. What does not ship is our internal task board, agent notes and infrastructure
details, none of which are needed to reproduce a run.

Questions, corrections and "your control is wrong because X" are welcome as GitHub issues, and a
demonstrated error in the method is worth more to us than a flattering one.
