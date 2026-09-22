# Pre-registration stamps

Two manifests, each a list of SHA-256 hashes, each timestamped with
[OpenTimestamps](https://opentimestamps.org) so that the Bitcoin blockchain carries independent
proof of when those bytes existed.

| Manifest | Stamped | Covers |
|---|---|---|
| `2026-09-18-preregistration.sha256` | 2026-09-18 23:16 UTC | the question set, the protocol and the population rules, before any question was ever run through a brain |
| `2026-09-18-pre-batch.sha256` | 2026-09-18 23:57 UTC | everything that determines the midterm forecasts: questions, reference sentences, config (seeds), populations, protocol, and the `src/` tree that ran them |

`predictions/SHA256SUMS` + `.ots` then seal the forecasts themselves, before the outcome is known.

## Verify it yourself

The manifests were written inside our working repository, so they say "verify with
`git show <commit>:<path>`". This public repository is published as squashed snapshots and has no
such commit, so `frozen/` carries the exact stamped bytes instead. They are the same bytes: that is
the thing the hashes check.

```bash
# 1. the proof: these bytes existed on that date, anchored in a Bitcoin block
ots verify stamps/2026-09-18-pre-batch.sha256.ots

# 2. the frozen files reproduce every hash in the manifest
cd stamps/frozen/2026-09-18-pre-batch && sha256sum questions.yaml reference.yaml config.yaml populations.yaml README.md
# compare with ../../2026-09-18-pre-batch.sha256 (same order, filenames there carry their full path)

# 3. the source tree that ran the forecasts, checked against the manifest's git tree id
mkdir /tmp/v && tar xf src-at-seal.tar.gz -C /tmp/v && cd /tmp/v
git init -q && git add src && git rev-parse "$(git write-tree)":src
# -> 6615fd2c2150161ba94890864b11c5d4384e0e20, the last line of the manifest
```

## What changed after sealing

Sealing is only worth something if drift is reported rather than hidden. Comparing today's files
against `frozen/2026-09-18-pre-batch/`:

| File | Status |
|---|---|
| `questions.yaml` | byte-identical to the seal |
| `config.yaml` (seeds, side mapping, trial counts) | byte-identical to the seal |
| `reference.yaml` (the 24 neutral handedness sentences) | byte-identical to the seal |
| `populations.yaml` | changed once, 2026-09-19 00:53 UTC, while the batch was running: two readout populations (`P1_pC1`, `pIP10`) were **added** for the live page, and one comment was reworded. On 2026-09-21, publishing this repository reworded one more comment (a pointer to an internal file became a pointer to `docs/DATA.md`). The run itself reads `sensory_all` as input and `descending_all` as readout (`config.yaml`), and neither definition was ever touched — `diff` against the frozen copy shows every one of these |
| `README.md` (this protocol) | status line updated after the run, plus later edits to the "known limits" and verification sections |

Two further facts a reviewer should have, because they are visible in the data anyway:

- Every one of the 429 sealed runs carries the same `config_hash` (`9edf541db9a9b5e3`), which is
  computed from `config.yaml`'s content. One constant value across the whole batch is machine
  evidence that the configuration did not move underneath the run.
- The runs record two code commits, both flagged `-dirty`: work on the live page was in progress
  in the same working tree while the batch ran in its own process. The sealed `src/` tree above is
  the committed state the batch started from. We would rather say this out loud than have someone
  find `-dirty` in the JSON and wonder what else was not mentioned.

## What a timestamp does not prove

It proves that a file existed on a date. It does not prove that the stated system produced it. The
defence against that is a different one: the pipeline is deterministic, the seeds are in
`config.yaml`, the data is public, and the code is here — a prediction can be recomputed from
scratch and compared. Where that is not enough, an outside reviewer checking the protocol is the
right answer, and we are asking for one.
