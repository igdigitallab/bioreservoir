# Data

Fetch everything with `python scripts/fetch_data.py` (stdlib only, no venv required). It reads
`scripts/datasets.json`, skips files whose SHA-256 already matches, and re-downloads anything
missing or corrupted. `--check-only` verifies without downloading; `--dataset <name>` restricts
to one dataset (`malecns-v1.0`, `banc-626`, `all-MiniLM-L6-v2`).

## MaleCNS v1.0 — male *Drosophila* CNS

Janelia FlyEM's complete adult male central-nervous-system connectome (brain + ventral nerve
cord), released 2026-06-08. On disk: 3 files, 1.03 GiB (`data/raw/malecns-v1.0/`).

- **License:** CC BY 4.0.
- **Attribution:** Berg, Beckett, Costa et al. *Sexual dimorphism in the complete connectome of
  the Drosophila male central nervous system.* bioRxiv (2025). doi:10.1101/2025.10.09.680999.
  Source: https://male-cns.janelia.org/. "Changes: converted into flat tables and re-mapped into
  a shared neuron/edge schema for this project." Not affiliated with or endorsed by Janelia,
  HHMI or the dataset authors.
- **Files used:** `body-annotations-*-minconf-0.5.feather` (cell type/class/side, filtered to
  `status == "Traced"`, the fully proofread subset: 165,122 of 211,577 rows), and
  `body-neurotransmitters-*.feather` (per-body predicted NT + confidence), and
  `connectome-weights-*-minconf-0.5.feather`, the plain (not `-significant-only` /
  `-traced-only`) variant — it is the one file the official download page documents as "the
  complete connection graph"; the two undocumented siblings were skipped rather than guessed at.
  We did **not** fetch `body-stats` (778 MB, per-body synapse totals) or any of `syn-partners` /
  `syn-points` / `tbar-neurotransmitters` (2.6–13 GB each, per-synapse-point tables — out of
  budget and out of scope for a neuron-level schema).

## BANC — female *Drosophila* brain and nerve cord

Female adult CNS reconstruction from the preprint *Distributed control circuits across a
brain-and-cord connectome*. Harvard Dataverse doi:10.7910/DVN/8TFGGB, license field **CC BY
4.0** (confirmed via the Dataverse API). All files used here share **CAVE materialization 626**,
snapshotted 2025-07-21 (per the dataset's own description) — a different, later snapshot,
`banc_meta_821.tab`, exists for cross-referencing but was not needed and not fetched. On disk: 3
files, 64.5 MiB (`data/raw/banc-626/`).

- **Attribution:** Bates, Phelps, Kim et al. *Distributed control circuits across a
  brain-and-cord connectome* (preprint). Harvard Dataverse, doi:10.7910/DVN/8TFGGB.
- **Files used:** `codex_annotations_flat_table.tab` (cell type/class/side; **only 78,621 of
  114,461 rows have a non-null `cell_type`** — beware quoted `"NA"` strings, which are a real
  null, not the text "NA"), `banc_neurotransmitter_prediction.csv` (predicted NT + confidence
  0-100, from the same table so the pair is internally consistent; 13 duplicate IDs deduplicated
  by keeping the first row), `connections_princeton.csv.gz` (synapse counts per
  pre/post/neuropil triple, summed over neuropil to one row per pre/post pair; **its own
  Dataverse description says pairs below 3 total synapses and all autapses are already
  excluded** — see "Harmonized graphs" below), and `backbone_proofread.tab` (proofreading
  status, materialization 626; used only by `harmonize.py`, not by `banc.load_neurons`). Skipped:
  `synapses_250226_human_readable.csv.gz` and `banc_synapses_to_neuropils_250226.csv` (12.2–12.7
  GB, per-synapse tables), `synapses_250226_backbone_proofread_counts.csv` (376 MB, a
  proofread-only edge table redundant with `connections_princeton.csv.gz` + the proofreading
  filter we already apply), `neuron_skeletons.zip` / `neuron_colormips.zip` (meshes/images),
  `banc_meta_821.tab` (different, later ID snapshot), NBLAST cross-dataset similarity files.
  **Note:** Dataverse's own recorded MD5 for `backbone_proofread.tab` does not match the bytes
  served by `/access/datafile` at fetch time, though file size matches exactly and two
  independent downloads were byte-identical — the same behaviour already existed (unnoticed
  until now) for `codex_annotations_flat_table.tab`; both are Dataverse-ingested `.tab` exports,
  a known quirk of Dataverse's tabular-file re-serving, not corruption. `scripts/datasets.json`
  records the self-computed sha256 of the actually-downloaded bytes, same as every other entry.

## Text model — `sentence-transformers/all-MiniLM-L6-v2`

Pinned to commit `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. On disk: 87.3 MiB
(`data/models/all-MiniLM-L6-v2/`) — only the files `sentence-transformers` needs to load and run
the model (`model.safetensors`, tokenizer, configs); the ONNX/OpenVINO/TF/rust duplicate export
formats on the model's HF repo (another ~570 MB) were skipped as redundant.

- **License:** Apache-2.0. Reimers, Gurevych. *Sentence-BERT: Sentence Embeddings using Siamese
  BERT-Networks.* EMNLP (2019).

## Common schema (`src/bioreservoir/connectomes/schema.py`)

**Neurons:** `neuron_id, cell_type, cell_class, super_class, side, nt, nt_conf, sex, sign`
(`nt_conf` in `[0, 1]`; BANC's native 0-100 score is rescaled to match MaleCNS). **Edges:**
`pre_id, post_id, syn_count`.

**Sign mapping** (changed 2026-09-18 to match Shiu et al. 2024 exactly — was: ACh → +1,
GABA/glutamate → -1, everything else → 0): Shiu's Methods (*Nature* 634, PMC11446845) state the rule as a **strict
binary partition, no third "unknown" category**: "We assume GABAergic and glutamatergic neurons
are inhibitory... each neuron is either exclusively inhibitory or excitatory... Neurons predicted
to be dopaminergic, octopaminergic or serotonergic are assigned to the excitatory category."
GABA/glutamate → -1 (glutamate is inhibitory in insects via GluCl channels — the opposite of the
usual vertebrate convention); **everything else → +1**, including acetylcholine, dopamine,
serotonin, octopamine, and — extending Shiu's rule to two NT classes their own classifier never
predicted — tyramine and histamine (histamine is biologically inhibitory in photoreceptors, but
Shiu's model has no exception for it, so neither does this port; see `schema.py` module
docstring), plus `nt == "unclear"` or null (no confident prediction, ~8.7% of MaleCNS Traced
neurons — Shiu's classifier always outputs one of 6 classes so this case cannot arise in Shiu's
own pipeline; treated as "not GABA/glutamate" per the same binary rule). Caveat: this is a
per-*neuron* prediction from EM morphology, not a validated per-synapse measurement (MaleCNS's
own docs quote ~6-13% class error). Before this change, `sign == 0` (dopamine/serotonin/
octopamine/histamine/unclear/null, ~13.3% of MaleCNS Traced neurons) silenced every *outgoing*
recurrent synapse those neurons make (`signed_weight = sign(pre) * syn_count`), not just their
classification — see docs/MODEL.md "Calibration" for the measured effect on the canonical
sugar-GRN → MN9 result.

## `inspect` output (read-only; `python -m bioreservoir.connectomes.inspect <malecns|banc>`)

| | MaleCNS (male) | BANC (female) |
|---|---|---|
| neurons | 165,122 | 114,461 |
| edges | 151,856,684 | 2,676,592 |
| total synapses | 311,833,243 | 22,243,310 |
| top NT | ACh 94,946 / Glu 28,055 / GABA 20,218 | ACh 66,334 / GABA 20,895 / Glu 16,392 |

Key cell types (exact `cell_type` match unless noted; "not found" means the name is genuinely
absent from that dataset's own vocabulary, not a search failure):

| cell type | MaleCNS | BANC | note |
|---|---|---|---|
| sugar GRN (`Gr5a`/`Gr64f`) | 0 | 0 | neither dataset names taste neurons after the receptor gene; closest proxy is the class-level count printed by `inspect` (MaleCNS `class=="gustatory"`: 1,428; BANC `cell_class=="taste_peg_neuron"`: 1,216). **Not the same as `populations.yaml`'s `gustatory_sugar`**, which cross-matches BANC's `cell_function_detailed=="sweet"` `cell_type` names (LB3a/LB3b, "labellar bristle") against MaleCNS's own `type` column — MaleCNS n=28 (both independently `class=="gustatory"` there too), see "Harmonized graphs" below |
| bitter GRN (`Gr66a`) | 0 | 0 | same caveat as above; `gustatory_bitter` cross-match (LB1a/LB1b) gives MaleCNS n=17 |
| MN9 | 2 | 0 | |
| giant fiber DNp01 | 2 | 2 | |
| DNa01 | 2 | 2 | |
| DNa02 | 2 | 2 | |
| MDN (moonwalker) | 4 | 4 | |
| P9 descending | 0 | 0 | no `cell_type` literally named "P9" in either dataset (a substring search hits unrelated `LoVP9*`/`P6-8P9` types) |
| P1 (male courtship) | 156 | 10 | via `pC1` prefix (P1 ≡ pC1 hemilineage is a standard synonym: von Philipsborn 2011, Kohatsu 2011). BANC's own `pC1a-e` are present (10) — expected, pC1 exists in both sexes — but are not labelled "P1", since that is the male-specific behavioural name, not used in BANC's female annotation |
| pIP10 | 2 | 0 | |

## Known differences that matter for a male-vs-female comparison

- **Annotation coverage differs a lot.** MaleCNS: 165,122/211,577 (78%) of annotated bodies are
  `Traced`, and 164,506 of those (>99%) carry a `cell_type`. BANC: only 78,621/114,461 (69%) of
  annotated rows carry a `cell_type` at all, out of a full segmentation of ~188k bodies (i.e.
  well under half of all BANC neurons are typed in this snapshot) — a much sparser public
  annotation than MaleCNS's.
- **Proofreading status lives in different places.** MaleCNS bakes a `status` column (Traced /
  Orphan / Glia / ...) directly into the same table used for cell typing, so "is this a real,
  complete neuron" is answerable from one file. BANC keeps proofreading status in a separate
  file (`backbone_proofread.tab`). `banc.load_neurons` (the raw, common-schema loader) still does
  not filter by it, by design — see "Harmonized graphs" below for the layer that does.
- **Neuron/edge counts are not directly comparable as "brain size".** MaleCNS is brain + VNC for
  a male; BANC is brain + VNC for a female; MaleCNS's edge table (151.9M rows) is the *complete*
  segment-to-segment graph including non-typed bodies, while BANC's edge table (2.7M rows, summed
  over neuropil) only covers pairs that appear in the Princeton connectivity export — the two
  edge tables were not filtered to the same criteria and should not be diffed row-for-row.
- **Sex-specific circuitry.** P1/pC1 courtship-command neurons and the male-specific dimorphism
  fields in MaleCNS have no equivalent structure in BANC's schema; BANC has its own
  female-specific circuits (e.g. oviposition control) with no MaleCNS counterpart.

- **Left/right imbalance in BANC.** The public snapshot has 66,745 right-side vs 46,107 left-side
  neurons (+45%), a proofreading artefact. Any left-vs-right input or readout on the female brain
  must use equal neuron counts per side (seeded down-sampling) or normalize the total drive.
  `side` is normalized to `L`/`R`/`M` in the common schema for both brains.
- **MaleCNS `somaSide` is null for primary sensory neurons** (16,080/165,122 Traced, 9.7%,
  checked 2026-09-18) — a soma side is only meaningful for neurons whose soma is inside the
  traced volume, so essentially every gustatory/olfactory/Johnston's-organ row has no `somaSide`.
  Fixed 2026-09-18 (was a `bench.py`-only workaround that missed sensory populations):
  `populations.raw_annotations` now coalesces `somaSide` with `rootSide` (the hemisphere a
  neuron's primary neurite enters on; its own extra value `"unknown"` is treated as null, not a
  fourth side) for every rule in `populations.yaml`, filling 15,864 of those 16,080 and leaving
  only 216 genuinely sideless.

## Harmonized graphs (`src/bioreservoir/connectomes/harmonize.py`, `populations.py`)

One identical rule on both brains: **neurons** = proofread only (MaleCNS `status == "Traced"`;
BANC = `neuron_id` in `backbone_proofread.tab`, materialization 626 — in practice this removes
almost nothing, since 114,456/114,456 already-annotated BANC neurons turn out to already be
proofread); **edges** = both ends in that set and `syn_count >= 5`. BANC's own
`connections_princeton.csv.gz` already drops pairs under 3 synapses server-side, so 5 is a real
shared floor on both sides, not just on MaleCNS. `signed_weight = sign(presynaptic neuron) *
syn_count`. Results cache as parquet under `data/processed/<dataset>-min<N>/`.

| | MaleCNS (male) | BANC (female) |
|---|---|---|
| neurons | 165,122 | 114,456 |
| edges | 6,235,682 | 1,395,876 |
| synapses | 89,731,551 | 17,903,013 |
| mean degree (edges/neurons) | 37.76 | 12.20 |
| excitatory / inhibitory edges (sign, updated 2026-09-18) | 3.96M / 2.27M | 0.82M / 0.57M |

Population counts (`populations.yaml`, rules per dataset's own vocabulary, run via `inspect
--harmonized`): gustatory sugar/bitter **functional label** (`cell_function_detailed`) exists
only in BANC (sweet=200, bitter=88); MaleCNS has no gene- or function-level split of its own, but
`gustatory_sugar`/`gustatory_bitter` are now cross-matched by reusing BANC's own `cell_type`
names for those neurons against MaleCNS's `type` column (task 3, 2026-09-18): MaleCNS sugar
(LB3a+LB3b) = 28, bitter (LB1a+LB1b) = 17 — both independently `class=="gustatory"` in MaleCNS
too, so corroborated, not just name-matched; BANC's other sweet/bitter type names (SAch02/SNch11;
SNxx19/20/21) either do not exist in MaleCNS's vocabulary or exist there tagged
`class=="unknown_sensory"`/abdomen, not gustatory, and were deliberately excluded. Pooled
`gustatory_all` 1,428 vs 1,491; olfactory 2,639 vs 1,429; Johnston's organ 672 vs 806;
photoreceptor 4,107 vs 68 (**60x, see caveat below**); `DNa01`/`DNa02` L/R = 1 each, `DNp01` = 2,
`MDN` = 4 — present and matched in both; `MN9` = 2 in MaleCNS, **absent in BANC** (closest is
`cell_function == "proboscis_motor"`, 35 opaquely-named neurons; checked cross-dataset via
`other_names`, `manc_121_match_id` and `fafb_783_match_id` too, 2026-09-18 — none carry an "MN9"
name, not a confirmed homolog, reported absent rather than guessed). New readout/input
populations added for calibration (task 5-6): `descending_all_left`/`right` (super_class-level
pool, not one named type — MaleCNS 656/648, BANC 648/655, a near-1:1 side balance without
downsampling, much larger than the 2+2-neuron DNa01/DNa02 pair) and `sensory_all` (every primary
sensory super_class pooled, MaleCNS 15,912 / BANC 12,985) — see docs/MODEL.md for why.

**What still differs after harmonizing:** mean degree is 3x higher in MaleCNS — the two datasets
were proofread and exported to different depths, not just different sizes. Photoreceptor counts
differ 60x, most likely retina reconstruction/annotation coverage, not biology. BANC's edge
export was already curated (min-3-synapse, autapse-free) before we ever touch it, MaleCNS's
was not; both now share the same *floor*, but not the same *upstream processing history*.

**Verdict:** the harmonized graphs are fair enough for a same-species, same-methodology
comparison of shared circuits (turning, escape, backward-walking readouts all present 1:1) —
but only with the sugar/bitter, MN9 and photoreceptor caveats stated per-population above, and
with mean-degree (density) reported alongside any result, since it is not equalized by this
rule and could by itself explain a "stronger"/"weaker" response in one brain.

## Third-party assets — fruit-fly specimen

`site/public/models/fly.glb` is derived from the anatomical **flybody** mesh assets by
Google DeepMind and HHMI Janelia Research Campus (Vaxenburg et al.,
[Whole-body physics simulation of fruit fly locomotion](https://doi.org/10.1038/s41586-025-09029-4),
Nature 643, 1312–1320, 2025).

- **Source:** [TuragaLab/flybody, fruitfly/assets](https://github.com/TuragaLab/flybody/tree/736608121847c3c025fd02a629bdb016a3294f9a/flybody/fruitfly/assets).
  Pinned commit: `736608121847c3c025fd02a629bdb016a3294f9a`.
  The brief's `google-deepmind/flybody` URL returned 404; TuragaLab is the official upstream
  linked by both the paper and DeepMind's model catalogue.
- **Licence:** Apache-2.0, including the model assets. Checked the upstream
  [LICENSE](https://github.com/TuragaLab/flybody/blob/736608121847c3c025fd02a629bdb016a3294f9a/LICENSE)
  and DeepMind's **model-specific**
  [flybody README licence statement](https://github.com/google-deepmind/mujoco_menagerie/blob/8161bba264d7fa7c99ca301e91e7fb44737676ad/flybody/README.md#license).
  That README identifies the exact upstream commit used here and states “This model is
  released under an Apache-2.0 License.” No asset-specific licence override or upstream
  NOTICE file is present in the pinned source. Full licence and attribution are distributed
  beside the GLB as `fly-LICENSE.txt` and `fly-NOTICE.txt`, without a visible UI credit.
- **Changes:** CPU mesh decimation, 16-bit positions and 8-bit normals using
  `KHR_mesh_quantization`, anatomical joint hierarchy retained, stone/ink materials and
  translucent membranes. The female-source abdomen is shortened 20% and terminal tergites
  darkened to suggest a male. This is a display adaptation, **not a male scan or a reconstruction
  of the male connectome donor**. Mesh names such as `head_red` refer to upstream material
  regions; the distributed eye material is ink, not red. No FlyWire asset is used.
- **Rebuild:** `scripts/build_fly_model.py` documents pinned CPU-only Python dependencies and
  downloads the pinned archive, or accepts a checkout at that commit with `--source`.
  See [fly specimen development notes](FLY_MODEL.md) for preview and verification instructions.
