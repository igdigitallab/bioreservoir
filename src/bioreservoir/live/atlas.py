"""`neuron_id -> atlas index` mapping for the Three.js soma-cloud visualisation.

`scripts/export_atlas.py` produces `site/public/data/atlas/{meta.json,positions.f32,neuron_ids.u64}`
(see `site/README.md`'s "Atlas export") — this module only *reads* that pair of files (never
`positions.f32`, which the backend has no use for). Documented format: `neuron_ids.u64` is a flat
little-endian `uint64` array, one MaleCNS `neuron_id` (dataset-native body ID, `schema.py`) per
atlas point, in the same order as `positions.f32`'s (x, y, z) triples — so an atlas *index* (this
module's output) is simply that array's position, and is what `frames.py` encodes per spike bin.

If the files do not exist yet (generated later, or tested against a fixture), every
function here raises `FileNotFoundError` with the expected path, not a bare `IOError` — worker.py
is expected to catch this at startup and run without frames (`frames=None` in the Answer) rather
than crash, since frames are a visual extra, not load-bearing for the answer itself.
"""

from __future__ import annotations

import json

import numpy as np

from bioreservoir.live import config


def atlas_files_exist(atlas_dir=config.ATLAS_DIR) -> bool:
    return (atlas_dir / "meta.json").exists() and (atlas_dir / "neuron_ids.u64").exists()


def load_meta(atlas_dir=config.ATLAS_DIR) -> dict:
    path = atlas_dir / "meta.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — atlas not generated yet (see module docstring)")
    return json.loads(path.read_text())


def load_neuron_ids(atlas_dir=config.ATLAS_DIR) -> np.ndarray:
    """The atlas's own `neuron_id` array, dtype `uint64`, index == atlas index."""
    path = atlas_dir / "neuron_ids.u64"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — atlas not generated yet (see module docstring)")
    return np.fromfile(path, dtype="<u8")  # little-endian uint64, per the documented format


def build_dense_to_atlas(id_to_dense: dict[int, int], neuron_ids: np.ndarray) -> dict[int, int]:
    """`{dense_brian2_index: atlas_index}` for every neuron present in BOTH the simulated graph
    (`id_to_dense`, `bioreservoir.sim.bench.graph_to_arrays`'s own map) and the atlas file. A
    neuron missing from one side (harmonization/atlas-export used slightly different filters) is
    simply absent from the result — `frames.py` treats an unmapped dense index as invisible, not
    an error, since the atlas is a rendering aid, not part of the scored pipeline.
    """
    dense_to_neuron_id = {dense: neuron_id for neuron_id, dense in id_to_dense.items()}
    neuron_id_to_atlas = {int(neuron_id): i for i, neuron_id in enumerate(neuron_ids.tolist())}
    return {
        dense: neuron_id_to_atlas[neuron_id]
        for dense, neuron_id in dense_to_neuron_id.items()
        if neuron_id in neuron_id_to_atlas
    }
