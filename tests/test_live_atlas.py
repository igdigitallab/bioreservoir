"""atlas.py: `neuron_id -> atlas index` mapping against a small fixture matching the documented
format (`meta.json` + `neuron_ids.u64`, little-endian uint64) — the real frontend-exported files do
not exist in this worktree yet (confirmed: `site/public/data/atlas/` is absent from the repo)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from bioreservoir.live import atlas


def _write_fixture(atlas_dir, neuron_ids):
    atlas_dir.mkdir(parents=True, exist_ok=True)
    (atlas_dir / "meta.json").write_text(json.dumps({"n_points": len(neuron_ids), "brain": "malecns"}))
    np.array(neuron_ids, dtype="<u8").tofile(atlas_dir / "neuron_ids.u64")


def test_atlas_files_exist_is_false_when_missing(tmp_path):
    assert atlas.atlas_files_exist(tmp_path / "nowhere") is False


def test_atlas_files_exist_is_true_once_both_files_are_present(tmp_path):
    _write_fixture(tmp_path / "atlas", [10, 20, 30])
    assert atlas.atlas_files_exist(tmp_path / "atlas") is True


def test_load_meta_raises_a_clear_error_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="not generated yet"):
        atlas.load_meta(tmp_path / "nowhere")


def test_load_neuron_ids_raises_a_clear_error_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="not generated yet"):
        atlas.load_neuron_ids(tmp_path / "nowhere")


def test_load_neuron_ids_reads_little_endian_uint64_in_order(tmp_path):
    atlas_dir = tmp_path / "atlas"
    _write_fixture(atlas_dir, [111, 222, 333])
    ids = atlas.load_neuron_ids(atlas_dir)
    assert ids.dtype == np.dtype("<u8")
    assert ids.tolist() == [111, 222, 333]


def test_build_dense_to_atlas_maps_intersection_only(tmp_path):
    atlas_dir = tmp_path / "atlas"
    _write_fixture(atlas_dir, [100, 200, 300])  # atlas indices 0, 1, 2
    id_to_dense = {100: 5, 200: 6, 999: 7}  # 999 is not in the atlas at all
    neuron_ids = atlas.load_neuron_ids(atlas_dir)
    result = atlas.build_dense_to_atlas(id_to_dense, neuron_ids)
    assert result == {5: 0, 6: 1}
    assert 7 not in result  # neuron_id 999 absent from the atlas -> silently dropped, not an error


def test_build_dense_to_atlas_empty_when_no_overlap(tmp_path):
    atlas_dir = tmp_path / "atlas"
    _write_fixture(atlas_dir, [1, 2, 3])
    result = atlas.build_dense_to_atlas({999: 0}, atlas.load_neuron_ids(atlas_dir))
    assert result == {}


