"""frames.py: per-25ms-bin active-neuron-index encoding — pure numpy/base64, no Brian2."""

from __future__ import annotations

import numpy as np

from bioreservoir.live import frames


def test_n_bins_for_250ms_at_25ms_is_10():
    assert frames.n_bins_for(250.0, 25.0) == 10


def test_bin_index_places_spikes_in_expected_bins():
    times = np.array([0.0, 24.9, 25.0, 49.0, 249.9])
    bins = frames.bin_index(times, duration_ms=250.0, bin_ms=25.0)
    assert bins.tolist() == [0, 0, 1, 1, 9]


def test_bin_index_clips_spike_at_exact_duration_to_last_bin():
    times = np.array([250.0])
    bins = frames.bin_index(times, duration_ms=250.0, bin_ms=25.0)
    assert bins.tolist() == [9]


def test_active_dense_indices_per_bin_dedupes_within_a_bin():
    idx = np.array([3, 3, 5, 3])
    times = np.array([1.0, 2.0, 3.0, 20.0])  # all in bin 0
    result = frames.active_dense_indices_per_bin(idx, times, duration_ms=250.0, bin_ms=25.0)
    assert result[0].tolist() == [3, 5]
    assert all(b.size == 0 for b in result[1:])


def test_active_dense_indices_per_bin_handles_no_spikes():
    result = frames.active_dense_indices_per_bin(np.array([]), np.array([]), duration_ms=250.0, bin_ms=25.0)
    assert len(result) == 10
    assert all(b.size == 0 for b in result)


def test_map_to_atlas_drops_unmapped_dense_indices():
    dense_to_atlas = {1: 100, 2: 200}
    result = frames.map_to_atlas(np.array([1, 2, 3]), dense_to_atlas)
    assert sorted(result.tolist()) == [100, 200]
    assert result.dtype == np.uint32


def test_cap_bin_leaves_small_arrays_untouched():
    arr = np.arange(10, dtype=np.uint32)
    capped, was_capped = frames.cap_bin(arr, cap=6000, seed=0)
    assert np.array_equal(capped, arr)
    assert was_capped is False


def test_cap_bin_subsamples_deterministically_when_over_cap():
    arr = np.arange(100, dtype=np.uint32)
    capped1, was_capped1 = frames.cap_bin(arr, cap=10, seed=42)
    capped2, was_capped2 = frames.cap_bin(arr, cap=10, seed=42)
    assert was_capped1 and was_capped2
    assert capped1.size == 10
    assert np.array_equal(capped1, capped2)  # deterministic given the same seed
    assert (np.diff(capped1) > 0).all()  # sorted


def test_encode_decode_bin_b64_roundtrip():
    arr = np.array([1, 5, 100000], dtype=np.uint32)
    encoded = frames.encode_bin_b64(arr)
    assert isinstance(encoded, str)
    decoded = frames.decode_bin_b64(encoded)
    assert np.array_equal(decoded, arr)


def test_encode_bin_b64_empty_array_roundtrips_to_empty():
    encoded = frames.encode_bin_b64(np.array([], dtype=np.uint32))
    decoded = frames.decode_bin_b64(encoded)
    assert decoded.size == 0


def test_frames_payload_shape_matches_answer_schema():
    dense_to_atlas = {i: i * 10 for i in range(50)}
    spike_idx = np.array([1, 2, 3, 1, 40])
    spike_time = np.array([1.0, 26.0, 3.0, 2.0, 240.0])
    payload = frames.frames_payload(spike_idx, spike_time, duration_ms=250.0, dense_to_atlas=dense_to_atlas)
    assert payload["bin_ms"] == 25.0
    assert payload["n_bins"] == 10
    assert len(payload["active_b64"]) == 10
    assert isinstance(payload["capped"], bool)
    # bin 0 should contain atlas indices for dense 1 and 3 (both fired in [0,25))
    decoded_bin0 = frames.decode_bin_b64(payload["active_b64"][0])
    assert sorted(decoded_bin0.tolist()) == sorted([dense_to_atlas[1], dense_to_atlas[3]])


def test_frames_payload_flags_capped_bins():
    dense_to_atlas = {i: i for i in range(20000)}
    spike_idx = np.arange(10000)
    spike_time = np.zeros(10000)  # all in bin 0
    payload = frames.frames_payload(
        spike_idx, spike_time, duration_ms=250.0, dense_to_atlas=dense_to_atlas, cap=6000
    )
    assert payload["capped"] is True
    decoded_bin0 = frames.decode_bin_b64(payload["active_b64"][0])
    assert decoded_bin0.size == 6000
