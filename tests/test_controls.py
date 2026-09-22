import numpy as np
import pytest

from bioreservoir.oracle import controls


def _tiny_simple_graph(seed: int, n: int = 60, n_edges: int = 300):
    """A small, already-simple (no self-loops, no duplicate edges) directed weighted graph, the
    precondition every control function assumes (harmonize.load_graph's own edges are simple:
    at most one row per (pre_id, post_id) pair)."""
    rng = np.random.default_rng(seed)
    pre = rng.integers(0, n, size=n_edges * 3)
    post = rng.integers(0, n, size=n_edges * 3)
    keep = pre != post
    pre, post = pre[keep], post[keep]
    keys = pre.astype(np.int64) * n + post.astype(np.int64)
    _, first = np.unique(keys, return_index=True)
    first = np.sort(first)[:n_edges]
    pre, post = pre[first], post[first]
    weight = rng.choice([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0], size=n_edges)
    return n, pre, post, weight


def _assert_simple_graph(n: int, pre: np.ndarray, post: np.ndarray) -> None:
    assert not np.any(pre == post), "self-loop present"
    keys = pre.astype(np.int64) * n + post.astype(np.int64)
    assert np.unique(keys).size == keys.size, "duplicate edge present"


@pytest.mark.parametrize("seed", range(10))
def test_rewire_degree_preserving_invariants(seed):
    n, pre, post, weight = _tiny_simple_graph(seed=seed)
    g = controls.rewire_degree_preserving(n, pre, post, weight, seed=1000 + seed, max_repair_rounds=500)

    assert g.n_neurons == n
    # out-degree: pre_idx (and its aligned weight) is untouched row-for-row.
    assert np.array_equal(pre, g.pre_idx)
    assert np.array_equal(weight, g.weight)
    # in-degree: post_idx is a permutation of the same multiset of values.
    assert np.array_equal(np.sort(post), np.sort(g.post_idx))
    _assert_simple_graph(n, g.pre_idx, g.post_idx)


def test_rewire_is_seeded_and_deterministic():
    n, pre, post, weight = _tiny_simple_graph(seed=0)
    g1 = controls.rewire_degree_preserving(n, pre, post, weight, seed=42)
    g2 = controls.rewire_degree_preserving(n, pre, post, weight, seed=42)
    g3 = controls.rewire_degree_preserving(n, pre, post, weight, seed=43)
    assert np.array_equal(g1.post_idx, g2.post_idx)
    assert not np.array_equal(g1.post_idx, g3.post_idx)


def test_rewire_actually_changes_topology():
    n, pre, post, weight = _tiny_simple_graph(seed=0)
    g = controls.rewire_degree_preserving(n, pre, post, weight, seed=1)
    assert not np.array_equal(post, g.post_idx)


@pytest.mark.parametrize("seed", range(5))
def test_erdos_renyi_like_invariants(seed):
    n, pre, post, weight = _tiny_simple_graph(seed=seed)
    er = controls.erdos_renyi_like(n, pre, post, weight, seed=2000 + seed)

    assert er.n_neurons == n
    assert er.pre_idx.size == pre.size  # same edge count
    assert np.array_equal(np.sort(weight), np.sort(er.weight))  # same weight multiset
    _assert_simple_graph(n, er.pre_idx, er.post_idx)


def test_erdos_renyi_like_has_no_relation_to_original_topology_in_general():
    n, pre, post, weight = _tiny_simple_graph(seed=0, n=30, n_edges=100)
    er = controls.erdos_renyi_like(n, pre, post, weight, seed=5)
    original_keys = set((pre.astype(np.int64) * n + post.astype(np.int64)).tolist())
    er_keys = set((er.pre_idx.astype(np.int64) * n + er.post_idx.astype(np.int64)).tolist())
    # An ER graph on a small dense-ish random graph should not happen to reproduce most of the
    # original edge set — this is a sanity check against an accidental no-op implementation, not
    # a strict probabilistic bound.
    assert len(original_keys & er_keys) < len(original_keys)


def test_cached_control_graph_writes_and_reads_back(tmp_path, monkeypatch):
    monkeypatch.setattr(controls, "PROCESSED_DIR", tmp_path)
    n, pre, post, weight = _tiny_simple_graph(seed=0)
    calls = {"n": 0}

    def build():
        calls["n"] += 1
        return controls.rewire_degree_preserving(n, pre, post, weight, seed=9)

    g1 = controls.cached_control_graph("banc", 5, "rewired", 9, build)
    g2 = controls.cached_control_graph("banc", 5, "rewired", 9, build)
    assert calls["n"] == 1, "second call should hit the cache, not rebuild"
    assert np.array_equal(g1.pre_idx, g2.pre_idx)
    assert np.array_equal(g1.post_idx, g2.post_idx)
    assert np.array_equal(g1.weight, g2.weight)
    assert g1.n_neurons == g2.n_neurons == n
    assert (tmp_path / "banc-min5-rewired-seed9" / "edges.parquet").exists()


def test_no_brain_baseline_bias_and_probability():
    left = np.array([30.0, 20.0])
    right = np.array([10.0])
    result = controls.no_brain_baseline(left, right)
    assert result["left_rate_total_hz"] == 50.0
    assert result["right_rate_total_hz"] == 10.0
    assert result["p_yes"] > 0.5  # left-heavy drive -> P(yes) > 0.5 under the left=yes default


def test_no_brain_baseline_zero_rates_is_neutral():
    result = controls.no_brain_baseline(np.array([0.0]), np.array([0.0]))
    assert result["bias"] is None
    assert result["p_yes"] == 0.5


def test_coin_control_deterministic_and_question_specific():
    a1 = controls.coin_control("2026-senate-ak", seed_base=424242)
    a2 = controls.coin_control("2026-senate-ak", seed_base=424242)
    b = controls.coin_control("2026-senate-ga", seed_base=424242)
    assert a1 == a2
    assert a1["p_yes"] in (0.0, 1.0)
    assert a1["seed"] != b["seed"]
