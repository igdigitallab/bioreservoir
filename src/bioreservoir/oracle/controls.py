"""Controls every question is scored against (README.md Pipeline step 4):

    (a) degree-preserving rewiring — same topology statistics, shuffled specific connections.
    (b) an Erdos-Renyi graph — same size and density, no real topology at all.
    (c) a no-brain baseline — the readout rule applied directly to the input drive.
    (d) a seeded coin flip.

(a) and (b) operate on the same `(n_neurons, pre_idx, post_idx, weight)` dense-index arrays
`bioreservoir.sim.bench.graph_to_arrays` returns for a real harmonized graph, and are cached to
`data/processed/` the same way `harmonize.load_graph` caches its own output (module docstring
there), keyed by dataset, `min_syn` and seed so different seeds never collide on disk.

Rewiring method
---------------
A literal edge-by-edge double-edge-swap Markov chain (pick two random edges, swap their
post-endpoints if that does not create a self-loop or a duplicate edge, repeat ~5-10x the edge
count) is the textbook approach, but at 6.2M edges (MaleCNS) that is tens of millions of Python-
level steps — not "reasonable time" in a CPU-only cage. This module instead does the same thing in
one shot: a uniformly random permutation of the `post_idx` column, holding `pre_idx` and `weight`
fixed per row.

That single permutation is mathematically the closure of applying a double-edge-swap to every pair
of edges simultaneously (swapping two rows' `post` values is exactly one double-edge-swap; a full
random permutation is the composition of many such swaps) — it preserves, for every neuron, both
its out-degree (each row keeps its original `pre_idx`, so the multiset of `pre_idx` values, and
therefore how many rows list a given neuron as source, is untouched) and its in-degree (the
`post_idx` column is *permuted*, not resampled, so the multiset of `post_idx` values, and thus the
count of rows landing on any given neuron, is exactly preserved). "Each edge's weight and
presynaptic sign" (the task's own wording) both follow immediately: `weight[i]` and `pre_idx[i]`
never move, so an edge's weight and its presynaptic neuron's sign (`sign(pre)`, folded into
`weight`'s sign already by `harmonize.load_graph`) travel with it unchanged; only which neuron it
lands on changes.

The one thing a single permutation does not guarantee is a *simple* graph: it can (rarely, for a
graph this sparse) reassign a row's post target to its own pre neuron (a self-loop) or to a target
another row already uses from the same pre neuron (a duplicate edge). `_repair_collisions` finds
and fixes those with real, local double-edge-swaps restricted to just the colliding rows — a tiny
fraction of the total, so this repair step is fast even though it is a genuine (not simultaneous)
double-edge-swap loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from bioreservoir.connectomes.harmonize import PROCESSED_DIR

CONTROL_EDGE_SCHEMA = pa.schema(
    [
        pa.field("pre_idx", pa.int64()),
        pa.field("post_idx", pa.int64()),
        pa.field("weight", pa.float64()),
    ]
)


@dataclass(frozen=True)
class GraphArrays:
    n_neurons: int
    pre_idx: np.ndarray
    post_idx: np.ndarray
    weight: np.ndarray


def _edge_keys(n_neurons: int, pre_idx: np.ndarray, post_idx: np.ndarray) -> np.ndarray:
    """One int64 per row uniquely identifying its `(pre, post)` pair. `n_neurons` (not the max
    observed index) as the multiplier keeps keys stable across calls on the same graph regardless
    of which rows happen to be present in a given subset."""
    return pre_idx.astype(np.int64) * np.int64(n_neurons) + post_idx.astype(np.int64)


def _find_collisions(n_neurons: int, pre_idx: np.ndarray, post_idx: np.ndarray) -> np.ndarray:
    """Row indices that are a self-loop or share their `(pre, post)` key with another row (all
    but the first occurrence of each duplicated key are flagged, per the module docstring)."""
    self_loop = pre_idx == post_idx
    keys = _edge_keys(n_neurons, pre_idx, post_idx)
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    is_dup_in_sorted = np.empty(sorted_keys.size, dtype=bool)
    is_dup_in_sorted[0] = False
    is_dup_in_sorted[1:] = sorted_keys[1:] == sorted_keys[:-1]
    duplicate = np.zeros(pre_idx.size, dtype=bool)
    duplicate[order[is_dup_in_sorted]] = True
    return np.where(self_loop | duplicate)[0]


def _repair_collisions(
    n_neurons: int,
    pre_idx: np.ndarray,
    post_idx: np.ndarray,
    rng: np.random.Generator,
    max_rounds: int,
) -> np.ndarray:
    """Fix self-loops/duplicate edges left by a global post-permutation with real, local
    double-edge-swaps against a fresh random partner row (module docstring). Returns a new
    `post_idx` array; `pre_idx`/`weight` are never touched by repair, so the invariants they carry
    stay exact throughout.

    Each bad row is paired against a partner drawn from *all* rows, not just other bad rows: a
    duplicate-edge pair's non-flagged sibling, or a lone self-loop with no other bad row left to
    swap with, both need a good row as a partner to fix, and restricting partners to the bad set
    alone can never resolve those (verified empirically: it stalls forever on exactly this case on
    small test graphs). Rows are processed one at a time (not batched) so `existing`'s counts are
    always consistent when the next row's candidate swap is evaluated.
    """
    post_idx = post_idx.copy()
    n_edges = pre_idx.size
    # Multiplicity, not membership: two colliding (duplicate) rows share one key, so a plain set
    # would forget that key is still legitimately occupied by whichever row of the pair does NOT
    # move. A `Counter` (int64 key -> how many rows currently use it) tracks that correctly as
    # rows are tentatively moved and, if invalid, moved back.
    from collections import Counter

    existing: Counter[int] = Counter(_edge_keys(n_neurons, pre_idx, post_idx).tolist())

    for _ in range(max_rounds):
        bad = _find_collisions(n_neurons, pre_idx, post_idx)
        if bad.size == 0:
            return post_idx
        partners = rng.integers(0, n_edges, size=bad.size)
        for a, b in zip(bad.tolist(), partners.tolist()):
            if a == b:
                continue
            old_key_a = int(pre_idx[a]) * n_neurons + int(post_idx[a])
            old_key_b = int(pre_idx[b]) * n_neurons + int(post_idx[b])
            new_post_a, new_post_b = post_idx[b], post_idx[a]
            new_key_a = int(pre_idx[a]) * n_neurons + int(new_post_a)
            new_key_b = int(pre_idx[b]) * n_neurons + int(new_post_b)

            # Tentatively vacate a's and b's own current keys before checking the target keys are
            # free, so a swap between two rows that happen to collide with only *each other*
            # (old_key_a == old_key_b, or new_key_a/new_key_b coinciding with an old key) is
            # judged correctly instead of always rejected by its own about-to-be-vacated entry.
            existing[old_key_a] -= 1
            existing[old_key_b] -= 1
            valid = (
                pre_idx[a] != new_post_a
                and pre_idx[b] != new_post_b
                and new_key_a != new_key_b
                and existing[new_key_a] == 0
                and existing[new_key_b] == 0
            )
            if not valid:
                existing[old_key_a] += 1
                existing[old_key_b] += 1
                continue  # left as still-bad; retried next round with a fresh random partner
            existing[new_key_a] += 1
            existing[new_key_b] += 1
            post_idx[a], post_idx[b] = new_post_a, new_post_b

    remaining = _find_collisions(n_neurons, pre_idx, post_idx)
    if remaining.size:
        raise RuntimeError(
            f"rewiring repair did not converge: {remaining.size} colliding edge(s) left after "
            f"{max_rounds} rounds — raise controls.rewire_max_repair_rounds in config.yaml"
        )
    return post_idx


def rewire_degree_preserving(
    n_neurons: int,
    pre_idx: np.ndarray,
    post_idx: np.ndarray,
    weight: np.ndarray,
    seed: int,
    max_repair_rounds: int = 200,
) -> GraphArrays:
    """Degree-, weight- and sign-preserving rewiring of one harmonized graph (module docstring)."""
    rng = np.random.default_rng(seed)
    permuted_post = rng.permutation(post_idx)
    repaired_post = _repair_collisions(n_neurons, pre_idx, permuted_post, rng, max_repair_rounds)
    return GraphArrays(n_neurons=n_neurons, pre_idx=pre_idx.copy(), post_idx=repaired_post, weight=weight.copy())


def erdos_renyi_like(
    n_neurons: int,
    pre_idx: np.ndarray,
    post_idx: np.ndarray,
    weight: np.ndarray,
    seed: int,
    oversample: float = 1.3,
) -> GraphArrays:
    """Erdos-Renyi control: same `n_neurons`, same edge count and the same weight multiset as the
    real graph, but every `(pre, post)` pair drawn uniformly at random (no degree structure at
    all) instead of taken from the connectome. Weights are a seeded permutation of the real
    graph's own weight array, then assigned to the random edges in that (also arbitrary) order —
    so the *set* of weights each neuron in the real graph produced is preserved network-wide, but
    no longer tied to which neuron produced which value, unlike the rewiring control.

    Generated by batched rejection sampling (numpy-vectorized: candidate pairs, self-loop removal
    and within-batch de-duplication are all array ops; only the small cross-batch "already used"
    check is a Python-level set) rather than one edge at a time — at MaleCNS's density (6.2M edges
    over ~165k^2 possible pairs, collision probability per draw is on the order of 1e-4) this
    reaches the target edge count in a small, bounded number of batches.
    """
    n_edges = pre_idx.size
    rng = np.random.default_rng(seed)
    shuffled_weight = rng.permutation(weight)

    seen: set[int] = set()
    pre_chunks: list[np.ndarray] = []
    post_chunks: list[np.ndarray] = []
    remaining = n_edges
    while remaining > 0:
        batch = max(int(remaining * oversample) + 64, 64)
        cand_pre = rng.integers(0, n_neurons, size=batch, dtype=np.int64)
        cand_post = rng.integers(0, n_neurons, size=batch, dtype=np.int64)
        keep = cand_pre != cand_post
        cand_pre, cand_post = cand_pre[keep], cand_post[keep]
        cand_key = cand_pre * np.int64(n_neurons) + cand_post
        _, first_idx = np.unique(cand_key, return_index=True)
        first_idx = np.sort(first_idx)
        cand_pre, cand_post, cand_key = cand_pre[first_idx], cand_post[first_idx], cand_key[first_idx]
        novel = np.fromiter((k not in seen for k in cand_key.tolist()), dtype=bool, count=cand_key.size)
        cand_pre, cand_post, cand_key = cand_pre[novel], cand_post[novel], cand_key[novel]
        take = min(cand_pre.size, remaining)
        pre_chunks.append(cand_pre[:take])
        post_chunks.append(cand_post[:take])
        seen.update(cand_key[:take].tolist())
        remaining -= take

    return GraphArrays(
        n_neurons=n_neurons,
        pre_idx=np.concatenate(pre_chunks),
        post_idx=np.concatenate(post_chunks),
        weight=shuffled_weight,
    )


def _cache_dir(dataset: str, min_syn: int, condition: str, seed: int) -> Path:
    return PROCESSED_DIR / f"{dataset}-min{min_syn}-{condition}-seed{seed}"


def cached_control_graph(
    dataset: str,
    min_syn: int,
    condition: str,
    seed: int,
    build: callable[[], GraphArrays],
) -> GraphArrays:
    """Read `(pre_idx, post_idx, weight)` for `condition` ("rewired" | "er") from
    `data/processed/<dataset>-min<N>-<condition>-seed<seed>/edges.parquet` if present, else call
    `build()` and write it there — same convention as `harmonize.load_graph`'s own cache, with the
    seed folded into the directory name (task brief: "with the seed in the path") so two different
    seeded rewirings of the same graph never collide on disk.
    """
    cache_dir = _cache_dir(dataset, min_syn, condition, seed)
    edges_path = cache_dir / "edges.parquet"
    if edges_path.exists():
        table = pq.read_table(edges_path)
        n_neurons = int(table.schema.metadata[b"n_neurons"]) if table.schema.metadata else None
        if n_neurons is None:
            raise RuntimeError(f"{edges_path} is missing its n_neurons metadata — delete and rebuild")
        return GraphArrays(
            n_neurons=n_neurons,
            pre_idx=table.column("pre_idx").to_numpy(),
            post_idx=table.column("post_idx").to_numpy(),
            weight=table.column("weight").to_numpy(),
        )

    graph = build()
    table = pa.table(
        {"pre_idx": graph.pre_idx, "post_idx": graph.post_idx, "weight": graph.weight},
        schema=CONTROL_EDGE_SCHEMA,
    ).replace_schema_metadata({b"n_neurons": str(graph.n_neurons).encode("ascii")})
    cache_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, edges_path)
    return graph


def no_brain_baseline(left_rate_hz: np.ndarray, right_rate_hz: np.ndarray) -> dict:
    """The same readout rule applied directly to the input drive, no connectome at all
    (README.md Pipeline step 4a): lateral bias of the encoder's own left/right Poisson rates.
    Meaningful even though nothing spikes here — it is a control for how much of the brain
    readout's apparent "signal" is already present in the raw embedding-to-rate mapping before any
    simulation runs."""
    from bioreservoir.oracle.readout import lateral_bias, probability_from_bias

    left_total = float(left_rate_hz.sum())
    right_total = float(right_rate_hz.sum())
    bias = lateral_bias(left_total, right_total)
    p_yes = 0.5 if bias is None else probability_from_bias(bias)
    return {"left_rate_total_hz": left_total, "right_rate_total_hz": right_total, "bias": bias, "p_yes": p_yes}


def coin_control(question_id: str, seed_base: int) -> dict:
    """A seeded coin flip (README.md Pipeline step 4d): one concrete, reproducible binary decision
    per question, not a constant P(yes)=0.5 forecast — the task brief's "coin = seeded by question
    id" wording describes flipping an actual coin, not stating a fixed probability. `p_yes` is
    correspondingly 1.0 or 0.0 (full confidence in whichever way the coin landed), so scoring
    treats it exactly like any other prediction rather than as a special always-undecided case."""
    from bioreservoir.oracle.seeding import stable_seed

    seed = stable_seed(question_id, base=seed_base)
    rng = np.random.default_rng(seed)
    heads = bool(rng.random() < 0.5)
    return {"seed": seed, "heads": heads, "p_yes": 1.0 if heads else 0.0}
