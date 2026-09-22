"""Question text -> per-neuron Poisson input rates (README.md Pipeline step 1).

`question + "\\n" + context` is embedded with the local `all-MiniLM-L6-v2` model (offline, no
network call — see `load_encoder`), then pushed through a fixed, seeded Gaussian random
projection onto the neurons of one configured input population, split left/right in equal counts
per side (`bioreservoir.sim.bench.balanced_lateral_population` — reused, not reimplemented, per
memory `banc-left-right-asymmetry.md`). The whole path is deterministic: same question text, same
dataset, same config -> the exact same `EncodedInput` every time, so `LIFNetwork.run_trial`'s
per-trial `seed` is the pipeline's only source of randomness (README.md Pipeline step 5,
"deterministic, so anyone can recompute every prediction").
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from bioreservoir.oracle.config import REPO_ROOT

MODEL_DIR = REPO_ROOT / "data" / "models" / "all-MiniLM-L6-v2"


@dataclass(frozen=True)
class EncodedInput:
    """Per-trial-invariant stimulus for one question variant on one (dataset, input population).

    `dense_idx`/`rate_hz` are what `LIFNetwork.run_trial` wants directly (`(idx, rate)` tuple).
    `left_idx`/`right_idx`/`left_rate_hz`/`right_rate_hz` are the same data split by hemisphere,
    needed by the no-brain baseline (`oracle.controls.no_brain_baseline`), which reads out the
    lateral bias of the *input drive itself*, with no connectome in the loop at all.
    """

    dense_idx: np.ndarray
    rate_hz: np.ndarray
    left_idx: np.ndarray
    right_idx: np.ndarray
    left_rate_hz: np.ndarray
    right_rate_hz: np.ndarray
    side_counts: dict[str, int]
    embedding_dim: int


def load_encoder():
    """Load `all-MiniLM-L6-v2` from the local, already-downloaded path (docs/DATA.md), fully
    offline — `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE` are set defensively in case some dependency
    still tries a network round-trip for a version check even when given a local folder path."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from sentence_transformers import SentenceTransformer

    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            f"{MODEL_DIR} not found — run `python3 scripts/fetch_data.py --dataset all-MiniLM-L6-v2`"
        )
    return SentenceTransformer(str(MODEL_DIR))


def embed_text(question: str, context: str, model=None) -> np.ndarray:
    """`question + "\\n" + context` -> a 384-dim embedding (all-MiniLM-L6-v2's native output
    size). Deterministic: `SentenceTransformer.encode` runs the model in eval mode (no dropout),
    CPU inference is bit-reproducible given fixed weights and input text."""
    if model is None:
        model = load_encoder()
    text = f"{question}\n{context}"
    embedding = model.encode(text, convert_to_numpy=True, show_progress_bar=False)
    return np.asarray(embedding, dtype=np.float64)


def gaussian_projection_matrix(seed: int, n_targets: int, embedding_dim: int) -> np.ndarray:
    """Fixed, seeded Gaussian random projection, shape `(n_targets, embedding_dim)`.

    `n_targets` is the size of the specific (dataset, input population) neuron set being driven,
    not a global constant — population sizes differ between MaleCNS and BANC (docs/DATA.md) and
    between input populations (populations.yaml). "Fixed seeded" (README.md) means: the same
    `(seed, n_targets, embedding_dim)` triple always produces the same matrix, not that one matrix
    is shared across datasets of different sizes.
    """
    rng = np.random.default_rng(seed)
    return rng.normal(loc=0.0, scale=1.0, size=(n_targets, embedding_dim))


def project_to_rates(
    embedding: np.ndarray, projection: np.ndarray, rate_hz_min: float, rate_hz_max: float
) -> np.ndarray:
    """`projection @ embedding` -> one Poisson rate per target neuron in `[rate_hz_min,
    rate_hz_max]`. Raw projection scores are z-scored across the target population before the
    logistic squash, so the rate spread is stable across questions regardless of small drifts in
    embedding norm (MiniLM embeddings are close to unit-norm but not exactly, and the projection's
    scale otherwise depends on `n_targets`)."""
    raw = projection @ embedding
    std = raw.std()
    z = (raw - raw.mean()) / std if std > 0 else np.zeros_like(raw)
    squashed = 1.0 / (1.0 + np.exp(-z))
    return rate_hz_min + (rate_hz_max - rate_hz_min) * squashed


def encode_question_to_input(
    question: str,
    context: str,
    dataset: str,
    id_to_dense: dict[int, int],
    input_population: str,
    projection_seed: int,
    balance_seed: int,
    rate_hz_min: float,
    rate_hz_max: float,
    n_per_side: int | None = None,
    model=None,
) -> EncodedInput:
    """Full README.md Pipeline step 1 for one question variant on one brain.

    `dataset`/`id_to_dense` identify the harmonized graph (`bioreservoir.sim.bench.graph_to_arrays`)
    this encoding targets; `input_population` is looked up in `populations.yaml` via
    `bioreservoir.sim.bench.balanced_lateral_population`, which the encoder reuses rather than
    reimplementing (module docstring).
    """
    from bioreservoir.sim import bench

    left_idx, right_idx, side_counts = bench.balanced_lateral_population(
        dataset, id_to_dense, input_population, seed=balance_seed
    )
    if n_per_side is not None:
        # Keep the drive inside the calibrated regime (docs/MODEL.md: 300-1000 driven neurons).
        rng = np.random.default_rng(balance_seed)
        left_idx = np.sort(rng.choice(left_idx, size=min(n_per_side, left_idx.size), replace=False))
        right_idx = np.sort(rng.choice(right_idx, size=min(n_per_side, right_idx.size), replace=False))
    combined_idx = np.concatenate([left_idx, right_idx])
    embedding = embed_text(question, context, model=model)
    projection = gaussian_projection_matrix(
        projection_seed, n_targets=combined_idx.size, embedding_dim=embedding.size
    )
    rates = project_to_rates(embedding, projection, rate_hz_min, rate_hz_max)
    n_left = left_idx.size
    return EncodedInput(
        dense_idx=combined_idx,
        rate_hz=rates,
        left_idx=left_idx,
        right_idx=right_idx,
        left_rate_hz=rates[:n_left],
        right_rate_hz=rates[n_left:],
        side_counts=side_counts,
        embedding_dim=embedding.size,
    )
