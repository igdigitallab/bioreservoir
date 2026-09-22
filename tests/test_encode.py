"""Encoder tests against the real, already-downloaded local MiniLM model and a small *real*
population (`gustatory_sugar` on BANC, ~86 neurons/side after balancing — see docs/DATA.md) rather
than a synthetic one: `populations.py`'s rules are written against each dataset's own annotation
vocabulary, so there is no dataset-agnostic way to fabricate a fake population for this layer.
Loading the model and a harmonized graph's cached parquet is fast (~1-2 s, no Brian2, no cage —
see /tmp/cardloop-scratch progress notes) and does not touch simulation at all.
"""

import numpy as np
import pytest

from bioreservoir.oracle import encode

DATASET = "banc"
POPULATION = "gustatory_sugar"


@pytest.fixture(scope="module")
def graph():
    from bioreservoir.sim import bench

    _n_neurons, _pre_idx, _post_idx, _weight, id_to_dense = bench.graph_to_arrays(DATASET, min_syn=5)
    return id_to_dense


@pytest.fixture(scope="module")
def model():
    return encode.load_encoder()


def test_embed_text_is_deterministic(model):
    e1 = encode.embed_text("Will X win?", "Some context.", model=model)
    e2 = encode.embed_text("Will X win?", "Some context.", model=model)
    assert np.array_equal(e1, e2)
    assert e1.shape == (384,)  # all-MiniLM-L6-v2's native embedding size


def test_embed_text_differs_for_different_text(model):
    e1 = encode.embed_text("Will X win?", "Some context.", model=model)
    e2 = encode.embed_text("Will X lose?", "Some context.", model=model)
    assert not np.array_equal(e1, e2)


def test_gaussian_projection_matrix_deterministic_shape_and_seed():
    m1 = encode.gaussian_projection_matrix(seed=42, n_targets=10, embedding_dim=384)
    m2 = encode.gaussian_projection_matrix(seed=42, n_targets=10, embedding_dim=384)
    m3 = encode.gaussian_projection_matrix(seed=43, n_targets=10, embedding_dim=384)
    assert m1.shape == (10, 384)
    assert np.array_equal(m1, m2)
    assert not np.array_equal(m1, m3)


def test_project_to_rates_stays_within_range():
    rng = np.random.default_rng(0)
    embedding = rng.normal(size=384)
    projection = encode.gaussian_projection_matrix(seed=1, n_targets=50, embedding_dim=384)
    rates = encode.project_to_rates(embedding, projection, rate_hz_min=50.0, rate_hz_max=300.0)
    assert rates.shape == (50,)
    assert rates.min() >= 50.0 - 1e-9
    assert rates.max() <= 300.0 + 1e-9
    assert rates.std() > 0  # not degenerate (would be constant if z-scoring were broken)


def test_encode_question_to_input_is_deterministic(graph, model):
    kwargs = {
        "question": "Will the Republican Party hold the Alaska U.S. Senate seat?",
        "context": "A neutral context sentence.",
        "dataset": DATASET,
        "id_to_dense": graph,
        "input_population": POPULATION,
        "projection_seed": 20260918,
        "balance_seed": 20260918,
        "rate_hz_min": 50.0,
        "rate_hz_max": 300.0,
        "model": model,
    }
    e1 = encode.encode_question_to_input(**kwargs)
    e2 = encode.encode_question_to_input(**kwargs)
    assert np.array_equal(e1.dense_idx, e2.dense_idx)
    assert np.array_equal(e1.rate_hz, e2.rate_hz)


def test_encode_question_to_input_balances_left_and_right_counts(graph, model):
    encoded = encode.encode_question_to_input(
        question="Will the Republican Party hold the Alaska U.S. Senate seat?",
        context="A neutral context sentence.",
        dataset=DATASET,
        id_to_dense=graph,
        input_population=POPULATION,
        projection_seed=20260918,
        balance_seed=20260918,
        rate_hz_min=50.0,
        rate_hz_max=300.0,
        model=model,
    )
    assert encoded.left_idx.size == encoded.right_idx.size
    assert encoded.left_idx.size == encoded.side_counts["used_per_side"]
    assert encoded.dense_idx.size == encoded.left_idx.size + encoded.right_idx.size
    assert encoded.rate_hz.size == encoded.dense_idx.size
    assert np.array_equal(encoded.rate_hz[: encoded.left_idx.size], encoded.left_rate_hz)
    assert np.array_equal(encoded.rate_hz[encoded.left_idx.size :], encoded.right_rate_hz)


def test_encode_question_to_input_differs_between_original_and_negation(graph, model):
    common = {
        "context": "A neutral context sentence.",
        "dataset": DATASET,
        "id_to_dense": graph,
        "input_population": POPULATION,
        "projection_seed": 20260918,
        "balance_seed": 20260918,
        "rate_hz_min": 50.0,
        "rate_hz_max": 300.0,
        "model": model,
    }
    original = encode.encode_question_to_input(question="Will the Republican Party hold the Alaska U.S. Senate seat?", **common)
    negation = encode.encode_question_to_input(question="Will the Republican Party lose the Alaska U.S. Senate seat?", **common)
    assert not np.array_equal(original.rate_hz, negation.rate_hz)
    # left/right index sets themselves are a property of the population + balance seed, not the
    # text, so they must stay identical across variants (only the drive on them changes).
    assert np.array_equal(original.left_idx, negation.left_idx)
    assert np.array_equal(original.right_idx, negation.right_idx)
