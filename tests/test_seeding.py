from bioreservoir.oracle.seeding import reference_trial_seed, stable_seed, trial_seed


def test_stable_seed_deterministic_across_calls():
    assert stable_seed("a", "b", base=1) == stable_seed("a", "b", base=1)


def test_stable_seed_distinguishes_field_boundaries():
    # "a"+"b" concatenated naively would collide with "ab" split differently; the delimiter must
    # prevent that.
    assert stable_seed("a", "b", base=0) != stable_seed("ab", base=0)
    assert stable_seed("a", "b", base=0) != stable_seed("a", "b", base=1)


def test_stable_seed_is_non_negative_and_fits_default_rng():
    import numpy as np

    seed = stable_seed("2026-senate-ak", "malecns", "real", "original", base=100000)
    assert seed >= 0
    np.random.default_rng(seed)  # must not raise


def test_trial_seed_varies_by_trial_index():
    seeds = [trial_seed(0, "q1", "malecns", "real", "original", t) for t in range(5)]
    assert len(set(seeds)) == 5


def test_trial_seed_varies_by_condition_and_variant():
    base = {"base": 0, "question_id": "q1", "brain": "malecns", "trial_index": 0}
    s_real_original = trial_seed(base["base"], base["question_id"], base["brain"], "real", "original", base["trial_index"])
    s_rewired_original = trial_seed(base["base"], base["question_id"], base["brain"], "rewired", "original", base["trial_index"])
    s_real_negation = trial_seed(base["base"], base["question_id"], base["brain"], "real", "negation", base["trial_index"])
    assert len({s_real_original, s_rewired_original, s_real_negation}) == 3


def test_trial_seed_fits_brian2_legacy_seeding():
    import numpy as np

    for t in range(50):
        seed = trial_seed(100000, "2026-senate-ak", "malecns", "real", "original", t)
        assert 0 <= seed < 2**32
        np.random.seed(seed)  # what brian2.seed() calls; must not raise


def test_reference_trial_seed_deterministic_and_fits_brian2_legacy_seeding():
    import numpy as np

    a = reference_trial_seed(100000, "ref-01", "malecns", "real", 0)
    b = reference_trial_seed(100000, "ref-01", "malecns", "real", 0)
    assert a == b
    assert 0 <= a < 2**32
    np.random.seed(a)  # must not raise


def test_reference_trial_seed_varies_by_reference_id_brain_condition_and_trial():
    base = {"base": 0, "brain": "malecns", "condition": "real", "trial_index": 0}
    seeds = {
        reference_trial_seed(base["base"], "ref-01", base["brain"], base["condition"], base["trial_index"]),
        reference_trial_seed(base["base"], "ref-02", base["brain"], base["condition"], base["trial_index"]),
        reference_trial_seed(base["base"], "ref-01", "banc", base["condition"], base["trial_index"]),
        reference_trial_seed(base["base"], "ref-01", base["brain"], "rewired", base["trial_index"]),
        reference_trial_seed(base["base"], "ref-01", base["brain"], base["condition"], 1),
    }
    assert len(seeds) == 5


def test_reference_trial_seed_never_collides_with_a_question_trial_seed_of_the_same_string_id():
    # A reference sentence id that happened to equal a question id must still get a different
    # seed -- the "ref" literal in reference_trial_seed's own stable_seed key guards this even
    # though the two id namespaces ("ref-NN" vs "2026-...") do not overlap in practice.
    q_seed = trial_seed(100000, "shared-id", "malecns", "real", "original", 0)
    ref_seed = reference_trial_seed(100000, "shared-id", "malecns", "real", 0)
    assert q_seed != ref_seed
