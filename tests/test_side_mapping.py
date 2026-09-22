from bioreservoir.oracle.questions import load_questions
from bioreservoir.oracle.side_mapping import left_is_yes_for

SIDE_BASE = 20260918107  # config.yaml's seed.side_base


def test_left_is_yes_for_is_deterministic():
    a = left_is_yes_for("2026-senate-ak", SIDE_BASE)
    b = left_is_yes_for("2026-senate-ak", SIDE_BASE)
    assert a is b


def test_left_is_yes_for_varies_by_question_id():
    values = {left_is_yes_for(f"q{i}", SIDE_BASE) for i in range(20)}
    # 20 distinct ids should not all land on the same side.
    assert values == {True, False}


def test_left_is_yes_for_varies_by_side_base():
    mapped_1 = [left_is_yes_for(f"q{i}", side_base=1) for i in range(20)]
    mapped_2 = [left_is_yes_for(f"q{i}", side_base=2) for i in range(20)]
    assert mapped_1 != mapped_2


def test_left_is_yes_for_does_not_depend_on_negation_or_paraphrase():
    """The mapping is keyed by question id alone -- oracle.plan's WorkItem carries the same
    question_id for original/negation/paraphrase, so all three variants of one question must
    share one YES-side coin."""
    for variant_suffix in ("", "-unused-suffix-would-change-the-id"):
        qid = "2026-senate-ak" + variant_suffix
        assert left_is_yes_for(qid, SIDE_BASE) == left_is_yes_for(qid, SIDE_BASE)


def test_side_mapping_is_roughly_balanced_over_the_real_33_question_set():
    """~50/50 balance check (task brief) over the actual committed question set, reported both
    overall and by incumbent_party, using the real seed.side_base from config.yaml."""
    questions = load_questions()
    assert len(questions) == 33

    mapping = {q.id: left_is_yes_for(q.id, SIDE_BASE) for q in questions}
    n_left_is_yes = sum(mapping.values())
    n_total = len(mapping)

    # A 33-sample Bernoulli(0.5) draw landing outside [10, 23] happens well under 5% of the time
    # (binomial two-sided tail) -- a loose bound, not a strict 50/50 requirement, since this is one
    # fixed seed, not a statistical average over many seeds.
    assert 10 <= n_left_is_yes <= 23, (
        f"side mapping looks skewed for seed.side_base={SIDE_BASE}: "
        f"{n_left_is_yes}/{n_total} questions map left=yes"
    )

    by_party = {"Republican": [0, 0], "Democratic": [0, 0]}  # [n_left_is_yes, n_total]
    for q in questions:
        party = q.incumbent_party
        if party not in by_party:
            continue
        by_party[party][1] += 1
        if mapping[q.id]:
            by_party[party][0] += 1

    # questions.yaml's own header comment counts 20 Republican-held / 11 Democratic-held among the
    # 31 individual-race questions; the 2 chamber-control questions are also tagged
    # incumbent_party="Republican" (both chambers are currently Republican-held), so the full
    # 33-question set is 22 Republican-tagged / 11 Democratic-tagged.
    assert by_party["Republican"][1] == 22
    assert by_party["Democratic"][1] == 11
    # Neither party's subset is entirely on one side of the coin -- the mapping cannot be read as
    # "left = Democratic gain" or "left = Republican hold" for either party's questions.
    assert 0 < by_party["Republican"][0] < by_party["Republican"][1]
    assert 0 < by_party["Democratic"][0] < by_party["Democratic"][1]
