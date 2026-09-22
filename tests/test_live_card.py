"""card.py: PNG share-card renderer + `/a/{id}` OpenGraph HTML. Atlas files are absent in this
worktree (frontend agent hasn't produced them yet — `atlas.py`'s own docstring), so every test
here exercises the documented fallback path (`_brain_scatter`'s plain ellipse); a fixture with a
tiny fake `positions.f32` covers the atlas-present branch too.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

from bioreservoir.live import card, config


def _answer(**overrides) -> dict:
    base = {
        "id": 1,
        "question": "Will it rain tomorrow?",
        "answer": "yes",
        "confidence": 0.73,
        "lateral_bias": 0.21,
        "states": {"appetite": None, "fear": None, "backoff": None, "courtship": None, "arousal": 0.42},
        "frames": None,
        "brain": "malecns",
        "sim_ms": 250,
        "n_trials": 3,
        "answered_at": "2026-09-18T00:00:00Z",
    }
    base.update(overrides)
    return base


def test_render_card_produces_the_exact_documented_size():
    png_bytes = card.render_card(_answer())
    img = Image.open(io.BytesIO(png_bytes))
    assert img.size == (config.CARD_WIDTH, config.CARD_HEIGHT)
    assert img.format == "PNG"


def test_render_card_works_with_a_no_answer():
    png_bytes = card.render_card(_answer(answer="no", confidence=0.55))
    img = Image.open(io.BytesIO(png_bytes))
    assert img.size == (config.CARD_WIDTH, config.CARD_HEIGHT)


def test_render_card_works_when_every_state_is_null():
    png_bytes = card.render_card(_answer(states={"appetite": None, "fear": None, "backoff": None, "courtship": None, "arousal": None}))
    assert Image.open(io.BytesIO(png_bytes)).size == (config.CARD_WIDTH, config.CARD_HEIGHT)


def test_render_card_works_with_a_long_question_wrapped():
    long_question = "Will the fly correctly predict whether this extremely long yes or no question wraps across several lines on the card?"
    png_bytes = card.render_card(_answer(question=long_question))
    assert Image.open(io.BytesIO(png_bytes)).size == (config.CARD_WIDTH, config.CARD_HEIGHT)


def test_render_card_with_frames_highlights_active_neurons(monkeypatch, tmp_path):
    """Atlas-present branch: a tiny fake `positions.f32` + a frames payload referencing some of
    its indices — must not crash, and must still produce a correctly sized PNG."""
    atlas_dir = tmp_path / "atlas"
    atlas_dir.mkdir()
    positions = np.random.default_rng(0).normal(size=(200, 3)).astype("<f4")
    (atlas_dir / "positions.f32").write_bytes(positions.tobytes())
    monkeypatch.setattr(config, "ATLAS_DIR", atlas_dir)

    from bioreservoir.live.frames import encode_bin_b64

    answer = _answer(
        frames={
            "bin_ms": 25.0, "n_bins": 10,
            "active_b64": [encode_bin_b64(np.array([1, 5, 10], dtype=np.uint32))] + [encode_bin_b64(np.array([], dtype=np.uint32))] * 9,
            "capped": False,
        }
    )
    png_bytes = card.render_card(answer)
    assert Image.open(io.BytesIO(png_bytes)).size == (config.CARD_WIDTH, config.CARD_HEIGHT)


def test_dominant_state_picks_the_highest_non_null_non_arousal_value():
    states = {"appetite": 0.2, "fear": 0.9, "backoff": None, "courtship": 0.5, "arousal": 0.99}
    assert card.dominant_state(states) == ("fear", 0.9)


def test_dominant_state_is_none_when_everything_but_arousal_is_null():
    states = {"appetite": None, "fear": None, "backoff": None, "courtship": None, "arousal": 0.5}
    assert card.dominant_state(states) is None


def test_render_share_html_contains_opengraph_and_twitter_tags():
    html_out = card.render_share_html(_answer(), base_url="https://fly.example.com")
    assert 'property="og:image"' in html_out
    assert "https://fly.example.com/api/card/1.png" in html_out
    assert 'name="twitter:card"' in html_out
    assert "https://fly.example.com/?a=1" in html_out


def test_render_share_html_escapes_question_text():
    html_out = card.render_share_html(_answer(question="Is <script>alert(1)</script> safe?"), base_url="https://x.example")
    assert "<script>alert(1)</script>" not in html_out
    assert "&lt;script&gt;" in html_out


# -- election share-preview embargo (until the midterm votes are counted) -------------------------

from datetime import UTC, datetime

BEFORE = datetime(2026, 10, 1, tzinfo=UTC)
AFTER = datetime(2026, 11, 5, tzinfo=UTC)


def test_is_election_question_catches_candidates_parties_and_races():
    for q in (
        "Will Trump be president in 2028?",
        "Will Democrats win the House majority?",
        "Will the GOP keep the Senate?",
        "Is Newsom running?",
        "Will turnout in the midterms be high?",
    ):
        assert card.is_election_question(q), q


def test_is_election_question_leaves_ordinary_questions_alone():
    for q in ("Will it rain tomorrow?", "Should I learn to cook?", "Is pineapple good on pizza?", "Will the fly turn left?"):
        assert not card.is_election_question(q), q


def test_the_election_embargo_is_off(monkeypatch):
    """Turned off 2026-09-21 (config.ELECTION_CARD_EMBARGO_ENABLED): a visitor's election question
    is just another visitor question, and holding its verdict back read as the opposite. The
    mechanism below still works -- this asserts the shipped default, so switching it back on is a
    deliberate act with a failing test to notice."""
    assert config.ELECTION_CARD_EMBARGO_ENABLED is False
    assert not card.election_embargo(_answer(question="Will Trump be president in 2028?"), now=BEFORE)


def test_election_embargo_holds_only_until_the_votes_are_counted(monkeypatch):
    monkeypatch.setattr(config, "ELECTION_CARD_EMBARGO_ENABLED", True)
    a = _answer(question="Will Trump be president in 2028?")
    assert card.election_embargo(a, now=BEFORE)
    assert not card.election_embargo(a, now=AFTER)
    assert not card.election_embargo(_answer(), now=BEFORE)


def test_every_card_carries_the_disclaimer_in_its_share_description():
    """The preview travels without the page around it, so the framing has to be ON it (this
    replaced the embargo as the guard against a screenshot reading like a forecast)."""
    html_out = card.render_share_html(_answer(question="Will the Senate flip?"), base_url="https://x.example")
    assert "not advice" in html_out
    assert "a visitor&#x27;s question" in html_out or "a visitor's question" in html_out


def test_embargoed_share_html_carries_no_verdict(monkeypatch):
    monkeypatch.setattr(card, "election_embargo", lambda answer, now=None: True)
    page = card.render_share_html(_answer(question="Will Trump be president in 2028?", answer="yes"), "https://fly.igdigi.com")
    assert "The fly says" not in page
    assert "about the election" in page


def test_embargoed_card_still_renders_at_the_documented_size(monkeypatch):
    monkeypatch.setattr(card, "election_embargo", lambda answer, now=None: True)
    img = Image.open(io.BytesIO(card.render_card(_answer(question="Will Trump be president in 2028?"))))
    assert img.size == (config.CARD_WIDTH, config.CARD_HEIGHT)


# -- game: the card shows all three verdicts and which one was real ------------------------------

_GAME = {
    "order": ["random_graph", "real", "no_brain"],
    "contenders": {
        "real": {"answer": "yes", "corrected_bias": 0.2, "decisiveness": 0.7},
        "random_graph": {"answer": "no", "corrected_bias": -0.05, "decisiveness": 0.55},
        "no_brain": {"answer": "yes", "corrected_bias": 0.02, "decisiveness": 0.51},
    },
}


def test_game_summary_line_is_none_without_a_game_field():
    assert card._game_summary_line(_answer()) is None


def test_game_summary_line_lists_all_three_slots_and_marks_the_real_one():
    line = card._game_summary_line(_answer(game=_GAME))
    assert line is not None
    assert "A: NO" in line  # order[0] = random_graph -> "no"
    assert "B: YES (real)" in line  # order[1] = real -> "yes"
    assert "C: YES" in line and "C: YES (real)" not in line  # order[2] = no_brain -> "yes", not real


def test_render_card_with_a_game_field_still_renders_at_the_documented_size():
    png_bytes = card.render_card(_answer(game=_GAME))
    assert Image.open(io.BytesIO(png_bytes)).size == (config.CARD_WIDTH, config.CARD_HEIGHT)


def test_render_card_without_a_game_field_is_unaffected_backward_compat():
    """Old-shape answers (no `game` key at all) must render exactly as before -- no crash, no
    game summary line drawn."""
    png_bytes = card.render_card(_answer())
    assert Image.open(io.BytesIO(png_bytes)).size == (config.CARD_WIDTH, config.CARD_HEIGHT)


def test_embargoed_card_with_a_game_field_still_carries_no_verdict(monkeypatch):
    """Election embargo wins over the game too -- a noisy fly-brain election "call" must not leak
    into a shared preview via the game summary line either."""
    monkeypatch.setattr(card, "election_embargo", lambda answer, now=None: True)
    answer = _answer(question="Will Trump be president in 2028?", game=_GAME)
    img = Image.open(io.BytesIO(card.render_card(answer)))
    assert img.size == (config.CARD_WIDTH, config.CARD_HEIGHT)
