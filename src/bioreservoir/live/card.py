"""Share card: `GET /api/card/{id}.png` (1200x630 PNG) and `GET /a/{id}` (OpenGraph/Twitter HTML
that points at it) — a shareable card designed as a viral loop, in the spirit of Akinator's own
share cards.

Colours follow the project's "warm paper" style reference: stone canvas background, ink
black / warm gray text, cyan as the one chromatic accent. No red/green — the answer word (YES/NO)
is not colour-coded, it is its own label.
"""

from __future__ import annotations

import html
import io
import re
import textwrap
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from bioreservoir.live import config
from bioreservoir.live.verdict import with_turn_strength

BG_COLOR = (250, 250, 249)  # Stone Canvas #fafaf9 — page/card background
FG_COLOR = (12, 10, 9)  # Ink Black #0c0a09 — primary text (headings, question, answer word)
DIM_COLOR = (120, 113, 108)  # Warm Gray #78716c — secondary/dim text
BORDER_COLOR = (232, 230, 229)  # Stone Border #e8e6e5 — hairline structure, never a heavy stroke
MUTED_DOT_COLOR = (214, 211, 209)  # Stone Muted #d6d3d1 — background soma scatter
CYAN_COLOR = (59, 166, 241)  # Cyan Signal #3ba6f1 — the only chromatic accent on this card

ATTRIBUTION = "IG Digital Lab · BioReservoir"
BRAIN_CREDIT = "Brain: Janelia MaleCNS v1.0 (CC BY 4.0)"
FRAMING_LINE = "A real fly brain answered:"

# Neutral, non-anthropomorphizing labels (2026-09-19: the old "looked
# hungry/startled/excited/wary/engaged" wording read as more claim than a spiking-population
# readout supports) -- same site-wide vocabulary the rewritten frontend uses for these states.
_STATE_LABELS = {
    "appetite": "Appetite",
    "fear": "Startle",
    "backoff": "Backed off",
    "courtship": "Courtship",
    "arousal": "Whole-brain buzz",
}

TURN_STRENGTH_CAVEAT = "It reacts to the pattern of the words, not their meaning."

# Rides on every card (2026-09-21, replacing the election embargo): the share preview is the one
# surface that travels without the page's framing around it, so the framing has to be ON it. Two
# lines, <=56 chars each -- that is what fits the right column (x=520..1140) at 18pt.
CARD_DISCLAIMER = (
    "Asked by a visitor, answered by a simulation.",
    "Entertainment and research — not advice.",
)

_FONT_CANDIDATES_REGULAR = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
)
_FONT_CANDIDATES_BOLD = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
)

# Fixed seed for the atlas-absent fallback brain silhouette (`_fallback_brain_dots`) — the
# scatter must be byte-identical across renders of the same box, not a fresh draw each time.
_FALLBACK_SEED = 20260918


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    """A real TTF if this host has DejaVu (most Debian/Ubuntu images do), else Pillow's bundled
    default bitmap font (always available — keeps this renderer working in a bare container with
    no system fonts installed, at the cost of a plainer look)."""
    for path in _FONT_CANDIDATES_BOLD if bold else _FONT_CANDIDATES_REGULAR:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


_ELECTION_RE = re.compile(
    r"\b(elect\w*|vot(e|es|ed|er|ers|ing)|ballot\w*|midterms?|polls?|primar(y|ies)|president\w*|"
    r"senat\w*|congress\w*|governor\w*|house (seat|race|majority)|democrat\w*|republican\w*|"
    r"gop|dnc|rnc|trump|harris|biden|vance|obama|desantis|newsom|pelosi|schumer|mcconnell)\b",
    re.IGNORECASE,
)


def is_election_question(question: str) -> bool:
    """Broad on purpose: a false positive only costs a less specific share preview, a miss puts a
    fly-brain "election call" into social feeds."""
    return bool(_ELECTION_RE.search(question))


def question_embargoed(question: str, now: datetime | None = None) -> bool:
    """The embargo test on the question text alone -- used by `election_embargo` below (a settled
    Answer already has `["question"]`) and by `api.py`'s `GET /api/now`'s `thinking` field, which
    only has the raw question text of a row the worker has not finished yet ("this
    applies to every PUBLIC surface")."""
    if not config.ELECTION_CARD_EMBARGO_ENABLED:
        return False
    now = now or datetime.now(UTC)
    until = datetime.fromisoformat(config.ELECTION_CARD_EMBARGO_UNTIL)
    return now < until and is_election_question(question)


def election_embargo(answer: dict, now: datetime | None = None) -> bool:
    # `.get`: a handful of legacy/minimal answer rows carry no question text of their own (the row
    # in `live_questions` has it) — an answer with no text cannot match the election pattern.
    return question_embargoed(answer.get("question", ""), now)


def dominant_state(states: dict[str, float | None]) -> tuple[str, float] | None:
    """The highest-valued non-null, non-`arousal` state (per-state captions, e.g.
    "Appetite", "Startle") — `None` if every state is currently unvalidated (all null)."""
    candidates = [(name, v) for name, v in states.items() if name != "arousal" and v is not None]
    if not candidates:
        return None
    return max(candidates, key=lambda pair: pair[1])


def _game_summary_line(answer: dict) -> str | None:
    """`Answer.game`'s three verdicts, in A/B/C slot order, which was real -- `None` if this
    answer has no game (old answers: "the game is simply absent"). Deliberately
    reveals which slot was real: unlike the live page's own suspense (the visitor guesses before
    seeing it), this card is generated for an already-settled answer being shared after the fact,
    so there is nothing left to spoil."""
    game = answer.get("game")
    if not game:
        return None
    order = game["order"]
    contenders = game["contenders"]
    parts = []
    for i, name in enumerate(order):
        slot = "ABC"[i]
        verdict = contenders[name]["answer"].upper()
        tag = " (real)" if name == "real" else ""
        parts.append(f"{slot}: {verdict}{tag}")
    return "   ".join(parts)


def _fallback_brain_dots(box: tuple[int, int, int, int]) -> list[tuple[float, float]]:
    """Deterministic (fixed-seed) stippled twin-lobe silhouette for when `positions.f32` (the
    atlas export, `scripts/export_atlas.py`) is not on disk yet — evokes two brain hemispheres side by
    side, in the spirit of the live site's real point-cloud brain hero, without claiming to be
    real neuron data. Same `box` size always yields the same points (rejection-sampled from a
    seeded RNG), so the rendered PNG is byte-identical run to run."""
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    cx, cy = x0 + w / 2, y0 + h / 2
    lobe_rx, lobe_ry = w * 0.29, h * 0.40
    gap = w * 0.06  # midline separation between the two lobes
    centers = ((cx - lobe_rx - gap / 2, cy), (cx + lobe_rx + gap / 2, cy))

    rng = np.random.default_rng(_FALLBACK_SEED)
    n_per_lobe = 700
    points: list[tuple[float, float]] = []
    for lcx, lcy in centers:
        collected = 0
        while collected < n_per_lobe:
            # Gaussian proposal rejection-sampled to the lobe ellipse — denser core, thinning
            # edge (closer to a real soma cloud than a uniform disc fill).
            dx = rng.normal(scale=lobe_rx * 0.42)
            dy = rng.normal(scale=lobe_ry * 0.42)
            if (dx / lobe_rx) ** 2 + (dy / lobe_ry) ** 2 > 1.0:
                continue
            points.append((lcx + dx, lcy + dy))
            collected += 1
    return points


def _brain_scatter(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], answer: dict) -> None:
    """A small scatter of atlas soma positions inside `box`, highlighting the answer's own
    active neurons (its `frames.active_b64` bins, unioned) in cyan — falls back to a baked,
    deterministic twin-lobe stipple if the atlas or this answer's frames aren't available
    (`atlas.py`'s own docstring: the atlas export may not exist yet)."""
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=10, outline=BORDER_COLOR, width=2)

    positions_path = config.ATLAS_DIR / "positions.f32"
    if not positions_path.exists():
        for px, py in _fallback_brain_dots(box):
            draw.ellipse((px - 1, py - 1, px + 1, py + 1), fill=MUTED_DOT_COLOR)
        return

    positions = np.fromfile(positions_path, dtype="<f4")
    if positions.size % 3 != 0 or positions.size == 0:
        for px, py in _fallback_brain_dots(box):
            draw.ellipse((px - 1, py - 1, px + 1, py + 1), fill=MUTED_DOT_COLOR)
        return
    positions = positions.reshape(-1, 3)
    xs, ys = positions[:, 0], positions[:, 1]
    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    x_span = (x_max - x_min) or 1.0
    y_span = (y_max - y_min) or 1.0

    active_idx: set[int] = set()
    frames = answer.get("frames")
    if frames:
        from bioreservoir.live.frames import decode_bin_b64

        for b64 in frames["active_b64"]:
            active_idx.update(decode_bin_b64(b64).tolist())

    # Subsample the background cloud for render speed (a PNG this size does not need every soma).
    n = positions.shape[0]
    step = max(1, n // 4000)
    for i in range(0, n, step):
        px = x0 + (xs[i] - x_min) / x_span * (x1 - x0)
        py = y0 + (ys[i] - y_min) / y_span * (y1 - y0)
        color = CYAN_COLOR if i in active_idx else MUTED_DOT_COLOR
        draw.ellipse((px - 1, py - 1, px + 1, py + 1), fill=color)


def render_card(answer: dict) -> bytes:
    """`answer` (the Answer dict, `pipeline.compute_answer`'s return value) -> PNG bytes, exactly
    `config.CARD_WIDTH` x `config.CARD_HEIGHT`."""
    img = Image.new("RGB", (config.CARD_WIDTH, config.CARD_HEIGHT), BG_COLOR)
    draw = ImageDraw.Draw(img)

    _brain_scatter(draw, (60, 60, 460, 460), answer)

    draw.text((520, 58), FRAMING_LINE, font=_font(22), fill=DIM_COLOR)

    if election_embargo(answer):
        lines = [
            "a question about the election.",
            "",
            "Its answer is on the site, not on",
            "this card: until the votes are",
            "counted, we keep a noisy fly",
            "brain's election calls out of",
            "social previews.",
        ]
        y = 98
        for i, line in enumerate(lines):
            draw.text((520, y), line, font=_font(40 if i == 0 else 30), fill=FG_COLOR if i == 0 else DIM_COLOR)
            y += 52 if i == 0 else 40
        draw.text((520, y + 24), "Ask your own: fly.igdigi.com", font=_font(30, bold=True), fill=FG_COLOR)
        draw.text((60, 560), ATTRIBUTION, font=_font(26, bold=True), fill=FG_COLOR)
        draw.text((60, 592), BRAIN_CREDIT, font=_font(20), fill=DIM_COLOR)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    answer = with_turn_strength(answer)  # old stored rows may predate this field

    question = answer["question"]
    wrapped = textwrap.wrap(question, width=34)[:4]
    y = 98
    for line in wrapped:
        draw.text((520, y), line, font=_font(40), fill=FG_COLOR)
        y += 52

    draw.text((520, y + 20), answer["answer"].upper(), font=_font(90, bold=True), fill=FG_COLOR)

    next_y = y + 130
    dom = dominant_state(answer.get("states", {}))
    if dom is not None:
        name, value = dom
        label = _STATE_LABELS.get(name, name)
        draw.text(
            (520, next_y), f"Readouts that fired: {label} ({round(value * 100)}%)",
            font=_font(24), fill=DIM_COLOR,
        )
        next_y += 34

    # Task brief: replace the 50-100% `confidence` scale (never shown on this card, but the
    # analogous "how sure was it" line) with a plain 0-100% fraction of a real, named biological
    # effect size, plus a caveat line so a screenshot never reads as more certain than it is.
    turn_pct = round(answer["turn_strength"] * 100)
    draw.text(
        (520, next_y), f"Turn strength: {turn_pct}% of a one-antenna sound cue",
        font=_font(24), fill=DIM_COLOR,
    )
    next_y += 32
    draw.text((520, next_y), TURN_STRENGTH_CAVEAT, font=_font(18), fill=DIM_COLOR)

    game_line = _game_summary_line(answer)
    if game_line is not None:
        draw.text((520, 494), "Which one is the real fly brain?", font=_font(18, bold=True), fill=DIM_COLOR)
        draw.text((520, 516), game_line, font=_font(22), fill=FG_COLOR)

    draw.text((60, 560), ATTRIBUTION, font=_font(26, bold=True), fill=FG_COLOR)
    draw.text((60, 592), BRAIN_CREDIT, font=_font(20), fill=DIM_COLOR)
    draw.text((520, 558), CARD_DISCLAIMER[0], font=_font(18), fill=DIM_COLOR)
    draw.text((520, 584), CARD_DISCLAIMER[1], font=_font(18), fill=DIM_COLOR)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def render_share_html(answer: dict, base_url: str) -> str:
    """Minimal OpenGraph/Twitter HTML for `GET /a/{id}` — crawlers (link-preview
    bots) read the meta tags and never execute JS/the meta-refresh; a real visitor's browser
    follows the meta-refresh (and the plain `<a>` fallback) straight to the live page."""
    id_ = answer["id"]
    question = html.escape(answer["question"])
    card_url = f"{base_url.rstrip('/')}/api/card/{id_}.png"
    # `page_url` is where a human (and the meta-refresh below) actually lands -- the home page
    # with `?a=` reopening this answer's reveal. `share_url` is the canonical `og:url` (task
    # brief: crawlers/dedup logic should key sharing this ANSWER on its own permalink, `/a/{id}`,
    # not on the home page's query-string form).
    page_url = f"{base_url.rstrip('/')}/?a={id_}"
    share_url = f"{base_url.rstrip('/')}/a/{id_}"
    if election_embargo(answer):
        title = html.escape("A real fly brain answered a question about the election")
        description = html.escape("See its answer on the site. Election calls stay out of share previews until the votes are counted.")
    else:
        title = html.escape(f'The fly says {answer["answer"].upper()}')
        # The disclaimer belongs in the description, not only on the page it links to: a link
        # preview is quoted, screenshotted and reposted on its own (2026-09-21, replacing the
        # election embargo -- label the preview instead of withholding the verdict).
        description = html.escape(
            f'"{answer["question"]}" — a visitor\'s question, answered by a real fly-brain '
            "simulation. Entertainment and research, not advice."
        )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<meta http-equiv="refresh" content="0; url={page_url}">
<meta property="og:title" content="{title}">
<meta property="og:description" content="{description}">
<meta property="og:image" content="{card_url}">
<meta property="og:url" content="{share_url}">
<meta property="og:type" content="website">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="{title}">
<meta name="twitter:description" content="{description}">
<meta name="twitter:image" content="{card_url}">
</head>
<body>
<p>{title}: &ldquo;{question}&rdquo;</p>
<p><a href="{page_url}">See the fly's answer</a></p>
</body>
</html>
"""
