#!/usr/bin/env python3
"""Renders the home page's link preview (`site/public/og.png`, 1200x630).

Every link to fly.igdigi.com posted anywhere — X, LinkedIn, Slack, iMessage — unfurls with this
image; without it the launch post is a bare URL (2026-09-19 review). Drawn from the SAME real
MaleCNS soma positions the page itself renders (`site/public/data/atlas/positions.f32`, produced by
scripts/export_atlas.py), with card.py's palette and fonts, so the preview is the actual data, not
a stock illustration.

usage: python scripts/export_og_image.py [--out site/public/og.png]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bioreservoir.live import config
from bioreservoir.live.card import (
    BG_COLOR,
    BORDER_COLOR,
    CYAN_COLOR,
    DIM_COLOR,
    FG_COLOR,
    MUTED_DOT_COLOR,
    _font,
)

HEADLINE = "Ask a real fly brain a yes/no question"
DECK = "165,000 mapped neurons, simulated live. Watch which way it turns."
DECK2 = "Not a chatbot, not a language model — a whole-brain simulation."
FOOT = "fly.igdigi.com  ·  IG Digital Lab  ·  Brain: Janelia MaleCNS v1.0 (CC BY 4.0)"


def brain_layer(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], positions_path: Path) -> None:
    """The soma cloud, drawn to fill `box`. A sixth of the neurons are drawn in cyan, chosen by a
    fixed stride — decoration, not a claim about which ones fired: no question was asked here."""
    positions = np.fromfile(positions_path, dtype="<f4")
    if positions.size == 0 or positions.size % 3 != 0:
        raise SystemExit(f"{positions_path} is not a float32 xyz array — re-run scripts/export_atlas.py")
    positions = positions.reshape(-1, 3)
    xs, ys = positions[:, 0], positions[:, 1]
    x0, y0, x1, y1 = box
    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    x_span = (x_max - x_min) or 1.0
    y_span = (y_max - y_min) or 1.0
    n = positions.shape[0]
    step = max(1, n // 9000)
    for i in range(0, n, step):
        px = x0 + (xs[i] - x_min) / x_span * (x1 - x0)
        py = y0 + (ys[i] - y_min) / y_span * (y1 - y0)
        color = CYAN_COLOR if (i // step) % 6 == 0 else MUTED_DOT_COLOR
        draw.ellipse((px - 1, py - 1, px + 1, py + 1), fill=color)


def wrapped(draw: ImageDraw.ImageDraw, text: str, font, width: int) -> list[str]:
    """Greedy word wrap to `width` pixels, so no line runs out of the card into the brain."""
    lines: list[str] = []
    for word in text.split():
        trial = f"{lines[-1]} {word}" if lines else word
        if lines and draw.textlength(trial, font=font) <= width:
            lines[-1] = trial
        else:
            lines.append(word)
    return lines


def render(out: Path) -> None:
    positions_path = config.ATLAS_DIR / "positions.f32"
    if not positions_path.exists():
        raise SystemExit(
            f"no atlas at {positions_path} — run `python3 scripts/export_atlas.py` first "
            "(it needs the downloaded MaleCNS data, see scripts/fetch_data.py)"
        )
    img = Image.new("RGB", (config.CARD_WIDTH, config.CARD_HEIGHT), BG_COLOR)
    draw = ImageDraw.Draw(img)

    # The brain fills the right half; the text sits on the left with room to breathe.
    brain_layer(draw, (620, 40, 1180, 590), positions_path)
    draw.rounded_rectangle((40, 40, 600, 590), radius=16, outline=BORDER_COLOR, width=2)

    draw.text((72, 96), "BIORESERVOIR", font=_font(20, bold=True), fill=DIM_COLOR)
    y = 150
    for line in ("Ask a real fly brain", "a yes/no question"):
        draw.text((72, y), line, font=_font(52, bold=True), fill=FG_COLOR)
        y += 64
    y += 24
    # The card's inner width: 72 px in from its left edge (40) to 32 px before its right edge (600).
    text_width = 600 - 32 - 72
    for paragraph in (DECK, DECK2):
        for line in wrapped(draw, paragraph, _font(24), text_width):
            draw.text((72, y), line, font=_font(24), fill=FG_COLOR)
            y += 34
        y += 10
    foot_lines = wrapped(draw, FOOT, _font(18), text_width)
    for i, line in enumerate(foot_lines):
        draw.text((72, 546 - 26 * (len(foot_lines) - i)), line, font=_font(18), fill=DIM_COLOR)

    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out, format="PNG", optimize=True)
    print(f"{out} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "site" / "public" / "og.png")
    render(parser.parse_args().out)
