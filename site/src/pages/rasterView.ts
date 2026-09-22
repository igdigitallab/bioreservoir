// Renders Answer.lab.raster as a real spike raster on a <canvas>: one row per neuron that FIRED
// (labelled by its real cell_type), x-axis = time within the trial in ms (0..provenance.sim_ms,
// both real numbers from the API), one tick per decoded spike. Rows whose cell_type is also one
// of the answer's top_cell_types are drawn in the highlight color ("readout neurons").
//
// Two things the first version got wrong, both visible on the live page: it drew a row for every
// recorded neuron, and in a 250 ms trial roughly three quarters of them never fire — so the
// canvas was ~1800px of mostly white. And it drew at a fixed pixel width that CSS then scaled to
// the panel, which smeared every 1px tick. Now the plot holds only the neurons that fired
// (raster.ts's buildRasterLayout, which also reports what it left out), and the canvas is drawn
// at the container's real size times devicePixelRatio, so a tick is a tick.
import { h } from "../dom";
import { buildRasterLayout, decodeRasterSpikes, describeRaster } from "../raster";
import type { AnswerLab } from "../api/types";

const BG = "#ffffff";
const ROW_COLOR = "rgba(12, 10, 9, 0.62)";
const HIGHLIGHT_COLOR = "#3ba6f1";
const AXIS_COLOR = "#e8e6e5";
const LABEL_COLOR = "#78716c";
const PADDING = { top: 8, right: 12, bottom: 20, left: 8 };
/** Plot height budget. Narrow viewports get the shorter one: on a phone the readout sits in a
 * scrolling column where a tall block reads as dead space. */
const MAX_PLOT_HEIGHT = 240;
const COMPACT_PLOT_HEIGHT = 150;
const COMPACT_WIDTH = 520;

export interface RasterView {
  el: HTMLElement;
  canvas: HTMLCanvasElement;
  destroy: () => void;
}

/** Nice round gridline spacing for a trial of `simMs` — 25 ms for short trials, 50 ms for the
 * 250 ms default, 100 ms for anything long. Never more than ~6 lines. */
function gridStepMs(simMs: number): number {
  if (simMs <= 120) return 25;
  if (simMs <= 600) return 50;
  return 100;
}

export function buildRasterView(lab: AnswerLab, opts: { width: number; interactive: boolean }): RasterView {
  const raster = lab.raster;
  const spikes = decodeRasterSpikes(raster);
  const highlighted = new Set(lab.top_cell_types.map((t) => t.cell_type));
  const simMs = lab.provenance.sim_ms;

  const canvas = h("canvas", {
    class: "raster-canvas",
    role: "img",
    "aria-label": "Real spike raster for this answer",
  }) as HTMLCanvasElement;
  const ctx = canvas.getContext("2d");
  const plot = h("div", { class: "raster-wrap" }, [canvas]);
  const caption = h("p", { class: "helper-text raster-caption" });
  const el = h("div", { class: "raster-view" }, [plot, caption]);

  // Layout of the last draw, so the tooltip maps a y position back to the right neuron.
  let layout = buildRasterLayout(raster, spikes, { maxPlotHeight: MAX_PLOT_HEIGHT });

  function draw() {
    if (!ctx) return;
    const cssWidth = plot.clientWidth || opts.width;
    const maxPlotHeight = cssWidth < COMPACT_WIDTH ? COMPACT_PLOT_HEIGHT : MAX_PLOT_HEIGHT;
    layout = buildRasterLayout(raster, spikes, { maxPlotHeight });
    caption.textContent = describeRaster(layout);

    const plotH = layout.rows.length * layout.rowHeight;
    const cssHeight = plotH + PADDING.top + PADDING.bottom;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.round(cssWidth * dpr));
    canvas.height = Math.max(1, Math.round(cssHeight * dpr));
    canvas.style.width = "100%";
    canvas.style.height = `${cssHeight}px`;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    ctx.fillStyle = BG;
    ctx.fillRect(0, 0, cssWidth, cssHeight);

    const plotW = cssWidth - PADDING.left - PADDING.right;
    if (layout.rows.length === 0 || simMs <= 0 || plotW <= 0) {
      ctx.fillStyle = LABEL_COLOR;
      ctx.font = "12px monospace";
      ctx.fillText(simMs <= 0 ? "no raster data" : "no spikes in this trial", PADDING.left, cssHeight / 2);
      return;
    }

    // Time gridlines first, so spikes sit on top of them.
    ctx.strokeStyle = AXIS_COLOR;
    ctx.lineWidth = 1;
    const step = gridStepMs(simMs);
    for (let t = step; t < simMs; t += step) {
      const x = Math.round(PADDING.left + (t / simMs) * plotW) + 0.5;
      ctx.beginPath();
      ctx.moveTo(x, PADDING.top);
      ctx.lineTo(x, PADDING.top + plotH);
      ctx.stroke();
    }

    const displayRowOf = new Map<number, number>();
    layout.rows.forEach((r, i) => displayRowOf.set(r.row, i));
    const tickHeight = Math.max(1, layout.rowHeight - (layout.rowHeight > 3 ? 1 : 0));
    for (const spike of spikes) {
      const i = displayRowOf.get(spike.row);
      if (i === undefined) continue; // a quietest-row casualty, counted in the caption
      const cellType = raster.cell_types[spike.row];
      ctx.fillStyle = cellType && highlighted.has(cellType) ? HIGHLIGHT_COLOR : ROW_COLOR;
      const x = PADDING.left + (spike.timeMs / simMs) * plotW;
      ctx.fillRect(x - 0.6, PADDING.top + i * layout.rowHeight, 1.4, tickHeight);
    }

    // Time axis, labelled from the real sim_ms (never a hardcoded "250ms" string).
    ctx.strokeStyle = AXIS_COLOR;
    ctx.beginPath();
    ctx.moveTo(PADDING.left, PADDING.top + plotH + 3.5);
    ctx.lineTo(PADDING.left + plotW, PADDING.top + plotH + 3.5);
    ctx.stroke();
    ctx.fillStyle = LABEL_COLOR;
    ctx.font = "10px monospace";
    ctx.fillText("0 ms", PADDING.left, cssHeight - 5);
    ctx.textAlign = "right";
    ctx.fillText(`${simMs} ms`, PADDING.left + plotW, cssHeight - 5);
    ctx.textAlign = "left";
  }
  draw();

  let tooltip: HTMLElement | undefined;
  let onMove: ((event: MouseEvent) => void) | undefined;
  let onLeave: (() => void) | undefined;

  if (opts.interactive) {
    tooltip = h("div", { class: "raster-tooltip", role: "status" });
    plot.appendChild(tooltip);
    onMove = (event: MouseEvent) => {
      if (layout.rows.length === 0) return;
      const rect = canvas.getBoundingClientRect();
      const y = event.clientY - rect.top;
      const i = Math.max(0, Math.min(layout.rows.length - 1, Math.floor((y - PADDING.top) / layout.rowHeight)));
      const entry = layout.rows[i];
      if (!entry) return;
      const cellType = raster.cell_types[entry.row] ?? "—";
      const atlasIdx = raster.atlas_indices[entry.row];
      const spikeWord = entry.count === 1 ? "spike" : "spikes";
      const where = atlasIdx !== undefined && atlasIdx >= 0 ? ` (atlas #${atlasIdx})` : "";
      tooltip!.textContent = `${cellType}${where} · ${entry.count} ${spikeWord}`;
      tooltip!.style.left = `${event.clientX - rect.left + 8}px`;
      tooltip!.style.top = `${event.clientY - rect.top + 8}px`;
      tooltip!.style.display = "block";
    };
    onLeave = () => {
      tooltip!.style.display = "none";
    };
    canvas.addEventListener("mousemove", onMove);
    canvas.addEventListener("mouseleave", onLeave);
  }

  // The canvas is sized from its container, so it has to be redrawn when the container changes —
  // including the first time it gets a real width, since at construction it is not in the DOM yet.
  let observer: ResizeObserver | undefined;
  let lastWidth = plot.clientWidth;
  if (typeof ResizeObserver !== "undefined") {
    observer = new ResizeObserver(() => {
      const width = plot.clientWidth;
      if (width > 0 && width !== lastWidth) {
        lastWidth = width;
        draw();
      }
    });
    observer.observe(plot);
  }

  return {
    el,
    canvas,
    destroy: () => {
      if (onMove) canvas.removeEventListener("mousemove", onMove);
      if (onLeave) canvas.removeEventListener("mouseleave", onLeave);
      observer?.disconnect();
    },
  };
}
