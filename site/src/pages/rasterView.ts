// Renders Answer.lab.raster as a real spike raster on a <canvas>: one row per recorded neuron
// (labelled by its real cell_type), x-axis = time within the trial in ms (0..provenance.sim_ms,
// both real numbers from the API), one tick per decoded spike. Rows whose cell_type is also one
// of the answer's top_cell_types are drawn in the highlight color ("readout neurons").
import { h } from "../dom";
import { decodeRasterSpikes } from "../raster";
import type { AnswerLab } from "../api/types";

const BG = "#ffffff";
const ROW_COLOR = "rgba(12, 10, 9, 0.55)";
const HIGHLIGHT_COLOR = "#3ba6f1";
const ROW_HEIGHT = 6;
const PADDING = { top: 8, right: 12, bottom: 22, left: 8 };

export interface RasterView {
  el: HTMLElement;
  canvas: HTMLCanvasElement;
  destroy: () => void;
}

export function buildRasterView(lab: AnswerLab, opts: { width: number; interactive: boolean }): RasterView {
  const raster = lab.raster;
  const rowCount = raster.cell_types.length;
  const height = Math.max(80, rowCount * ROW_HEIGHT + PADDING.top + PADDING.bottom);
  const width = opts.width;

  const canvas = h("canvas", {
    width: String(width),
    height: String(height),
    class: "raster-canvas",
    role: "img",
    "aria-label": "Real spike raster for this answer",
  }) as HTMLCanvasElement;
  const ctx = canvas.getContext("2d");

  const highlighted = new Set(lab.top_cell_types.map((t) => t.cell_type));
  const simMs = lab.provenance.sim_ms;
  const plotW = width - PADDING.left - PADDING.right;
  const plotH = rowCount * ROW_HEIGHT;

  function draw() {
    if (!ctx) return;
    ctx.fillStyle = BG;
    ctx.fillRect(0, 0, width, height);
    if (rowCount === 0 || simMs <= 0) {
      ctx.fillStyle = "#78716c";
      ctx.font = "12px monospace";
      ctx.fillText("no raster data", PADDING.left, height / 2);
      return;
    }
    const spikes = decodeRasterSpikes(raster);
    for (const spike of spikes) {
      const cellType = raster.cell_types[spike.row];
      ctx.strokeStyle = cellType && highlighted.has(cellType) ? HIGHLIGHT_COLOR : ROW_COLOR;
      ctx.lineWidth = 1.5;
      const x = PADDING.left + (spike.timeMs / simMs) * plotW;
      const y = PADDING.top + spike.row * ROW_HEIGHT;
      ctx.beginPath();
      ctx.moveTo(x, y + 1);
      ctx.lineTo(x, y + ROW_HEIGHT - 1);
      ctx.stroke();
    }
    // Time axis, labelled from the real sim_ms (never a hardcoded "250ms" string).
    ctx.strokeStyle = "#e8e6e5";
    ctx.beginPath();
    ctx.moveTo(PADDING.left, PADDING.top + plotH + 4);
    ctx.lineTo(PADDING.left + plotW, PADDING.top + plotH + 4);
    ctx.stroke();
    ctx.fillStyle = "#78716c";
    ctx.font = "10px monospace";
    ctx.fillText("0 ms", PADDING.left, height - 6);
    ctx.textAlign = "right";
    ctx.fillText(`${simMs} ms`, PADDING.left + plotW, height - 6);
    ctx.textAlign = "left";
  }
  draw();

  let tooltip: HTMLElement | undefined;
  let onMove: ((event: MouseEvent) => void) | undefined;
  let onLeave: (() => void) | undefined;
  const wrap = h("div", { class: "raster-wrap" }, [canvas]);

  if (opts.interactive && rowCount > 0) {
    tooltip = h("div", { class: "raster-tooltip", role: "status" });
    wrap.appendChild(tooltip);
    onMove = (event: MouseEvent) => {
      const rect = canvas.getBoundingClientRect();
      const y = ((event.clientY - rect.top) / rect.height) * height;
      const row = Math.max(0, Math.min(rowCount - 1, Math.floor((y - PADDING.top) / ROW_HEIGHT)));
      const cellType = raster.cell_types[row] ?? "—";
      const atlasIdx = raster.atlas_indices[row];
      tooltip!.textContent = atlasIdx !== undefined && atlasIdx >= 0 ? `${cellType} (atlas #${atlasIdx})` : cellType;
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

  return {
    el: wrap,
    canvas,
    destroy: () => {
      if (onMove) canvas.removeEventListener("mousemove", onMove);
      if (onLeave) canvas.removeEventListener("mouseleave", onLeave);
    },
  };
}
