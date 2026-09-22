// The original top-down canvas icon remains available when WebGL or the model cannot load.
// Procedural Canvas 2D drawing only, no image assets, no third-party model.
// Every animation is driven only by non-null behavioural readouts (see states.ts) — this module
// has no notion of "emotion" beyond the named drives it is handed.
//
// Style (design-style-reference.md "Mascot sticker"): outline-only line art, 1.5px stroke ink
// black, eyes a solid ink fill; cyan is reserved for the selected DOM answer label. The sticker drop-shadow is CSS
// (`.fly-panel canvas`, style.css), not drawn here.
import type { AnimationTrigger, FlySide, TurnTrigger } from "./states";
import { ANSWER_HOLD_SECONDS, answerTurnAt, answerYaw, deliberateYaw, envelope, IDLE_SEED, triggerDrive } from "./flyPose";
import { createAnswerLabels } from "./flyAnswer";

interface Drive {
  /** -1 = turned fully left, +1 = turned fully right, 0 = facing forward. */
  turn: number;
  appetite: number;
  fear: number;
  backoff: number;
  courtship: number;
  arousal: number;
}

const ZERO_DRIVE: Drive = { turn: 0, appetite: 0, fear: 0, backoff: 0, courtship: 0, arousal: 0 };

const COLOR_INK = "#0c0a09";
const COLOR_WING_FILL = "rgba(12, 10, 9, 0.05)";
const STROKE_PX = 1.5;

export interface FlyIcon {
  el: HTMLElement;
  /** Start playing a new set of animations, synced to last ~durationMs (matches the brain replay). */
  play: (triggers: AnimationTrigger[], durationMs: number) => void;
  /** This question is still in the simulator: both labels up, the fly looking around. */
  deliberate: (yesSide: FlySide) => void;
  release: () => void;
  dispose: () => void;
}

/** Half-width of an axis-aligned ellipse (center cy, radii rx/ry) at a given y — used to draw
 * stripe lines that stay inside the abdomen's own outline instead of a fixed guessed width. */
function ellipseHalfWidthAt(y: number, cy: number, rx: number, ry: number): number {
  const t = Math.max(-1, Math.min(1, (y - cy) / ry));
  return rx * Math.sqrt(1 - t * t);
}

/** Mounts a self-contained, self-animating top-down fly icon into `container`. Resizes with the
 * container (ResizeObserver), same pattern as the brain's WebGL canvas. */
export function createFlyIcon(container: HTMLElement, replay?: {
  triggers: AnimationTrigger[]; startMs: number; durationMs: number;
}, hold = ANSWER_HOLD_SECONDS): FlyIcon {
  let holdSeconds = hold;
  const canvas = document.createElement("canvas");
  canvas.setAttribute("role", "img");
  canvas.setAttribute("aria-label", "Top-down illustration of the fly, showing its current behavioural state");
  container.appendChild(canvas);
  const labels = createAnswerLabels(container);
  const ctx = canvas.getContext("2d");
  const motion = matchMedia("(prefers-reduced-motion: reduce)");
  let visible = false;
  let disposed = false;

  let cssSize = 0;

  function resize() {
    const w = container.clientWidth;
    const h = container.clientHeight;
    const size = Math.min(w, h);
    if (size === 0) return;
    cssSize = size;
    const dpr = Math.min(window.devicePixelRatio, 2);
    canvas.width = Math.round(size * dpr);
    canvas.height = Math.round(size * dpr);
    canvas.style.width = `${size}px`;
    canvas.style.height = `${size}px`;
    wake();
  }
  const resizeObserver = new ResizeObserver(resize);
  resizeObserver.observe(container);

  let target: Drive = { ...ZERO_DRIVE };
  let startMs = 0;
  let durationMs = 0;
  let playing = false;
  let turn: TurnTrigger | undefined;
  let fromYaw = 0;
  let answerPose = answerTurnAt(undefined, 0, 0);
  let pendingSide: FlySide | undefined;
  /** Same three rules as the 3D icon: the labels belong to the question being simulated (never to
   * an answer still easing off screen), they come up at once, and the wander waits for a previous
   * answer to finish easing out. */
  let labelsPending = false;
  let labelStartMs = 0;
  let wanderStartMs = 0;

  /** Decorative look-around, but only when no answer owns the heading. */
  function wanderYaw(): number {
    if (playing || pendingSide === undefined) return 0;
    return deliberateYaw((performance.now() - wanderStartMs) / 1000, IDLE_SEED, motion.matches);
  }

  function play(triggers: AnimationTrigger[], newDurationMs: number) {
    holdSeconds = hold;
    labelsPending = false;
    const now = performance.now();
    // Start the turn from wherever the fly points now — an interrupted turn or a wander look.
    const interrupted = answerTurnAt(turn, (now - startMs) / 1000, durationMs / 1000, motion.matches, fromYaw, holdSeconds);
    fromYaw = interrupted.active ? interrupted.yaw : wanderYaw();
    turn = triggers.find((trigger): trigger is TurnTrigger => trigger.kind === "turn");
    target = triggerDrive(triggers);
    startMs = now;
    durationMs = Number.isFinite(newDurationMs) ? Math.max(0, newDurationMs) : 0;
    playing = true;
    wake();
  }

  function currentDrive(): Drive {
    if (!playing) return ZERO_DRIVE;
    const now = performance.now();
    const progress = durationMs > 0 ? (now - startMs) / durationMs : 1;
    answerPose = answerTurnAt(turn, (now - startMs) / 1000, durationMs / 1000, motion.matches, fromYaw, holdSeconds);
    playing = progress < 1 || answerPose.active;
    if (progress >= 1) {
      return ZERO_DRIVE;
    }
    const e = motion.matches ? 1 : envelope(progress);
    return {
      turn: target.turn * e,
      appetite: target.appetite * e,
      fear: target.fear * e,
      backoff: target.backoff * e,
      courtship: target.courtship * e,
      arousal: target.arousal * e,
    };
  }

  // --- drawing helpers, all in a local unit space of roughly [-1, 1] with forward = -Y ---

  function drawLeg(c: CanvasRenderingContext2D, x0: number, y0: number, x1: number, y1: number, jitter: number) {
    const jx = (Math.random() - 0.5) * jitter;
    const jy = (Math.random() - 0.5) * jitter;
    c.beginPath();
    c.moveTo(x0, y0);
    c.quadraticCurveTo((x0 + x1) / 2 + jx, (y0 + y1) / 2 + jy, x1 + jx, y1 + jy);
    c.stroke();
  }

  function drawWing(c: CanvasRenderingContext2D, side: -1 | 1, spreadRad: number, extend: number) {
    c.save();
    c.translate(side * 0.06, -0.04);
    c.rotate(side * spreadRad);
    c.beginPath();
    c.ellipse(0, 0.32 * extend, 0.15, 0.34 * extend, 0, 0, Math.PI * 2);
    c.fillStyle = COLOR_WING_FILL;
    c.fill();
    c.stroke();
    c.restore();
  }

  function draw(elapsedSeconds: number) {
    if (!ctx || cssSize === 0) return;
    const dpr = Math.min(window.devicePixelRatio, 2);
    ctx.save();
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, cssSize, cssSize);

    const drive = currentDrive();
    const deliberating = pendingSide !== undefined;
    if (deliberating && answerPose.active) wanderStartMs = performance.now();
    const answering = turn !== undefined && answerPose.active && !labelsPending;
    const pendingOpacity = !deliberating ? 0
      : motion.matches ? 1 : Math.min(1, (performance.now() - labelStartMs) / 300);
    labels.render(answering ? turn : deliberating ? { yesSide: pendingSide! } : undefined,
      answering ? Math.max(answerPose.opacity, pendingOpacity) : pendingOpacity, cssSize, side => {
        const angle = -answerYaw(side);
        return { x: .5 + Math.sin(angle) * .43, y: .5 - Math.cos(angle) * .43 };
      }, answering ? answerPose.chosen : 0);
    if (answering && answerPose.chosen >= 1) pendingSide = undefined;
    const scale = (cssSize / 2) * 0.82;
    // Physical CSS-pixel stroke width, converted into the local unit space established by the
    // `ctx.scale(scale, scale)` below — kept constant on screen regardless of canvas size.
    const strokeLocal = STROKE_PX / scale;

    ctx.translate(cssSize / 2, cssSize / 2);

    // Idle breathing, always present (independent of `playing`).
    const breathe = 1 + (playing || motion.matches ? 0 : Math.sin(elapsedSeconds * 1.4) * 0.025);

    // Turn: rotate the whole fly toward the physical side that produced the answer. While the
    // question is still being simulated, a much smaller decorative wander instead (flyPose).
    const turnAngle = -(answerPose.yaw + wanderYaw());
    ctx.rotate(turnAngle);

    // Backoff: slide backward along the fly's own -forward axis (screen-space, post-rotation is
    // fine since backing off is small and mostly reads as "moved down").
    ctx.translate(0, drive.backoff * 0.22 * scale);

    // Fear: a brief startle scale-pulse standing in for a jump (this is a flat icon, no ground
    // plane to jump above).
    const fearPulse = 1 + drive.fear * 0.18 * Math.abs(Math.sin(elapsedSeconds * 14));
    ctx.scale(scale * fearPulse, scale * fearPulse);
    ctx.scale(breathe, breathe);

    ctx.lineCap = "round";
    ctx.strokeStyle = COLOR_INK;
    ctx.lineWidth = strokeLocal;

    // Arousal jitter amplitude (local units).
    const jitter = motion.matches ? 0 : drive.arousal * 0.05;

    // Legs: 3 pairs from the thorax, front angled forward, middle out to the side, hind angled
    // backward — [attachY, dx, dy] is the leg-tip offset from its attachment point, mirrored for
    // the left/right side.
    const legSpecs: Array<[number, number, number]> = [
      [-0.2, 0.42, -0.26],
      [-0.05, 0.46, 0.02],
      [0.14, 0.4, 0.32],
    ];
    for (const [attachY, dx, dy] of legSpecs) {
      for (const side of [-1, 1] as const) {
        drawLeg(ctx, side * 0.22, attachY, side * (0.22 + dx), attachY + dy, jitter);
      }
    }

    // Abdomen: one outlined shape plus 3 stripe lines, each spanning exactly the abdomen's own
    // width at that height (computed from the ellipse itself, not a guessed constant).
    const abdomenCenterY = 0.48;
    const abdomenRx = 0.22;
    const abdomenRy = 0.4;
    ctx.beginPath();
    ctx.ellipse(0, abdomenCenterY, abdomenRx, abdomenRy, 0, 0, Math.PI * 2);
    ctx.stroke();
    for (const t of [-0.45, 0, 0.45]) {
      const y = abdomenCenterY + t * abdomenRy;
      const halfW = ellipseHalfWidthAt(y, abdomenCenterY, abdomenRx, abdomenRy);
      ctx.beginPath();
      ctx.moveTo(-halfW, y);
      ctx.lineTo(halfW, y);
      ctx.stroke();
    }

    // Wings: spread in a resting V, flare wider under fear, one extends+vibrates for courtship.
    const restSpread = Math.PI / 5.2;
    const flare = restSpread + drive.fear * 0.35;
    drawWing(ctx, -1, flare, 1);
    const courtshipVibe = drive.courtship > 0 ? Math.sin(elapsedSeconds * 45) * 0.12 * drive.courtship : 0;
    drawWing(ctx, 1, flare + drive.courtship * 0.3 + courtshipVibe, 1 + drive.courtship * 0.25);

    // Thorax.
    ctx.beginPath();
    ctx.ellipse(0, -0.1, 0.27, 0.3, 0, 0, Math.PI * 2);
    ctx.stroke();

    // Head.
    ctx.beginPath();
    ctx.ellipse(0, -0.55, 0.16, 0.15, 0, 0, Math.PI * 2);
    ctx.stroke();

    // Antennae, jittering slightly with arousal.
    for (const side of [-1, 1] as const) {
      const ax = side * (0.08 + jitter * 0.4);
      ctx.beginPath();
      ctx.moveTo(side * 0.06, -0.66);
      ctx.lineTo(ax * 1.6, -0.82 + jitter * 0.3);
      ctx.stroke();
    }

    // Compound eyes — large, dominating the head sides, the one solid-fill feature (spec:
    // "eyes can stay a solid ink fill").
    ctx.fillStyle = COLOR_INK;
    for (const side of [-1, 1] as const) {
      ctx.beginPath();
      ctx.ellipse(side * 0.16, -0.56, 0.12, 0.125, 0, 0, Math.PI * 2);
      ctx.fill();
    }

    // Proboscis: extends forward only when appetite is non-null and driving.
    if (drive.appetite > 0.001) {
      const len = 0.06 + drive.appetite * 0.32;
      ctx.beginPath();
      ctx.moveTo(0, -0.68);
      ctx.lineTo(0, -0.68 - len);
      ctx.stroke();
    }

    ctx.restore();
  }

  let raf = 0;
  const startTime = performance.now();
  function frame() {
    raf = 0;
    if (disposed || !visible || document.hidden) return;
    draw(motion.matches ? 0 : (performance.now() - startTime) / 1000);
    if (!motion.matches || playing) raf = requestAnimationFrame(frame);
  }
  function wake() {
    cancelAnimationFrame(raf);
    if (!disposed && visible && !document.hidden) raf = requestAnimationFrame(frame);
  }
  const intersection = new IntersectionObserver(entries => {
    visible = entries.some(entry => entry.isIntersecting);
    wake();
  });
  intersection.observe(container);
  document.addEventListener("visibilitychange", wake);
  motion.addEventListener("change", wake);
  resize();

  function dispose() {
    disposed = true;
    cancelAnimationFrame(raf);
    resizeObserver.disconnect();
    intersection.disconnect();
    document.removeEventListener("visibilitychange", wake);
    motion.removeEventListener("change", wake);
    labels.dispose();
    canvas.remove();
  }

  if (replay) {
    play(replay.triggers, replay.durationMs);
    startMs = replay.startMs;
  }
  function deliberate(yesSide: FlySide) {
    if (disposed || pendingSide === yesSide) return;
    pendingSide = yesSide;
    labelsPending = true;
    labelStartMs = performance.now();
    wanderStartMs = labelStartMs;
    wake();
  }

  function release() {
    pendingSide = undefined;
    labelsPending = false;
    if (!turn) return wake();
    holdSeconds = Math.max(0, (performance.now() - startMs - durationMs) / 1000);
    wake();
  }

  return { el: canvas, play, deliberate, release, dispose };
}
