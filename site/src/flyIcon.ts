// Each specimen owns its renderer: sharing the brain camera loses the anatomy at panel size.
// The renderer and model are loaded only when this panel enters the viewport.
import type { AnimationTrigger, FlySide, TurnTrigger } from "./states";
import { createFlyIcon as createFlyIcon2d } from "./flyIcon2d";
import { answerTurnAt, ANSWER_HOLD_SECONDS, ANSWER_RELEASE_SECONDS, canSwitchFlurryPeriod, createFlyPose, DELIBERATE_FLURRY_PERIOD, deliberateYaw, driveAt, FLURRY_PERIOD, IDLE_SEED, poseAt, triggerDrive, ZERO_DRIVE } from "./flyPose";
import { createAnswerLabels } from "./flyAnswer";
import type { FlyScene } from "./flyScene";

export interface FlyIcon {
  el: HTMLElement;
  /** Replay behaviours for durationMs; hold the answer heading and label for six more seconds. */
  play: (triggers: AnimationTrigger[], durationMs: number) => void;
  /** This question is in the simulator now: put both answer labels on their real sides and let
   * the fly look around while it waits. Decoration only — see flyPose's `deliberateYaw`. */
  deliberate: (yesSide: FlySide) => void;
  /** Ease a held answer back to neutral now (the live stage moving on to the next question). */
  release: () => void;
  dispose: () => void;
}

export interface FlyIconOptions {
  /** Keep the answer heading and labels for good instead of releasing them after six seconds:
   * the reveal's caption ("It turned toward YES.") and verdict stay on screen, so the fly must too. */
  holdAnswer?: boolean;
}

export function createFlyIcon(container: HTMLElement, options: FlyIconOptions = {}): FlyIcon {
  let holdSeconds = options.holdAnswer ? Infinity : ANSWER_HOLD_SECONDS;
  const el = document.createElement("div");
  el.style.cssText = "width:100%;height:100%;position:relative";
  container.appendChild(el);
  const labels = createAnswerLabels(el);
  const motion = matchMedia("(prefers-reduced-motion: reduce)");
  const modelAbort = new AbortController();
  let scene: FlyScene | undefined;
  let fallback: FlyIcon | undefined;
  let disposed = false;
  let visible = false;
  let loading = false;
  let raf = 0;
  let target = { ...ZERO_DRIVE };
  let triggers: AnimationTrigger[] = [];
  let startMs = 0;
  let durationMs = 0;
  let turn: TurnTrigger | undefined;
  let fromYaw = 0;
  let size = 0;
  const pose = createFlyPose();
  const drive = { ...ZERO_DRIVE };
  const idle = { seconds: 0, playing: false, recovery: 1, seed: IDLE_SEED, period: FLURRY_PERIOD };
  let idleStartMs = performance.now();
  /** Set while the question on screen is still being simulated: labels up, fly looking around. */
  let pendingSide: FlySide | undefined;
  /** Two clocks, deliberately: the labels come up the moment the question does, while the wander
   * waits for a previous answer to finish easing out (otherwise the two fight over the heading). */
  let labelStartMs = 0;
  let wanderStartMs = 0;
  /** The labels belong to the question being simulated, not to the answer still easing off
   * screen: without this, the previous answer's lit label survives under the new question's
   * "deciding" caption for the length of its release. */
  let labelsPending = false;

  /** Where the fly is pointing this instant, whether that comes from an answer or from the
   * deliberation wander — `play()` starts its turn here so the heading never jumps. */
  function yawAt(now: number): number {
    const answer = answerTurnAt(turn, (now - startMs) / 1000, durationMs / 1000, motion.matches, fromYaw, holdSeconds);
    if (answer.active) return answer.yaw;
    return pendingSide === undefined ? 0 : deliberateYaw((now - wanderStartMs) / 1000, idle.seed, motion.matches);
  }

  function draw(now: number) {
    raf = 0;
    if (disposed || !visible || document.hidden || !scene) return;
    const progress = durationMs > 0 ? (now - startMs) / durationMs : 1;
    const seconds = (now - startMs) / 1000;
    const answer = answerTurnAt(turn, seconds, durationMs / 1000, motion.matches, fromYaw, holdSeconds);
    idle.playing = progress < 1 || answer.active;
    idle.seconds = Math.max(0, (now - idleStartMs) / 1000);
    idle.recovery = (now - idleStartMs) / 1000;
    // A fly that is being asked something takes off more often: it should look busy, not frozen.
    // Switching cadence mid-hop would teleport it, so the change waits for a moment when both
    // cadences agree the fly is on the ground.
    const wantPeriod = pendingSide === undefined ? FLURRY_PERIOD : DELIBERATE_FLURRY_PERIOD;
    if (wantPeriod !== idle.period
      && canSwitchFlurryPeriod(idle.seconds, idle.seed, motion.matches, idle.playing, idle.period, wantPeriod)) {
      idle.period = wantPeriod;
    }
    // Reduced motion uses static behaviour poses as well as a static answer heading.
    poseAt(motion.matches && progress < 1 ? target : driveAt(target, progress, drive), seconds, motion.matches, idle, pose);
    const deliberating = pendingSide !== undefined;
    // The previous answer is still easing out: hold the wander at zero so the heading has one
    // owner at a time. Only this clock waits — the labels are already up (see labelStartMs).
    if (deliberating && answer.active) wanderStartMs = now;
    pose.yaw = answer.active
      ? answer.yaw
      : deliberating ? deliberateYaw((now - wanderStartMs) / 1000, idle.seed, motion.matches) : 0;
    scene.render(pose);
    // The labels are already up when the turn begins, so the hand-off must not dip their opacity;
    // once the fly has committed, the answer owns them alone (and may fade out with it).
    const pendingOpacity = !deliberating ? 0
      : motion.matches ? 1 : Math.min(1, (now - labelStartMs) / 300);
    const answering = turn !== undefined && answer.active && !labelsPending;
    labels.render(
      answering ? turn : deliberating ? { yesSide: pendingSide! } : undefined,
      answering ? Math.max(answer.opacity, pendingOpacity) : pendingOpacity,
      size, scene.projectAnswer, answering ? answer.chosen : 0,
    );
    if (answering && answer.chosen >= 1) pendingSide = undefined;
    // A held answer is a still pose once the turn and label have settled: stop redrawing it
    // (wake() still repaints one frame on resize or when it scrolls back into view). Never while
    // deliberating: the previous answer is still fading on the frame the next question arrives,
    // so without this the loop would stop exactly there and the fly would sit frozen under the
    // "deciding" caption, waiting for an unrelated resize to wake it.
    const settled = options.holdAnswer && holdSeconds === Infinity && turn !== undefined
      && progress >= 1 && answer.chosen >= 1 && pendingSide === undefined;
    if ((!motion.matches || idle.playing) && !settled) raf = requestAnimationFrame(draw);
  }

  function wake() {
    if (disposed) return;
    cancelAnimationFrame(raf);
    raf = 0;
    if (visible && !document.hidden) {
      if (!scene && !fallback && !loading) void load();
      if (scene) raf = requestAnimationFrame(draw);
    }
  }

  function useFallback() {
    if (disposed || fallback) return;
    labels.render(undefined, 0, size, () => ({ x: 0, y: 0 }));
    modelAbort.abort();
    scene?.dispose();
    scene = undefined;
    fallback = createFlyIcon2d(el, { triggers, startMs, durationMs }, holdSeconds);
    if (pendingSide !== undefined) fallback.deliberate(pendingSide);
  }

  async function load() {
    loading = true;
    try {
      const { createFlyScene } = await import("./flyScene");
      if (disposed) return;
      const loaded = await createFlyScene(el, useFallback, modelAbort.signal);
      if (disposed || fallback) loaded.dispose();
      else { scene = loaded; resize(); wake(); }
    } catch {
      // A failed model request or unavailable WebGL context still leaves a useful icon.
      useFallback();
    }
  }

  function resize() {
    size = Math.min(container.clientWidth, container.clientHeight);
    if (size > 0) scene?.resize(size);
    wake();
  }
  const resizeObserver = new ResizeObserver(resize);
  resizeObserver.observe(container);
  const intersection = new IntersectionObserver(entries => {
    visible = entries.some(entry => entry.isIntersecting);
    wake();
  });
  intersection.observe(container);
  document.addEventListener("visibilitychange", wake);
  motion.addEventListener("change", wake);

  function deliberate(yesSide: FlySide) {
    if (disposed || pendingSide === yesSide) return;
    pendingSide = yesSide;
    labelsPending = true;
    labelStartMs = performance.now();
    wanderStartMs = labelStartMs;
    fallback?.deliberate(yesSide);
    wake();
  }

  function play(next: AnimationTrigger[], ms: number) {
    if (disposed) return;
    holdSeconds = options.holdAnswer ? Infinity : ANSWER_HOLD_SECONDS;
    const now = performance.now();
    // Start from wherever the fly is pointing now — an interrupted turn or a deliberation look —
    // even if its last frame was offscreen.
    fromYaw = yawAt(now);
    triggers = next.map(trigger => ({ ...trigger }));
    turn = triggers.find((trigger): trigger is TurnTrigger => trigger.kind === "turn");
    target = triggerDrive(triggers);
    startMs = now;
    durationMs = Number.isFinite(ms) ? Math.max(0, ms) : 0;
    // Restart the decorative schedule after playback, so no partial hover resumes.
    idleStartMs = startMs + durationMs + (turn ? (holdSeconds + ANSWER_RELEASE_SECONDS) * 1000 : 0);
    // A different question's labels go at once; the ones already up for THIS question stay,
    // so the options do not blink as the fly starts to turn.
    labelsPending = false;
    if (pendingSide !== undefined && pendingSide !== turn?.yesSide) pendingSide = undefined;
    if (pendingSide === undefined) labels.render(undefined, 0, size, () => ({ x: 0, y: 0 }));
    fallback?.play(triggers, durationMs);
    wake();
  }

  function release() {
    if (disposed) return;
    pendingSide = undefined;
    labelsPending = false;
    fallback?.release();
    if (!turn) {
      labels.render(undefined, 0, size, () => ({ x: 0, y: 0 }));
      wake();
      return;
    }
    const now = performance.now();
    // Start the usual release (heading and labels ease out) from this moment, then idle again.
    holdSeconds = Math.max(0, (now - startMs - durationMs) / 1000);
    idleStartMs = startMs + durationMs + (holdSeconds + ANSWER_RELEASE_SECONDS) * 1000;
    wake();
  }

  function dispose() {
    if (disposed) return;
    disposed = true;
    modelAbort.abort();
    cancelAnimationFrame(raf);
    resizeObserver.disconnect();
    intersection.disconnect();
    document.removeEventListener("visibilitychange", wake);
    motion.removeEventListener("change", wake);
    scene?.dispose();
    fallback?.dispose();
    labels.dispose();
    el.remove();
  }
  return { el, play, deliberate, release, dispose };
}
