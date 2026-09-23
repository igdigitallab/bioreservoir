// Orchestrates the brain point cloud into a renderer, with a camera that frames the SAME
// world-space extent regardless of the container's aspect ratio — this is what keeps home and
// /stream visually consistent (same "camera") and guarantees the brain is never clipped, whether
// the container is wide (hero) or narrow (stream's center-brain panel). Technique: derive the
// vertical FOV each resize from two independent "must show at least this much world-space
// extent" constraints (horizontal and vertical), take the larger requirement — the standard
// aspect-independent "contain" fit — instead of hardcoding one fov+distance pair that only looks
// right at one aspect ratio.
// https://threejs.org/docs/#api/en/cameras/PerspectiveCamera (fov is the *vertical* field of
// view in degrees; horizontal FOV is derived from fov+aspect, not set directly).
//
// The fly used to share this canvas and camera; per the 2026-09-18 review it reads as a red dot
// at this scale/angle, so it now lives in its own top-down 2D icon (flyIcon.ts) instead. The
// camera framing below was originally derived from the brain AND the (now-removed) fly's combined
// bounding box, which left the brain off-center inside its card — reserving room on the right for
// a fly that is no longer drawn here. Re-centred on the brain alone on 2026-09-19 after the UI
// was flagged as "crooked" — the first pass after the fly's removal kept the old off-center
// framing by oversight; this is the deliberate follow-up fix.
import * as THREE from "three";
import { loadAtlas } from "../atlas";
import { buildBrainCloud, type BrainCloud } from "./brain";
import { buildReplaySchedule, decodeFrames } from "../frames";
import type { Answer } from "../api/types";

export interface BrainScene {
  playAnswer: (answer: Answer) => void;
  dispose: () => void;
}

const CAMERA_DISTANCE = 3.1;
const SWAY_SPEED = 0.25; // rad/sec argument to sin() — a slow back-and-forth, not a spin.
const SWAY_MAX = 0.1; // radians (~6deg) either side of dead-frontal.

// Layout, in world units. The atlas normalizes the non-VNC brain to roughly fill [-1, 1] on X
// (its longest axis) with a Y half-extent of ~0.54x that (scripts/export_atlas.py's
// bbox_brain) — BRAIN_SCALE below is deliberately large so the brain reads as the dominant
// element (per the 2026-09-18 review: "brain should fill ~70% of canvas width").
const BRAIN_LOCAL_HALF_X = 1.0;
const BRAIN_LOCAL_HALF_Y = 0.54;
const BRAIN_SCALE = 1.7;
// Brain centered at world X=0 — with the fly gone, nothing else shares this canvas, so there is
// no reason to reserve space on either side.
const BRAIN_X = 0;

// "Contain" fit targets: half the world-space extent that must always be visible, with an 8%
// margin so nothing touches the container edge. Symmetric around the brain's own center now
// that it is the only thing in frame (previously asymmetric, biased right to leave room for the
// removed fly — see the module docstring).
const compositionCenterX = BRAIN_X;
// 1.2, not 1.08: the brain read as too bulky (2026-09-18); a little more air around it.
const FRAME_MARGIN = 1.2;
// Vertically the point cloud reaches past the 0.54 bbox estimate once perspective enlarges the
// nearer (anterior) points, and in a ~2:1 stage the height is the binding constraint — at 1.08
// the top of the brain was clipped by the card edge (2026-09-18 screenshots).
const FRAME_MARGIN_Y = 1.3;
const TARGET_HALF_WIDTH = BRAIN_SCALE * BRAIN_LOCAL_HALF_X * FRAME_MARGIN;
const TARGET_HALF_HEIGHT = BRAIN_SCALE * BRAIN_LOCAL_HALF_Y * FRAME_MARGIN_Y;

export function createBrainScene(
  container: HTMLElement,
  opts: { interactive?: boolean; showVnc?: boolean } = {},
): BrainScene {
  const interactive = opts.interactive ?? true;
  const showVnc = opts.showVnc ?? false;

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100);

  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  container.appendChild(renderer.domElement);

  let brainCloud: BrainCloud | undefined;

  function resize() {
    const w = container.clientWidth;
    const h = container.clientHeight;
    if (w === 0 || h === 0) return;
    renderer.setSize(w, h, false);
    const aspect = w / h;
    camera.aspect = aspect;
    const hHalf = Math.atan(TARGET_HALF_WIDTH / CAMERA_DISTANCE);
    const vHalfForWidth = Math.atan(Math.tan(hHalf) / aspect);
    const vHalfForHeight = Math.atan(TARGET_HALF_HEIGHT / CAMERA_DISTANCE);
    const vHalf = Math.max(vHalfForWidth, vHalfForHeight);
    camera.fov = THREE.MathUtils.clamp(THREE.MathUtils.radToDeg(2 * vHalf), 20, 100);
    camera.updateProjectionMatrix();
  }
  resize();
  const resizeObserver = new ResizeObserver(resize);
  resizeObserver.observe(container);

  loadAtlas()
    .then((atlas) => {
      brainCloud = buildBrainCloud(atlas, { showVnc });
      brainCloud.points.position.set(BRAIN_X, 0, 0);
      brainCloud.points.scale.setScalar(BRAIN_SCALE);
      scene.add(brainCloud.points);
      // Drops the CSS loading placeholder (style.css `.hero-canvas`): the canvas is transparent,
      // so the placeholder blobs would otherwise stay visible behind the sparse point cloud.
      container.classList.add("is-loaded");
    })
    .catch((err) => {
      console.error("atlas load failed", err);
    });

  let raf = 0;
  // Draw only while the canvas is on screen and the tab is visible: a WebGL point cloud redrawn
  // at 60 fps off screen costs a phone its battery and a CPU-only machine a whole core (a tab left
  // open in a background browser burned ~660% CPU for 9 h, measured 2026-09-19). The replay itself runs
  // on timers (brain.ts), so pausing the draw loop loses nothing but the frames nobody sees.
  let onScreen = true;
  function running(): boolean {
    return onScreen && !document.hidden;
  }
  function wake() {
    if (running() && raf === 0) raf = requestAnimationFrame(frame);
  }
  const visibility = new IntersectionObserver((entries) => {
    onScreen = entries.some((e) => e.isIntersecting);
    wake();
  });
  visibility.observe(container);
  document.addEventListener("visibilitychange", wake);

  function frame(now: number) {
    raf = 0;
    if (!running()) return;
    const elapsed = now / 1000;

    const sway = interactive ? Math.sin(elapsed * SWAY_SPEED) * SWAY_MAX : 0;
    camera.position.x = compositionCenterX + Math.sin(sway) * CAMERA_DISTANCE;
    camera.position.y = 0.05;
    camera.position.z = Math.cos(sway) * CAMERA_DISTANCE;
    camera.lookAt(compositionCenterX, 0, 0);

    renderer.render(scene, camera);
    raf = requestAnimationFrame(frame);
  }
  wake();

  function playAnswer(answer: Answer) {
    const bins = decodeFrames(answer.frames);
    const schedule = buildReplaySchedule(bins);
    brainCloud?.playReplay(schedule);
  }

  function dispose() {
    cancelAnimationFrame(raf);
    onScreen = false;
    visibility.disconnect();
    document.removeEventListener("visibilitychange", wake);
    resizeObserver.disconnect();
    brainCloud?.dispose();
    renderer.dispose();
    renderer.domElement.remove();
  }

  return { playAnswer, dispose };
}
