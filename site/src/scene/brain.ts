// The brain point cloud: ~140k real MaleCNS soma positions rendered as THREE.Points
// (BufferGeometry + Float32Array position/color attributes — the standard large-point-cloud
// pattern, https://threejs.org/docs/#api/en/objects/Points). Monochrome on white per
// design-style-reference.md: resting points are stone/ink (#a8a29e -> #0c0a09) by local
// soma-position density (a real, computed property of the real data, not a decorative gradient);
// real spikes are the only color, cyan #3ba6f1.
import * as THREE from "three";
import type { Atlas } from "../atlas";
import type { ReplayStep } from "../frames";

// design-style-reference.md colors.
const ASH_GRAY = new THREE.Color(0xa8a29e);
const INK_BLACK = new THREE.Color(0x0c0a09);
const FLASH_COLOR = new THREE.Color(0x3ba6f1);

export interface BrainCloud {
  points: THREE.Points;
  /** Flash a set of atlas indices cyan for one replay bin; call from an rAF-driven loop. */
  playReplay: (schedule: ReplayStep[], onDone?: () => void) => void;
  dispose: () => void;
}

export interface BrainCloudOptions {
  /** false hides the VNC (points with three.z < atlas.meta.vnc_z_threshold) so the hero/stream
   * view reads instantly as a fly brain (the "butterfly" of the two optic lobes) instead of
   * having the VNC merge into a blob underneath it. Default true (full brain + VNC). */
  showVnc?: boolean;
}

const DENSITY_GRID = 22; // cells per axis — coarse enough to read as regional structure, not noise.

/** Local soma-position density per included point, one real 3D histogram pass over the actual
 * positions (not a synthetic/decorative value) — used only to choose a shade of stone/ink. */
function computeDensityColors(positions: Float32Array, n: number): Float32Array {
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  for (let i = 0; i < n; i++) {
    const x = positions[i * 3]!, y = positions[i * 3 + 1]!, z = positions[i * 3 + 2]!;
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
    if (z < minZ) minZ = z;
    if (z > maxZ) maxZ = z;
  }
  const spanX = Math.max(1e-6, maxX - minX);
  const spanY = Math.max(1e-6, maxY - minY);
  const spanZ = Math.max(1e-6, maxZ - minZ);

  const cellOf = (i: number) => {
    const gx = Math.min(DENSITY_GRID - 1, Math.floor(((positions[i * 3]! - minX) / spanX) * DENSITY_GRID));
    const gy = Math.min(DENSITY_GRID - 1, Math.floor(((positions[i * 3 + 1]! - minY) / spanY) * DENSITY_GRID));
    const gz = Math.min(DENSITY_GRID - 1, Math.floor(((positions[i * 3 + 2]! - minZ) / spanZ) * DENSITY_GRID));
    return (gx * DENSITY_GRID + gy) * DENSITY_GRID + gz;
  };

  const counts = new Uint32Array(DENSITY_GRID * DENSITY_GRID * DENSITY_GRID);
  for (let i = 0; i < n; i++) counts[cellOf(i)]!++;

  // log1p compresses the long tail of a few very dense cells so most of the cloud still spans
  // the visible gray range instead of clipping to near-black.
  let maxLog = 0;
  for (let c = 0; c < counts.length; c++) {
    const v = Math.log1p(counts[c]!);
    if (v > maxLog) maxLog = v;
  }
  maxLog = Math.max(maxLog, 1e-6);

  const colors = new Float32Array(n * 3);
  const mixed = new THREE.Color();
  for (let i = 0; i < n; i++) {
    const density = Math.log1p(counts[cellOf(i)]!) / maxLog; // 0..1
    mixed.copy(ASH_GRAY).lerp(INK_BLACK, density);
    colors[i * 3] = mixed.r;
    colors[i * 3 + 1] = mixed.g;
    colors[i * 3 + 2] = mixed.b;
  }
  return colors;
}

export function buildBrainCloud(atlas: Atlas, opts: BrainCloudOptions = {}): BrainCloud {
  const showVnc = opts.showVnc ?? true;
  const totalN = atlas.meta.n;

  // Build the included-row list up front so we can both compact the GPU buffers (skip work for
  // hidden points, not just hide them) and remap original atlas indices -> local buffer rows for
  // flashBin (spike indices from the API always refer to the full, un-filtered atlas).
  const included: number[] = [];
  for (let i = 0; i < totalN; i++) {
    if (!showVnc && atlas.positions[i * 3 + 2]! < atlas.meta.vnc_z_threshold) continue;
    included.push(i);
  }
  const n = included.length;
  const origToLocal = new Int32Array(totalN).fill(-1);

  const positions = new Float32Array(n * 3);
  included.forEach((origIdx, local) => {
    origToLocal[origIdx] = local;
    positions[local * 3] = atlas.positions[origIdx * 3]!;
    positions[local * 3 + 1] = atlas.positions[origIdx * 3 + 1]!;
    positions[local * 3 + 2] = atlas.positions[origIdx * 3 + 2]!;
  });

  const baseColors = computeDensityColors(positions, n);

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  const colorAttr = new THREE.BufferAttribute(baseColors.slice(), 3);
  geometry.setAttribute("color", colorAttr);

  const material = new THREE.PointsMaterial({
    size: 0.011,
    vertexColors: true,
    transparent: true,
    opacity: 0.9,
    sizeAttenuation: true,
  });
  const points = new THREE.Points(geometry, material);
  points.name = "brain-cloud";

  let activeTimeouts: number[] = [];

  function resetColors() {
    const arr = colorAttr.array as Float32Array;
    arr.set(baseColors);
    colorAttr.needsUpdate = true;
  }

  function flashBin(indices: Uint32Array) {
    const arr = colorAttr.array as Float32Array;
    for (const origIdx of indices) {
      const local = origIdx < totalN ? origToLocal[origIdx]! : -1;
      if (local < 0) continue; // out of range, or a VNC neuron hidden in this view.
      arr[local * 3] = FLASH_COLOR.r;
      arr[local * 3 + 1] = FLASH_COLOR.g;
      arr[local * 3 + 2] = FLASH_COLOR.b;
    }
    colorAttr.needsUpdate = true;
  }

  function playReplay(schedule: ReplayStep[], onDone?: () => void) {
    activeTimeouts.forEach((t) => clearTimeout(t));
    activeTimeouts = [];
    resetColors();
    for (const step of schedule) {
      activeTimeouts.push(
        window.setTimeout(() => {
          resetColors();
          flashBin(step.indices);
        }, step.atMs),
      );
    }
    const totalMs = schedule.length > 0 ? schedule[schedule.length - 1]!.atMs + 250 : 0;
    activeTimeouts.push(
      window.setTimeout(() => {
        resetColors();
        onDone?.();
      }, totalMs),
    );
  }

  function dispose() {
    activeTimeouts.forEach((t) => clearTimeout(t));
    geometry.dispose();
    material.dispose();
  }

  return { points, playReplay, dispose };
}
