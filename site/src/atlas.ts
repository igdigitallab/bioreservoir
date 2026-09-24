// Loads the MaleCNS soma-position atlas exported by scripts/export_atlas.py (see that file's
// docstring for the exact binary layout). Fetched lazily — only when a page actually mounts the
// brain scene — so the initial page paint never waits on ~3MB of binary data.

export interface AtlasMeta {
  n: number;
  groups: string[];
  bbox: { min: [number, number, number]; max: [number, number, number] };
  bbox_brain: { min: [number, number, number]; max: [number, number, number] };
  center: [number, number, number];
  scale: number;
  /** Points with three.z below this are the VNC (anatomical z-gap split, not the superclass
   * label — see scripts/export_atlas.py's docstring, "Axis mapping"). */
  vnc_z_threshold: number;
  axis_mapping: string;
  source: string;
  license: string;
  attribution: string;
  files: Record<string, { bytes: number; sha256: string }>;
}

export interface Atlas {
  meta: AtlasMeta;
  /** N x 3 float32, already centered and uniformly scaled to roughly [-1, 1]. */
  positions: Float32Array;
  /** N uint8, index into meta.groups. */
  groups: Uint8Array;
  /** N uint64 body IDs, same order as positions/groups — the atlas index IS the array index. */
  neuronIds: BigUint64Array;
}

const ATLAS_BASE = "/data/atlas";

async function fetchBinary(path: string): Promise<ArrayBuffer> {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`atlas fetch failed: ${path} (${res.status})`);
  return res.arrayBuffer();
}

let cached: Promise<Atlas> | null = null;

/** Fetch and decode the atlas once; subsequent calls reuse the same in-flight/resolved promise. */
export function loadAtlas(): Promise<Atlas> {
  if (!cached) {
    cached = (async () => {
      const meta: AtlasMeta = await fetch(`${ATLAS_BASE}/meta.json`).then((r) => r.json());
      const [posBuf, groupBuf, idBuf] = await Promise.all([
        fetchBinary(`${ATLAS_BASE}/positions.f32`),
        fetchBinary(`${ATLAS_BASE}/groups.u8`),
        fetchBinary(`${ATLAS_BASE}/neuron_ids.u64`),
      ]);
      return {
        meta,
        positions: new Float32Array(posBuf),
        groups: new Uint8Array(groupBuf),
        neuronIds: new BigUint64Array(idBuf),
      };
    })();
  }
  return cached;
}
