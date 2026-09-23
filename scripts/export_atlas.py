"""Export the MaleCNS soma-position atlas the live-fly site's brain point cloud renders.

Reads data/raw/malecns-v1.0/body-annotations-male-cns-v1.0-minconf-0.5.feather (fetched by
fetch_data.py, see docs/DATA.md), keeps the fully proofread neurons with a known soma position
(`status == "Traced"` and non-null `somaLocation`; ~140,024 of 211,577 rows as of the v1.0
release fetched 2026-09-18), and writes three flat binary files plus a JSON manifest to
site/public/data/atlas/ for the Three.js point cloud to fetch directly (no parsing library
needed in the browser):

    positions.f32    N x 3 little-endian float32, already in Three.js display space (see "Axis
                      mapping" below), scaled by one uniform factor (no per-axis stretching) so
                      the central-brain + optic-lobe cluster (i.e. everything except the VNC)
                      fits in roughly [-1, 1]; the VNC extends further along -Z when included.
    groups.u8         N uint8, index into meta.json["groups"] (one entry per distinct
                      `superclass` value, sorted; a null superclass maps to the literal string
                      "unknown", appended after the sort so the index list itself stays sorted
                      for every other entry).
    neuron_ids.u64    N little-endian uint64 `bodyId`, same row order as the two files above —
                      the live backend maps its per-neuron spike output to these IDs.
    meta.json         {n, groups[], bbox (raw, full dataset), bbox_brain (raw, VNC excluded —
                       the basis for center/scale), center, scale, axis_mapping, source,
                       license, attribution, files: {name: {bytes, sha256}}}.

Row order is `bodyId` ascending, independent of on-disk feather row order, so re-running this
script against the same input feather always produces byte-identical output (checked by
comparing sha256 across two runs in CI/local testing, not asserted here).

Axis mapping (verified empirically 2026-09-18 against this release's own somaLocation values,
not assumed from the general "8nm voxel, x=L/R, y=D/V, z=A/P" MaleCNS convention alone): raw
`somaLocation` x separates L/R somaSide/rootSide cleanly (L mean x=79,133, R mean x=17,726; y/z
means barely differ between sides), and a histogram of raw z has a hard, near-empty gap from
~45,400 to ~59,900 (7 consecutive 2,073-wide bins holding 1 row total out of 140,024) separating
the central-brain+optic-lobe cluster (z < Z_GAP) from the VNC (z >= Z_GAP) — confirming
x=left-right, z=anterior-posterior (brain=anterior=smaller z), and, from the same split, raw y
is higher for the VNC side of that gap (more ventral) than the brain side, confirming
y=dorsal-ventral (larger=ventral). `superclass` alone is NOT a reliable brain/VNC split: 1,848
rows labelled ascending_neuron/efferent_ascending/null have a soma physically in the VNC
(z >= Z_GAP) despite the label — the z-gap threshold, not the label, decides brain vs. VNC
everywhere in this script. Exported as:
    three.x = (raw.x - center.x) * scale   [no flip: subject-left (larger raw x) renders on the
                                             viewer's right, the standard "facing the subject"
                                             anatomical frontal-view convention]
    three.y = -(raw.y - center.y) * scale  [flipped: dorsal (smaller raw y) renders as up]
    three.z = -(raw.z - center.z) * scale  [flipped: anterior (smaller raw z) renders toward
                                             +Z, i.e. toward a camera placed on +Z looking at
                                             the origin sees the face; the VNC (large raw z)
                                             ends up at strongly negative Z, behind the brain]
`center`/`scale` are derived from the brain-proper (z < Z_GAP) bounding box only so the
optic-lobe "butterfly" — the shape that makes this instantly readable as a fly brain — fills
the frame; the VNC (present in every exported row, never dropped) then sits below and behind
it, extending outside [-1, 1] along -Y/-Z. `meta.json["vnc_z_threshold"]` is the same z-gap
cutoff already converted to output three.z units, so a renderer can filter `three.z <
vnc_z_threshold` to hide the VNC without re-deriving anything.

Usage: python3 scripts/export_atlas.py [--input PATH] [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import pyarrow.compute as pc
from pyarrow import feather

DEFAULT_INPUT = Path("data/raw/malecns-v1.0/body-annotations-male-cns-v1.0-minconf-0.5.feather")
DEFAULT_OUT_DIR = Path("site/public/data/atlas")

# Anatomical brain/VNC split point in raw somaLocation z, from the histogram gap documented in
# the module docstring's "Axis mapping" section (empty from ~45,400 to ~59,900) — the midpoint.
Z_GAP_THRESHOLD = 52650.0

SOURCE = "Janelia MaleCNS v1.0 body-annotations-male-cns-v1.0-minconf-0.5.feather (status == 'Traced', non-null somaLocation)"
LICENSE = "CC BY 4.0"
ATTRIBUTION = (
    "Berg, Beckett, Costa et al. \"Sexual dimorphism in the complete connectome of the "
    "Drosophila male central nervous system.\" bioRxiv (2025). doi:10.1101/2025.10.09.680999. "
    "Source: https://male-cns.janelia.org/. Not affiliated with or endorsed by Janelia, HHMI "
    "or the dataset authors."
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build(input_path: Path, out_dir: Path) -> None:
    table = feather.read_table(input_path)
    table = table.filter(pc.equal(table.column("status"), "Traced"))
    table = table.filter(pc.is_valid(table.column("somaLocation")))
    table = table.sort_by("bodyId")

    n = table.num_rows
    body_ids = table.column("bodyId").to_pylist()
    soma = table.column("somaLocation").to_pylist()  # list of [x, y, z] per row
    superclass = table.column("superclass").to_pylist()

    # groups: sorted distinct non-null superclass values, "unknown" appended last for nulls.
    distinct = sorted({s for s in superclass if s is not None})
    groups = distinct + ["unknown"]
    group_index = {name: i for i, name in enumerate(groups)}
    if len(groups) > 256:
        raise ValueError(f"{len(groups)} groups exceed uint8 range")

    xs = [p[0] for p in soma]
    ys = [p[1] for p in soma]
    zs = [p[2] for p in soma]
    bbox_min = (min(xs), min(ys), min(zs))
    bbox_max = (max(xs), max(ys), max(zs))

    # center/scale basis: brain proper only (z < Z_GAP_THRESHOLD), see "Axis mapping" above —
    # the anatomical z-gap, not the (partially mislabeled) superclass, decides brain vs. VNC.
    is_brain = [z < Z_GAP_THRESHOLD for z in zs]
    brain_xs = [v for v, b in zip(xs, is_brain) if b]
    brain_ys = [v for v, b in zip(ys, is_brain) if b]
    brain_zs = [v for v, b in zip(zs, is_brain) if b]
    if not brain_xs:
        raise ValueError("no brain-proper (z < Z_GAP_THRESHOLD) rows found to compute center/scale from")
    bbox_brain_min = (min(brain_xs), min(brain_ys), min(brain_zs))
    bbox_brain_max = (max(brain_xs), max(brain_ys), max(brain_zs))
    center = tuple((lo + hi) / 2.0 for lo, hi in zip(bbox_brain_min, bbox_brain_max))
    half_extent = max((hi - lo) / 2.0 for lo, hi in zip(bbox_brain_min, bbox_brain_max))
    scale = 1.0 / half_extent if half_extent > 0 else 1.0
    vnc_z_threshold_three = -(Z_GAP_THRESHOLD - center[2]) * scale

    positions = bytearray(n * 3 * 4)
    for i, (x, y, z) in enumerate(soma):
        struct.pack_into(
            "<fff",
            positions,
            i * 12,
            (x - center[0]) * scale,
            -(y - center[1]) * scale,
            -(z - center[2]) * scale,
        )

    groups_bytes = bytearray(n)
    for i, s in enumerate(superclass):
        groups_bytes[i] = group_index[s if s is not None else "unknown"]

    ids_bytes = bytearray(n * 8)
    for i, bid in enumerate(body_ids):
        struct.pack_into("<Q", ids_bytes, i * 8, bid)

    out_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "positions.f32": bytes(positions),
        "groups.u8": bytes(groups_bytes),
        "neuron_ids.u64": bytes(ids_bytes),
    }
    file_meta = {}
    for name, data in files.items():
        (out_dir / name).write_bytes(data)
        file_meta[name] = {"bytes": len(data), "sha256": sha256_bytes(data)}

    meta = {
        "n": n,
        "groups": groups,
        "bbox": {"min": list(bbox_min), "max": list(bbox_max)},
        "bbox_brain": {"min": list(bbox_brain_min), "max": list(bbox_brain_max)},
        "center": list(center),
        "scale": scale,
        "vnc_z_threshold": vnc_z_threshold_three,
        "axis_mapping": (
            "raw somaLocation (x=left-right, larger=subject-left; y=dorsal-ventral, "
            "larger=ventral; z=anterior-posterior, larger=posterior) -> "
            "three.x=(x-cx)*scale, three.y=-(y-cy)*scale, three.z=-(z-cz)*scale; "
            "center/scale from the brain-proper bounding box z<Z_GAP_THRESHOLD (bbox_brain); "
            "a point is VNC iff its three.z < vnc_z_threshold — see module docstring"
        ),
        "source": SOURCE,
        "license": LICENSE,
        "attribution": ATTRIBUTION,
        "files": file_meta,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")

    print(f"wrote {n} neurons, {len(groups)} groups -> {out_dir}")
    for name, m in file_meta.items():
        print(f"  {name}: {m['bytes']} bytes sha256={m['sha256'][:12]}...")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    build(args.input, args.out_dir)


if __name__ == "__main__":
    main()
