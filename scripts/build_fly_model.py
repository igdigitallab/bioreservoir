#!/usr/bin/env python3
"""Build the site's flybody specimen on CPU, without Blender or MuJoCo.

python3 -m venv /tmp/fly-model-venv
/tmp/fly-model-venv/bin/pip install numpy==2.2.6 trimesh==4.7.4 fast-simplification==0.1.12 pillow==11.3.0 scipy==1.15.3
/tmp/fly-model-venv/bin/python scripts/build_fly_model.py [--source /tmp/flybody-source]

Without --source the pinned archive is downloaded to a temporary directory. The
optional checkout must be at COMMIT. Geometry is decimated before 16-bit position
and 8-bit normal quantisation (KHR_mesh_quantization); no external decoder needed.
The MJCF hierarchy supplies real anatomical pivots, rather than guessed joints.
Model-specific licence evidence: https://github.com/google-deepmind/mujoco_menagerie/blob/main/flybody/README.md#license
Anatomy: Vaxenburg et al., https://doi.org/10.1038/s41586-025-09029-4
"""
import argparse
import io
import json
import struct
import subprocess
import tarfile
import tempfile
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh

COMMIT = "736608121847c3c025fd02a629bdb016a3294f9a"
REPO = "https://github.com/TuragaLab/flybody"
ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "site/public/models"
# MuJoCo: +X forward, +Y left, +Z up. glTF: -Z forward, -X left, +Y up.
BASIS = np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]], dtype=float)


def transform(element):
    w, x, y, z = np.fromstring(element.get("quat", "1 0 0 0"), sep=" ")
    q = np.array([w, x, y, z]); w, x, y, z = q / np.linalg.norm(q)
    matrix = np.eye(4)
    matrix[:3, :3] = [
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ]
    matrix[:3, 3] = np.fromstring(element.get("pos", "0 0 0"), sep=" ")
    return matrix


def build(source):
    assets = source / "flybody/fruitfly/assets"
    xml = ET.parse(assets / "fruitfly.xml").getroot()
    mesh_files = {m.get("name"): m.get("file") for m in xml.findall("asset/mesh")}
    materials = []
    # Colours are linear-light equivalents of warm stone / ink; no source red survives.
    palette = {
        "body": ([.29, .265, .24, 1], .52),
        "lower": ([.42, .39, .355, 1], .62),
        "black": ([.025, .023, .021, 1], .58),
        "bristle-brown": ([.045, .04, .035, 1], .65),
        "red": ([.018, .017, .016, 1], .32),
        "ocelli": ([.012, .011, .01, 1], .24),
        "brown": ([.14, .125, .11, 1], .48),
        "membrane": ([.68, .665, .64, .26], .3),
    }
    for name, (color, roughness) in palette.items():
        material = {"name": name, "pbrMetallicRoughness": {
            "baseColorFactor": color, "metallicFactor": .08, "roughnessFactor": roughness}}
        if name == "membrane":
            material.update(alphaMode="BLEND", doubleSided=True)
        materials.append(material)
    gltf = {"asset": {"version": "2.0", "generator": "BioReservoir build_fly_model.py",
        "copyright": "flybody contributors, Google DeepMind and HHMI Janelia; Apache-2.0",
        "extras": {"source": REPO, "commit": COMMIT,
                  "changes": "Decimated, quantised, converted MJCF joints to glTF; monochrome materials; shortened female-source abdomen with dark male-like terminal tergites."}},
        "extensionsUsed": ["KHR_mesh_quantization"], "extensionsRequired": ["KHR_mesh_quantization"],
        "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [], "meshes": [],
        "materials": materials, "accessors": [], "bufferViews": [], "buffers": []}
    binary = bytearray()
    triangles = 0

    def specimen_points(points, name):
        # The source is a female reconstruction (paper Fig. 1). A shorter abdomen and dark
        # terminal tergites suggest a male at icon size; this is not a male anatomical scan.
        if name.startswith("abdomen"):
            points = points.copy()
            points[..., 2] = .447 + (points[..., 2] - .447) * .8
        return points

    def accessor(data, component, kind, target, normalized=False, bounds=False):
        while len(binary) % 4: binary.append(0)
        view = {"buffer": 0, "byteOffset": len(binary), "byteLength": data.nbytes, "target": target}
        if kind == "VEC3": view["byteStride"] = data.strides[0]
        gltf["bufferViews"].append(view)
        binary.extend(data.tobytes())
        item = {"bufferView": len(gltf["bufferViews"])-1, "componentType": component,
                "count": len(data), "type": kind}
        if normalized: item["normalized"] = True
        if bounds:
            item["min"] = data[:, :3].min(axis=0).tolist()
            item["max"] = data[:, :3].max(axis=0).tolist()
        gltf["accessors"].append(item)
        return len(gltf["accessors"])-1

    def body(element, parent_matrix, parent_origin):
        nonlocal triangles
        world = parent_matrix @ transform(element)
        origin = specimen_points(BASIS @ world[:3, 3] * 10, element.get("name"))
        name = element.get("name")
        node = {"name": "proboscis" if name == "rostrum" else name,
                "translation": (origin-parent_origin).tolist(), "children": []}
        index = len(gltf["nodes"]); gltf["nodes"].append(node)
        for geom in element.findall("geom"):
            if geom.get("mesh") not in mesh_files: continue
            mesh_name = geom.get("mesh")
            mesh = trimesh.load(assets / mesh_files[mesh_name], force="mesh", process=True)
            # Spend triangles on the silhouette and vein network, not invisible inner surfaces.
            budget = 1100
            if name == "thorax": budget = 6500
            elif name == "head": budget = 6500 if mesh_name == "head_red" else 2800
            elif name.startswith("wing"): budget = 7500 if "brown" in mesh_name else 1600
            elif name.startswith("abdomen"): budget = 1400
            elif name.startswith("antenna"): budget = 1600
            elif "tarsus" in name or "claw" in name: budget = 220
            if len(mesh.faces) > budget:
                mesh = mesh.simplify_quadric_decimation(face_count=budget, aggression=5)
            triangles += len(mesh.faces)
            matrix = world @ transform(geom)
            points = (mesh.vertices * .1) @ matrix[:3, :3].T + matrix[:3, 3]
            points = specimen_points(points @ BASIS.T * 10, name) - origin
            mesh = trimesh.Trimesh(points, mesh.faces, process=False)
            center = (points.max(axis=0) + points.min(axis=0)) / 2
            extent = max(float(np.abs(points-center).max()), 1e-8)
            # VEC3 attributes have padded four-byte strides, including quantised normals.
            positions = np.zeros((len(points), 4), dtype="<i2")
            positions[:, :3] = np.rint((points-center)/extent*32767).astype("<i2")
            normals = np.zeros((len(points), 4), dtype="i1")
            smooth_normals = mesh.vertex_normals.copy()
            # Back-to-back bristle faces can cancel averaged normals. Use an incident face
            # there so every glTF normal remains unit length after quantisation.
            missing = np.linalg.norm(smooth_normals, axis=1) < .5
            smooth_normals[missing] = mesh.face_normals[np.maximum(mesh.vertex_faces[missing, 0], 0)]
            smooth_normals[np.linalg.norm(smooth_normals, axis=1) < .5] = [0, 1, 0]
            normals[:, :3] = np.rint(smooth_normals*127).astype("i1")
            indices = np.asarray(mesh.faces.flatten(), dtype="<u2")
            attrs = {"POSITION": accessor(positions, 5122, "VEC3", 34962, True, True),
                     "NORMAL": accessor(normals, 5120, "VEC3", 34962, True)}
            material_name = geom.get("material", "body")
            # Male terminal tergites are dark. Keep the anterior segment edges visible too.
            if name.startswith("abdomen") and material_name == "body":
                material_name = "black" if name in ("abdomen_5", "abdomen_6", "abdomen_7") else "body"
            primitive = {"attributes": attrs, "indices": accessor(indices, 5123, "SCALAR", 34963),
                         "material": list(palette).index(material_name)}
            gltf["meshes"].append({"name": mesh_name, "primitives": [primitive]})
            node["children"].append(len(gltf["nodes"]))
            gltf["nodes"].append({"name": mesh_name+"_surface", "mesh": len(gltf["meshes"])-1,
                                  "translation": center.tolist(), "scale": [extent]*3})
        for child in element.findall("body"):
            node["children"].append(body(child, world, origin))
        return index

    body(xml.find("worldbody/body"), np.eye(4), np.zeros(3))
    while len(binary) % 4: binary.append(0)
    gltf["buffers"] = [{"byteLength": len(binary)}]
    encoded = json.dumps(gltf, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 4)
    result = struct.pack("<III", 0x46546C67, 2, 28+len(encoded)+len(binary))
    result += struct.pack("<II", len(encoded), 0x4E4F534A)+encoded
    result += struct.pack("<II", len(binary), 0x004E4942)+binary
    assert len(result) <= 1_500_000, f"Model exceeds budget: {len(result):,} bytes"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "fly.glb").write_bytes(result)
    (OUTPUT / "fly-LICENSE.txt").write_bytes((source / "LICENSE").read_bytes())
    print(f"{OUTPUT / 'fly.glb'}: {len(result):,} bytes, {triangles:,} triangles, {len(gltf['nodes'])} nodes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    if args.source:
        actual = subprocess.check_output(["git", "-C", str(args.source), "rev-parse", "HEAD"], text=True).strip()
        if actual != COMMIT: parser.error(f"--source must be at {COMMIT}, found {actual}")
        build(args.source)
    else:
        with tempfile.TemporaryDirectory(prefix="flybody-build-") as directory:
            url = f"https://codeload.github.com/TuragaLab/flybody/tar.gz/{COMMIT}"
            with urllib.request.urlopen(url) as response:
                payload = io.BytesIO(response.read())
            with tarfile.open(fileobj=payload, mode="r:gz") as archive:
                archive.extractall(directory, filter="data")
            build(Path(directory) / f"flybody-{COMMIT}")
