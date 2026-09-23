# Fruit-fly specimen

The existing `createFlyIcon(container)` API lazy-loads a three.js scene and one local GLB.
The pages and layout are unchanged. The original drawing is retained in `flyIcon2d.ts` for
WebGL failure, context loss or a failed asset request. Both renderers pause offscreen and
when the document is hidden. Pixel ratio is capped at 2. Reduced motion removes idle and
periodic movement, easing into the triggered pose and back to rest with the original replay
envelope (20% attack, 60% sustain, 20% release).

`flyPose.ts` contains the pure drive/envelope, pose and two-bone leg IK functions. The gait
alternates L1/L3/R2 with L2/R1/R3; the measured femur and tibia lengths are preserved while
lifting the feet. `play()` replaces the previous drives; an empty trigger list clears them.
Cyan is reserved for a physical turn. Left/right are **the fly's anatomical sides**, so from
the front three-quarter camera its left is on the viewer's right. These poses illustrate
the readouts; they are not a MuJoCo simulation or a new inference from the connectome.

The meshes preserve the head, articulated proboscis, antennae, wings and veins, halteres,
thorax, abdominal segments, six coxae/femora/tibiae and all tarsal segments/claws. Warm
monochrome materials replace the source colours. The camera, hemisphere/key/rim lights and
analytic contact shadow are independent of the brain visualization.

Source/licence evidence and the female-source anatomical caveat are in [DATA.md](DATA.md#third-party-assets--fruit-fly-specimen).
The source abdomen is shortened 20% and darkened at the tip to suggest a male; it is not a
male scan. No FlyWire geometry or data is used in this asset. Licence and attribution files
are copied into the production site's `models/` directory alongside the model.

## Rebuild (CPU only)

From the repository root:

```sh
python3 -m venv /tmp/fly-model-venv
/tmp/fly-model-venv/bin/pip install numpy==2.2.6 trimesh==4.7.4 fast-simplification==0.1.12 pillow==11.3.0 scipy==1.15.3
/tmp/fly-model-venv/bin/python scripts/build_fly_model.py
```

The script fetches commit `736608121847c3c025fd02a629bdb016a3294f9a` from TuragaLab/flybody.
Alternatively, pass `--source /path/to/flybody` with that commit checked out. It reads the
OBJ parts and MJCF hierarchy, simplifies geometry, quantises positions/normals and writes
`site/public/models/fly.glb` and `fly-LICENSE.txt`. The checked-in `fly-NOTICE.txt` accompanies
these. The GLB uses `KHR_mesh_quantization`, with no Draco/meshopt decoder or external textures.
The script fails if the output exceeds 1,500,000 bytes.

Verified output: **1,222,384 bytes**, 93,668 triangles, 152 nodes including 67 articulated
body nodes. Two consecutive pinned builds produced identical SHA-256 hashes. Khronos
glTF Validator reported **0 errors and 0 warnings**.

## Preview and verification

```sh
cd site
npx vite --port 4790
```

Open `http://localhost:4790/dev/fly.html?anim=appetite&t=.5`.
The dev HTML is not a Vite production entry and is absent from `dist/`. Controls select each
behaviour and scrub the replay. Supported queries:

- `anim=idle|turn-left|turn-right|appetite|fear|backoff|courtship|arousal`
- `t=0..1`: deterministic replay progress across a six-second preview
- `size=200`: mobile panel (default 320)
- `reduced`: static reduced-motion pose
- `live`: exercise the production loader, observers and timed replay
- `fallback`: exercise the original 2D canvas

```sh
mkdir -p /tmp/fly3d
$(ls -d ~/.cache/ms-playwright/chromium_headless_shell-1243/*/ | head -1)chrome-headless-shell \
  --no-sandbox --hide-scrollbars --use-angle=swiftshader --enable-unsafe-swiftshader \
  --window-size=640,640 --timeout=12000 --screenshot=/tmp/fly3d/idle.png \
  'http://localhost:4790/dev/fly.html?anim=idle&t=.5'
```

No virtual-time budget is used. Chromium's CLI can capture before the asynchronous model
finishes; for reliable automation wait for `html[data-ready="true"]` in Playwright before
capturing, using the same local Chromium executable and SwiftShader flags. Verification
did both: the requested CLI captures, followed by ready-aware captures inspected visually.

Final captures: `/tmp/fly3d/{idle,turn-left,turn-right,appetite,fear,backoff,courtship,arousal,idle-200}.png`.
Additional lifecycle checks exercised lazy loading, DPR 3 capped at 2, offscreen pause,
visibility changes, reduced-motion idle/replay, missing WebGL, context disposal and disposal
during a pending model request. Physical iOS Safari hardware was not available for testing.

```sh
cd site
npm test
npm run build
```

The build retains the existing warning for the shared three.js chunk over 500 kB.
