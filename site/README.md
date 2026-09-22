# Live Fly site

Interactive page: a visitor asks a yes/no question, a real MaleCNS connectome simulation (run by
the backend, `bioreservoir.oracle`) answers it, and the page replays the actual spikes that fired.

Stack: Vite + TypeScript (strict) + Three.js, no framework. Vanilla DOM (`src/dom.ts`) + a tiny
History API router for four routes: `/`, `/stream`, `/a/:id`.

## Dev

```bash
npm install
cp .env.example .env.local   # fill in values, see "Env vars" below
npm run dev                  # http://localhost:5173
npm run typecheck
npm test                     # vitest: frames.test.ts, states.test.ts
npm run build                # tsc --noEmit && vite build -> dist/
npm run preview              # serve dist/ locally
```

Set `VITE_USE_MOCK=1` in `.env.local` to develop with no backend running — patches
`window.fetch`/`EventSource` with an in-browser fixture (`src/mock/mockApi.ts`), gated behind
`import.meta.env.DEV` so it is dead-code-eliminated from `dist/` (never shipped).

## Env vars (names only — see `.env.example`)

- `VITE_TURNSTILE_SITEKEY` — Cloudflare Turnstile site key (public). For local dev use
  Cloudflare's documented "always passes" test key `1x00000000000000000000AA`
  (https://developers.cloudflare.com/turnstile/troubleshooting/testing/) — it renders a real
  widget and issues a dummy token accepted by no real backend, so pair it with `VITE_USE_MOCK=1`.
- `VITE_API_BASE` — oracle API origin, no trailing slash.
- `VITE_PUBLIC_URL` — canonical public URL, used for the `/stream` QR code and share links.
- `VITE_USE_MOCK` — `1` to use the dev fixture instead of a real backend. Dev only.

## Atlas export

The brain point cloud reads `public/data/atlas/` (gitignored, generated — see repo root
`.gitignore`), built from the MaleCNS body-annotation feather file:

```bash
cd <repo root>
~/.local/bin/uv run --no-sync python3 scripts/export_atlas.py
```

Deterministic: same input feather -> byte-identical output. Output: `positions.f32` (already in
Three.js display space — frontal view, VNC hidden below/behind by default, see the script's
"Axis mapping" docstring), `groups.u8`, `neuron_ids.u64` (spike-index lookup), `meta.json`.
140,024 neurons, ~2.9 MB total.

## `/stream` route

Fixed 1920x1080, `overflow: hidden` (`src/pages/stream.ts`, `.stream-page`) for a headless-browser
capture -> ffmpeg -> RTMP pipeline (stage B, not built yet). No interactive controls or Turnstile.

## Deploy

Static host must serve `index.html` for unmatched paths (SPA fallback) so a hard refresh on
`/stream`, `/a/:id` still loads the router (nginx: `try_files $uri /index.html;`).
Deploy path not decided yet.
