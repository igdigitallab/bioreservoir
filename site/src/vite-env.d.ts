/// <reference types="vite/client" />

// Client-exposed env vars, see site/README.md "Env vars" and
// https://vite.dev/guide/env-and-mode.html ("Env Variables and Modes" — only VITE_-prefixed
// names are exposed to client code; import.meta.env.DEV/PROD are statically replaced at build
// time so `if (import.meta.env.DEV)` branches are dead-code-eliminated from the production
// bundle).
interface ImportMetaEnv {
  /** Cloudflare Turnstile site key (public, safe to ship — the secret key stays server-side). */
  readonly VITE_TURNSTILE_SITEKEY: string;
  /** Base URL of the oracle API, e.g. https://fly.bioreservoir.example — no trailing slash. */
  readonly VITE_API_BASE: string;
  /** Canonical public URL of this site, used for the /stream QR code and share links. */
  readonly VITE_PUBLIC_URL: string;
  /** "1" to route API calls through the in-browser dev mock instead of VITE_API_BASE. Dev only. */
  readonly VITE_USE_MOCK?: string;
  /** Public GitHub repository; the site links to it only when set (aboutLab.ts). */
  readonly VITE_REPO_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
