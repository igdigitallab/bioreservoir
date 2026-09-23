/// <reference types="vitest/config" />
import { defineConfig } from "vitest/config";

// https://vite.dev/config/
export default defineConfig({
  build: {
    target: "es2022",
    sourcemap: true,
  },
  server: {
    // Dev proxy so `npm run dev` can preview against a real backend (DEV_API_PROXY, default the
    // public site) instead of only the VITE_USE_MOCK fixture — set VITE_API_BASE to "" (default,
    // see .env.example) so requests go to same-origin "/api/..." and hit this proxy rather than
    // the dev server itself. Never POST questions through this — /api/ask is proxied like every
    // other route, so a real submit here would hit production for real.
    proxy: {
      "/api": {
        target: process.env.DEV_API_PROXY ?? "https://fly.igdigi.com",
        changeOrigin: true,
      },
    },
  },
  test: {
    environment: "node",
    include: ["src/**/*.test.ts"],
  },
});
