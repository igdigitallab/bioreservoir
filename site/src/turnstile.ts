// Thin wrapper around the Cloudflare Turnstile widget (script tag loaded in index.html with
// ?render=explicit, https://developers.cloudflare.com/turnstile/get-started/client-side-rendering/).
// The widget is rendered explicitly into a container and reports its token via a callback;
// server-side verification of that token happens in the backend (POST /api/ask), never here.
//
// Explicit render options (2026-09-18 review): Cloudflare's docs
// (https://developers.cloudflare.com/turnstile/get-started/client-side-rendering/widget-configurations/)
// list `appearance`/`size`/`retry`/`refresh-expired`/`error-callback`/`expired-callback` as
// configurable; this module sets them explicitly instead of relying on defaults so behaviour is
// the same regardless of future Cloudflare default changes, and so failures are diagnosable
// (widget id logged, error/expired states surfaced to a visible status line) rather than a
// silently-empty box. NOTE: Turnstile is *designed* to challenge or refuse automated/headless
// browsers (network + fingerprint signals, not just this widget's config) — that is not a bug
// in this client and is not something to work around here; server-side verification
// (moderation.py) is untouched.
//
// appearance (2026-09-19 layout audit): "always" rendered a full-size widget iframe permanently
// sitting between the character count and the submit button, reading as an unexplained floating
// box in the middle of the ask row. "interaction-only" (Cloudflare's own name for this — renders
// nothing until a visible challenge is actually required, which is most sessions) fixes that:
// the widget takes zero layout space until Cloudflare needs to show something, and the visible
// `turnstileStatus` line (driven by the callbacks below, not by widget chrome) already tells the
// visitor what is happening either way. `.turnstile-mount` in style.css gives it a dedicated
// full-width row under the textarea for the rare case it does need to render a challenge.

declare global {
  interface Window {
    turnstile?: {
      render: (
        container: string | HTMLElement,
        options: {
          sitekey: string;
          callback: (token: string) => void;
          "error-callback"?: (errorCode?: string) => void;
          "expired-callback"?: () => void;
          appearance?: "always" | "execute" | "interaction-only";
          size?: "normal" | "flexible" | "compact";
          retry?: "auto" | "never";
          "refresh-expired"?: "auto" | "manual" | "never";
        },
      ) => string;
      reset: (widgetId?: string) => void;
      getResponse: (widgetId?: string) => string | undefined;
      remove: (widgetId?: string) => void;
    };
  }
}

function waitForTurnstile(timeoutMs = 8000): Promise<NonNullable<Window["turnstile"]>> {
  return new Promise((resolve, reject) => {
    const start = performance.now();
    const poll = () => {
      if (window.turnstile) {
        resolve(window.turnstile);
        return;
      }
      if (performance.now() - start > timeoutMs) {
        reject(new Error("Turnstile script did not load in time"));
        return;
      }
      setTimeout(poll, 50);
    };
    poll();
  });
}

export interface TurnstileHandle {
  getToken: () => string | undefined;
  reset: () => void;
}

/** Status for the small visible line under the ask box — see home.ts's `turnstileStatus`. */
export type TurnstileStatus = "verifying" | "verified" | "expired" | "error";

export const TURNSTILE_STATUS_TEXT: Record<TurnstileStatus, string> = {
  verifying: "verifying you’re human…",
  verified: "verified",
  expired: "verification expired — retrying…",
  error: "verification failed — reload the page",
};

/** Render a Turnstile widget into `container` and resolve once it is mounted. `onStatus`, if
 * given, is called on every state change so the caller can drive a visible status line —
 * "verifying you're human…" / "verified" / "verification failed — reload". */
export async function mountTurnstile(
  container: HTMLElement,
  onToken: (token: string) => void,
  onStatus?: (status: TurnstileStatus) => void,
): Promise<TurnstileHandle> {
  const sitekey = import.meta.env.VITE_TURNSTILE_SITEKEY;
  onStatus?.("verifying");
  const ts = await waitForTurnstile();
  const widgetId = ts.render(container, {
    sitekey,
    callback: (token) => {
      onStatus?.("verified");
      onToken(token);
    },
    "error-callback": (errorCode) => {
      // Diagnosable, not just silent: which widget, which Cloudflare error code
      // (https://developers.cloudflare.com/turnstile/troubleshooting/client-side-errors/).
      console.error("turnstile error-callback", { widgetId, errorCode });
      onStatus?.("error");
    },
    "expired-callback": () => {
      console.warn("turnstile expired-callback", { widgetId });
      onStatus?.("expired");
    },
    // Explicit, not defaults (widget-configurations doc): render nothing until an actual
    // challenge is needed (not "execute" mode, which requires manually calling execute() — this
    // widget still auto-runs on mount, it just stays invisible when it doesn't need to show a
    // challenge), normal size when it does show, auto-retry a failed challenge, auto-refresh an
    // expired token.
    appearance: "interaction-only",
    size: "normal",
    retry: "auto",
    "refresh-expired": "auto",
  });
  console.log("turnstile widget mounted", { widgetId });
  return {
    getToken: () => ts.getResponse(widgetId),
    reset: () => {
      onStatus?.("verifying");
      ts.reset(widgetId);
    },
  };
}
