// GA4 for fly.igdigi.com.
//
// Same GA4 property as igdigi.com (G-DV5C3E37KT), on purpose: gtag.js's default cookie_domain
// is "auto" (highest-level domain it can set a cookie on), so the `_ga` cookie lands on
// `.igdigi.com` and a visit that starts here and ends on igdigi.com stays ONE session — no
// self-referral, original source (Hacker News, Reddit, ...) survives all the way to a lead.
// Subdomains need no extra "linker" config for this; that is only for genuinely different
// domains. https://support.google.com/analytics/answer/10071811,
// https://developers.google.com/tag-platform/devguides/cross-domain
//
// Every page_view from here carries `site_section: "fly"` so GA4 reports can isolate this
// subdomain's traffic inside the shared property.
//
// The gtag script + `gtag('config', ID, {send_page_view: false, site_section: 'fly'})` call
// live in index.html (loaded before this module). Automatic page_view on `gtag('config', ...)`
// only fires once, for the very first load; a client-side router swap (pushState) does not
// re-trigger it — GA4's own guidance for SPAs is to disable send_page_view and fire page_view
// manually on every route change:
// https://developers.google.com/analytics/devguides/collection/ga4/views
import type { Route } from "./router";

declare global {
  interface Window {
    dataLayer?: unknown[];
    gtag?: (...args: unknown[]) => void;
  }
}

function gtag(...args: unknown[]): void {
  if (typeof window.gtag === "function") window.gtag(...args);
}

/** Call once per route render (main.ts's `render()`), including the very first one — the
 * config call's automatic page_view is suppressed (send_page_view:false in index.html) so this
 * is the only source of page_view events, and the only one guaranteed to carry site_section. */
export function trackPageView(_route: Route): void {
  gtag("event", "page_view", {
    page_location: window.location.href,
    page_path: window.location.pathname + window.location.search,
    page_title: document.title,
    site_section: "fly",
  });
}

// Every outbound link to igdigi.com from this site is written with `?ref=fly` (aboutLab.ts,
// footer.ts) instead of a `utm_source` — a utm parameter on our OWN domain would overwrite the
// visitor's real acquisition source the moment they land on igdigi.com, which is exactly the
// attribution this analytics setup exists to protect (see module docstring above). `ref=fly` is
// inert to GA4's own source/medium logic, so the original source survives.
//
// `fly_outbound` fires on click, independent of whether igdigi.com's own `fly_referral` (fired
// there when it sees `?ref=fly`) ever arrives — a GA4 exploration can then compare "clicks sent"
// against "landings recorded" to see how much the ad/tracker-blocking crowd (measured
// separately, server-side, by tools/fly_traffic.py) is costing this specific handoff.
const OUTBOUND_RE = /^https:\/\/(www\.)?igdigi\.com(\/|$|\?)/;

export function initOutboundTracking(): void {
  document.addEventListener(
    "click",
    (event) => {
      const target = event.target;
      if (!(target instanceof Element)) return;
      const anchor = target.closest("a");
      const href = anchor?.getAttribute("href") || "";
      if (!OUTBOUND_RE.test(href)) return;
      gtag("event", "fly_outbound", { link_url: href });
    },
    true,
  );
}
