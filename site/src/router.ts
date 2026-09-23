// A minimal client-side router using the History API — no framework needed for four routes.
// https://developer.mozilla.org/en-US/docs/Web/API/History_API
// The static host must serve index.html for any unknown path (SPA fallback) so a hard refresh
// on /a/<id> or /stream still loads this router — see site/README.md.
export type Route =
  | { name: "home" }
  | { name: "stream" }
  | { name: "share"; id: string }
  | { name: "legal"; doc: "terms" | "privacy" }
  | { name: "not-found" };

// `search` (default "") is `window.location.search`, e.g. "?a=4". The server's OG share page
// (`GET /a/{id}` in card.py) meta-refreshes a real visitor's browser to `/?a={id}` — a QUERY
// param on the home path, not the `/a/:id` PATH route below (that path is reserved for the
// backend's own crawler-facing OG route, so the SPA can't also claim it) — so `/?a=4` used to
// fall through to plain "home" and silently ignore `a`, landing a shared-link visitor on the
// idle homepage instead of the answer they were sent. Routing it to the
// same "share" route as `/a/:id` reuses the existing mountShare() page (answer card + brain
// replay) instead of building a second answer-rendering path.
export function parseRoute(pathname: string, search = ""): Route {
  if (pathname === "/" || pathname === "") {
    const shareId = new URLSearchParams(search).get("a");
    if (shareId) return { name: "share", id: shareId };
    return { name: "home" };
  }
  if (pathname === "/stream") return { name: "stream" };
  // The legal documents (2026-09-21) — linked from the footer of every page, kept off the live
  // page itself. Trailing slash accepted: a footer link shared by hand often grows one.
  if (pathname === "/terms" || pathname === "/terms/") return { name: "legal", doc: "terms" };
  if (pathname === "/privacy" || pathname === "/privacy/") return { name: "legal", doc: "privacy" };
  // Retired 2026-09-19 with the rest of the forecast surface: an old link or bookmark lands on
  // the live page instead of a dead end.
  if (pathname === "/scoreboard") return { name: "home" };
  const shareMatch = pathname.match(/^\/a\/([^/]+)\/?$/);
  if (shareMatch) return { name: "share", id: decodeURIComponent(shareMatch[1]!) };
  return { name: "not-found" };
}

export function navigate(path: string): void {
  window.history.pushState({}, "", path);
  window.dispatchEvent(new PopStateEvent("popstate"));
}

export function onRouteChange(handler: (route: Route) => void): void {
  const render = () => handler(parseRoute(window.location.pathname, window.location.search));
  window.addEventListener("popstate", render);
  render();
}
