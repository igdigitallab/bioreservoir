import "./style.css";
// Self-hosted fonts (Fontsource, OFL), latin subset only, no CDN — design-style-reference.md
// "Typography": Inter Tight (display, 400/500) + Inter (body, 400/500/600). Roobert is
// commercial and is not used anywhere in this codebase.
// https://fontsource.org/docs/getting-started/install ("import only the weights/styles you
// need"); this package's own `files/` directory ships one CSS file per subset+weight
// (`latin-400.css` etc.), which is the subset-scoping mechanism used here.
import "@fontsource/inter-tight/latin-400.css";
import "@fontsource/inter-tight/latin-500.css";
import "@fontsource/inter/latin-400.css";
import "@fontsource/inter/latin-500.css";
import "@fontsource/inter/latin-600.css";
import { onRouteChange, type Route } from "./router";
import { copy } from "./content";
import { h, clear } from "./dom";

document.title = copy.meta.title;
const descTag = document.querySelector('meta[name="description"]');
if (descTag) descTag.setAttribute("content", copy.meta.description);

// Explicit HTMLElement annotation (not the getElementById's HTMLElement|null return type) so the
// null check below is the only place that type is narrowed — TypeScript does not otherwise carry
// narrowing across the function-closure boundaries used throughout this file.
const app: HTMLElement = document.getElementById("app") as HTMLElement;
if (!app) throw new Error("#app root element missing from index.html");

let unmount: (() => void) | undefined;
// Bumped on every render() call so a slow dynamic import for a route the user has since
// navigated away from can't mount itself over whatever loaded after it.
let renderToken = 0;

/** Nav links point at "/#section-id" (works from any route); after a same-page SPA route swap
 * the browser does not auto-scroll to a URL hash the way a full page load would, so do it
 * ourselves once the target section exists in the DOM. */
function scrollToHashIfAny() {
  const hash = window.location.hash;
  if (!hash) return;
  const target = document.getElementById(hash.slice(1));
  target?.scrollIntoView({ behavior: "smooth", block: "start" });
}

// Dynamic imports per route so Three.js (pulled in by home/stream/share) is not downloaded by a
// visitor who only ever loads a share link, and so the initial JS needed before first paint of
// any page is just the router itself.
function render(route: Route) {
  const token = ++renderToken;
  unmount?.();
  unmount = undefined;
  clear(app);

  if (route.name === "home") {
    import("./pages/home").then(({ mountHome }) => {
      if (token === renderToken) {
        unmount = mountHome(app);
        scrollToHashIfAny();
      }
    });
  } else if (route.name === "stream") {
    import("./pages/stream").then(({ mountStream }) => {
      if (token === renderToken) unmount = mountStream(app);
    });
  } else if (route.name === "share") {
    import("./pages/share").then(({ mountShare }) => {
      if (token === renderToken) unmount = mountShare(app, route.id);
    });
  } else if (route.name === "legal") {
    import("./pages/legal").then(({ mountLegal }) => {
      if (token === renderToken) unmount = mountLegal(app, route.doc);
    });
  } else {
    app.appendChild(h("div", { class: "container section" }, [h("h1", {}, "Not found"), h("a", { href: "/" }, "Back home")]));
  }
}

onRouteChange(render);

// Intercept same-origin link clicks so internal navigation stays client-side (no full reload).
document.addEventListener("click", (event) => {
  const target = event.target;
  if (!(target instanceof Element)) return;
  const anchor = target.closest("a");
  if (!anchor) return;
  const href = anchor.getAttribute("href");
  if (!href || !href.startsWith("/") || anchor.target === "_blank") return;
  event.preventDefault();
  window.history.pushState({}, "", href);
  window.dispatchEvent(new PopStateEvent("popstate"));
});
