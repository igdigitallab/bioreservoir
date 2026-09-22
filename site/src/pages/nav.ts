// Top nav: small logo mark + wordmark left, a few section links, "Ask the fly" cyan pill right —
// design-style-reference.md's "Top nav" component. Shared across /, /scoreboard, /a/:id (not
// /stream, which is its own full-bleed capture layout with no chrome).
import { h } from "../dom";

/** Minimal abstract mark: a node with two synapse dots — matches the "brand icon strokes" role
 * (cyan) from the style reference's color table, not a literal fly/brain illustration (that is
 * the hero and the mascot sticker's job). Inline SVG, no external asset. */
function logoMark(): SVGSVGElement {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("class", "nav-mark");
  svg.setAttribute("aria-hidden", "true");
  svg.innerHTML =
    '<circle cx="12" cy="12" r="7" fill="none" stroke="#3ba6f1" stroke-width="1.6"/>' +
    '<circle cx="12" cy="12" r="2" fill="#3ba6f1"/>' +
    '<circle cx="4.5" cy="7" r="1.4" fill="#0c0a09"/>' +
    '<circle cx="19.5" cy="17" r="1.4" fill="#0c0a09"/>';
  return svg;
}

/** Hamburger glyph — three bars, no external icon set for one glyph. */
function menuMark(): SVGSVGElement {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("aria-hidden", "true");
  svg.innerHTML =
    '<path d="M4 6h16M4 12h16M4 18h16" stroke="#0c0a09" stroke-width="1.8" stroke-linecap="round"/>';
  return svg;
}

const NAV_LINKS: Array<[string, string]> = [
  ["/#live", "Live"],
  ["/#how-it-works", "How it works"],
  ["/#validation", "Lab data"],
  ["/#about-lab", "About the lab"],
];

export function buildNav(): HTMLElement {
  const brand = h("a", { class: "nav-brand", href: "/" }, [logoMark(), "BioReservoir"]);
  const links = h(
    "nav",
    { class: "nav-links" },
    NAV_LINKS.map(([href, label]) => h("a", { href }, label)),
  );
  const cta = h("a", { class: "btn", href: "/#ask-box" }, "Ask the fly");

  // Mobile nav (below 860px, where .nav-links is display:none, see style.css): a toggle button
  // that opens a dropdown panel with the same links, restoring "How it works"/"Lab data"/
  // "About the lab", which were previously just gone with nothing replacing them
  // (2026-09-19 layout audit).
  const panel = h(
    "nav",
    { class: "nav-panel" },
    NAV_LINKS.map(([href, label]) => h("a", { href }, label)),
  );
  const toggle = h(
    "button",
    { class: "nav-toggle", type: "button", "aria-label": "Menu", "aria-expanded": "false" },
    menuMark(),
  ) as HTMLButtonElement;
  toggle.addEventListener("click", () => {
    const open = panel.classList.toggle("is-open");
    toggle.setAttribute("aria-expanded", String(open));
  });
  // A same-page nav link click always closes the panel — otherwise it
  // stays open over the next page since main.ts's route swap doesn't rebuild this closed nav.
  panel.addEventListener("click", (e) => {
    if (e.target instanceof Element && e.target.closest("a")) {
      panel.classList.remove("is-open");
      toggle.setAttribute("aria-expanded", "false");
    }
  });

  return h("header", { class: "site-nav" }, [
    h("div", { class: "site-nav-inner container" }, [brand, links, h("div", { class: "nav-actions" }, [toggle, cta])]),
    panel,
  ]);
}
