// The two legal documents (/terms and /privacy), rendered from content/legal.json — the same
// "every editorial string lives in content/" rule content.ts states for copy.json.
//
// They are deliberately OFF the live page (2026-09-21: the page itself stays clean and
// only about the fly); the footer links here, and the ask box points at the Terms in one line.
// Same nav/footer chrome as every other route so a visitor who lands on /privacy from a search
// result is still obviously on this site.
import { h, clear } from "../dom";
import { buildNav } from "./nav";
import { buildFooter } from "./footer";
import { legal, type LegalDoc } from "../legal";

function buildSection(section: LegalDoc["sections"][number]): HTMLElement {
  const children: (HTMLElement | string)[] = [h("h2", {}, section.h)];
  for (const paragraph of section.p ?? []) children.push(h("p", {}, paragraph));
  if (section.list) children.push(h("ul", { class: "legal-list" }, section.list.map((item) => h("li", {}, item))));
  for (const paragraph of section.after ?? []) children.push(h("p", {}, paragraph));
  return h("section", { class: "legal-section" }, children);
}

export function buildLegalDoc(doc: LegalDoc): HTMLElement {
  return h("article", { class: "legal-doc container section" }, [
    h("h1", {}, doc.title),
    h("p", { class: "label-mono" }, `Last updated: ${doc.updated}`),
    h("p", { class: "legal-intro" }, doc.intro),
    ...doc.sections.map(buildSection),
    h("p", { class: "legal-footer-link" }, [
      h("a", { href: doc === legal.terms ? "/privacy" : "/terms" }, doc === legal.terms ? "Privacy Policy" : "Terms of Use"),
      " · ",
      h("a", { href: "/" }, "Back to the fly"),
    ]),
  ]);
}

export function mountLegal(root: HTMLElement, which: "terms" | "privacy"): () => void {
  clear(root);
  const doc = legal[which];
  // Restored on unmount: main.ts sets the site title once at load, so an SPA hop from /terms back
  // to / would otherwise leave "Terms of Use" in the tab.
  const previousTitle = document.title;
  document.title = `${doc.title} | BioReservoir by IG Digital Lab`;
  root.appendChild(buildNav());
  root.appendChild(buildLegalDoc(doc));
  root.appendChild(buildFooter());
  window.scrollTo(0, 0);
  return () => {
    document.title = previousTitle;
  };
}
