import { h } from "../dom";
import { copy } from "../content";

/** Shared attribution/disclaimer footer — required on every page that carries the answer feed.
 * The text lives inside a `.container` div (2026-09-19 layout audit: the footer previously had
 * no gutter or max-width at all, so its text sat flush at x=0 on every page) while the border-top
 * and background stay on the full-bleed `<footer>` itself.
 *
 * 2026-09-21: the long "not affiliated with Kalshi/Polymarket/Manifold" line moved out of here
 * into the Terms (§9). It still needs saying, but saying it in the footer of a page that has
 * nothing to do with prediction markets was the only thing on the page introducing the
 * association. The two document links replace it. */
export function buildFooter(): HTMLElement {
  return h(
    "footer",
    { class: "site-footer" },
    h("div", { class: "container" }, [
      h("p", {}, copy.footer.attribution),
      h("p", {}, copy.footer.disclaimer),
      h("p", { class: "footer-links" }, [
        h("a", { href: "/terms" }, copy.footer.terms_link),
        " · ",
        h("a", { href: "/privacy" }, copy.footer.privacy_link),
        " · ",
        h("a", { href: "mailto:hello@igdigi.com" }, "hello@igdigi.com"),
      ]),
    ]),
  );
}
