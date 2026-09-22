// Who runs this and why it is a real fly brain (2026-09-19: link the company site, say
// this is an IT company doing research, add GitHub and the scientific sources). Citations come from
// NOTICE and docs/DATA.md; the Shiu et al. DOI was checked against nature.com on 2026-09-19.
import { h } from "../dom";
import { copy } from "../content";
import { renderHeadline } from "../headline";

const COMPANY_URL = "https://igdigi.com";
const COMPANY_EMAIL = "hello@igdigi.com";

function external(href: string, text: string, cls = ""): HTMLElement {
  const internal = href.startsWith("/") || href.startsWith("#");
  return h("a", { href, class: cls, ...(internal ? {} : { target: "_blank", rel: "noopener" }) }, text);
}

/** The code link appears only once the public repository exists (VITE_REPO_URL at build time):
 * a link that 404s reads worse than no link. */
export function repoUrl(): string | undefined {
  const url = import.meta.env.VITE_REPO_URL as string | undefined;
  return url && /^https:\/\/github\.com\//.test(url) ? url : undefined;
}

export function buildAboutLab(): HTMLElement {
  const a = copy.about_lab;
  const links: HTMLElement[] = [
    external(COMPANY_URL, a.links.site, "btn"),
    external(`mailto:${COMPANY_EMAIL}`, a.links.email, "btn btn-ghost"),
  ];
  const repo = repoUrl();
  if (repo) links.push(external(repo, a.links.code, "btn btn-ghost"));

  const sources = h(
    "ol",
    { class: "sources-list" },
    a.sources.map((s) =>
      h("li", {}, [
        h("span", { class: "label-mono" }, s.what),
        h("p", {}, [external(s.url, s.cite)]),
        ...(s.extra && s.extra_url ? [h("p", { class: "helper-text" }, external(s.extra_url, s.extra))] : []),
      ]),
    ),
  );

  return h("section", { class: "section", id: "about-lab" }, [
    h("div", { class: "editorial-grid" }, [
      h("h2", {}, renderHeadline(a.title)),
      h("div", { class: "editorial-grid-body" }, [
        h("p", {}, a.body),
        h("div", { class: "about-links" }, links),
        h("h3", { class: "sources-title" }, a.sources_title),
        sources,
      ]),
    ]),
  ]);
}
