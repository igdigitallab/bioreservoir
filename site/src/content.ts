// Typed access to content/copy.json — every editorial string on the site comes from here.
// Two "wired exactly as ... measured/mapped" phrases were replaced with "wired as scientists
// mapped them/it" (2026-09-18, to avoid overclaiming precision), plus a small "sections" object
// this frontend added on top (2026-09-18 redesign) for the three structural section headings the
// copy schema has no field for (How it works / Lab data / Live counters) — see that field's own
// comment. `{{double braces}}` in any headline field mark the one phrase the site's headline
// highlight span applies to (parsed by headline.ts); not hardcoded in TS.
// A handful of short mechanical/technical UI labels this schema has no field for (table column
// headers in the lab readout panel, the "Copy" button on the reproduce block, etc.) are NOT
// editorial copy and stay as plain string literals in the page modules that use them.
import raw from "../content/copy.json";

export interface Copy {
  /** Frontend-authored (2026-09-19, live page v2): the live stage, the visitor's own questions
   * and the recent-answers list. `{placeholders}` are filled with `.replace()`. */
  stage: {
    label_live: string; label_deciding: string; label_yours_deciding: string; label_visitor: string;
    label_replay: string; label_yours: string; label_yours_repeat: string; label_shared: string;
    deciding: string; deciding_short: string; turned: string; sealed: string;
    status_waiting: string; status_empty: string; idle_empty: string; idle_offline: string; loading: string;
  };
  mine: {
    title: string; queued: string; joined: string; thinking: string; answered: string; repeat: string;
    keep_link: string; copy: string; copied: string; watch: string; rejected: string;
  };
  recent: { title: string; sealed: string; empty: string };
  /** The archive of answered questions with its two tabs (2026-09-19). */
  questions: {
    title: string; tab_recent: string; tab_top: string; total: string; more: string;
    like: string; like_failed: string; empty: string; loading: string; offline: string;
  };
  meta: { title: string; description: string; og_title: string; og_description: string };
  hero: { headline: string; subheadline: string; cta: string };
  ask: {
    placeholder: string;
    helper: string;
    rules: string[];
    rules_toggle: string;
    rejected_messages: Record<string, string>;
    /** Shown under the ask box, before anyone types: what becomes public, and what an answer is
     * not (2026-09-21). `terms_note` carries the inline link to /terms. */
    disclaimer: string;
    terms_note: string;
  };
  queue: { title: string; empty: string; position_template: string };
  answer: {
    yes_label: string;
    no_label: string;
    confidence_label: string;
    /** Follows the turn-strength percentage: what 100% means. */
    strength_reference: string;
    caveat: string;
    states: Record<string, { label: string; tooltip: string }>;
    share_text_template: string;
  };
  how_it_works: Array<{ title: string; body: string }>;
  honesty: { title: string; body: string };
  about_lab: {
    title: string;
    body: string;
    links: { site: string; email: string; code: string };
    sources_title: string;
    /** Every citation here is copied from NOTICE / docs/DATA.md / docs/MODEL.md (checked DOIs). */
    sources: Array<{ what: string; cite: string; url: string; extra?: string; extra_url?: string }>;
  };
  footer: { attribution: string; disclaimer: string; terms_link: string; privacy_link: string };
  stream: { headline: string; qr_caption: string; ticker: string[] };
  /** Frontend-added (not from the copywriter's original file, see this file's header comment):
   * plain structural section titles, each with one {{highlight}} marker. */
  sections: { how_it_works: string; validation: string; lab_stats: string };
}

export const copy = raw as Copy;

// Fail loudly in dev if content/copy.json drifts from the shape this file (and every page
// module) assumes — a silently-undefined field used to render as an empty pill/label with no
// error anywhere (see the empty hero CTA bug, 2026-09-18 review). Every path here is a field a
// page module actually reads.
const REQUIRED_PATHS = [
  "meta.title",
  "meta.description",
  "hero.headline",
  "hero.subheadline",
  "hero.cta",
  "ask.placeholder",
  "ask.helper",
  "ask.rules",
  "ask.rejected_messages",
  "ask.disclaimer",
  "ask.terms_note",
  "queue.position_template",
  "answer.yes_label",
  "answer.no_label",
  "answer.confidence_label",
  "answer.states",
  "answer.share_text_template",
  "how_it_works",
  "questions.title",
  "questions.tab_recent",
  "questions.tab_top",
  "questions.total",
  "questions.more",
  "questions.like",
  "questions.like_failed",
  "questions.empty",
  "questions.loading",
  "questions.offline",
  "honesty.title",
  "honesty.body",
  "about_lab.title",
  "about_lab.body",
  "footer.attribution",
  "footer.disclaimer",
  "footer.terms_link",
  "footer.privacy_link",
  "stream.headline",
  "stream.qr_caption",
  "stream.ticker",
  "sections.how_it_works",
  "sections.validation",
  "sections.lab_stats",
];

function resolvePath(obj: unknown, path: string): unknown {
  return path.split(".").reduce<unknown>((acc, key) => {
    if (acc === null || typeof acc !== "object") return undefined;
    return (acc as Record<string, unknown>)[key];
  }, obj);
}

if (import.meta.env.DEV) {
  const missing = REQUIRED_PATHS.filter((path) => resolvePath(copy, path) === undefined);
  if (missing.length > 0) {
    // eslint-disable-next-line no-console
    console.error("content/copy.json is missing required keys:", missing);
    throw new Error(`content/copy.json is missing required keys: ${missing.join(", ")}`);
  }
}
