// Parses the `{{highlighted phrase}}` marker copy.json headline fields use to say which phrase
// gets the one-per-headline highlight span (design-style-reference.md: "Exactly one per
// headline... never hardcode [in code] which word is highlighted"). Renders plain text nodes
// around a single `<span class="highlight">`.
import { h } from "./dom";

const MARKER = /\{\{(.+?)\}\}/;

/** Build the child nodes for a headline element (h1/h2/etc.) from marked-up text. If the text
 * has no `{{...}}` marker, renders it as plain text (no highlight) rather than throwing — not
 * every heading needs one, but at most one marker is honoured; a second is left literal so a
 * copy mistake is visible instead of silently eaten. */
export function renderHeadline(text: string): Array<string | HTMLElement> {
  const match = MARKER.exec(text);
  if (!match) return [text];
  const [full, phrase] = match;
  const before = text.slice(0, match.index);
  const after = text.slice(match.index + full.length);
  const nodes: Array<string | HTMLElement> = [];
  if (before) nodes.push(before);
  nodes.push(h("span", { class: "highlight" }, phrase));
  if (after) nodes.push(after);
  return nodes;
}
