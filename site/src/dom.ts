// Tiny DOM-builder helper so page modules stay readable without a templating dependency. Always
// uses textContent for dynamic strings (never innerHTML) so user-submitted question text can
// never be interpreted as markup.
type Children = Array<Node | string> | Node | string | undefined;

export function h<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  props: Record<string, string> = {},
  children: Children = [],
): HTMLElementTagNameMap[K] {
  const el = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    // setAttribute (not IDL property assignment) so lowercase HTML attribute names like
    // "maxlength"/"readonly" work correctly regardless of their camelCased DOM property name —
    // https://developer.mozilla.org/en-US/docs/Web/API/Element/setAttribute
    el.setAttribute(key, value);
  }
  const list = Array.isArray(children) ? children : [children];
  for (const child of list) {
    if (child === undefined) continue;
    el.appendChild(typeof child === "string" ? document.createTextNode(child) : child);
  }
  return el;
}

export function clear(el: HTMLElement): void {
  while (el.firstChild) el.removeChild(el.firstChild);
}
